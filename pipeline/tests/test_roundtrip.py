"""The round-trip check must catch the failures this project has actually shipped.

Each test writes the logs a particular federation would have written and asserts the
verdict. Synthetic on purpose: a real run cannot be asked to drop a client on command, and
the point is to prove the check FAILS when it should, which a healthy run can never show.
"""
import pytest

from pipeline import roundtrip


def _stamp(minute: int, second: int) -> str:
    return f"2026-09-26 12:{minute:02d}:{second:02d},000"


def _logs(tmp_path, aggregates, per_round_sent, received=None, one_process=False):
    """Write the logs a run would leave, timestamped, because rounds are read from time.

    `one_process=True` puts every client's lines in a SINGLE file, which is what Ray
    actually does at `--gpu-fraction 1.0`: one `ClientAppActor` serves all six vehicles in
    turn. Measured on a real run -- 12 `Sending back` lines in one file for six clients over
    two rounds.
    """
    d = tmp_path / "logs"
    d.mkdir(exist_ok=True)
    # Round r is aggregated at 12:r0:30; every client line for that round precedes it.
    (d / "server.1.log").write_text("\n".join(
        f"{_stamp(i, 30)} - INFO - [Server] Aggregated parameters with checksum: {a}"
        for i, a in enumerate(aggregates)), encoding="utf-8")

    n_clients = len(per_round_sent[0])
    per_file: dict[str, list[str]] = {}
    for c in range(n_clients):
        for r, sent in enumerate(per_round_sent):
            got = received[r][c] if received else (aggregates[r - 1] if r else 0.0)
            checksum, examples = sent[c]
            name = "client.10.log" if one_process else f"client.{c + 10}.log"
            per_file.setdefault(name, []).extend([
                f"{_stamp(r, c)} - INFO - [Client] Received weights with checksum: {got}",
                f"{_stamp(r, c + 1)} - INFO - [Client] Sending back weights with "
                f"checksum: {checksum} (num_examples={examples})",
            ])
    for name, lines in per_file.items():
        (d / name).write_text("\n".join(sorted(lines)), encoding="utf-8")
    return d


def _verdict(tmp_path, monkeypatch, **kw):
    d = _logs(tmp_path, **kw)
    # `collect()` takes the directory, so no patching is needed -- and passing it explicitly
    # is also what keeps this test off whatever logs happen to be on the machine.
    rounds = roundtrip.collect(d)
    return roundtrip.check(rounds), rounds


def test_an_honest_federation_passes():
    """Equal shards, so the weighted mean is the plain mean: (10+20+30)/3 = 20."""
    r = roundtrip.Round(1, aggregate=20.0, sent=[(10.0, 1400), (20.0, 1400), (30.0, 1400)],
                        received=[0.0, 0.0, 0.0])
    ok, lines = roundtrip.check([r])
    assert ok, lines
    assert r.deviation < roundtrip.TOLERANCE


def test_the_weighting_is_by_num_examples_not_by_client_count():
    """FedAvg weights by shard size. A server that averaged evenly would be wrong here by
    3.3 %, and this is the arithmetic that says so: (10*1000 + 30*3000)/4000 = 25, not 20."""
    weighted = roundtrip.Round(1, aggregate=25.0, sent=[(10.0, 1000), (30.0, 3000)],
                               received=[0.0, 0.0])
    evenly = roundtrip.Round(1, aggregate=20.0, sent=[(10.0, 1000), (30.0, 3000)],
                             received=[0.0, 0.0])

    assert roundtrip.check([weighted])[0]
    assert not roundtrip.check([evenly])[0], "an unweighted mean must not pass"


def test_a_dropped_client_is_caught():
    """The failure that looks like a working run: the server aggregates five of six, the
    metrics are fine, and the fleet is quietly training on less than it thinks."""
    sent = [(10.0, 1400), (20.0, 1400), (30.0, 1400)]
    dropped_mean = (10.0 + 20.0) / 2          # the third client never made it in
    r = roundtrip.Round(1, aggregate=dropped_mean, sent=sent, received=[0.0, 0.0, 0.0])

    ok, lines = roundtrip.check([r])
    assert not ok
    assert any("MISMATCH" in l for l in lines)


def test_a_client_returning_what_it_was_sent_is_caught():
    """B4, exactly: the client hands back the aggregate it received. The mean still equals
    the aggregate -- the identity alone would pass this -- so the check needs the second
    question as well as the third."""
    r = roundtrip.Round(1, aggregate=10.0, sent=[(10.0, 1400), (10.0, 1400)],
                        received=[10.0, 10.0])

    ok, lines = roundtrip.check([r])
    assert not ok
    assert any("returned exactly what they were sent" in l for l in lines)


def test_clients_that_did_not_start_from_the_last_aggregate_are_caught():
    """The hand-off. If round 2 begins from anything but round 1's aggregate, the rounds
    are not a sequence and the federation is a set of unrelated local trainings."""
    r1 = roundtrip.Round(1, aggregate=20.0, sent=[(10.0, 1), (30.0, 1)], received=[0.0, 0.0])
    r2 = roundtrip.Round(2, aggregate=40.0, sent=[(30.0, 1), (50.0, 1)],
                         received=[999.0, 999.0])      # not 20.0

    ok, lines = roundtrip.check([r1, r2])
    assert not ok
    assert any("did not start from round 1's aggregate" in l for l in lines)


def test_a_run_from_before_the_check_says_so_instead_of_failing():
    """A false alarm teaches people to ignore the check. Logs with no `num_examples` cannot
    be tested, and that is a third state, not a failure."""
    old = roundtrip.Round(1, aggregate=1.0, sent=[(2.0, 0)], received=[0.0])
    assert not roundtrip.checkable([old])
    assert roundtrip.CANNOT_CHECK not in (0, 1), "it must not collide with pass or fail"


def test_it_reads_real_log_lines_end_to_end(tmp_path, monkeypatch):
    """The parsing half. Everything above builds `Round` objects by hand; this one proves
    the regexes match what the client and server actually write."""
    (ok, lines), rounds = _verdict(
        tmp_path, monkeypatch,
        aggregates=[20.0, 40.0],
        per_round_sent=[[(10.0, 1400), (30.0, 1400)], [(30.0, 1400), (50.0, 1400)]])

    assert len(rounds) == 2
    assert [len(r.sent) for r in rounds] == [2, 2]
    assert rounds[1].received == [20.0, 20.0], "round 2 must start from round 1's aggregate"
    assert ok, lines


def test_six_clients_sharing_one_ray_actor_are_still_six_clients(tmp_path, monkeypatch):
    """Ray does not give each client its own process.

    At `--gpu-fraction 1.0` a single `ClientAppActor` serves all six vehicles in turn, so
    one log file holds 6 x rounds send lines -- measured: 12 in one file for a six-client
    two-round run. The first version of this reader took the k-th line in a file as round k,
    which folded the fleet into one vehicle and reported MISMATCH against a healthy run.
    Rounds are decided by timestamp now, and this is the layout that proves it.
    """
    d = _logs(tmp_path,
              aggregates=[20.0, 40.0],
              per_round_sent=[[(10.0, 1400), (30.0, 1400)], [(30.0, 1400), (50.0, 1400)]],
              one_process=True)

    assert len(list(d.glob("client*.log"))) == 1, "the fixture must put them in one file"
    rounds = roundtrip.collect(d)

    assert [len(r.sent) for r in rounds] == [2, 2], "both clients seen in both rounds"
    ok, lines = roundtrip.check(rounds)
    assert ok, lines
