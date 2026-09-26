"""Did the weights actually make the round trip, and is the aggregate really their mean?

Every other check in this project asks whether a run *finished*. This one asks whether the
thing that finished was a federation at all:

1. **Down.** Each client began the round from the aggregate the server had just published.
2. **Up.** Each client returned something different from what it was handed.
3. **The mean.** The published aggregate is the ``num_examples``-weighted mean of what the
   clients sent -- FedAvg's actual arithmetic, not a description of it.

The third is the one that cannot be faked, and it is checkable for one reason: a checksum
here is a **sum over the weight tensors**, so it is linear. If

    w_agg = Σ (n_i / N) · w_i          (what FedAvg claims to compute)

then summing both sides gives

    checksum(w_agg) = Σ (n_i / N) · checksum(w_i)

exactly, in floating point to within accumulation error. A server that dropped a client,
double-counted one, weighted by the wrong number, or returned any client's weights
untouched would break that identity. So would a strategy that is not averaging at all.

**This is a diagnostic, not a proof of correctness.** A checksum is one number: a server
that swapped two tensors between clients would satisfy it. It catches the failures this
project has actually shipped -- a client returning what it was sent, one shard training for
the whole fleet, an aggregate that never moved -- and it catches them from the logs of a
real run, with no instrumentation and no extra GPU time.

Both sides must compute the checksum the same way for the identity to mean anything; they
share ``my_project.get_set_model.learning_checksum``, which sums floating-point tensors
only. Integer BatchNorm counters are excluded there because they climb on their own.

    python -m pipeline.roundtrip            # the last run
    python -m pipeline.roundtrip --json     # for a report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from . import logparse

#: The three lines this reads, each with the timestamp that decides which round it
#: belongs to. `logparse` owns the aggregate's value; the round boundaries are here.
STAMP = r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})"
RECEIVED = re.compile(STAMP + r".*\[Client\] Received weights with checksum: (-?[\d.eE+]+)",
                      re.M)
SENT = re.compile(STAMP + r".*\[Client\] Sending back weights with checksum: "
                  r"(-?[\d.eE+]+)(?: \(num_examples=(\d+)\))?", re.M)
AGGREGATED = re.compile(STAMP + r".*\[Server\] Aggregated parameters with checksum: "
                        r"(-?[\d.eE+]+)", re.M)


def _ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f").timestamp()

#: Floating-point summation over 355 tensors and six clients does not associate, so the
#: identity holds to a relative tolerance rather than exactly. 1e-6 is far tighter than any
#: real defect -- a dropped client moves the mean by percent, not by parts per million.
TOLERANCE = 1e-6


@dataclass
class Round:
    number: int
    aggregate: float
    sent: list[tuple[float, int]] = field(default_factory=list)   # (checksum, num_examples)
    received: list[float] = field(default_factory=list)

    @property
    def weighted_mean(self) -> float | None:
        total = sum(n for _, n in self.sent)
        if not total:
            return None
        return sum(c * n for c, n in self.sent) / total

    @property
    def deviation(self) -> float | None:
        mean = self.weighted_mean
        if mean is None:
            return None
        scale = max(abs(self.aggregate), abs(mean), 1e-9)
        return abs(self.aggregate - mean) / scale


def collect(log_dir: Path | None = None) -> list[Round]:
    """One entry per round of the most recent run, from that run's logs only.

    **Rounds are decided by timestamp, not by position in a file**, and that is not a
    refinement -- the first version of this counted the k-th `Sending back` line in each
    client log as round k, and reported "1 client" for a six-client run. Ray does not give
    each client its own process: at `--gpu-fraction 1.0` one `ClientAppActor` serves all
    six in turn, so a single log holds 6 x rounds send lines and a positional read collapses
    the fleet to one vehicle. A line belongs to the first round whose aggregation happened
    after it.

    `logparse.current_run_logs` bounds the run by the newest server log that recorded an
    aggregation, which is what keeps a previous run's clients out of this one's arithmetic.
    """
    server = logparse.latest_run_log(log_dir)
    if server is None:
        return []
    marks = [(_ts(m.group(1)), float(m.group(2)))
             for m in AGGREGATED.finditer(server.read_text(errors="replace"))]
    if not marks:
        return []
    rounds = [Round(i + 1, value) for i, (_, value) in enumerate(marks)]

    def bucket(when: float) -> Round | None:
        for r, (at, _) in zip(rounds, marks):
            if when <= at:
                return r
        return None            # after the last aggregation: evaluation, not a fit round

    for f in logparse.current_run_logs("client*.log", log_dir):
        text = f.read_text(errors="replace")
        for m in SENT.finditer(text):
            r = bucket(_ts(m.group(1)))
            if r is not None:
                r.sent.append((float(m.group(2)), int(m.group(3) or 0)))
        for m in RECEIVED.finditer(text):
            r = bucket(_ts(m.group(1)))
            if r is not None:
                r.received.append(float(m.group(2)))
    return rounds


#: Distinct from pass and from fail. A run recorded before the client logged `num_examples`
#: cannot be checked, and saying FAIL there would be a false alarm -- as bad as a silent
#: pass, and the faster way to teach someone to ignore the check.
CANNOT_CHECK = 2


def checkable(rounds: list[Round]) -> bool:
    """Whether these logs carry what the identity needs. Old runs do not."""
    return bool(rounds) and any(n for r in rounds for _, n in r.sent)


def check(rounds: list[Round]) -> tuple[bool, list[str]]:
    """The three questions, answered against `rounds`. Returns (ok, lines to print)."""
    out: list[str] = []
    if not rounds:
        return False, ["no rounds found in this run's logs -- nothing to check"]

    ok = True
    for r in rounds:
        mean = r.weighted_mean
        if mean is None:
            ok = False
            out.append(f"  round {r.number}: no client `Sending back` lines carrying "
                       f"num_examples. Re-run: the identity needs FedAvg's own weights")
            continue

        moved = [c for c, _ in r.sent if not any(abs(c - got) < 1e-12 for got in r.received)]
        dev = r.deviation
        agrees = dev is not None and dev < TOLERANCE
        ok = ok and agrees and len(moved) == len(r.sent)

        out.append(
            f"  round {r.number}: {len(r.sent)} client(s), "
            f"weighted mean {mean:.9f} vs aggregate {r.aggregate:.9f}, "
            f"relative deviation {dev:.2e} {'OK' if agrees else 'MISMATCH'}")
        if len(moved) != len(r.sent):
            out.append(f"    ! {len(r.sent) - len(moved)} client(s) returned exactly what "
                       f"they were sent -- that is the B4 failure")

    # Down: round k+1's clients must start from round k's aggregate.
    for prev, nxt in zip(rounds, rounds[1:]):
        started_from = [got for got in nxt.received
                        if abs(got - prev.aggregate) < max(abs(prev.aggregate), 1.0) * 1e-9]
        if nxt.received and not started_from:
            ok = False
            out.append(f"    ! round {nxt.number}'s clients did not start from round "
                       f"{prev.number}'s aggregate ({prev.aggregate:.6f}); "
                       f"they received {sorted(set(nxt.received))[:3]}")
    return ok, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable, for the report")
    args = ap.parse_args(argv)

    rounds = collect()
    if rounds and not checkable(rounds):
        msg = ("this run predates the round-trip check: its client logs carry no "
               "`num_examples`, and its checksums were summed over integer buffers too, "
               "so neither the identity nor the hand-off can be tested against them. "
               "Re-run to check it.")
        if args.json:
            print(json.dumps({"ok": None, "cannot_check": msg}, indent=2))
        else:
            print("ROUND TRIP: CANNOT CHECK")
            print(f"  {msg}")
        return CANNOT_CHECK

    ok, lines = check(rounds)

    if args.json:
        print(json.dumps({
            "ok": ok,
            "tolerance": TOLERANCE,
            "rounds": [{"round": r.number, "aggregate": r.aggregate,
                        "weighted_mean": r.weighted_mean, "deviation": r.deviation,
                        "clients": len(r.sent),
                        "num_examples": [n for _, n in r.sent]} for r in rounds],
        }, indent=2))
        return 0 if ok else 1

    print("Did the weights make the round trip, and is the aggregate their mean?")
    for line in lines:
        print(line)
    print(f"ROUND TRIP: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
