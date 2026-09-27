"""Assert the five pass criteria. Same checks the CI simulation-smoke job uses.

Criterion 1 is the one that matters: identical round-over-round aggregate checksums
mean the clients returned the weights they were handed and FedAvg averaged its own
input. The run still reports success in that state, which is exactly the trap.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import logparse, paths, roundtrip


def metrics_csv() -> Path:
    """logs/metrics.csv from whichever log dir actually has one."""
    for d in paths.log_dirs():
        if (d / "metrics.csv").exists():
            return d / "metrics.csv"
    return paths.PROJECT / "logs" / "metrics.csv"


def check(project: Path = paths.PROJECT) -> tuple[bool, list[str]]:
    results = []
    ok = True

    learned, detail = logparse.federation_learned()
    results.append(f"[{'PASS' if learned else 'FAIL'}] global weights change between rounds: {detail}")
    ok &= learned

    sent, recv = [], []
    for f in logparse.iter_logs("client*.log"):
        evs = logparse.parse_text(f.read_text(errors="replace"))
        sent += [e.value for e in evs if e.kind == "client_sent_checksum"]
        recv += [e.value for e in evs if e.kind == "client_received_checksum"]
    c2 = bool(sent and recv and set(sent) != set(recv))
    results.append(f"[{'PASS' if c2 else 'FAIL'}] clients return something other than what they got "
                   f"({len(recv)} received, {len(sent)} sent)")
    ok &= c2

    rows = logparse.read_metrics_csv(metrics_csv())
    maps = [r.get("mAP50") for r in rows if r.get("mAP50") is not None]
    c3 = len(rows) >= 2 and any(m and m > 0 for m in maps)
    results.append(f"[{'PASS' if c3 else 'FAIL'}] metrics.csv has rows with non-zero mAP50 "
                   f"({len(rows)} rows, max mAP50 {max(maps) if maps else 0})")
    ok &= c3

    ckpts = sorted((project / "checkpoints").glob("global_*.pt"))
    c4 = any(p.name == "global_last.pt" for p in ckpts) and len(ckpts) >= 2
    results.append(f"[{'PASS' if c4 else 'FAIL'}] checkpoints written ({len(ckpts)}: "
                   f"{', '.join(p.name for p in ckpts[:4])})")
    ok &= c4

    # Criterion 5: the aggregate really is the weighted mean of what the clients sent.
    # Criteria 1 and 2 say the numbers moved and differ; this one says the server did
    # FedAvg's arithmetic on them. A run recorded before the client logged num_examples
    # cannot be checked, and that is reported as a WARN rather than counted as a failure --
    # a false alarm is the fastest way to teach someone to ignore a check.
    rounds = roundtrip.collect()
    if not roundtrip.checkable(rounds):
        results.append("[WARN] round trip not checkable: this run's client logs carry no "
                       "num_examples, so FedAvg's own weights are unknown")
    else:
        c5, detail = roundtrip.check(rounds)
        results.append(f"[{'PASS' if c5 else 'FAIL'}] the aggregate is the weighted mean of "
                       f"what the clients sent")
        results += [f"   {line.strip()}" for line in detail]
        ok &= c5

    # Not a pass criterion, but a silent-failure warning worth surfacing.
    noop = [e for f in logparse.iter_logs("client*.log")
            for e in logparse.parse_text(f.read_text(errors="replace"))
            if e.kind == "no_optimizer_step"]
    if noop:
        results.append(f"[WARN] {len(noop)} round(s) could not take an optimizer step -- "
                       f"raise local_epochs or lower the batch size")
    return ok, results


def main() -> int:
    ok, results = check()
    for line in results:
        print(line)
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
