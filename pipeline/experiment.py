"""Run a set of configurations back to back and put the results side by side.

One run is an anecdote. The questions this project actually has -- does the strategy
matter, does the partition matter, how much of the difference is just the seed --
are all comparisons, and doing them by hand means remembering to change one thing at
a time, to re-evaluate on the holdout each time, and to keep the reports apart.

This does that. It is a thin driver over ``pipeline.runner``: every arm is a normal
run that writes a normal report, so nothing here is a second code path that can drift
from the real one.

    python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes
    python -m pipeline.experiment --preset strategies --strategies fedavg,fedadam --yes
    python -m pipeline.experiment --preset partitions --profile demo --yes
    python -m pipeline.experiment --arms arms.json --yes        # anything else

Every arm runs the full chain including `evaluate`, so each one ends with a number
measured on the shared holdout -- the only number that can honestly be compared
between arms.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import compare, holdout, paths, vehicles
from .stages import Config

RESULTS = paths.STATE / "experiments"


def arms_for(preset: str, base: dict, seeds: list[int], strategies: list[str],
             partitions: list[str], alphas: list[float],
             skews: list[float] | None = None) -> list[dict]:
    """Expand a preset into arms that differ in exactly one setting."""
    if preset == "seeds":
        return [{**base, "seed": s, "label": f"seed={s}"} for s in seeds]
    if preset == "strategies":
        return [{**base, "strategy": s, "label": f"strategy={s}"} for s in strategies]
    if preset == "partitions":
        return [{**base, "partition": p, "label": f"partition={p}"} for p in partitions]
    if preset == "alpha":
        return [{**base, "partition": "dirichlet", "alpha": a, "label": f"alpha={a}"}
                for a in alphas]
    if preset == "skew":
        # Every arm makes the same image-visits -- skewed_sizes preserves the fleet
        # total -- so this sweep isolates unequal `num_examples`, which is FedAvg's
        # aggregation weight, from the amount of data.
        return [{**base, "size_skew": s, "label": f"skew={s}"} for s in (skews or [])]
    raise SystemExit(
        f"unknown preset {preset!r}; known: seeds, strategies, partitions, alpha, skew")


def command(arm: dict, confirm: bool) -> list[str]:
    cmd = [sys.executable, "-m", "pipeline.runner", "--all",
           "--profile", str(arm.get("profile", "demo")),
           "--vehicles", str(arm.get("vehicles", 6)),
           "--rounds", str(arm.get("rounds", 2)),
           "--epochs", str(arm.get("epochs", 1)),
           "--seed", str(arm.get("seed", 0)),
           # Config.partition, not a literal. This was hardcoded "condition"; when the
           # default moved to "random" every arm of every preset would have silently
           # rebuilt the fleet as non-IID -- including the seed sweep whose entire job
           # is to hold everything except the seed constant.
           "--partition", str(arm.get("partition", Config.partition)),
           "--alpha", str(arm.get("alpha", 0.5)),
           "--size-skew", str(arm.get("size_skew", 0.0)),
           "--strategy", str(arm.get("strategy", "fedavg")),
           # Never passed before, so every arm ran at 1.0 and took ~1.9x longer than
           # the same run launched by hand.
           "--gpu-fraction", str(arm.get("gpu_fraction", 1.0))]
    if arm.get("per_vehicle"):
        cmd += ["--per-vehicle", str(arm["per_vehicle"])]
    if arm.get("local_bn"):
        cmd.append("--local-bn")
    # `--all` includes the gated `baseline` stage, which trains a whole centralised
    # model. With --yes that fires once per arm: a three-seed sweep would train three
    # identical ceilings to measure a spread the ceiling plays no part in.
    if arm.get("skip"):
        cmd += ["--skip", str(arm["skip"])]
    if confirm:
        cmd.append("--yes")
    return cmd


def run(arms: list[dict], confirm: bool, dry_run: bool = False) -> list[dict]:
    """Run each arm in turn. A failed arm is recorded, not fatal: the rest still
    produce numbers, and an experiment that dies on arm 2 of 5 has wasted the GPU
    time already spent on arm 1."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = []
    for i, arm in enumerate(arms, 1):
        cmd = command(arm, confirm)
        print(f"\n=== arm {i}/{len(arms)}: {arm.get('label', '')} ===")
        print("    " + " ".join(cmd))
        if dry_run:
            done.append({**arm, "status": "dry-run"})
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=paths.REPO)
        elapsed = round(time.time() - t0, 1)
        report = latest_report()
        done.append({**arm, "status": "ok" if proc.returncode == 0 else "failed",
                     "exit": proc.returncode, "seconds": elapsed,
                     "report": report.name if report else None})
        print(f"    -> {done[-1]['status']} in {elapsed}s, report {done[-1]['report']}")
    return done


def latest_report() -> Path | None:
    if not paths.REPORTS.is_dir():
        return None
    dirs = [d for d in sorted(paths.REPORTS.iterdir()) if (d / "report.json").is_file()]
    return dirs[-1] if dirs else None


def summarise(done: list[dict]) -> str:
    """The arms, their holdout numbers, and what differed between them."""
    runs = []
    for arm in done:
        if not arm.get("report"):
            continue
        f = paths.REPORTS / arm["report"] / "report.json"
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append({"run": arm.get("label") or arm["report"], **compare.summarise(data)})
    if not runs:
        return "No arm produced a readable report."
    return compare.table(runs, markdown=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=("seeds", "strategies", "partitions", "alpha", "skew"),
                    help="what to vary; every arm changes exactly one setting")
    ap.add_argument("--arms", type=Path, help="a JSON list of arms, for anything else")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--strategies", default="fedavg,fedadam,fedavgm")
    ap.add_argument("--partitions", default="condition,random,dirichlet")
    ap.add_argument("--alphas", default="0.05,0.5,100")
    ap.add_argument("--skews", default="0,0.8,1.5",
                    help="quantity-skew sweep; every arm keeps the same fleet total")
    ap.add_argument("--profile", default="demo", choices=("demo", "full"))
    ap.add_argument("--vehicles", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per-vehicle", type=int, default=0)
    ap.add_argument("--partition", default=Config.partition, choices=vehicles.PARTITIONS,
                    help="held constant across arms, except in the `partitions` preset")
    ap.add_argument("--gpu-fraction", type=float, default=1.0,
                    help="passed to every arm; 0.5 runs two clients at once")
    ap.add_argument("--local-bn", action="store_true",
                    help="FedBN in every arm")
    ap.add_argument("--skip", default="baseline",
                    help="stages to leave out of every arm. Defaults to `baseline`: "
                         "`--all` would otherwise retrain the centralised ceiling once "
                         "per arm, and the ceiling does not vary with the thing being "
                         "swept. Pass an empty string to include it")
    ap.add_argument("--yes", action="store_true", help="confirm the gated stages in every arm")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = ap.parse_args(argv)

    base = {"profile": args.profile, "vehicles": args.vehicles, "rounds": args.rounds,
            "epochs": args.epochs, "per_vehicle": args.per_vehicle,
            "partition": args.partition, "gpu_fraction": args.gpu_fraction,
            "local_bn": args.local_bn, "skip": args.skip}
    if args.arms:
        arms = json.loads(args.arms.read_text())
        arms = [{**base, **a} for a in arms]
    elif args.preset:
        arms = arms_for(args.preset, base,
                        [int(s) for s in args.seeds.split(",") if s.strip()],
                        [s.strip() for s in args.strategies.split(",") if s.strip()],
                        [s.strip() for s in args.partitions.split(",") if s.strip()],
                        [float(a) for a in args.alphas.split(",") if a.strip()],
                        [float(s) for s in args.skews.split(",") if s.strip()])
    else:
        ap.error("pass --preset or --arms")

    if not holdout.names() and not args.dry_run:
        print("No holdout carved yet. Every arm would produce a number that cannot be "
              "compared with the others. Run:\n  python -m pipeline.holdout --build\n")
        return 2

    print(f"{len(arms)} arm(s); each runs the full chain and ends with a holdout score.")
    done = run(arms, confirm=args.yes, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = summarise(done)
    (RESULTS / f"{stamp}.json").write_text(json.dumps(done, indent=1))
    (RESULTS / f"{stamp}.md").write_text(table + "\n")

    print("\n" + table)
    print(f"\nwritten: {RESULTS / (stamp + '.md')}")
    failed = [a for a in done if a.get("status") == "failed"]
    if failed:
        print(f"\n{len(failed)} arm(s) FAILED: " +
              ", ".join(a.get("label", "?") for a in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
