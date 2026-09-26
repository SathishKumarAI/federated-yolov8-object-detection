"""Every run this project has done, as one comparable record.

A run's facts are scattered across the report it wrote, the holdout curve, the
centralised ceiling and the stage timings. Answering "which approach was worth it"
meant opening four files per run and holding the arithmetic in your head.

This reads them into one row per run: what approach it was, how much data it used,
how long it took, what it cost, and what it produced -- including the per-epoch
learning inside each round, which is where a plateau shows up long before the
round-level metric admits it.

Nothing here recomputes anything. It reads what the runs already wrote, so a row can
never disagree with the report it came from.

    python -m pipeline.ledger
    python -m pipeline.ledger --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import paths

#: What distinguishes one approach from another. Two runs with the same signature
#: are repeats; anything else is a different experiment.
APPROACH_KEYS = ("strategy", "partition", "alpha", "size_skew", "profile", "n_vehicles",
                 "rounds", "local_epochs", "per_vehicle", "seed")

#: Run levers that only exist after 2026-09-26. Their presence in a report's config is
#: the discriminator for which transport produced the run -- see `era` below.
LEVER_KEYS = ("freeze_round1", "server_ema", "fix_bn_from_round", "imgsz", "local_bn")


def era(cfg: dict) -> dict:
    """Which weights this run's clients actually sent back.

    Until 2026-09-26 every client returned the fp16-rounded EMA of its own `best.pt`
    rather than the weights it had trained, and FedAvg averaged that. Runs from before
    and after that fix are measurements of two different systems, so plotting them as one
    series compares nothing -- and there is no timestamp to sort on, because a report's
    `generated` field says when the report was written, not what code wrote it.

    The lever fields are the usable discriminator: they cannot appear in a config a
    pre-fix runner produced.
    """
    new_levers = [k for k in LEVER_KEYS if k in cfg]
    if new_levers:
        return {"transport": "trained-fp32", "since": "2026-09-26",
                "evidence": f"config carries {', '.join(sorted(new_levers))}",
                "comparable_with_older": False,
                "note": "Clients return the fp32 weights they trained."}
    return {"transport": "ema-fp16-best", "since": None,
            "evidence": "config carries none of the post-2026-09-26 run levers",
            "comparable_with_older": False,
            "note": "Clients returned the fp16-rounded EMA of their own best.pt, chosen "
                    "on a 280-image local val split, and FedAvg averaged that. Not "
                    "comparable with a run made after the 2026-09-26 transport fix."}


def _report_files(limit: int | None = None) -> list[Path]:
    if not paths.REPORTS.is_dir():
        return []
    files = [d / "report.json" for d in sorted(paths.REPORTS.iterdir())
             if (d / "report.json").is_file()]
    return files[-limit:] if limit else files


def approach(cfg: dict) -> str:
    """A short label for the thing that was tried."""
    bits = [str(cfg.get("strategy") or "fedavg"), str(cfg.get("partition") or "?")]
    if cfg.get("partition") == "dirichlet":
        bits.append(f"a={cfg.get('alpha')}")
    if cfg.get("size_skew"):
        bits.append(f"skew={cfg['size_skew']:g}")
    bits.append(f"{cfg.get('n_vehicles', '?')}v")
    bits.append(f"{cfg.get('rounds', '?')}x{cfg.get('local_epochs', '?')}")
    bits.append(f"{cfg.get('per_vehicle', '?')}img")
    return " · ".join(bits)


def row(data: dict, name: str) -> dict:
    """One run, reduced to what a comparison actually needs."""
    cfg = data.get("config") or {}
    gpu = data.get("gpu") or {}
    stages = data.get("stages") or []
    learning = data.get("learning") or {}
    holdout_rounds = ((data.get("holdout") or {}).get("rounds")) or []
    self_eval = [r.get("mAP50") for r in (data.get("metrics") or [])
                 if r.get("stage") == "evaluate" and r.get("mAP50") is not None]

    images = (cfg.get("n_vehicles") or 0) * (cfg.get("per_vehicle") or 0)
    visits = images * (cfg.get("rounds") or 0) * (cfg.get("local_epochs") or 0)
    seconds = sum(s.get("seconds") or 0 for s in stages)

    # Per-epoch rows are the finest grain this project records: within one round, is
    # the loss still falling, or has the client stopped learning and started
    # drifting from the others?
    epochs = {}
    for vid, rows in (learning.get("epochs") or {}).items():
        epochs[vid] = [{"epoch": i + 1,
                        "box": r.get("box_loss"), "cls": r.get("cls_loss"),
                        "dfl": r.get("dfl_loss"), "mAP50": r.get("mAP50")}
                       for i, r in enumerate(rows)]

    best = max((r["mAP50"] for r in holdout_rounds), default=None)
    ceiling = (data.get("baseline") or {})
    return {
        "run": name,
        "generated": data.get("generated"),
        "approach": approach(cfg),
        "config": {k: cfg.get(k) for k in APPROACH_KEYS},
        "levers": {k: cfg[k] for k in LEVER_KEYS if k in cfg},
        # Not cosmetic: two rows in different eras are two different systems, and the
        # Metrics tab refuses to plot them as one series.
        "era": era(cfg),
        "data": {"images": images, "image_visits": visits,
                 "per_vehicle": cfg.get("per_vehicle"),
                 "vehicles": cfg.get("n_vehicles"),
                 "effective_epochs": (cfg.get("rounds") or 0) * (cfg.get("local_epochs") or 0)},
        "time": {"seconds": round(seconds, 1),
                 "per_stage": [{"name": s.get("name"), "seconds": round(s.get("seconds") or 0, 1),
                                "status": s.get("status")} for s in stages],
                 "seconds_per_kvisit": round(seconds / (visits / 1000), 2) if visits else None},
        "cost": {"energy_wh": gpu.get("energy_wh"), "peak_mem_mib": gpu.get("peak_mem_mib"),
                 "mean_util_pct": gpu.get("mean_util_pct"),
                 "wh_per_point": (round(gpu["energy_wh"] / best, 1)
                                  if best and gpu.get("energy_wh") else None)},
        "result": {"holdout_mAP50": best,
                   "holdout_curve": [r["mAP50"] for r in holdout_rounds],
                   "holdout_mAP50_95": max((r["mAP50-95"] for r in holdout_rounds), default=None),
                   "self_mAP50": max(self_eval, default=None),
                   "checksums": data.get("checksums") or [],
                   "learned": bool(data.get("learned")),
                   "ceiling_mAP50": ceiling.get("centralised_mAP50"),
                   "retained": ceiling.get("retained"),
                   "budget_matched": ceiling.get("matched"),
                   # A federation above its own ceiling means the ceiling is not one:
                   # a stale baseline.json from a different scale, or a budget the
                   # centralised model never got. Said out loud rather than plotted.
                   "ceiling_suspect": bool(ceiling.get("retained") and ceiling["retained"] > 1.0)},
        "learning": {"per_vehicle_rounds": learning.get("rounds") or {},
                     "conditions": learning.get("conditions") or {},
                     "epochs": epochs,
                     "trained": learning.get("trained") or []},
    }


def load(limit: int | None = None) -> list[dict]:
    out = []
    for f in _report_files(limit):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(row(data, f.parent.name))
    return out


def by_approach(rows: list[dict] | None = None) -> list[dict]:
    """Runs grouped by what was tried, so repeats sit together.

    One run is an anecdote; this is where a seed repeat stops looking like a result.
    """
    rows = load() if rows is None else rows
    groups: dict[str, dict] = {}
    for r in rows:
        key = r["approach"]
        g = groups.setdefault(key, {"approach": key, "runs": [], "scores": []})
        g["runs"].append(r["run"])
        if r["result"]["holdout_mAP50"] is not None:
            g["scores"].append(r["result"]["holdout_mAP50"])
    for g in groups.values():
        s = g["scores"]
        g["n"] = len(g["runs"])
        g["best"] = max(s) if s else None
        g["mean"] = round(sum(s) / len(s), 4) if s else None
        g["spread"] = round(max(s) - min(s), 4) if len(s) > 1 else None
    return sorted(groups.values(), key=lambda g: (g["mean"] is None, -(g["mean"] or 0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--last", type=int, default=0, help="only the N most recent runs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rows = load(args.last or None)
    if args.json:
        print(json.dumps({"runs": rows, "approaches": by_approach(rows)}, indent=1))
        return 0
    if not rows:
        print("No runs recorded yet. Every run writes one to pipeline/reports/.")
        return 0

    print(f"{len(rows)} run(s)\n")
    head = f"{'run':<17}{'approach':<44}{'visits':>10}{'min':>7}{'Wh':>7}{'holdout':>9}{'of ceiling':>12}"
    print(head)
    print("-" * len(head))
    for r in rows:
        res, cost = r["result"], r["cost"]
        retained = (f"{100 * res['retained']:.1f}%" if res.get("retained") else "-")
        if res.get("retained") and res.get("budget_matched") is False:
            retained += "*"
        print(f"{r['run']:<17}{r['approach'][:43]:<44}"
              f"{r['data']['image_visits']:>10,}"
              f"{r['time']['seconds'] / 60:>7.1f}"
              f"{(cost['energy_wh'] or 0):>7.1f}"
              f"{(res['holdout_mAP50'] if res['holdout_mAP50'] is not None else float('nan')):>9.4f}"
              f"{retained:>12}")
    print("\n* the ceiling had a different budget, so the figure is a bound not a result")

    groups = by_approach(rows)
    if any(g["n"] > 1 for g in groups):
        print("\nby approach:")
        for g in groups:
            spread = f", spread {g['spread']}" if g["spread"] is not None else ""
            print(f"  {g['approach'][:52]:<54} n={g['n']} mean {g['mean']}{spread}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
