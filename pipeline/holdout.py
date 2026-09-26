"""The shared holdout: one val set no vehicle trains or self-evaluates on.

Every number this project reported until now came from a client scoring itself on
its own val split. Vehicle 3 trains on rain and is measured on rain; vehicle 1
trains on daytime city and is measured on daytime city. Those numbers cannot be
compared with each other, and averaging them is not a global metric -- it is a mean
over different distributions.

This module carves a fixed slice out of the val pool **before** the fleet is
assigned, materialises it once, and scores the global checkpoints on it out of
band. Out of band matters: server-side evaluation lives in ``my-project`` and this
component is not allowed to modify it, but the checkpoints are already on disk and
Ultralytics can read them without the federation knowing.

    python -m pipeline.holdout --build --size 1000
    python -m pipeline.holdout --evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from . import paths, vehicles

HOLDOUT_ROOT = paths.VEHICLE_ROOT / "holdout"          # gitignored with the rest
NAMES_FILE = paths.STATE / "holdout.json"
METRICS_FILE = paths.STATE / "holdout_metrics.json"


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select(size: int = 1000, seed: int = 0, val_pool: set[str] | None = None) -> list[str]:
    """Pick the holdout deterministically from the val pool.

    Sorted before shuffling on purpose: ``set`` iteration order varies between
    processes, so shuffling the raw set would give a different holdout on every
    machine and make two runs incomparable without either of them looking wrong.
    """
    pool = sorted(vehicles._available(Path("val")) if val_pool is None else val_pool)
    if len(pool) < size:
        raise SystemExit(f"holdout needs {size} val images, the pool has {len(pool)}. "
                         f"Run the populate stage, or ask for fewer.")
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:size])


def fingerprint(chosen: list[str] | None = None) -> str:
    """Content hash of exactly which images the holdout holds.

    ``size`` and ``seed`` describe how it was *asked* for, not what it *is*: the same
    size and seed drawn from a val pool that has since grown gives a different holdout
    and records identical metadata. Every number this project reports is measured on
    this slice, so two runs claiming the same holdout need something checkable, on the
    same principle as ``Vehicle.fingerprint``.

    Names, not bytes -- the images are hardlinks onto a read-only cache.
    """
    chosen = sorted(names()) if chosen is None else sorted(chosen)
    return hashlib.sha256("\n".join(chosen).encode()).hexdigest()[:12]


def names() -> set[str]:
    """The current holdout, or an empty set if none has been carved yet."""
    if not NAMES_FILE.exists():
        return set()
    try:
        return set(json.loads(NAMES_FILE.read_text()).get("names", []))
    except (OSError, json.JSONDecodeError):
        return set()


def meta() -> dict:
    if not NAMES_FILE.exists():
        return {}
    try:
        data = json.loads(NAMES_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if k != "names"}


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------
def data_yaml() -> Path:
    return HOLDOUT_ROOT / "data.yaml"


def build(size: int = 1000, seed: int = 0, class_names: list[str] | None = None,
          nc: int = 13) -> Path:
    """Write pipeline/vehicles/holdout/ and record which images are in it."""
    from .build_fleet import class_names as read_class_names

    chosen = select(size, seed)
    if class_names is None:
        class_names, nc = read_class_names()

    images, labels = vehicles.image_index(), vehicles.label_index()
    (HOLDOUT_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (HOLDOUT_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    linked = 0
    for name in chosen:
        if name in images:
            vehicles.link(images[name], HOLDOUT_ROOT / "images" / "val" / name)
            linked += 1
        stem = name.rsplit(".", 1)[0]
        if stem in labels:
            vehicles.link(labels[stem], HOLDOUT_ROOT / "labels" / "val" / f"{stem}.txt")

    # `train` points at the same directory because Ultralytics wants the key to
    # exist. Nothing trains on this file -- only `val` is ever read, by val().
    data_yaml().write_text(
        f"path: {HOLDOUT_ROOT}\ntrain: images/val\nval: images/val\n"
        f"nc: {nc}\nnames:\n" + "".join(f"- {n}\n" for n in class_names)
    )
    NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NAMES_FILE.write_text(json.dumps(
        {"size": size, "seed": seed, "linked": linked,
         "fingerprint": fingerprint(chosen), "names": chosen}, indent=1))
    return HOLDOUT_ROOT


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def checkpoints() -> list[Path]:
    """This run's global checkpoints, in round order.

    ``global_last.pt`` is left out: it duplicates the highest round and would draw a
    false extra point.

    Checkpoints from *earlier* runs are left out too, and that is the subtle part.
    The directory is not cleared between runs, so a 2-round run leaves rounds 3..6
    from a previous 6-round one sitting beside its own. Scoring all of them produced
    a curve that jumped from 0.014 to 0.217 between round 2 and round 3 -- two
    different models plotted as one learning curve. A run rewrites round 1 first, so
    anything older than round 1 belongs to a run that is over.
    """
    ckpt_dir = paths.PROJECT / "checkpoints"
    rounds = sorted(ckpt_dir.glob("global_round_*.pt"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not rounds:
        return []
    first = rounds[0].stat().st_mtime
    fresh, stale = [], []
    for p in rounds:
        (fresh if p.stat().st_mtime >= first - 1 else stale).append(p)
    if stale:
        print(f"ignoring {len(stale)} checkpoint(s) older than round 1 "
              f"({', '.join(p.name for p in stale)}): they belong to a previous run")
    return fresh


def class_names_on_disk() -> dict[int, str]:
    """The holdout's own class names, from the yaml that defines the set being scored.

    Not from the checkpoint. ``YOLO(weights).val()`` names its classes from the
    **model**, and this project's checkpoints are built from ``yolov8s-13.yaml``,
    which declares no ``names:`` -- so Ultralytics defaults them to ``{0: "0",
    1: "1", ...}`` and every per-class row came out labelled with its own index.
    The holdout's ``data.yaml`` has the real list, written by ``build()`` from the
    same source the shards use.
    """
    try:
        import yaml
        data = yaml.safe_load(data_yaml().read_text())
        return {i: str(n) for i, n in enumerate(data.get("names") or [])}
    except (OSError, ImportError, AttributeError, ValueError):
        return {}


def per_class(result, names: dict[int, str] | None = None) -> list[dict]:
    """AP per class, for the classes the holdout actually contains.

    Three traps here, all of which produce a plausible wrong table rather than an
    error. Each cost a run to find:

    - ``box.ap50`` is indexed by *position within* ``box.ap_class_index``, not by
      class id. Zipping it against ``names`` in class order mislabels every row the
      moment one class is absent from the holdout.
    - ``box.maps`` **is** indexed by class id, but it pre-fills every absent class
      with the overall ``map``. A class with no instances would be reported as
      scoring the fleet average.
    - ``nt_per_class`` is on the **result**, not on ``result.box``. Reading it from
      the box silently yields ``None`` for every count, and the table renders with
      "?" in the column that says how much any of the other numbers mean.

    So iterate ``ap_class_index`` and name each row from it. A class that is not in
    the list is not in the holdout, and is left out rather than given a number.
    """
    box = result.box
    if names is None:
        names = class_names_on_disk() or {
            int(k): str(v) for k, v in (getattr(result, "names", {}) or {}).items()}
    counts = getattr(result, "nt_per_class", None)
    rows = []
    for i, c in enumerate(getattr(box, "ap_class_index", [])):
        c = int(c)
        rows.append({
            "class_id": c,
            "name": names.get(c, str(c)),
            "instances": int(counts[c]) if counts is not None and c < len(counts) else None,
            "AP50": float(box.ap50[i]),
            "AP50-95": float(box.ap[i]),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
        })
    return sorted(rows, key=lambda r: r["class_id"])


def evaluate(weights: Path, imgsz: int = 640, batch: int = 8, device: str = "0") -> dict:
    """Score one checkpoint on the holdout. Raises if it cannot -- never returns 0."""
    if not weights.exists():
        raise SystemExit(f"no such checkpoint: {weights}")
    if not data_yaml().exists():
        raise SystemExit("no holdout built yet; run `python -m pipeline.holdout --build`")

    from ultralytics import YOLO   # imported lazily: selection and tests need no torch

    result = YOLO(str(weights)).val(
        data=str(data_yaml()), imgsz=imgsz, batch=batch, device=device, workers=0,
        plots=False, verbose=False, project=str(paths.STATE / "holdout_runs"),
        name=weights.stem, exist_ok=True,
    )
    box = result.box
    return {"checkpoint": weights.name,
            "round": int(weights.stem.rsplit("_", 1)[-1]) if weights.stem[-1].isdigit() else None,
            "mAP50": float(box.map50), "mAP50-95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr),
            # `car` is 55.4 % of objects and `train` has 29 instances fleet-wide, so a
            # single averaged mAP is close to a car detector's report card. Per-class is
            # what says whether federation helped the rare classes or only the common one.
            "per_class": per_class(result)}


def evaluate_all(imgsz: int = 640, batch: int = 8, device: str = "0") -> list[dict]:
    """Score every global checkpoint and write the curve to .state/holdout_metrics.json."""
    ckpts = checkpoints()
    if not ckpts:
        raise SystemExit("no global_round_*.pt checkpoints; run the federate stage first")
    rows = [evaluate(c, imgsz=imgsz, batch=batch, device=device) for c in ckpts]
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps({"holdout": meta(), "rounds": rows}, indent=1))
    return rows


def best_round(rows: list[dict] | None = None, metric: str = "mAP50") -> dict:
    """Which round produced the best model, and what the rounds after it cost.

    A long run's result is **not** its last round. This project has measured the
    opposite twice: a warm-started model scored 0.2582 untrained against 0.2073 after
    two rounds, and a 1.667x-budget centralised ceiling scored *lower* than a smaller
    one. Reporting the final checkpoint turns "we trained for longer" into "we reported
    a worse model", silently.

    Ties go to the EARLIER round. Two rounds of equal score are not equally good --
    the earlier one cost less GPU, and preferring it makes `wasted_rounds` honest.
    """
    rows = (curve().get("rounds") or []) if rows is None else rows
    scored = [r for r in rows if r.get(metric) is not None]
    if not scored:
        return {}
    best = max(scored, key=lambda r: (r[metric], -(r.get("round") or 0)))
    last = scored[-1]
    after = [r for r in scored if (r.get("round") or 0) > (best.get("round") or 0)]
    return {
        "metric": metric,
        "best_round": best.get("round"),
        "best": best[metric],
        "best_checkpoint": best.get("checkpoint"),
        "final_round": last.get("round"),
        "final": last[metric],
        # Positive means the run peaked early and then got worse.
        "regression": round(best[metric] - last[metric], 4),
        "wasted_rounds": len(after),
        "rounds_scored": len(scored),
    }


def promote(rows: list[dict] | None = None, metric: str = "mAP50") -> Path | None:
    """Copy the best-scoring checkpoint to ``global_best.pt``.

    Named for what it is, and deliberately a copy rather than a rename: the per-round
    checkpoints stay where they are so the curve remains reproducible, and nothing that
    already reads ``global_round_*.pt`` changes behaviour.
    """
    info = best_round(rows, metric)
    if not info.get("best_checkpoint"):
        return None
    src = paths.PROJECT / "checkpoints" / info["best_checkpoint"]
    if not src.exists():
        return None
    dst = src.parent / "global_best.pt"
    shutil.copy2(src, dst)
    return dst


def curve() -> dict:
    """What the dashboard and the report read. ``{}`` before anything is scored."""
    try:
        return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="carve and materialise the holdout")
    ap.add_argument("--evaluate", action="store_true", help="score every global checkpoint on it")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--promote", action="store_true",
                    help="copy the best-scoring round to checkpoints/global_best.pt. "
                         "A long run's result is its best round, not its last")
    args = ap.parse_args(argv)

    if not (args.build or args.evaluate):
        ap.error("pass --build, --evaluate, or both")

    if args.build:
        root = build(args.size, args.seed)
        info = meta()
        print(f"holdout: {info.get('linked')} of {info.get('size')} images materialised at {root}")
        print(f"fingerprint: {info.get('fingerprint')} — quote this, not size and seed; "
              f"they describe how it was asked for, not what it is")
        print("no vehicle can train or self-evaluate on these: the fleet is assigned "
              "from the val pool with them removed")

    if args.evaluate:
        rows = evaluate_all(imgsz=args.imgsz, batch=args.batch, device=args.device)
        info = meta()
        print(f"\nGlobal model on the shared holdout ({info.get('size')} images "
              f"no vehicle saw, fingerprint {info.get('fingerprint') or fingerprint()}):\n")
        for r in rows:
            print(f"  round {r['round']:<3} mAP50 {r['mAP50']:.4f}  "
                  f"mAP50-95 {r['mAP50-95']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}")
        if len(rows) > 1:
            delta = rows[-1]["mAP50"] - rows[0]["mAP50"]
            print(f"\n  {delta:+.4f} mAP50 across {len(rows)} rounds, measured on data "
                  f"no client trained on.")

        last = rows[-1].get("per_class") or []
        if last:
            first = {c["class_id"]: c for c in (rows[0].get("per_class") or [])}
            print(f"\n  Per class, final round — an averaged mAP over a set that is "
                  f"mostly `car` hides which classes federation actually moved:\n")
            print(f"    {'class':<16}{'inst':>7}{'AP50':>9}{'AP50-95':>10}{'Δ AP50':>10}")
            for c in sorted(last, key=lambda r: -(r["instances"] or 0)):
                was = first.get(c["class_id"], {}).get("AP50")
                delta = f"{c['AP50'] - was:+.4f}" if was is not None and len(rows) > 1 else "—"
                inst = c["instances"] if c["instances"] is not None else "?"
                print(f"    {c['name']:<16}{inst:>7}{c['AP50']:>9.4f}"
                      f"{c['AP50-95']:>10.4f}{delta:>10}")
            print("\n    Classes absent from the holdout are omitted, not scored zero: "
                  "Ultralytics' `maps` would have given them the fleet average.")
        info = best_round(rows)
        if info and info["rounds_scored"] > 1:
            print(f"\n  Best round: **{info['best_round']}** at {info['best']:.4f} "
                  f"{info['metric']}; final round {info['final_round']} scored "
                  f"{info['final']:.4f}.")
            if info["regression"] > 0:
                print(f"  The run PEAKED EARLY: the last {info['wasted_rounds']} round(s) "
                      f"cost {info['regression']:.4f} {info['metric']}. Longer is not "
                      f"better here — report the best round, not the last one.")
            else:
                print(f"  Still improving at the final round; more rounds may pay.")
            if args.promote:
                dst = promote(rows)
                print(f"  promoted: {dst}" if dst else
                      "  could not promote: the best checkpoint is missing")
            else:
                print(f"  Pass --promote to copy it to checkpoints/global_best.pt.")
        print(f"\nwritten: {METRICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
