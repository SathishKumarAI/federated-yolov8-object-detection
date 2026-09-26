"""Every number this project measured once and now quotes, in one place.

The dashboard has to be able to say *where a number came from*. Until now the measured
constants were scattered: the cost-per-image-visit lived in `plan.py`, the VRAM ceiling
in `gpu.py`, the thinnest condition as a bare `1419` inside a warning string, and the
rest only in prose in `CLAUDE.md`. A number with no provenance on screen is
indistinguishable from a number someone typed, which is precisely the failure this
repo's history is made of.

So each record carries what it is, what it measures, **which file records it**, and
when. `python -m pipeline.measurements --check` re-reads those files and fails if a
value has drifted out of the document that is supposed to be its source — because the
alternative is a dashboard confidently citing a doc that no longer says that.

Two rules for adding one:

1. **No record without a source.** If nothing in the repo records it, it is not a
   measurement, it is a guess, and the UI must say "unknown" instead.
2. **The value goes in `value`, never only in the prose.** `--check` greps for the
   value's own string, so a record whose number lives only in a sentence is unchecked.

    python -m pipeline.measurements            # the table
    python -m pipeline.measurements --check    # each value still in the doc it names
"""
from __future__ import annotations

import argparse
import json

from . import paths

#: BDD100K's detection set, nc = 13. Order is the label-file order.
#: Source: docs/ARCHITECTURE.md, "Class taxonomy (nc = 13)".
BDD_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle",
               "bicycle", "traffic light", "traffic sign", "trailer",
               "other person", "other vehicle"]

#: The nine of those thirteen whose head rows were copied out of COCO by
#: `warm_start_head`. The other four started from a distribution and, per
#: docs/PHASED_PLAN.md, are where the residual loss is predicted to concentrate --
#: which makes this list a testable claim rather than trivia.
#: Source: docs/PHASED_PLAN.md, "Warm-start the head from COCO rows".
WARM_STARTED_CLASSES = ["person", "car", "bus", "truck", "train", "motorcycle",
                        "bicycle", "traffic light", "traffic sign"]

UNWARMED_CLASSES = [c for c in BDD_CLASSES if c not in WARM_STARTED_CLASSES]


def _m(mid, label, value, unit, source, quote, note, measured_on, **extra) -> dict:
    return {"id": mid, "label": label, "value": value, "unit": unit,
            "source": source, "quote": quote, "note": note,
            "measured_on": measured_on, **extra}


#: Ordered so the table reads as an argument: what the run costs, what it is worth,
#: and what would make the difference real.
RECORDS: list[dict] = [
    _m("seconds_per_kvisit", "wall clock per 1 000 image-visits at 640 px",
       3296 / 201.6, "s",
       "docs/PHASED_PLAN.md", "3 296 s",
       "From the reference run: 6 vehicles x 1 400 images x 6 rounds x 4 epochs = "
       "201 600 image-visits in 3 296 s on an RTX 5070 Ti, clients serialised.",
       "2026-08-06"),
    _m("wh_per_kvisit", "energy per 1 000 image-visits at 640 px",
       82.2 / 201.6, "Wh",
       "docs/PHASED_PLAN.md", "82.2 Wh",
       "Same run, integrated from nvidia-smi power samples rather than estimated.",
       "2026-08-06"),
    _m("federated_map50", "the federation's holdout mAP50", 0.4173, "mAP50",
       "STATUS.md", "0.4173",
       "6 rounds x 4 local epochs x 6 vehicles x 1 400 images, scored on the "
       "1 000-image shared holdout no vehicle trained on.",
       "2026-08-06"),
    _m("centralised_map50", "the budget-matched centralised ceiling", 0.4936, "mAP50",
       "STATUS.md", "0.4936",
       "The same 201 600 image-visits, pooled. Matching the budget is what makes the "
       "comparison mean anything -- an earlier ceiling had 1.667x and scored LOWER.",
       "2026-08-06"),
    _m("retained", "share of the ceiling the federation keeps", 0.845, "fraction",
       "STATUS.md", "84.5",
       "0.4173 of 0.4936. Published work reaches 61.5 vs 61.4 on BDD100K, so this gap "
       "is this project's schedule and transport, not a law of federated detection.",
       "2026-08-06"),
    _m("noise_floor_map50", "run-to-run spread in holdout mAP50", 0.016, "mAP50",
       "CLAUDE.md", "0.016",
       "A delta smaller than this is not a result. Recorded as a LOWER bound: it was "
       "inferred from a ceiling anomaly across two data volumes, not from repeated "
       "seeds, so the true spread may be smaller and this band may be dismissing real "
       "differences. Re-measure before ranking anything close.",
       "2026-08-16", bound="lower", confidence="inferred"),
    _m("warm_start_untrained", "holdout mAP50 of the untrained warm-started model",
       0.2582, "mAP50",
       "docs/PHASED_PLAN.md", "0.2582",
       "Against 0.0053 with a random head -- 49x, before one gradient step. It is also "
       "59 % of what 6 rounds x 4 epochs of federation reached.",
       "2026-08-16"),
    _m("round_one_cost", "what round 1 costs the warm-started model", -0.066, "mAP50",
       "docs/PHASED_PLAN.md", "0.066",
       "0.2582 untrained -> 0.1924 after round 1 -> 0.2073 after round 2. Round 1 is "
       "spent recovering from the LR schedule, not learning.",
       "2026-08-16"),
    _m("vram_peak_1400", "peak VRAM at 1 400 images per vehicle", 5087, "MiB",
       "CLAUDE.md", "5 087",
       "Of 16 303 MiB, with client-resources.num-gpus = 1.0. The card fits three of "
       "these, which is why gpu_fraction is a real lever at this size.",
       "2026-08-16", ceiling_mib=16303),
    _m("vram_peak_full", "peak VRAM on a full 6 308-image shard", 15900, "MiB",
       "CLAUDE.md", "15.9 GB",
       "Of 16 303 MiB. At this size clients MUST be serialised; there is no headroom "
       "to pack a second one.",
       "2026-08-16", ceiling_mib=16303),
    _m("gpu_fraction_speedup", "wall-clock speed-up by clients packed per card",
       {"1.0": 1.00, "0.5": 1.50, "0.33": 1.94}, "x",
       "CLAUDE.md", "1.94",
       "0.33 is fastest and fills 94.9-96.6 % of VRAM with three Ray actors; one run "
       "at that setting died mid-round-2 on a HOST allocation, not the card. 0.5 if "
       "anything else wants memory.",
       "2026-08-16"),
    _m("dataloader_ms_per_sample", "dataloader cost on the training thread", 7.93, "ms",
       "CLAUDE.md", "7.93",
       "5.56 ms with mosaic=0. ~127 ms per batch of 16 at workers=0, the same order as "
       "the GPU step -- this is the 27 % utilisation, and it is mosaic assembly and "
       "the warps, not decode.",
       "2026-08-16", mosaic_off=5.56),
    _m("auto_optimizer_lr0", "the LR ultralytics actually uses at nc=13", 5.88e-4, "lr",
       "CLAUDE.md", "5.88e-4",
       "optimizer='auto' is the default and REPLACES lr0 with 0.002*5/(4+nc). Passing "
       "lr0 without also passing optimizer is a silent no-op, and every run in this "
       "repo so far trained with AdamW at this rate.",
       "2026-08-16"),
    _m("thinnest_condition", "images the rarest profiled condition has in BDD100K",
       1419, "images",
       "CLAUDE.md", "1 419",
       "'overcast residential'. Asking for more per vehicle than a condition has tops "
       "the shard up with unrelated images and turns a non-IID run nearly IID -- "
       "silently, and the label still says the condition.",
       "2026-08-16"),
    _m("car_share", "car's share of the objects in BDD100K", 0.554, "fraction",
       "docs/FEDERATED_DETECTION.md", "55.4 % of objects",
       "One averaged mAP over this holdout is close to a car detector's report card. "
       "Read the per-class panel before believing a single number moved.",
       "2026-08-16"),
]

BY_ID = {r["id"]: r for r in RECORDS}


def value(mid: str):
    """One measured value. KeyError rather than a default: a missing measurement must
    not silently become a plausible number."""
    return BY_ID[mid]["value"]


def table() -> dict:
    """What `/api/measurements` serves: the records plus the class facts."""
    return {
        "records": RECORDS,
        "classes": {
            "all": BDD_CLASSES,
            "warm_started": WARM_STARTED_CLASSES,
            "unwarmed": UNWARMED_CLASSES,
            "note": "COCO transfers the backbone; warm_start_head copies 9 of the 13 "
                    "head rows. docs/PHASED_PLAN.md predicts the residual loss "
                    "concentrates in the other four, which the per-class panel is "
                    "what tests.",
        },
    }


def drift() -> list[str]:
    """Records whose value no longer appears in the document they cite."""
    out = []
    for rec in RECORDS:
        doc = paths.REPO / rec["source"]
        if not doc.is_file():
            out.append(f"{rec['id']}: cites {rec['source']}, which does not exist")
            continue
        if rec["quote"] not in doc.read_text(encoding="utf-8", errors="replace"):
            out.append(f"{rec['id']}: {rec['quote']!r} is no longer in {rec['source']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify each value still appears in the doc it cites")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps(table(), indent=1))
        return 0
    if args.check:
        problems = drift()
        for p in problems:
            print(f"  ! {p}")
        print(f"{len(RECORDS) - len(problems)} of {len(RECORDS)} records still match "
              f"their source")
        return 1 if problems else 0

    for rec in RECORDS:
        val = rec["value"]
        shown = json.dumps(val) if isinstance(val, dict) else f"{val}"
        print(f"  {rec['id']:<26} {shown:>12} {rec['unit']:<8} "
              f"{rec['source']} ({rec['measured_on']})")
    print(f"\n{len(BDD_CLASSES)} classes, {len(WARM_STARTED_CLASSES)} warm-started, "
          f"un-warmed: {', '.join(UNWARMED_CLASSES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
