"""A recorded run, replayable with no GPU and no data on disk.

The dashboard is unreadable on a clean checkout: every panel is empty, so nothing
demonstrates what any of them is for. This serves one run in the shape `/api/state`
returns, so the *real* panels render it -- no demo-only views, which would rot the
moment a panel changed.

**Every number here is transcribed from a document in this repository, and carries the
document's name.** Nothing is invented and nothing is interpolated. That rules out the
obvious way to build a nice demo, and it is the whole point: a walkthrough that made
plausible numbers up would be the exact failure this dashboard exists to catch, shipped
inside the thing that catches it.

The consequence is a sparse demo. Several panels will say "not measured", because for
this run nobody measured that -- no centralised ceiling, no per-class breakdown, no GPU
telemetry. The walkthrough says so at each one. A panel correctly refusing to show a
number it does not have is the most useful thing a new reader can be shown.

The run being replayed is the head warm-start probe of 2026-08-16: 6 vehicles x 1 400
images, condition-partitioned, seed 0, 2 rounds x 1 local epoch, scored on the
1 000-image shared holdout.

    python -m pipeline.demo_run          # the payload, and what each field cites
"""
from __future__ import annotations

import argparse
import json

from . import holdout, ledger, vehicles

#: field -> the document it is transcribed from. Rendered by the walkthrough, so a
#: reader can check any number on screen against the repo without asking.
SOURCES = {
    "fleet": "STATUS.md -- 'The fleet on disk is 1 400 images/vehicle, "
             "condition-partitioned, seed 0'; conditions are the first six entries of "
             "pipeline/vehicles.py PROFILES, in order",
    "holdout": "docs/PHASED_PLAN.md -- the warm-start table: untrained 0.2582, after "
               "round 1 0.1924, after round 2 0.2073, on the 1 000-image holdout",
    "checksums": "pipeline/tests/test_pipeline.py CAPTURED -- two consecutive "
                 "'Aggregated parameters with checksum' lines taken from a real server "
                 "log. They are real aggregates, and they are NOT from the same run as "
                 "the holdout curve above; they are here because the heartbeat panel "
                 "cannot be explained without a pair of them",
    "criteria": "pipeline/verify.py -- the wording of the four pass criteria, evaluated "
                "against the two checksums above and nothing else",
}

#: What is deliberately absent, and why. The walkthrough reads this out at the panel
#: that stays empty, so an empty panel is a lesson rather than a defect.
ABSENT = {
    "baseline": "No centralised ceiling was trained for this 2-round probe, so the "
                "'of the centralised ceiling' readout has nothing to divide by. Without "
                "it the holdout number has no scale -- which is why the panel says so "
                "instead of showing a percentage.",
    "per_class": "No per-class AP was recorded for this run. The per-class panel stays "
                 "hidden rather than showing thirteen zeros, because zeros there would "
                 "read as 'the model detects nothing'.",
    "mAP50-95": "Only mAP50 was written down for this run. The second series on the "
                "holdout chart is therefore missing rather than flat at zero.",
    "gpu": "No GPU telemetry was kept. Every readout in the GPU panel says 'not "
           "measured', which is a different fact from a card sitting at 0 %.",
    "metrics": "The per-client self-evaluation rows were not transcribed; only the "
               "holdout curve was, and the holdout is the honest number anyway.",
}

#: The two real aggregates. Equal consecutive values are what "nothing is being learned"
#: looks like; these differ, which is what a working round looks like.
CHECKSUMS = [-1032.5395936965942, -2646.913425683975]

#: mAP50 on the 1 000-image holdout. `untrained` is round 0 in the recorded table and is
#: the single most surprising number this project has: the warm-started model is better
#: before federating than after two rounds of it.
HOLDOUT_ROUNDS = [
    {"round": 1, "mAP50": 0.1924, "checkpoint": "global_round_1.pt"},
    {"round": 2, "mAP50": 0.2073, "checkpoint": "global_round_2.pt"},
]
UNTRAINED_MAP50 = 0.2582


def fleet() -> list[dict]:
    """Six vehicles with the six real condition profiles, 1 400 images each.

    Fingerprints are omitted rather than faked: a fingerprint is a hash of exactly which
    images a vehicle holds, and a made-up one would be a lie about data provenance in
    the field whose only job is data provenance.
    """
    return [{"vid": i + 1, "condition": name, "n_train": 1400, "n_val": 280,
             "fingerprint": None}
            for i, (name, _) in enumerate(vehicles.PROFILES[:6])]


def state() -> dict:
    """The recorded run in the shape `/api/state` returns."""
    return {
        "demo": True,
        "busy": False,
        "current": None,
        "config": {"profile": "full", "n_vehicles": 6, "rounds": 2, "local_epochs": 1,
                   "seed": 0, "partition": "condition", "strategy": "fedavg",
                   "per_vehicle": 1400, "imgsz": 640, "holdout_size": 1000,
                   "gpu_fraction": 1.0, "alpha": 0.5, "size_skew": 0.0,
                   "proximal_mu": 0.0, "cache": "", "local_bn": False,
                   "per_vehicle_override": 1400, "ray_address": None},
        "stages": [],
        "fleet": fleet(),
        # Omitted, not zeroed. renderGpu prints "not measured" for a null.
        "gpu": {"util_pct": None, "mem_used_mib": None, "power_w": None,
                "energy_wh": None, "temp_c": None, "mem_ceiling_mib": 16303,
                "history": []},
        "links": {"mlflow": "http://127.0.0.1:5000", "ray": "http://127.0.0.1:8265"},
        "options": {"partitions": list(vehicles.PARTITIONS), "strategies": ["fedavg"]},
        "results": [],
        "reports": [],
        "live": {
            "checksums": CHECKSUMS,
            "rounds_done": len(CHECKSUMS),
            "metrics": [],
            "map50": [],
            "loss": [],
            "per_vehicle": {str(v["vid"]): {"rounds": 2, "received": None, "sent": None,
                                            "device": None} for v in fleet()},
            "training_now": None,
            "no_optimizer_steps": 0,
            "criteria": [
                "[PASS] the aggregate checksum changed between rounds "
                f"({CHECKSUMS[0]:.4f} -> {CHECKSUMS[1]:.4f}), so the global model moved",
                "[WARN] no centralised baseline was trained, so the holdout number has "
                "no scale",
            ],
            "criteria_ok": True,
            "checkpoints": [r["checkpoint"] for r in HOLDOUT_ROUNDS],
            "learning": None,
            "holdout": {"holdout": {"size": 1000, "seed": 0, "fingerprint": None},
                        "rounds": HOLDOUT_ROUNDS},
            "baseline": {},
        },
    }


def payload() -> dict:
    """What `/api/demo` serves: the recorded state, its provenance, and its gaps.

    `live_run_available` decides which the walkthrough narrates. A machine with a real
    run on disk should be taught on its own numbers -- the recording exists for the
    clean checkout, not in preference to reality.
    """
    return {
        "state": state(),
        "sources": SOURCES,
        "absent": ABSENT,
        "untrained_mAP50": UNTRAINED_MAP50,
        "run": "head warm-start probe, 2026-08-16: 6 vehicles x 1 400 images, "
               "condition-partitioned, seed 0, 2 rounds x 1 local epoch",
        "live_run_available": bool(holdout.curve().get("rounds") or ledger.load()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    p = payload()
    if args.json:
        print(json.dumps(p, indent=1))
        return 0
    print(f"replaying: {p['run']}\n")
    print("transcribed from:")
    for field, src in p["sources"].items():
        print(f"  {field:<12} {src}")
    print("\ndeliberately absent:")
    for field, why in p["absent"].items():
        print(f"  {field:<12} {why}")
    print(f"\na real run is on this machine: {p['live_run_available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
