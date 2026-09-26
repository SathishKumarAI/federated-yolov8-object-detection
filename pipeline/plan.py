"""What a configuration will actually do, before it does it.

The run form's estimate was one line from one measured constant. It did not say how
many image-visits the configuration implies, that a centralised ceiling needs the
same number to be worth comparing, which stages will really run, or what the
equivalent command is.

That arithmetic being in somebody's head rather than on screen is exactly how a
baseline shipped this morning with 1.667x the federation's budget.

It does two things, and the second is what the Simulate tab is:

* `plan(cfg)` -- the budget, the stages, the warnings and the equivalent commands for
  the configuration the server is holding.
* `project(cfg, imgsz)` -- the same arithmetic for an *arbitrary* configuration, with
  every projection naming the measurement it rests on and a refusal for every lever
  pushed outside the range that measurement covers. It projects cost, never accuracy:
  one configuration's end-to-end result is recorded, and predicting another's from it
  would be a fabricated number carrying a measured one's confidence.

    python -m pipeline.plan --profile full --vehicles 6 --rounds 6 --epochs 4
    python -m pipeline.plan --project --vehicles 6 --per-vehicle 1400 --rounds 6 --epochs 4
"""
from __future__ import annotations

import argparse

from . import holdout, measurements, stages, vehicles
from .stages import Config

# Measured on an RTX 5070 Ti, 6 vehicles x 1 400 images x 6 rounds x 4 epochs: 3 296 s
# and 82.2 Wh for 201 600 image-visits at 640 px. Imported rather than restated -- the
# same two numbers drive the projection panel and the report, and a measured constant
# that lives in three files is a constant that will disagree with itself.
SECONDS_PER_KVISIT = measurements.value("seconds_per_kvisit")
WH_PER_KVISIT = measurements.value("wh_per_kvisit")


def budget(cfg: Config) -> dict:
    """The arithmetic both sides of the comparison have to agree on."""
    visits = cfg.n_vehicles * cfg.per_vehicle * cfg.rounds * cfg.local_epochs
    pooled = cfg.n_vehicles * cfg.per_vehicle
    # Scaled by resolution: cost goes with pixels, and demo runs at 320 not 640.
    scale = (cfg.imgsz / 640) ** 2
    return {
        "vehicles": cfg.n_vehicles,
        "images_per_vehicle": cfg.per_vehicle,
        "rounds": cfg.rounds,
        "local_epochs": cfg.local_epochs,
        "image_visits": visits,
        "effective_epochs": cfg.rounds * cfg.local_epochs,
        "pooled_images": pooled,
        "centralised_epochs_to_match": cfg.rounds * cfg.local_epochs,
        "seconds_estimate": round(visits / 1000 * SECONDS_PER_KVISIT * scale),
        "wh_estimate": round(visits / 1000 * WH_PER_KVISIT * scale, 1),
        "imgsz": cfg.imgsz,
    }


def commands(cfg: Config) -> list[dict]:
    """The exact CLI for this configuration, and the comparisons it enables."""
    base = (f"python -m pipeline.runner --all --profile {cfg.profile} "
            f"--vehicles {cfg.n_vehicles} --rounds {cfg.rounds} --epochs {cfg.local_epochs} "
            f"--seed {cfg.seed} --partition {cfg.partition} --strategy {cfg.strategy}")
    if cfg.per_vehicle_override:
        base += f" --per-vehicle {cfg.per_vehicle_override}"
    if cfg.partition == "dirichlet":
        base += f" --alpha {cfg.alpha}"
    per_vehicle = f" --per-vehicle {cfg.per_vehicle_override}" if cfg.per_vehicle_override else ""
    sweep = (f"--profile {cfg.profile} --vehicles {cfg.n_vehicles} --rounds {cfg.rounds} "
             f"--epochs {cfg.local_epochs}{per_vehicle} --yes")
    return [
        {"label": "this run", "cmd": base + " --yes --skip baseline"},
        {"label": "this run, with the matched centralised ceiling", "cmd": base + " --yes"},
        {"label": "is it just the seed?", "cmd": f"python -m pipeline.experiment --preset seeds --seeds 0,1,2 {sweep}"},
        {"label": "does the strategy matter?", "cmd": f"python -m pipeline.experiment --preset strategies --strategies fedavg,fedadam,fedavgm {sweep}"},
        {"label": "does non-IID matter?", "cmd": f"python -m pipeline.experiment --preset partitions --partitions condition,random,dirichlet {sweep}"},
        {"label": "how much does skew move it?", "cmd": f"python -m pipeline.experiment --preset alpha --alphas 0.05,0.5,100 {sweep}"},
        {"label": "compare what you already have", "cmd": "python -m pipeline.compare --last 10 --md"},
    ]


def warnings(cfg: Config) -> list[str]:
    """What this configuration will do that the person may not have intended."""
    out = []
    if not holdout.names():
        out.append("No holdout carved. Every number this run produces will be a client "
                   "scoring itself on its own distribution, and comparable with nothing.")
    if cfg.partition == "condition":
        # BDD's rarest profiled condition; asking for more silently tops up.
        thinnest = measurements.value("thinnest_condition")
        if cfg.per_vehicle > thinnest:
            out.append(f"{cfg.per_vehicle} images per vehicle exceeds the rarest condition "
                       f"(~{thinnest} images in all of BDD100K), so those shards will top up "
                       f"with unrelated images and the run will be closer to IID than it looks.")
    if cfg.rounds < 2:
        out.append("One round cannot show whether the global model moves, which is the "
                   "single signal that says the federation is learning at all.")
    if cfg.n_vehicles > 8:
        out.append("More vehicles than condition profiles; the extras repeat conditions.")
    b = budget(cfg)
    if b["seconds_estimate"] > 3600:
        out.append(f"About {b['seconds_estimate'] // 60} minutes of GPU time. Vehicles train "
                   f"serialised, so wall clock scales with vehicle count.")
    return out


def plan(cfg: Config) -> dict:
    return {
        "config": cfg.to_dict(),
        "budget": budget(cfg),
        "stages": stages.snapshot(cfg),
        "commands": commands(cfg),
        "warnings": warnings(cfg),
        "fleet_on_disk": vehicles.load_fleet_meta(),
        "holdout": holdout.meta(),
    }


# --------------------------------------------------------------------------
# Projection: what a configuration would cost, before it costs it
#
# Everything below is arithmetic over `measurements.RECORDS`. Nothing here predicts an
# mAP, and that is deliberate: the only end-to-end result this project has recorded is
# one configuration, so a predicted accuracy for a different one would be a fabricated
# number wearing a measured one's clothes. What is projected is cost -- which really was
# measured -- plus what size of difference the run could resolve at all.
# --------------------------------------------------------------------------

#: The one configuration whose end-to-end result is recorded. Every projection says how
#: far the configuration being asked about is from this, because that distance is the
#: honest answer to "what will I get".
#: Source: STATUS.md, "The result this project exists to produce".
REFERENCE_RUN = {
    "vehicles": 6, "per_vehicle": 1400, "rounds": 6, "local_epochs": 4,
    "partition": "condition", "strategy": "fedavg", "imgsz": 640, "gpu_fraction": 1.0,
    "holdout_mAP50": 0.4173, "retained": 0.845, "seconds": 3296, "wh": 82.2,
    "source": "STATUS.md",
}

#: Where each lever was actually measured. Outside one of these the projection is
#: REFUSED rather than extrapolated -- and it names the measurement that would be
#: needed, so the refusal is an instruction rather than a shrug.
MEASURED_RANGES = {
    "imgsz": {
        "points": [320, 640],
        "source": "docs/PHASED_PLAN.md",
        "needed": "a timed run at that resolution; cost is assumed proportional to "
                  "pixel count and that assumption has only been checked at 320 and 640",
    },
    "per_vehicle": {
        "lo": 300, "hi": 6308,
        "source": "CLAUDE.md",
        "needed": "a run at that shard size; VRAM is measured at 1 400 and at 6 308 and "
                  "is not linear between them",
    },
    "vehicles": {
        "lo": 2, "hi": 8,
        "source": "pipeline/vehicles.py PROFILES",
        "needed": "more condition profiles: beyond 8 the extras repeat conditions, so "
                  "a 9-vehicle fleet is not a 9-condition fleet",
    },
    "gpu_fraction": {
        "points": [0.33, 0.5, 1.0],
        "source": "CLAUDE.md",
        "needed": "a timed run at that packing; the three measured settings are the "
                  "ones Ray was actually run at",
    },
}


def _refusal(lever: str, asked, allowed: str) -> dict:
    r = MEASURED_RANGES[lever]
    return {"lever": lever, "asked": asked, "measured": allowed,
            "source": r["source"], "needed": r["needed"]}


def _out_of_range(cfg: Config, imgsz: int) -> list[dict]:
    """Every lever the caller pushed outside what was measured."""
    out = []
    if imgsz not in MEASURED_RANGES["imgsz"]["points"]:
        out.append(_refusal("imgsz", imgsz,
                            " or ".join(str(p) for p in MEASURED_RANGES["imgsz"]["points"])))
    pv = MEASURED_RANGES["per_vehicle"]
    if not pv["lo"] <= cfg.per_vehicle <= pv["hi"]:
        out.append(_refusal("per_vehicle", cfg.per_vehicle, f"{pv['lo']}-{pv['hi']}"))
    ve = MEASURED_RANGES["vehicles"]
    if not ve["lo"] <= cfg.n_vehicles <= ve["hi"]:
        out.append(_refusal("vehicles", cfg.n_vehicles, f"{ve['lo']}-{ve['hi']}"))
    if cfg.gpu_fraction not in MEASURED_RANGES["gpu_fraction"]["points"]:
        out.append(_refusal("gpu_fraction", cfg.gpu_fraction,
                            ", ".join(str(p) for p in MEASURED_RANGES["gpu_fraction"]["points"])))
    return out


def _vram(per_vehicle: int, gpu_fraction: float) -> dict:
    """Peak VRAM, as the bracket it was measured as rather than a fitted line.

    5 087 MiB at 1 400 images and ~15 900 at 6 308 are not two points on a line -- the
    300-image demo also peaks near 5 GB, so the curve is flat and then steep and nobody
    here has measured its shape. Reporting a single interpolated number would be
    inventing the shape, so a shard between the two measured sizes gets the bracket.
    """
    small = measurements.value("vram_peak_1400")
    large = measurements.value("vram_peak_full")
    ceiling = measurements.BY_ID["vram_peak_1400"]["ceiling_mib"]
    clients = max(1, round(1.0 / gpu_fraction))

    if per_vehicle <= 1400:
        lo = hi = small
        how = (f"measured at 1 400 images/vehicle, and flat below it -- the 300-image "
               f"demo peaks near the same {small} MiB")
    else:
        lo, hi = small, large
        how = (f"between the two sizes it was measured at: {small} MiB at 1 400 and "
               f"{large} MiB at 6 308. The curve between them is NOT measured, so this "
               f"is a bracket, not an estimate")

    return {
        "clients_on_the_card": clients,
        "per_client_lo_mib": lo, "per_client_hi_mib": hi,
        "total_lo_mib": lo * clients, "total_hi_mib": hi * clients,
        "ceiling_mib": ceiling,
        "fits": hi * clients <= ceiling,
        "may_not_fit": lo * clients <= ceiling < hi * clients,
        "how": how,
        "rests_on": ["vram_peak_1400", "vram_peak_full"],
    }


def _difference_from_reference(cfg: Config, imgsz: int) -> list[str]:
    """Which settings differ from the one run whose result is recorded."""
    mine = {"vehicles": cfg.n_vehicles, "per_vehicle": cfg.per_vehicle,
            "rounds": cfg.rounds, "local_epochs": cfg.local_epochs,
            "partition": cfg.partition, "strategy": cfg.strategy, "imgsz": imgsz,
            "gpu_fraction": cfg.gpu_fraction}
    return [f"{k}: {v} instead of {REFERENCE_RUN[k]}"
            for k, v in mine.items() if v != REFERENCE_RUN[k]]


def project(cfg: Config, imgsz: int | None = None) -> dict:
    """What this configuration would cost, from numbers that were measured.

    Returns projections that each name the measurements they rest on, and refusals for
    every lever pushed outside the range those measurements cover. A refusal is not a
    missing projection: it is the answer, and it says which measurement is missing.
    """
    imgsz = cfg.imgsz if imgsz is None else int(imgsz)
    b = budget(cfg)
    refusals = _out_of_range(cfg, imgsz)
    floor = measurements.value("noise_floor_map50")
    packing = measurements.packing(cfg.gpu_fraction)
    speedup = packing.get("speedup")

    projections: list[dict] = []

    def add(name, value, unit, how, rests_on, status="projected"):
        projections.append({"name": name, "value": value, "unit": unit, "how": how,
                            "rests_on": rests_on, "status": status})

    add("image-visits", b["image_visits"], "visits",
        "vehicles x images x rounds x local epochs. Counted, not measured -- this is "
        "the definition of the budget both sides of a comparison must match",
        [], status="exact")

    if any(r["lever"] == "imgsz" for r in refusals):
        add("wall clock", None, "s",
            f"refused: the cost of {imgsz} px has never been timed here. Cost is "
            f"assumed proportional to pixel count and that has only been checked at "
            f"320 and 640", ["seconds_per_kvisit"], status="refused")
        add("energy", None, "Wh", "refused for the same reason as the wall clock",
            ["wh_per_kvisit"], status="refused")
    else:
        scale = (imgsz / 640) ** 2
        secs = b["image_visits"] / 1000 * SECONDS_PER_KVISIT * scale
        packed = None if speedup is None else secs / speedup
        ratio = b["image_visits"] / (REFERENCE_RUN["vehicles"] * REFERENCE_RUN["per_vehicle"]
                                     * REFERENCE_RUN["rounds"] * REFERENCE_RUN["local_epochs"])
        add("wall clock", round(packed if packed is not None else secs), "s",
            f"{b['image_visits']:,} image-visits at {SECONDS_PER_KVISIT:.2f} s per "
            f"thousand, x{scale:.2f} for {imgsz} px"
            + (f", / {speedup} for gpu_fraction {cfg.gpu_fraction:g}" if speedup else "")
            + f". That is {ratio:.2f}x the reference run's budget, along a relationship "
              f"measured at one point",
            ["seconds_per_kvisit", "gpu_fraction_speedup"])
        add("energy", round(b["image_visits"] / 1000 * WH_PER_KVISIT * scale, 1), "Wh",
            "integrated from nvidia-smi power on the reference run, scaled the same way",
            ["wh_per_kvisit"])

    # imgsz belongs in this list, and its absence was a projection that would have
    # misled: both VRAM records were taken at 640, and activation memory scales with the
    # pixel count, so 5 087 MiB at imgsz=1024 is 2.56x understated. The wall clock was
    # already refused for the same reason; the number beside it was not.
    vram = _vram(cfg.per_vehicle, cfg.gpu_fraction) if not any(
        r["lever"] in ("per_vehicle", "gpu_fraction", "imgsz") for r in refusals) else None
    if vram is None:
        add("peak VRAM", None, "MiB",
            "refused: outside the shard sizes, packings or image sizes VRAM was "
            "measured at",
            ["vram_peak_1400", "vram_peak_full"], status="refused")
    else:
        add("peak VRAM", [vram["total_lo_mib"], vram["total_hi_mib"]], "MiB",
            f"{vram['clients_on_the_card']} client(s) on the card; {vram['how']}",
            vram["rests_on"])

    rec = measurements.BY_ID["noise_floor_map50"]
    add("smallest difference this run could resolve", floor, "mAP50",
        f"the recorded run-to-run spread, measured at {rec['conditions']} and a "
        f"{rec['bound']} bound. A delta smaller than this is not a measured difference, "
        f"whatever the curve looks like. It is also already stale: {rec['reopened_by']}",
        ["noise_floor_map50"])

    hazard = packing.get("hazard")
    if hazard:
        projections.append({"name": "packing hazard", "value": f"{cfg.gpu_fraction:g}",
                            "unit": "gpu fraction", "how": hazard,
                            "rests_on": ["gpu_fraction_speedup"], "status": "hazard"})

    return {
        "config": cfg.to_dict(),
        "imgsz": imgsz,
        "budget": b,
        "projections": projections,
        "refusals": refusals,
        "vram": vram,
        "warnings": warnings(cfg),
        "reference": REFERENCE_RUN,
        "differs_from_reference": _difference_from_reference(cfg, imgsz),
        # Derived, not hard-coded: resolution became a real run lever on 2026-09-26, and
        # whether THIS server can apply it is a fact about the Config it is holding.
        "imgsz_reachable": "imgsz" in Config.__dataclass_fields__ or imgsz == 640,
        "imgsz_note": ("Resolution is a run lever: one number drives the federation, the "
                       "clients' own validation, the centralised baseline and the "
                       "holdout together, so it cannot become an unfair comparison. The "
                       "cost of anything but 320 or 640 is still unmeasured here."
                       if "imgsz" in Config.__dataclass_fields__ else
                       "This server has no imgsz lever: the federation trains at "
                       "my-project's DEFAULT_IMAGE_SIZE of 640, and changing it means "
                       "editing my-project/my_project/client_app.py, which pipeline/ is "
                       "not allowed to do. Any other value here is a what-if."),
        "levers": {name: name in Config.__dataclass_fields__
                   for name in ("freeze_round1", "server_ema", "fix_bn_from_round",
                                "imgsz", "local_bn")},
        "lever_notes": measurements.LEVER_NOTES,
        "accuracy_note": "No mAP is projected. One configuration's end-to-end result is "
                         "recorded, and predicting another's from it would be a "
                         "fabricated number with a measured number's confidence. What "
                         "is offered instead is the distance from that recorded run and "
                         "the size of difference this budget could resolve.",
    }


def _print_projection(pr: dict) -> int:
    """The projection as text. Refusals first: they are the answer, not an omission."""
    c = pr["config"]
    print(f"{c['n_vehicles']} vehicles x {c['per_vehicle']} images x {c['rounds']} rounds "
          f"x {c['local_epochs']} local epochs at {pr['imgsz']} px")
    print()
    for row in pr["projections"]:
        val = row["value"]
        if row["status"] == "refused":
            shown = "REFUSED"
        elif isinstance(val, list):
            shown = f"{val[0]:,} - {val[1]:,}"
        elif isinstance(val, int):
            shown = f"{val:,}"
        else:
            shown = str(val)
        print(f"  {row['name']:<44}{shown:>20} {row['unit']}")
        print(f"      {row['how']}")
        if row["rests_on"]:
            print(f"      rests on: {', '.join(row['rests_on'])}")
    if pr["refusals"]:
        print()
        print("refused, and what would lift each refusal:")
        for r in pr["refusals"]:
            print(f"  ! {r['lever']} = {r['asked']}, measured only at {r['measured']} "
                  f"({r['source']})")
            print(f"    needs: {r['needed']}")
    if pr["differs_from_reference"]:
        ref = pr["reference"]
        print()
        print(f"differs from the one recorded end-to-end run ({ref['source']}, "
              f"{ref['holdout_mAP50']} mAP50, {ref['retained']:.3f} of its ceiling) in:")
        for d in pr["differs_from_reference"]:
            print(f"  - {d}")
    if not pr["imgsz_reachable"]:
        print()
        print(f"  ! {pr['imgsz_note']}")
    print()
    print(pr["accuracy_note"])
    for w in pr["warnings"]:
        print(f"  ! {w}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="demo", choices=("demo", "full"))
    ap.add_argument("--vehicles", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per-vehicle", type=int, default=0)
    ap.add_argument("--partition", default="condition", choices=vehicles.PARTITIONS)
    ap.add_argument("--strategy", default="fedavg", choices=stages.STRATEGIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-fraction", type=float, default=1.0)
    ap.add_argument("--imgsz", type=int, default=0,
                    help="project a resolution other than the profile's; 0 = the profile's")
    ap.add_argument("--project", action="store_true",
                    help="print the projection -- cost, VRAM, and what it cannot answer")
    args = ap.parse_args(argv)

    cfg = Config(profile=args.profile, n_vehicles=args.vehicles, rounds=args.rounds,
                 local_epochs=args.epochs, per_vehicle_override=args.per_vehicle,
                 partition=args.partition, strategy=args.strategy, seed=args.seed,
                 gpu_fraction=args.gpu_fraction)
    if args.project:
        return _print_projection(project(cfg, args.imgsz or None))
    p = plan(cfg)
    b = p["budget"]

    print(f"{b['vehicles']} vehicles x {b['images_per_vehicle']} images x {b['rounds']} rounds "
          f"x {b['local_epochs']} local epochs")
    print(f"  = {b['image_visits']:,} image-visits at {b['imgsz']}px "
          f"({b['effective_epochs']} effective epochs per vehicle)")
    print(f"  ~ {b['seconds_estimate'] // 60} min, ~{b['wh_estimate']} Wh")
    print(f"  a matched centralised ceiling: {b['pooled_images']:,} pooled images x "
          f"{b['centralised_epochs_to_match']} epochs\n")

    print("stages:")
    for s in p["stages"]:
        mark = "skip" if s["satisfied"] else ("GATE" if s["gated"] else "run ")
        print(f"  [{mark}] {s['name']:<9} {s['est']:<18} {s['detail'][:60]}")

    if p["warnings"]:
        print("\nwarnings:")
        for w in p["warnings"]:
            print(f"  ! {w}")

    print("\ncommands:")
    for c in p["commands"]:
        print(f"  # {c['label']}\n  {c['cmd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
