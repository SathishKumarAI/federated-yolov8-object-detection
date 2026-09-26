"""What a configuration will actually do, before it does it.

The run form's estimate was one line from one measured constant. It did not say how
many image-visits the configuration implies, that a centralised ceiling needs the
same number to be worth comparing, which stages will really run, or what the
equivalent command is.

That arithmetic being in somebody's head rather than on screen is exactly how a
baseline shipped this morning with 1.667x the federation's budget.

    python -m pipeline.plan --profile full --vehicles 6 --rounds 6 --epochs 4
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
    args = ap.parse_args(argv)

    cfg = Config(profile=args.profile, n_vehicles=args.vehicles, rounds=args.rounds,
                 local_epochs=args.epochs, per_vehicle_override=args.per_vehicle,
                 partition=args.partition, strategy=args.strategy, seed=args.seed)
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
