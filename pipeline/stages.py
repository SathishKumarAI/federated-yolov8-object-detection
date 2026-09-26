"""The pipeline stages: what each one runs, when it can be skipped, and what it costs.

Every stage is a subprocess with ``cwd=my-project``. That is not incidental — flwr's
detached SuperLink caches the CWD of whichever run started it, so launching from
anywhere else makes every relative path in the project resolve somewhere wrong.

A stage that fails halts the chain. This project's history is a catalogue of silent
no-ops (weights returned untrained, every client handed the same shard, a round with
no optimizer step), so nothing here is allowed to continue past a failure.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import baseline, holdout, paths, validate, vehicles

PY = sys.executable

#: Aggregation strategies this pipeline offers, mirroring server_app.STRATEGIES.
#: Mirrored rather than imported: reaching into my-project would drag flwr and
#: ultralytics into a package that is meant to stay stdlib-light. The two lists are
#: asserted equal by my-project/tests/test_strategy_registry.py, so they cannot drift.
STRATEGIES = ("fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad", "fedavgm",
              "fedmedian", "fedtrimmedavg", "krum", "bulyan", "qfedavg",
              "faulttolerantfedavg")


@dataclass
class Config:
    """Everything a run is parameterised by. Also the 'inputs' half of the report."""

    profile: str = "demo"        # demo | full
    n_vehicles: int = 6
    rounds: int = 2
    local_epochs: int = 1
    seed: int = 0
    # random = IID shards, the setting for "train the best model on this data": FedAvg
    # over IID clients converges closest to the centralised ceiling. `condition` is the
    # non-IID setting the FL techniques in docs/PHASED_PLAN.md are aimed at, and it is
    # still one flag away. This default moved 2026-09-02 and it is NOT cosmetic: the
    # fleet stage rebuilds whenever the fleet on disk disagrees with it, so a default
    # that disagrees with the data silently repartitions it on the next run.
    partition: str = "random"        # any key of vehicles.PARTITIONERS
    strategy: str = "fedavg"         # any key of server_app.STRATEGIES
    proximal_mu: float = 0.0         # FedProx only; >0 turns the proximal term on
    holdout_size: int = 1000         # images no vehicle may train or self-evaluate on
    # Deliberately NOT `seed`. The holdout is the ruler; the run seed is the thing being
    # measured. Tying them together meant a seed sweep -- the one experiment whose job
    # is to hold everything but the seed constant -- rebuilt the holdout for every arm
    # and compared three numbers taken on three different sets of images.
    holdout_seed: int = 0
    alpha: float = 0.5               # dirichlet only: smaller = more skewed
    size_skew: float = 0.0           # 0 = every vehicle the same size; ~1 = 10x spread
    per_vehicle_override: int = 0    # 0 = use the profile default
    gpu_fraction: float = 1.0        # Ray's per-client GPU share; <1 packs clients
    cache: str = ""                  # "" | ram | disk -- ultralytics' dataset cache
    local_bn: bool = False           # FedBN: each vehicle keeps its own BatchNorm
    # Round 1 only, 0 = off. `.load()` cannot transfer the classification convs, so
    # the head starts part-random and round 1 backpropagates its error into a backbone
    # that already knew: the warm-started model scored 0.2582 on the holdout untrained
    # and 0.2073 after two rounds. 10 is the YOLOv8s backbone.
    freeze_round1: int = 0           # layers frozen on round 1, so the head settles
    server_ema: float = 0.0          # >0: EMA the aggregate across rounds (FedSWA-style)
    fix_bn_from_round: int = 0       # >0: FixBN -- pin BatchNorm statistics from that round
    imgsz_override: int = 0          # 0 = the profile default (320 demo / 640 full)
    # Share of clients that re-score themselves each round. FedAvg's default is 1.0 and
    # the pipeline never sent anything, so every vehicle re-evaluated every round --
    # phase 0 measured that at 13.8 % of wall clock, spent on the per-client metric that
    # is NOT the headline (the shared holdout is). Must stay > 0: `pipeline.verify`'s
    # third criterion reads metrics.csv, which only has rows because clients evaluate.
    fraction_evaluate: float = 1.0   # 1.0 = every vehicle, as before
    ray_address: str | None = None   # set => attach to an existing head node

    @property
    def per_vehicle(self) -> int:
        """Images per vehicle.

        Worth overriding for condition partitioning: BDD100K only holds ~5 800 rainy
        and ~6 300 snowy images in total, so asking for 6 308 per vehicle exhausts the
        condition and the shard tops up with whatever is left -- which quietly turns a
        non-IID run into a nearly-IID one. Keep it below the rarest condition's count
        if the bias is meant to be real.
        """
        if self.per_vehicle_override:
            return self.per_vehicle_override
        return 300 if self.profile == "demo" else 6308

    @property
    def imgsz(self) -> int:
        """Input resolution, for every stage that has one.

        No longer the sanity stage's alone. The federation's size used to be
        my-project's ``DEFAULT_IMAGE_SIZE`` and unreachable from here -- the client took
        it from a module constant, and changing it meant editing ``client_app.py``, which
        this component may not do. It is now a run-config key the server broadcasts, so
        one number covers the federation, the clients' own validation, the centralised
        baseline and the holdout score. ``_cmd_baseline`` and ``_cmd_evaluate`` already
        passed this value as a flag, so raising it cannot silently produce a federation
        measured against a ceiling at a different resolution.

        Why it is worth raising: BDD100K frames are 1280x720, and at 640 a traffic light
        or a rider is sub-20px. Published YOLOv8 results on this dataset are 0.470 mAP50
        for yolov8n at 640 against 0.625 for yolov8s at 1024 -- and changing it touches no
        data. Costs activation memory as the square of the size, so the client scales its
        batch by ``(640/imgsz)**2``.
        """
        if self.imgsz_override:
            return self.imgsz_override
        return 320 if self.profile == "demo" else 640

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(per_vehicle=self.per_vehicle, imgsz=self.imgsz)
        return d


@dataclass
class Check:
    satisfied: bool
    detail: str


# flwr can exit 0 having crashed: it prints "Simulation Runtime crashed" and
# "Exit Code: 700" and still returns success. Trusting the exit code alone marked a
# dead run "ok" -- exactly the silent-failure pattern this project keeps producing --
# so the output is scanned too.
CRASH_MARKERS = (
    "Simulation Runtime crashed",
    "An error was encountered. Ending simulation",
    "Simulation raised an exception",
    "Traceback (most recent call last)",
)


@dataclass
class Stage:
    name: str
    title: str
    gated: bool                      # costs real time or GPU -> needs confirmation
    check: callable                  # (Config) -> Check
    command: callable                # (Config) -> list[str] | None
    cwd: Path = paths.PROJECT
    data_root: Path | None = None    # overrides FL_AV_DATA_ROOT for this stage
    est: str = ""
    crash_markers: tuple[str, ...] = ()   # output that means failure despite exit 0


def scan_for_crash(lines: list[str], markers: tuple[str, ...]) -> str | None:
    """Return the first crash marker present in the output, or None."""
    for marker in markers:
        for line in lines:
            if marker in line:
                return marker
    return None


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def _check_env(_: Config) -> Check:
    # Deliberately never "satisfied": it is a two-second probe and the answer
    # (right torch build? GPU visible?) is the single most useful line of output.
    return Check(False, "always runs; probes torch + CUDA capability")


def _check_dataset(_: Config) -> Check:
    pool = paths.find_pool()
    if pool is None:
        return Check(False, "BDD100K pool not found in the kagglehub cache")
    n_tr = paths.count_files(pool / "train", {".jpg"})
    n_va = paths.count_files(pool / "val", {".jpg"})
    # The Kaggle mirror nests train into trainA/trainB/..., so a flat count reads low
    # even when the download is complete. Val is flat and is the reliable signal.
    if n_va >= 10000:
        return Check(True, f"pool present ({n_va} val, {n_tr} loose train + nested)")
    return Check(False, f"pool incomplete ({n_va} val images)")


def _check_populate(_: Config) -> Check:
    missing = []
    for bid in paths.BATCH_IDS:
        batch = paths.PROJECT / "batch" / f"batch_{bid}"
        listing = batch / "train.txt"
        if not listing.exists():
            missing.append(f"batch_{bid}: no train.txt")
            continue
        want = sum(1 for l in listing.read_text().splitlines() if l.strip())
        have = paths.count_files(batch / "images" / "train", {".jpg"})
        if have < want:
            missing.append(f"batch_{bid}: {have}/{want}")
    if missing:
        return Check(False, "shards incomplete -- " + ", ".join(missing[:3]))
    return Check(True, "all 10 shards match their split lists")


def _check_fleet(cfg: Config) -> Check:
    # A shard must exist for every id the server can pick, not just for the clients
    # that will run -- DEFAULT_BATCH_ID_RANGE is (1, 10) and is not configurable.
    want = max(cfg.n_vehicles, len(list(paths.BATCH_IDS)))
    fleet = vehicles.load_fleet()
    if len(fleet) < want:
        return Check(False, f"fleet has {len(fleet)} shards, need {want} "
                            f"(the server can assign any id in {tuple(paths.BATCH_IDS)[0]}.."
                            f"{tuple(paths.BATCH_IDS)[-1]})")
    meta = vehicles.load_fleet_meta()
    held = len(holdout.names())
    if meta:
        # The manifest says how this fleet was built, so intent is compared rather
        # than guessed. Alpha counts only where it means something.
        differs = [k for k, want in (("partition", cfg.partition), ("seed", cfg.seed),
                                     ("per_vehicle", cfg.per_vehicle)) if meta.get(k) != want]
        if cfg.partition == "dirichlet" and meta.get("alpha") != cfg.alpha:
            differs.append("alpha")
        # Absent means zero: every fleet.meta.json written before quantity skew
        # existed lacks the key, and reading that as a difference would rmtree and
        # rebuild a perfectly good fleet on the first run after this change.
        if (meta.get("size_skew") or 0.0) != (cfg.size_skew or 0.0):
            differs.append("size_skew")
        # A fleet built against a different holdout may hold images the global model
        # is about to be scored on. That would make the one honest metric partly
        # self-referential, so it forces a rebuild.
        if meta.get("holdout", 0) != held:
            differs.append("holdout")
        if differs:
            return Check(False, "fleet on disk differs in " + ", ".join(differs) +
                                f" (built {meta.get('partition')!r}, want {cfg.partition!r})")
    else:
        # Pre-manifest fleet: fall back to the old inference, which can only
        # distinguish random from everything else.
        if held:
            return Check(False, f"fleet predates the {held}-image holdout and cannot be shown "
                                f"to exclude it; rebuild")
        want_random = cfg.partition == "random"
        is_random = all(v.get("condition") == "random mix" for v in fleet)
        if want_random != is_random:
            return Check(False, f"fleet was built with a different partition than {cfg.partition!r}")

    # Under quantity skew a shard is *meant* to be smaller than per_vehicle, so the
    # bar becomes the floor skew itself clamps to. Comparing against per_vehicle would
    # call every skewed fleet stale and rebuild it on every run -- and a rebuild
    # rmtree's the shard directory, including during a federation.
    least = vehicles.size_floor(cfg.per_vehicle) if cfg.size_skew else cfg.per_vehicle
    short = [v for v in fleet if v.get("n_train", 0) < least]
    if short:
        return Check(False, f"{len(short)} vehicle(s) below {least} images")
    return Check(True, f"{len(fleet)} vehicles: " + ", ".join(v["condition"] for v in fleet))


def _check_sanity(_: Config) -> Check:
    marker = paths.STATE / "sanity.ok"
    return Check(marker.exists(), "passed previously" if marker.exists() else "not yet run")


def _check_federate(_: Config) -> Check:
    return Check(False, "always runs; this is the point of the pipeline")


def _check_verify(_: Config) -> Check:
    return Check(False, "always runs; asserts the four pass criteria")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
_ENV_PROBE = (
    "import torch;"
    "cap=torch.cuda.get_device_capability() if torch.cuda.is_available() else None;"
    "print('torch',torch.__version__,'cuda',torch.version.cuda,'capability',cap);"
    "print('device',torch.cuda.get_device_name(0) if cap else 'CPU ONLY');"
    "assert cap is not None,'no CUDA device visible -- the run would silently use CPU'"
)


def _cmd_env(_: Config) -> list[str]:
    return [PY, "-c", _ENV_PROBE]


def _cmd_dataset(_: Config) -> list[str]:
    return [PY, "-c",
            "import kagglehub;"
            f"print('path:', kagglehub.dataset_download('{paths.KAGGLE_DATASET}'))"]


def _cmd_populate(_: Config) -> list[str] | None:
    pool = paths.find_pool()
    if pool is None:
        raise RuntimeError("cannot populate: dataset pool not found; run the dataset stage")
    return [PY, "scripts/populate_images.py", "--pool", str(pool)]


def _cmd_fleet(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.build_fleet",
            "--vehicles", str(cfg.n_vehicles),
            "--per-vehicle", str(cfg.per_vehicle),
            "--seed", str(cfg.seed),
            "--partition", cfg.partition,
            "--alpha", str(cfg.alpha),
            "--size-skew", str(cfg.size_skew)]


#: One epoch on one shard, through the Python API rather than a console script.
#: `python -m ultralytics.cfg` worked until ultralytics 8.4 made cfg a package with
#: no __main__, and the `yolo` console script is not guaranteed to be on PATH for the
#: interpreter running the pipeline. Calling the same API the clients call cannot
#: drift from what a client does, which is the point of a sanity stage.
_SANITY = """
from ultralytics import YOLO
YOLO({model!r}).load({pretrained!r}).train(
    data={data!r}, epochs=1, imgsz={imgsz}, batch=4, device={device!r}, workers=2,
    plots=False, project="runs/pipeline", name="sanity", exist_ok=True)
"""


def _cmd_sanity(cfg: Config) -> list[str]:
    # The fleet's own shard, by absolute path. The previous target,
    # my-project/batch/batch_1/data.runtime.yaml, is written by a *client* at
    # runtime and does not exist until one has run -- so the stage could only pass
    # on a machine that had already done the thing the stage exists to check.
    data = paths.VEHICLE_BATCHES / "batch_1" / "data.yaml"
    if not data.exists():
        raise RuntimeError(f"no shard to sanity-check at {data}; run the fleet stage first")
    return [PY, "-c", _SANITY.format(
        model="models/yolov8s-13.yaml", pretrained="models/yolov8s.pt",
        data=str(data), imgsz=cfg.imgsz, device="0")]


#: What `flwr.exe` does, done by the interpreter itself. The console-script shim is a
#: generated .exe wrapper around exactly this import, and the .exe is the part Windows
#: Application Control blocks.
_FLWR_CLI = "from flwr.cli.app import app; app()"


def flwr_launcher() -> list[str]:
    r"""How to run flwr from *this* interpreter -- as a command list, not a path.

    Resolved next to ``sys.executable`` rather than taken from PATH. A shell a person
    types into has the venv's Scripts directory on PATH; a non-interactive one started by
    a script does not, and the stage died with "[WinError 2] The system cannot find the
    file specified", a message that says nothing about which file. Matching the
    interpreter also matters in its own right: flwr spawns its clients with the
    interpreter it was installed against.

    **It returns the module invocation, not the shim, and that is the point.** On
    2026-09-26 a run halted at the federate stage with

        [WinError 4551] An Application Control policy has blocked this file

    on `Scripts\flwr.exe`. Smart App Control has done this to this project before -- it is
    what makes conda unusable here (`docs/ENV_WINDOWS.md`) -- and a generated console
    script is an unsigned executable produced on this machine, which is exactly what those
    policies stop. `python -c "from flwr.cli.app import app; app()"` is what the shim
    wraps, runs the same CLI in the same environment, and has no .exe to block.

    The shim's *existence* is still what proves flwr is installed in this environment, so
    it is still looked for -- it is used as evidence, not as the launcher. Without it the
    error stays a sentence someone can act on rather than a ModuleNotFoundError from three
    frames deep.
    """
    here = Path(PY).parent
    installed = any((here / name).exists()
                    for name in ("flwr.exe", "flwr", "flwr-script.py"))
    if not installed and shutil.which("flwr") is None:
        raise RuntimeError(
            f"flwr not found next to {PY} or on PATH. Install it into this environment: "
            f"{PY} -m pip install flwr")
    return [PY, "-c", _FLWR_CLI]


def _cmd_federate(cfg: Config) -> list[str]:
    # num-supernodes MUST track the vehicle count. It lives in the federation config,
    # which flwr migrates out of pyproject.toml into ~/.flwr/config.toml on first run
    # -- so editing pyproject would be silently ignored and the run would hang
    # forever waiting for clients that never arrive. Override it on the CLI instead.
    # client-resources-num-gpus is a *scheduling* fraction, not a memory limit: Ray
    # runs floor(1 / fraction) clients at once and none of them is capped. Phase 0
    # measured 72 client episodes never overlapping and 99.1 % of the wall clock
    # inside a client, so this is the one lever whose ceiling is the client count --
    # and mathematically a no-op, because clients are independent within a round.
    # It defaults to 1.0 because the ceiling is VRAM, which depends on the profile:
    # a 6 308-image shard peaks at 15.9 GB of 16.3, a 1 400-image one at 5.1 GB.
    fed = (f"num-supernodes={cfg.n_vehicles} "
           f"client-resources-num-gpus={cfg.gpu_fraction} client-resources-num-cpus=2")
    if not cfg.ray_address:
        # Ray refuses num_cpus/num_gpus when attaching to a cluster that already
        # exists ("When connecting to an existing cluster, num_cpus and num_gpus must
        # not be provided"), so these are only valid when flwr starts Ray itself.
        fed += " init-args-num-gpus=1 init-args-num-cpus=8"
    return [*flwr_launcher(), "run", ".", "--stream", "--federation-config", fed,
            "--run-config",
            f'num_server_rounds={cfg.rounds} local_epochs={cfg.local_epochs} '
            f'min_clients={cfg.n_vehicles} fraction_fit=1.0 '
            # Quoted because flwr parses run-config values as TOML: an unquoted
            # fedadam is a bare word, not a string, and the run dies on parse.
            f'strategy="{cfg.strategy}" proximal_mu={cfg.proximal_mu} '
            # TOML booleans are lowercase; Python's True is a bare word flwr cannot
            # parse, and the run would die before the first round.
            f'cache="{cfg.cache}" local_bn={str(cfg.local_bn).lower()} '
            f'freeze_round1={cfg.freeze_round1} server_ema={cfg.server_ema} '
            f'fix_bn_from_round={cfg.fix_bn_from_round} imgsz={cfg.imgsz} '
            f'fraction_evaluate={cfg.fraction_evaluate}']


def _cmd_verify(_: Config) -> list[str]:
    return [PY, "-m", "pipeline.verify"]


def _check_holdout(cfg: Config) -> Check:
    info = holdout.meta()
    if not info:
        return Check(False, "not yet carved")
    if info.get("size") != cfg.holdout_size or info.get("seed") != cfg.holdout_seed:
        return Check(False, f"holdout on disk is size={info.get('size')} seed={info.get('seed')}, "
                            f"config wants size={cfg.holdout_size} seed={cfg.holdout_seed}")
    fp = info.get("fingerprint")
    return Check(True, f"{info.get('linked')} images held out, no vehicle sees them"
                       + (f" (fingerprint {fp})" if fp else ""))


def _cmd_holdout(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.holdout", "--build",
            "--size", str(cfg.holdout_size), "--seed", str(cfg.holdout_seed)]


def _check_validate(_: Config) -> Check:
    # Never satisfied: it is seconds of scanning, and the whole point is to run it
    # against the shards a federation is about to train on, not against a memory of
    # the last time they were sound.
    return Check(False, "always runs; nothing measured on bad shards is worth reading")


def _cmd_validate(_: Config) -> list[str]:
    return [PY, "-m", "pipeline.validate"]


def _check_evaluate(_: Config) -> Check:
    return Check(False, "always runs; the only honest global metric")


def _cmd_evaluate(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.holdout", "--evaluate", "--imgsz", str(cfg.imgsz)]


def _check_baseline(cfg: Config) -> Check:
    row = baseline.result()
    if not row:
        return Check(False, "no centralised run yet; federated numbers have no scale without it")
    want = cfg.rounds * cfg.local_epochs
    if row.get("epochs") != want:
        return Check(False, f"baseline was {row.get('epochs')} epochs, this run's budget is {want}")
    return Check(True, f"mAP50 {row['mAP50']:.4f} on {row['images']} pooled images")


def _cmd_baseline(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.baseline", "--rounds", str(cfg.rounds),
            "--local-epochs", str(cfg.local_epochs), "--imgsz", str(cfg.imgsz)]


STAGES: list[Stage] = [
    Stage("env", "Environment probe", False, _check_env, _cmd_env, est="~5 s"),
    Stage("dataset", "Download BDD100K", True, _check_dataset, _cmd_dataset, est="~10 min, 7.6 GB"),
    Stage("populate", "Populate shards", False, _check_populate, _cmd_populate, est="~1 min"),
    # Before fleet, always. A holdout carved afterwards is already inside somebody's
    # val split, and the "global" metric measured on it would be partly
    # self-referential -- a silent failure of exactly the kind this project collects.
    Stage("holdout", "Carve the shared holdout", False, _check_holdout, _cmd_holdout,
          cwd=paths.REPO, est="~20 s"),
    Stage("fleet", "Build vehicle fleet", False, _check_fleet, _cmd_fleet,
          cwd=paths.REPO, est="~30 s"),
    Stage("validate", "Validate the shards", False, _check_validate, _cmd_validate,
          cwd=paths.REPO, est="~10 s"),
    Stage("sanity", "Single-client GPU sanity", True, _check_sanity, _cmd_sanity, est="~2 min"),
    Stage("federate", "Federated run", True, _check_federate, _cmd_federate,
          data_root=paths.VEHICLE_ROOT, est="minutes to hours",
          crash_markers=CRASH_MARKERS),
    Stage("evaluate", "Score the global model on the holdout", False,
          _check_evaluate, _cmd_evaluate, cwd=paths.REPO, est="~10 s per round"),
    Stage("verify", "Verify pass criteria", False, _check_verify, _cmd_verify,
          cwd=paths.REPO, est="~5 s"),
    # Last, and gated: it trains a whole model. Not part of --all by accident.
    Stage("baseline", "Centralised ceiling on pooled data", True,
          _check_baseline, _cmd_baseline, cwd=paths.REPO, est="as long as one full run"),
]

BY_NAME = {s.name: s for s in STAGES}


def resolve(names: str | None, skip: str | None = None) -> list[Stage]:
    """'fleet,federate' -> [Stage, Stage]; None -> the whole chain, minus `skip`.

    Skipping by name rather than by listing everything else keeps a caller from
    silently missing a stage added later -- the failure mode a hardcoded chain in a
    shell script would have.
    """
    if names:
        out = []
        for n in [x.strip() for x in names.split(",") if x.strip()]:
            if n not in BY_NAME:
                raise SystemExit(f"unknown stage {n!r}; known: {', '.join(BY_NAME)}")
            out.append(BY_NAME[n])
    else:
        out = list(STAGES)

    for n in [x.strip() for x in (skip or "").split(",") if x.strip()]:
        if n not in BY_NAME:
            raise SystemExit(f"unknown stage {n!r}; known: {', '.join(BY_NAME)}")
        out = [s for s in out if s.name != n]
    return out


def snapshot(cfg: Config) -> list[dict]:
    """Current state of every stage, for --list and for the control dashboard."""
    rows = []
    for s in STAGES:
        try:
            c = s.check(cfg)
        except Exception as e:                      # a broken check must not hide the stage
            c = Check(False, f"check failed: {e}")
        rows.append({"name": s.name, "title": s.title, "gated": s.gated,
                     "satisfied": c.satisfied, "detail": c.detail, "est": s.est})
    return rows
