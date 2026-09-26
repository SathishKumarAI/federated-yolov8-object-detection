"""Where things live, and the environment every stage subprocess needs.

Single place that knows the repo layout, so the rest of the package never
hardcodes a path or an env var that has already bitten someone.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "my-project"          # invoked, never modified
HERE = Path(__file__).resolve().parent
STATE = HERE / ".state"            # markers + attribute cache; gitignored
MLFLOW_STORE = HERE / "mlruns"     # gitignored
MLFLOW_DB = MLFLOW_STORE / "mlflow.db"
MLFLOW_ARTIFACTS = MLFLOW_STORE / "artifacts"
#: One experiment for both writers -- ultralytics' per-vehicle training curves and
#: the sink's federation-level facts. Lives here rather than in mlflow_sink because
#: subprocess_env has to name it too, and a constant in two places is a constant that
#: will disagree with itself.
MLFLOW_EXPERIMENT = "federated-yolov8"
REPORTS = HERE / "reports"         # gitignored
# Vehicle shards live here, NOT in my-project. task.get_batch_path() resolves
# FL_AV_DATA_ROOT/batch/batch_<id>, so pointing that env var at VEHICLE_ROOT is all
# it takes to run the federation on simulated vehicles without touching my-project.
VEHICLE_ROOT = HERE / "vehicles"
VEHICLE_BATCHES = VEHICLE_ROOT / "batch"

BATCH_IDS = range(1, 11)

# Where kagglehub puts solesensei/solesensei_bdd100k. The archive nests a second
# bdd100k inside the first -- that is the archive's own layout, not a typo.
KAGGLE_DATASET = "solesensei/solesensei_bdd100k"
_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "solesensei" / "solesensei_bdd100k"


def find_pool() -> Path | None:
    """Locate the images/100k directory holding train/ and val/, or None."""
    if not _CACHE.is_dir():
        return None
    for candidate in sorted(_CACHE.glob("versions/*/**/images/100k")):
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    return None


LABELS_DIR_NAME = "bdd100k_labels_release"


def find_label_jsons() -> list[Path]:
    """The raw BDD100K attribute JSONs (weather/scene/timeofday), if downloaded."""
    if not _CACHE.is_dir():
        return []
    return sorted(_CACHE.glob(f"versions/*/{LABELS_DIR_NAME}/**/bdd100k_labels_images_*.json"))


def mlflow_uri() -> str:
    """SQLite, not the filesystem store.

    mlflow 3.15 refuses the file:// backend outright -- "The filesystem tracking
    backend (e.g., './mlruns') is in maintenance mode and will not receive further
    updates" -- so the sink this project wired up logged nothing for a whole six-round
    run and the failure was a line in a stage log.

    Absolute and POSIX-slashed on purpose: SQLAlchemy reads everything after
    ``sqlite:///`` as a path, a Windows backslash escapes inside it, and a relative one
    would put a database wherever the subprocess happened to be standing -- which is
    the trap that has already truncated this project's metrics.csv once.
    """
    MLFLOW_STORE.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{MLFLOW_DB.as_posix()}"


def log_dirs() -> list[Path]:
    """Every directory a run's logs can land in.

    **`VEHICLE_ROOT / "logs"` is where a federation's logs actually are**, and leaving it
    out cost three weeks of blind diagnostics. `utils/logging_setup.project_root` resolves
    a relative log path against `FL_AV_DATA_ROOT` first, and `subprocess_env` sets that to
    `VEHICLE_ROOT` so my-project reads the simulated fleet -- so the server log follows the
    data root, not the package. Measured 2026-09-26: every federation since the fleet
    existed wrote `pipeline/vehicles/logs/server.<pid>.log`, this function listed neither
    that directory nor anything under it, and so

        latest_run_log()      -> my-project/logs/server.33044.log   (Sep 2 13:43)
        federation_learned()  -> (True, 'weights moved every round: [-600.77, -1021.01]')

    reported PASS from a three-week-old log, today, for every run. The B4 guard -- the one
    signal this project says to trust above the metrics -- was inert, and it was inert in
    the direction that passes.

    Ordered with the fleet first, because that is where a current run writes and
    `latest_run_log` breaks ties by mtime across all of them.

    `PROJECT / "logs"` still holds the logs of anything run with cwd=my-project and no data
    root -- the sanity stage, the baseline, a bare `pytest`. `REPO / "logs"` is history from
    before relative paths were anchored at all; finding nothing there is the expected state.
    """
    return [d for d in (VEHICLE_ROOT / "logs", PROJECT / "logs", REPO / "logs")
            if d.is_dir()]


def subprocess_env(ray_address: str | None = None, data_root: Path | None = None) -> dict:
    """Environment for every stage subprocess.

    Two of these are load-bearing and cost hours when missing:

    * ``FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION`` -- without it flwr >= 1.31
      builds its own runtime env via ``uv sync`` and installs the CPU-only torch
      wheel, so every client trains on CPU at ~5.5x the wall clock, silently.
    * ``FL_AV_DATA_ROOT`` -- pins the shard root to this checkout rather than the
      path baked into the committed ``data.yaml``.
    """
    env = os.environ.copy()
    env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    # Every child writes UTF-8 whatever the console codepage is. Without it, a
    # subprocess whose stdout is a redirected pipe gets cp1252 on Windows, and flwr
    # -- which prints a flower emoji in its banner -- dies with "'charmap' codec
    # can't encode character '\U0001f338'". The federation then fails for a reason
    # that has nothing to do with federated learning, and only when run from a
    # script rather than a terminal.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # The interpreter's own Scripts/bin directory goes first on PATH. An interactive
    # shell has it because the venv is activated; a shell started by a script does
    # not, and flwr then fails to launch `flower-superlink`, which it resolves from
    # PATH rather than from its own location:
    #   Unable to launch `flower-superlink` for local simulation: [WinError 2]
    # Resolving our own entry points is not enough -- the children we spawn spawn
    # children of their own.
    # It goes *first*, not merely present. The earlier version skipped the prepend
    # whenever the directory was already anywhere on PATH, which is a different and
    # much weaker guarantee: on a GitHub Windows runner `setup-python` puts it on PATH
    # behind PowerShell's own directory, and "somewhere on PATH" is precisely the
    # state in which another environment's launcher resolves first. Any existing copy
    # is dropped rather than left behind, so the entry cannot appear twice.
    scripts = str(Path(sys.executable).parent)
    same = os.path.normcase(scripts)
    rest = [p for p in env.get("PATH", "").split(os.pathsep)
            if p and os.path.normcase(p) != same]
    env["PATH"] = os.pathsep.join([scripts, *rest])
    env["FL_AV_DATA_ROOT"] = str(data_root or PROJECT)
    # Ultralytics' own MLflow callback reads both of these. The URI has to name the
    # same backend the pipeline's sink writes to, or the two halves of a run --
    # per-vehicle training curves and federation-level facts -- land in two stores and
    # neither is complete.
    #
    # The experiment name matters for a second reason. Without it the callback falls
    # back to `trainer.args.project`, creates an experiment of its own with no artifact
    # location, and MLflow then resolves artifacts against the CWD -- which for every
    # stage in this pipeline is my-project. One federation left 24 run directories under
    # my-project/mlruns/, outside the store, ungitignored and invisible to the sink.
    # Naming the sink's experiment reuses its explicit artifact location instead.
    env.setdefault("MLFLOW_TRACKING_URI", mlflow_uri())
    env.setdefault("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT)
    if ray_address:
        # Makes flwr's ray.init() attach to an already-running head node instead
        # of creating its own with include_dashboard=False hardcoded.
        env["RAY_ADDRESS"] = ray_address
    else:
        # Explicitly clear it. Inheriting a stale RAY_ADDRESS makes flwr attach to a
        # cluster it did not configure, and Ray then rejects the num_cpus/num_gpus
        # that flwr passes: "When connecting to an existing cluster, num_cpus and
        # num_gpus must not be provided."
        env.pop("RAY_ADDRESS", None)
    return env


def count_files(directory: Path, suffixes: set[str] | None = None) -> int:
    """Count files in a directory without materialising the list."""
    if not directory.is_dir():
        return 0
    n = 0
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and (suffixes is None or Path(entry.name).suffix.lower() in suffixes):
                n += 1
    return n
