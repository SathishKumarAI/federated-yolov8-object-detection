from __future__ import annotations  # lazy annotations: YOLO type hints need no import

import os
import logging
import warnings
import numpy as np
import torch
import yaml
import platform
import sys
from pathlib import Path
from collections import OrderedDict
# ultralytics (YOLO) is only referenced in type hints below; with future
# annotations it is never evaluated at import time, so dataset/path helpers like
# count_shard_examples stay importable without the heavy ultralytics dependency.

from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = configure_logging("task", "logs/task.log")

# Detect operating system
IS_WINDOWS = platform.system() == "Windows"
OS_NAME = platform.system()
logger.info(f"[Task] Detected operating system: {OS_NAME}")

# Constants for model paths and URLs
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
DEFAULT_IMAGE_SIZE = 640  # Default image size for training/validation
DEFAULT_BATCH_ID_RANGE = (1, 10)  # Consistent with server and client

# Base path for data. Defaults to the package's parent dir (so the repo layout
# works out-of-the-box on any OS); override with FL_AV_DATA_ROOT to point at a
# mounted data volume in containers/production. No hardcoded per-machine paths.
BASE_DATA_PATH = os.environ.get(
    "FL_AV_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
logger.info(f"[Task] Using base data path: {BASE_DATA_PATH}")

def download_model():
    """Download YOLO model if not found."""
    logger.info(f"[Task] Checking if model exists at {MODEL_PATH}...")

    # Ensure directory exists
    model_dir = os.path.dirname(MODEL_PATH) or "."  # Use current directory if empty
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(MODEL_PATH):  # Check before downloading
        logger.info(f"[Task] Model not found. Downloading from {MODEL_URL}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("[Task] Model downloaded successfully.")
        except Exception as e:
            logger.error(f"[Task] Failed to download the model from {MODEL_URL}: {e}", exc_info=True)
            raise RuntimeError(f"Cannot download a valid YOLO model to {MODEL_PATH}.") from e
    else:
        logger.info(f"[Task] Model already exists at {MODEL_PATH}, skipping download.")

def get_batch_path(batch_id):
    """
    Get the path to a batch directory.
    
    Args:
        batch_id: The batch ID
        
    Returns:
        Path: Path to the batch directory
    """
    batch_dir = Path(os.path.join(BASE_DATA_PATH, "batch", f"batch_{batch_id}"))
    logger.debug(f"[Task] Batch path for ID {batch_id}: {batch_dir}")
    return batch_dir

def get_data_yaml_path(batch_id):
    """
    Get the path to a batch's data.yaml file.
    
    Args:
        batch_id: The batch ID
        
    Returns:
        Path: Path to the data.yaml file
    """
    return get_batch_path(batch_id) / "data.yaml"


def should_checkpoint(server_round, save_every, num_rounds):
    """
    Whether to persist the global model after a given round.

    True on the configured cadence (every ``save_every`` rounds) and always on
    the final round, so the federated result is never lost. Pure/dependency-free
    so it is unit-testable without the flwr/ultralytics runtime.
    """
    save_every = max(int(save_every), 1)
    return (server_round % save_every == 0) or (server_round >= num_rounds)


def count_shard_examples(batch_id, split="train"):
    """
    Count the number of examples in a client's data shard for a given split.

    FedAvg weights each client's update by ``num_examples``; reporting a real
    count (instead of a constant) makes aggregation correctly proportional to
    shard size. Counts the images actually present under ``images/<split>/``, and
    only falls back to the committed split list (``train.txt`` / ``val.txt`` /
    ``test.txt``) when that directory is missing. The split list is a manifest of
    what *should* be there; when the two disagree the files on disk are what get
    trained on, and weighting by the manifest silently skews aggregation.

    Args:
        batch_id: The batch ID.
        split: "train", "val", or "test".

    Returns:
        int: Number of examples (>= 1, so a client never gets zero weight).
    """
    base = get_batch_path(batch_id)

    # Primary: the images actually on disk -- that is what Ultralytics trains on.
    img_dir = base / "images" / split
    if img_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        count = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in exts)
        if count > 0:
            logger.debug(f"[Task] batch {batch_id} {split}: {count} examples (from {img_dir})")
            return count

    # Fallback: the committed split list, when the images are not populated yet.
    list_file = base / f"{split}.txt"
    if list_file.exists():
        try:
            with open(list_file, "r") as f:
                count = sum(1 for line in f if line.strip())
            if count > 0:
                logger.debug(f"[Task] batch {batch_id} {split}: {count} examples (from {list_file.name})")
                return count
        except Exception as e:
            logger.warning(f"[Task] Failed reading {list_file}: {e}")

    logger.warning(f"[Task] Could not count examples for batch {batch_id} split {split}; defaulting to 1.")
    return 1


def materialize_data_yaml(batch_id):
    """
    Produce a runtime data.yaml with an absolute, machine-correct ``path``,
    WITHOUT mutating the tracked ``data.yaml``.

    The committed ``batch/*/data.yaml`` may carry a stale/foreign ``path``. Rather
    than rewriting that tracked file on every run (which dirtied the git tree and
    baked machine-specific paths into version control), we read it, set ``path`` to
    the resolved absolute batch directory, and write a sibling ``data.runtime.yaml``
    (gitignored). Training/validation point at this runtime copy.

    Args:
        batch_id: The batch ID.

    Returns:
        Path: Path to the runtime data.yaml, or the original path if generation
              fails (so callers still have something to try).
    """
    src = get_data_yaml_path(batch_id)
    if not src.exists():
        logger.warning(f"[Task] data.yaml not found at {src}, cannot materialize runtime copy")
        return src

    try:
        with open(src, "r") as f:
            data = yaml.safe_load(f) or {}

        # Absolute path to this batch dir, resolved for the current machine/OS.
        data["path"] = str(get_batch_path(batch_id).resolve())

        runtime = src.parent / "data.runtime.yaml"
        with open(runtime, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"[Task] Materialized runtime data.yaml for batch {batch_id} at {runtime} (path={data['path']})")
        return runtime
    except Exception as e:
        logger.error(f"[Task] Failed to materialize data.yaml for batch {batch_id}: {e}", exc_info=True)
        return src

# ----------------------------------------------------------
# 1) Configuration Loader
# ----------------------------------------------------------
def load_config(config_path):
    """
    Loads a YAML (or JSON) configuration file and returns a Python dict.

    :param config_path: Path to your config file (e.g. 'config.yaml').
    :return: Dictionary with parsed configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"[Task] Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"[Task] Failed to load config from {config_path}: {e}", exc_info=True)
        return {}


# ----------------------------------------------------------
# 3) GPU-based Batch Size Estimation
# ----------------------------------------------------------
def get_optimal_batch_size(imgsz: int = DEFAULT_IMAGE_SIZE):
    """
    Attempts to pick a suitable batch size based on GPU free memory.

    Scaled by ``(DEFAULT_IMAGE_SIZE / imgsz) ** 2``, because activation memory grows with
    the pixel count and this heuristic used to ignore the image size entirely: at
    ``imgsz=1024`` a batch of 16 is 2.56x the activations it was sized for, and the run
    dies part-way through a round rather than at its start. Never below 2 -- a batch of 1
    makes BatchNorm meaningless, and Ultralytics' nominal batch of 64 already means small
    batches accumulate before stepping. The caller still warns when the arithmetic leaves
    a shard too small for any optimizer step at all.
    """
    try:
        batch = 4
        if torch.cuda.is_available():
            # mem_get_info()[0] -> total free memory on current GPU in bytes
            free_memory = torch.cuda.mem_get_info()[0]
            free_gb = free_memory / 1e9
            logger.info(f"[Task] Detected GPU with {free_gb:.2f} GB free memory.")

            # Simple heuristic for batch size
            if free_memory > 10e9:
                batch = 16
            elif free_memory > 6e9:
                batch = 8

        scaled = max(2, int(batch * (DEFAULT_IMAGE_SIZE / max(int(imgsz), 1)) ** 2))
        if scaled != batch:
            logger.info(f"[Task] Batch {batch} -> {scaled} for imgsz={imgsz}: activation "
                        f"memory scales with the pixel count, not with the image count.")
        return scaled
    except Exception as e:
        logger.warning(f"[Task] Batch size detection failed: {str(e)}. Falling back to 4.")
        return 4


# ----------------------------------------------------------
# 5) Optional YOLO Training / Validation Helpers
# ----------------------------------------------------------
def train_custom(model: YOLO, device: str, epochs: int = 1, data_yaml: str = "data.yaml"):
    """
    An example custom training wrapper that calls YOLO's built-in train method.
    :param model:     YOLO model instance.
    :param device:    "cuda:0" or "cpu".
    :param epochs:    Number of local epochs to train.
    :param data_yaml: Path to the dataset config (YAML) describing train/val sets.
    :return:          Float representing the final training loss.
    """
    try:
        # data_yaml is expected to already point at a valid (absolute-path) config,
        # e.g. produced by materialize_data_yaml(); no in-place mutation here.
        logger.info(f"[Task] Training YOLO model for {epochs} epoch(s) on device: {device}")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=DEFAULT_IMAGE_SIZE,
            device=device,
            verbose=False
        )
        final_loss = results.results_dict.get("train/loss", float("inf"))
        logger.info(f"[Task] Training completed with loss={final_loss}")
        return final_loss
    except Exception as e:
        logger.error(f"[Task] train_custom error: {e}", exc_info=True)
        return float("inf")


def test_custom(model: YOLO, device: str, data_yaml: str = "data.yaml"):
    """
    An example custom validation wrapper that calls YOLO's built-in val method.
    :param model:     YOLO model instance.
    :param device:    "cuda:0" or "cpu".
    :param data_yaml: Path to the dataset config (YAML) describing val set.
    :return:          (loss, mAP50) as float and float.
    """
    try:
        # data_yaml is expected to already point at a valid (absolute-path) config,
        # e.g. produced by materialize_data_yaml(); no in-place mutation here.
        logger.info(f"[Task] Evaluating YOLO model on device: {device}")
        results = model.val(
            data=data_yaml,
            imgsz=DEFAULT_IMAGE_SIZE,
            device=device,
            verbose=False
        )
        loss = results.results_dict.get("val/loss", float("inf"))
        map50 = results.results_dict.get("metrics/mAP50", 0.0)
        logger.info(f"[Task] Validation completed: loss={loss}, mAP50={map50}")
        return float(loss), map50
    except Exception as e:
        logger.error(f"[Task] test_custom error: {e}", exc_info=True)
        return float("inf"), 0.0
