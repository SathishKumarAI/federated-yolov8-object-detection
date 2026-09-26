"""num_examples must reflect real shard size — it is FedAvg's aggregation weight.

Built on a temporary shard rather than the repo's own `batch/`, so the result does
not change depending on whether someone has populated their images yet.
"""
import importlib
import sys


def _task(monkeypatch, root):
    monkeypatch.setenv("FL_AV_DATA_ROOT", str(root))
    sys.modules.pop("my_project.task", None)
    return importlib.import_module("my_project.task")


def _shard(root, batch_id, *, images, listed):
    """Build batch_<id> with `images` real JPEGs and a split list naming `listed`."""
    batch = root / "batch" / f"batch_{batch_id}"
    train = batch / "images" / "train"
    train.mkdir(parents=True)
    for i in range(images):
        (train / f"img_{i}.jpg").write_bytes(b"")
    (batch / "train.txt").write_text("".join(f"img_{i}.jpg\n" for i in range(listed)))
    return batch


def test_counts_the_images_actually_on_disk(tmp_path, monkeypatch):
    _shard(tmp_path, 1, images=7, listed=7)
    assert _task(monkeypatch, tmp_path).count_shard_examples(1, "train") == 7


def test_disk_wins_when_the_split_list_disagrees(tmp_path, monkeypatch):
    """The manifest says 6308, the shard holds 10. Ultralytics trains on the 10, so
    weighting the update by 6308 would silently skew aggregation."""
    _shard(tmp_path, 1, images=10, listed=6308)
    assert _task(monkeypatch, tmp_path).count_shard_examples(1, "train") == 10


def test_falls_back_to_the_split_list_when_images_are_not_populated(tmp_path, monkeypatch):
    batch = tmp_path / "batch" / "batch_1"
    batch.mkdir(parents=True)
    (batch / "train.txt").write_text("".join(f"img_{i}.jpg\n" for i in range(42)))
    assert _task(monkeypatch, tmp_path).count_shard_examples(1, "train") == 42


def test_not_the_old_hardcoded_constant(tmp_path, monkeypatch):
    _shard(tmp_path, 1, images=23, listed=23)
    assert _task(monkeypatch, tmp_path).count_shard_examples(1, "train") != 10


def test_missing_batch_returns_at_least_one(tmp_path, monkeypatch):
    # A non-existent shard must never yield 0 — that would zero its FedAvg weight.
    assert _task(monkeypatch, tmp_path).count_shard_examples(99999, "train") >= 1


def test_the_batch_shrinks_when_the_images_grow():
    """Activation memory scales with the pixel count, and this heuristic used to read
    only free VRAM. At imgsz=1024 a batch of 16 is 2.56x the activations it was sized
    for, and the run dies part-way through a round rather than refusing at its start."""
    from my_project.task import get_optimal_batch_size

    at_640 = get_optimal_batch_size(640)
    at_1024 = get_optimal_batch_size(1024)
    at_320 = get_optimal_batch_size(320)

    assert at_1024 < at_640 <= at_320
    # Never 1: BatchNorm over a single image is meaningless, and Ultralytics' nominal
    # batch of 64 already accumulates before it steps.
    assert at_1024 >= 2
