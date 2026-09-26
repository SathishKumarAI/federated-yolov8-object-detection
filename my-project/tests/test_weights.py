"""Weight-transfer correctness tests for get_set_model.

These prove the FedAvg-critical property: a get_weights -> set_weights round-trip
restores the FULL model state, including BatchNorm running buffers — not just the
learnable parameters. A toy BatchNorm-containing module is used so the tests run
without downloading the full YOLOv8 model (a YOLO-specific check lives in the
integration tests / conftest model fixture).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from my_project.get_set_model import get_weights, set_weights


class TinyBNNet(nn.Module):
    """Conv + BatchNorm + Linear: has params AND BN buffers, like YOLO blocks."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.fc(x.mean(dim=(2, 3)))


def _train_a_step(model):
    """Run a forward/backward in train mode so BN buffers diverge from defaults."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 3, 8, 8)
    out = model(x)
    out.sum().backward()
    opt.step()
    # Several forward passes update running_mean/var and num_batches_tracked.
    for _ in range(3):
        model(torch.randn(8, 3, 8, 8))


def test_state_dict_includes_buffers():
    """get_weights must serialize the whole state_dict, strictly more than params."""
    model = TinyBNNet()
    n_state = len(get_weights(model))
    n_params = len(list(model.parameters()))
    assert n_state == len(model.state_dict())
    assert n_state > n_params, (
        "get_weights returned only parameters — BatchNorm buffers are missing, "
        "which breaks FedAvg correctness."
    )


def test_weight_roundtrip_restores_bn_buffers():
    """A full round-trip restores every state tensor incl. BN running stats."""
    src = TinyBNNet()
    _train_a_step(src)  # make BN buffers non-trivial

    # Sanity: source BN running_mean is no longer the default zeros.
    assert not torch.allclose(src.bn.running_mean, torch.zeros_like(src.bn.running_mean))

    weights = get_weights(src)

    dst = TinyBNNet()  # fresh model with default buffers
    assert not torch.allclose(dst.bn.running_mean, src.bn.running_mean)

    ok = set_weights(dst, weights)
    assert ok is True

    # Every tensor in the state dict must match after the round-trip.
    for key, src_t in src.state_dict().items():
        dst_t = dst.state_dict()[key]
        assert dst_t.dtype == src_t.dtype, f"dtype drift at {key}"
        assert torch.allclose(dst_t.float(), src_t.float()), f"value drift at {key}"

    # num_batches_tracked is an int64 buffer — confirm it stayed integer and equal.
    assert dst.bn.num_batches_tracked.dtype == torch.int64
    assert int(dst.bn.num_batches_tracked) == int(src.bn.num_batches_tracked)


def test_set_weights_length_mismatch_returns_false():
    """Wrong-length input must fail safely (False), not partially load."""
    model = TinyBNNet()
    weights = get_weights(model)
    assert set_weights(model, weights[:-1]) is False


def test_roundtrip_through_numpy_list():
    """Weights must survive as a plain list of numpy arrays (the wire format)."""
    src = TinyBNNet()
    _train_a_step(src)
    weights = get_weights(src)
    assert all(isinstance(w, np.ndarray) for w in weights)
    dst = TinyBNNet()
    assert set_weights(dst, [w.copy() for w in weights]) is True
    assert torch.allclose(dst.bn.running_var, src.bn.running_var)


def test_the_checksum_ignores_the_batchnorm_counters():
    """The B4 guard is "equal consecutive checksums mean nothing is being learned". That
    only works if the number cannot move on its own.

    `num_batches_tracked` is int64 and climbs by one per optimizer step in every BatchNorm
    layer -- 57 of them, 88 steps a round. A naive sum therefore rises about +5 000 every
    round whatever the weights do, and a frozen federation would look like a learning one.

    It was float-only by accident until 2026-09-26: the clients sent `best.pt`'s EMA and
    `ModelEMA.update` lerps only floating-point tensors, so every counter arrived as 0.
    Sending the trained module brought the real counters with it.
    """
    import numpy as np

    from my_project.server_app import learning_checksum

    floats = [np.array([1.5, -0.5], dtype=np.float32)]
    counters = [np.array([88], dtype=np.int64)]

    assert learning_checksum(floats) == 1.0
    assert learning_checksum(floats + counters) == 1.0,         "an integer buffer must not be able to move the checksum"
    # And a round later, with the counters bumped and nothing learned:
    assert learning_checksum(floats + [np.array([176], dtype=np.int64)]) == 1.0


def test_the_checksum_still_moves_when_a_weight_does():
    """The guard on the guard: a filter that excluded everything would also be stable."""
    import numpy as np

    from my_project.server_app import learning_checksum

    before = [np.array([1.0], dtype=np.float32), np.array([1], dtype=np.int64)]
    after = [np.array([1.0001], dtype=np.float32), np.array([1], dtype=np.int64)]

    assert learning_checksum(before) != learning_checksum(after)
