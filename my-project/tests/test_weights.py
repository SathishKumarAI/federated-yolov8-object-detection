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


def test_integer_buffers_survive_aggregation():
    """FedAvg's in-place path silently zeroes every int64 buffer.

    `aggregate_inplace` scales each client's array by `num_examples / total` using
    `np.multiply(x, f, out=x)`. For an int64 array that truncates in place: 89 * 0.5
    written back into int64 is 0 -- and it is 0 for ANY fleet larger than one, because the
    scaling factor is always below 1.

    Measured on a real 2-client round: clients logged `BN counters min=89 max=89`, and the
    model the server was about to save logged `min=0 max=0`. `num_batches_tracked` is inert
    for this model (PyTorch divides by it only when `momentum is None`; Ultralytics uses
    0.03), so nothing trained wrongly -- but it is part of the payload being corrupted in
    transit, and the server therefore asks for the copying path.
    """
    import numpy as np
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
    from flwr.server.strategy.aggregate import aggregate, aggregate_inplace

    def one(n):
        return (None, FitRes(status=Status(Code.OK, ""), num_examples=n, metrics={},
                             parameters=ndarrays_to_parameters(
                                 [np.array(89, dtype=np.int64)])))

    assert int(aggregate_inplace([one(1400), one(1400)])[0]) == 0, \
        "if this ever stops being 0, flwr fixed it and `inplace=False` can go"
    assert float(aggregate([([np.array(89, dtype=np.int64)], 1400)] * 2)[0]) == 89.0, \
        "the copying path is the one that keeps the counter"


def test_the_aggregate_keeps_the_architecture_dtypes():
    """FedAvg's copying path returns every array as float64, counters included.

    That breaks the round-trip identity rather than merely looking untidy: a float-only
    checksum then counts 57 values the clients excluded. Measured the day it appeared --
    the server published 4643.564230 against a weighted mean of -429.435242, a difference
    of exactly 57 x 89. The aggregate of a state_dict is a state_dict.
    """
    import numpy as np

    from my_project.get_set_model import learning_checksum
    from my_project.server_app import as_state_dict_dtypes

    reference = [np.zeros(3, dtype=np.float32), np.array(0, dtype=np.int64)]
    aggregated = [np.array([1.5, 2.5, 3.0], dtype=np.float64),
                  np.array(89.0, dtype=np.float64)]      # what aggregate() hands back

    assert learning_checksum(aggregated) == 96.0, "the counter is being counted: 7 + 89"

    restored = as_state_dict_dtypes(aggregated, reference)
    assert [a.dtype for a in restored] == [np.dtype(np.float32), np.dtype(np.int64)]
    assert learning_checksum(restored) == 7.0, "and now it is not"
    assert int(restored[1]) == 89, "while the counter itself survives"
