"""The server averages the aggregate across rounds, because the client cannot.

Ultralytics builds a fresh `ModelEMA` for every `train()` with `updates = 0`
(`engine/trainer.py:404`) and ramps decay as `0.9999*(1 - exp(-updates/2000))`
(`utils/torch_utils.py:734`). 88 client steps reach d = 0.043 and 352 reach 0.161, while
the centralised ceiling's 3 168 steps reach 0.795. So the fleet ends every round on an
essentially unsmoothed model and the thing it is compared against does not, and no
client-side setting can change that -- the counter restarts by construction. The
averaging has to happen where the rounds are.

The failure these guard is the one that makes an EMA look like a regression: an
uncorrected average drags round 1 back toward the initial model, and the run then reads
as "smoothing hurts" when what hurt was the bias.
"""
import numpy as np
import pytest

from my_project.server_app import CustomBatchStrategy


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """`__init__` truncates logs/metrics.csv and mkdirs checkpoints/ relative to CWD."""
    monkeypatch.chdir(tmp_path)


def _strategy(decay):
    return CustomBatchStrategy(server_ema=decay, num_rounds=4)


def test_round_one_is_exactly_the_aggregate():
    """Bias correction, and it is the whole reason the method is usable at round 1.
    Without it the fleet would continue from `d*initial + (1-d)*aggregate` -- a model
    pulled back toward the weights it started from, in the round that moves furthest."""
    s = _strategy(0.9)
    first = [np.array([1.0, 2.0], dtype=np.float32)]

    out = s._apply_server_ema(first, server_round=1)

    np.testing.assert_allclose(out[0], first[0])


def test_a_constant_aggregate_is_returned_unchanged_every_round():
    """The sharpest property of a bias-corrected EMA: if nothing changes, it reports no
    change. An off-by-one in the correction shows up here as drift."""
    s = _strategy(0.7)
    w = [np.array([3.0, -4.0], dtype=np.float32)]

    for rnd in range(1, 6):
        out = s._apply_server_ema([w[0].copy()], server_round=rnd)
        np.testing.assert_allclose(out[0], w[0], rtol=1e-6)


def test_it_lags_a_step_change_instead_of_following_it():
    """What the method is *for*: a round that jumps is damped, not obeyed. The value
    must land strictly between the old and the new, or there is no smoothing."""
    s = _strategy(0.5)
    s._apply_server_ema([np.array([0.0], dtype=np.float32)], server_round=1)

    out = s._apply_server_ema([np.array([1.0], dtype=np.float32)], server_round=2)

    # ema = 0.5*0 + 0.5*1 = 0.5, corrected by 1 - 0.5^2 = 0.75  ->  0.6667
    assert 0.0 < out[0][0] < 1.0
    np.testing.assert_allclose(out[0][0], (0.5 * 0.0 + 0.5 * 1.0) / 0.75, rtol=1e-6)


def test_dtypes_survive_the_blend():
    """`num_batches_tracked` is int64 in the state_dict and the positional wire format
    carries it like everything else. An EMA that returned float64 for it would only fail
    later, inside a client's `load_state_dict`."""
    s = _strategy(0.5)
    arrays = [np.array([1.0, 2.0], dtype=np.float32), np.array([7], dtype=np.int64)]

    s._apply_server_ema([a.copy() for a in arrays], server_round=1)
    out = s._apply_server_ema([a.copy() for a in arrays], server_round=2)

    assert [a.dtype for a in out] == [np.dtype(np.float32), np.dtype(np.int64)]


def test_off_by_default_means_the_aggregate_is_untouched():
    """Every number this repo has measured was taken with no server EMA. Turning the
    lever on must be a decision, not a side effect of upgrading."""
    s = CustomBatchStrategy(num_rounds=2)
    assert s.server_ema == 0.0
    assert s._ema_weights is None


def test_a_changed_tensor_list_raises_rather_than_restarting_the_average():
    """Averaging two different architectures would produce a model that is neither, and
    would do it silently. This repo's whole failure catalogue is silent."""
    s = _strategy(0.5)
    s._apply_server_ema([np.zeros(3, dtype=np.float32)], server_round=1)

    with pytest.raises(RuntimeError, match="two different models"):
        s._apply_server_ema([np.zeros(4, dtype=np.float32)], server_round=2)
