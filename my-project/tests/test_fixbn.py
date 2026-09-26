"""FixBN: the normalisation statistics come from the aggregate and do not move.

Why this and not FedBN, given the repo already has FedBN: FedBN keeps each vehicle's
BatchNorm local, which leaves **no single global model** -- the saved checkpoint carries an
averaged BatchNorm that is no vehicle's, so a holdout score on it measures a model that
never existed. FixBN (arXiv:2303.06530) pins the *shared* statistics instead, so every
client normalises the same way, nothing drifts, and the holdout keeps measuring the model
that was trained. The paper measures it beating GroupNorm across most federated settings.

The failure these guard is a freeze that lasts one epoch. `BaseTrainer._model_train` calls
`self.model.train()` at the start of every epoch (`engine/trainer.py:701`), so an `eval()`
set from outside the trainer is undone on epoch 2 -- leaving a frozen first epoch and three
unfrozen ones, which is a warm-up phase nobody chose, inside the round, invisible in every
log.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from my_project.server_app import round_config
from my_project.trainers import freeze_batchnorm_stats

from tests.test_weights import TinyBNNet, _train_a_step


def test_frozen_statistics_do_not_move_while_the_weights_do():
    """The point of the method: the layer keeps learning its affine parameters, and
    stops rewriting the statistics that every other client also has to live with."""
    model = TinyBNNet()
    _train_a_step(model)  # give the running estimates something other than 0/1

    frozen = freeze_batchnorm_stats(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(3):
        _train_a_step(model)
    after = model.state_dict()

    assert frozen == 1
    torch.testing.assert_close(after["bn.running_mean"], before["bn.running_mean"])
    torch.testing.assert_close(after["bn.running_var"], before["bn.running_var"])
    assert not torch.equal(after["conv.weight"], before["conv.weight"]), \
        "the network must still be learning, or this is not FixBN, it is a stop"


def test_the_counter_keeps_climbing_and_that_is_inert():
    """Measured, against my own first guess, which this test failed on. `num_batches_tracked`
    increments whenever the layer is in training mode with `track_running_stats` on, whatever
    `momentum` is (torch `nn/modules/batchnorm.py:171-177`). Momentum only decides what the
    count is USED for, and it feeds the average solely when `momentum is None`. So under
    FixBN the counter climbs and means nothing -- worth a test rather than a surprise,
    because it travels to the server like every other buffer."""
    model = TinyBNNet()
    _train_a_step(model)
    freeze_batchnorm_stats(model)
    counter_before = model.state_dict()["bn.num_batches_tracked"].clone()
    mean_before = model.state_dict()["bn.running_mean"].clone()

    for _ in range(3):
        _train_a_step(model)

    assert model.state_dict()["bn.num_batches_tracked"] > counter_before
    assert model.bn.momentum == 0.0, "momentum is not None, so the count is not divided by"
    torch.testing.assert_close(model.state_dict()["bn.running_mean"], mean_before)


def test_normalisation_uses_the_stored_statistics_not_the_batch():
    """The half that makes local training see what the holdout will see. In train mode a
    BatchNorm normalises with the mini-batch's own statistics; frozen, it must use the
    stored ones, so the same input gives the same output whatever it is batched with."""
    model = TinyBNNet()
    for _ in range(5):
        _train_a_step(model)
    freeze_batchnorm_stats(model)

    x = torch.randn(1, 3, 8, 8)
    alone = model(x)
    with_company = model(torch.cat([x, torch.randn(7, 3, 8, 8) * 50 + 20]))[0:1]

    torch.testing.assert_close(alone, with_company, rtol=1e-4, atol=1e-5)


def test_momentum_zero_survives_a_train_call_even_though_the_mode_does_not():
    """Belt and braces, and the braces are load-bearing: Ultralytics puts the model back
    into train() every epoch, so the mode is lost and the attribute is what is left."""
    model = TinyBNNet()
    freeze_batchnorm_stats(model)
    model.train()  # what _model_train does at the start of each epoch

    bn = model.bn
    assert bn.momentum == 0.0
    assert bn.training is True, "the mode really is lost; this is why the subclass exists"

    before = bn.running_mean.clone()
    _train_a_step(model)
    # momentum=0 must hold the estimates still even back in training mode.
    torch.testing.assert_close(bn.running_mean, before)


def test_a_model_without_batchnorm_reports_zero_rather_than_claiming_success():
    assert freeze_batchnorm_stats(nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())) == 0


def test_fixbn_is_on_from_its_round_to_the_end_of_the_run():
    """A per-round flag, but not a one-round flag: the warm-up phase before it is the
    method. Pinning on round 1 would pin COCO's initial statistics."""
    sent = [round_config(r, 6, 1, fix_bn_from_round=3)["fix_bn"] for r in range(1, 7)]
    assert sent == [False, False, True, True, True, True]


def test_fixbn_is_off_unless_asked_for():
    assert all(round_config(r, 6, 1)["fix_bn"] is False for r in range(1, 7))
