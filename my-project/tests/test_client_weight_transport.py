"""What leaves the client must be what the optimizer stepped.

`YOLO.train()` does not hand back the module it trained. It reloads a checkpoint
(`engine/model.py:828`) and rebinds `yolo.model` to that, and the checkpoint holds only
`deepcopy(ema).half()` -- `"model"` is written as `None` (`engine/trainer.py:725,740`)
and the loader prefers `ckpt["ema"]` (`nn/tasks.py:1899`). So reading `yolo.model` at
send time serialises an fp16-rounded EMA of whichever epoch `best.pt` chose, scored on
this vehicle's own val split.

Measured on one epoch of batch_1 before the fix: 355 of 355 tensors reaching
`get_weights` were exactly fp16-representable, against 58 of 355 in the live trained
model. That is the `.half().float()` signature, and it is what these tests reproduce in
miniature.

This is the third time this project has shipped a transport bug that a green run hid
(B4: the weights it was sent; B9: one shard for the whole fleet), so the tests are named
for the failure rather than for the function.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from my_project.client_app import FlowerClient
from my_project.get_set_model import get_weights

from tests.test_weights import TinyBNNet, _train_a_step


def _client(yolo):
    """A FlowerClient with nothing but a `yolo` attribute.

    `__init__` downloads a model and builds a 13-class net; neither is needed to test
    which attribute the send path reads, and requiring them would make this a slow test
    that only runs where the weights are on disk.
    """
    client = object.__new__(FlowerClient)
    client.yolo = yolo
    return client


def _as_ultralytics_left_it(trained):
    """The module `yolo.model` points at after train(): fp16-rounded, from the EMA.

    Reproduces `.half()` on save and `.float()` on load. The EMA part is not modelled --
    the fp16 part alone is enough to tell the two modules apart, and it is the half that
    is unconditional.
    """
    reloaded = TinyBNNet()
    reloaded.load_state_dict({k: v.half().float() for k, v in trained.state_dict().items()})
    return reloaded


def test_the_weights_that_travel_are_the_ones_the_optimizer_stepped():
    trained = TinyBNNet()
    _train_a_step(trained)
    yolo = SimpleNamespace(model=_as_ultralytics_left_it(trained),
                           trainer=SimpleNamespace(model=trained))

    sent = get_weights(_client(yolo).trained_model)
    expected = get_weights(trained)

    assert len(sent) == len(expected)
    for a, b in zip(sent, expected):
        np.testing.assert_array_equal(a, b)


def test_what_ultralytics_rebinds_is_not_what_was_trained():
    """The guard on the guard: if the fixture cannot tell the two modules apart, the
    test above passes for the wrong reason and would keep passing after a regression."""
    trained = TinyBNNet()
    _train_a_step(trained)
    reloaded = _as_ultralytics_left_it(trained)

    def exactly_fp16(module):
        return sum(1 for v in module.state_dict().values()
                   if v.is_floating_point() and torch.equal(v, v.half().float()))

    floats = sum(1 for v in trained.state_dict().values() if v.is_floating_point())
    assert exactly_fp16(reloaded) == floats, "the rebound module must be fp16-rounded"
    assert exactly_fp16(trained) < floats, (
        "a trained fp32 module should not be exactly fp16-representable; if it is, this "
        "fixture proves nothing")


def test_batchnorm_counters_survive_the_send():
    """`ModelEMA.update` lerps only floating-point tensors, so `num_batches_tracked`
    never left 0 in the old path -- measured as a 119-count difference after a
    119-batch epoch, in a transport whose docstring advertises sending buffers."""
    trained = TinyBNNet()
    for _ in range(3):
        _train_a_step(trained)
    counter = trained.state_dict()["bn.num_batches_tracked"].item()
    assert counter > 0, "the fixture must actually have stepped BatchNorm"

    frozen = TinyBNNet()  # never trained: its counter is still 0
    yolo = SimpleNamespace(model=frozen, trainer=SimpleNamespace(model=trained))

    keys = list(trained.state_dict().keys())
    sent = dict(zip(keys, get_weights(_client(yolo).trained_model)))
    assert sent["bn.num_batches_tracked"] == counter


def test_no_trainer_yet_falls_back_to_the_model_and_says_so(caplog):
    """Before any train() -- and in evaluate() -- there is no trainer. Falling back is
    correct; falling back silently is not, because it looks like the fix working."""
    model = TinyBNNet()
    client = _client(SimpleNamespace(model=model, trainer=None))

    with caplog.at_level("WARNING"):
        assert client.trained_model is model
    assert any("reloaded fp16 EMA" in r.message for r in caplog.records), \
        "the fallback must be logged loudly enough to find in a run's logs"


def test_a_client_with_no_yolo_at_all_does_not_raise():
    """Model loading can fail; `__init__` sets `self.yolo = None` and the fit path has
    to reach its error return rather than dying in a property."""
    assert _client(None).trained_model is None
