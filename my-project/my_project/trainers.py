"""DetectionTrainer subclasses — the hooks Ultralytics' high-level API does not expose.

`YOLO.train()` accepts `trainer=<class>` and instantiates it, which is the only honest way
to change what happens *inside* a round. The callback route does not work for this:
`"optimizer_step"` is a key in `default_callbacks` but `BaseTrainer.optimizer_step` never
calls `run_callbacks` for it, so registering one imports, registers, and never fires --
a silent no-op that looks like it works. Verified against the installed 8.4.115.

This module owns those subclasses. It imports ultralytics lazily, inside the factory, so
`my_project` stays importable for the fast unit tests without the heavy dependency --
the same reason `get_set_model` keeps to torch and numpy.
"""
from __future__ import annotations

import torch.nn as nn

from utils.logging_setup import configure_logging

logger = configure_logging("trainers", "logs/trainers.log")


def freeze_batchnorm_stats(model) -> int:
    """Stop every BatchNorm in ``model`` from moving, and normalise with what it has.

    Two changes per layer, and they do different jobs:

    - ``eval()`` makes the forward pass normalise with the stored ``running_mean`` /
      ``running_var`` instead of the current mini-batch's own statistics. This is the
      half that makes local training see the same normalisation the holdout will.
    - ``momentum = 0.0`` stops the running estimates being updated at all. PyTorch's
      update is ``running = (1 - momentum) * running + momentum * batch``, so zero is
      exactly "do not move". Redundant while the layer stays in eval, and not redundant
      at all across an Ultralytics round: ``BaseTrainer._model_train`` calls
      ``self.model.train()`` at the start of every epoch, which puts the layers back into
      training mode. The attribute survives that; the mode does not.

    One thing it does *not* stop, measured rather than assumed -- a test failed on my
    first guess: ``num_batches_tracked`` still increments, because torch bumps it whenever
    the layer is training with ``track_running_stats`` on, whatever the momentum
    (``nn/modules/batchnorm.py:171-177``). The count feeds the average only when
    ``momentum is None``, so under FixBN it climbs and means nothing. It still travels to
    the server like every other buffer.

    Returns the number of layers frozen, so the caller can log what happened rather than
    that it was attempted. A partial freeze is the failure mode worth catching -- it
    trains, it logs, and it is not the method.
    """
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.momentum = 0.0
            frozen += 1
    return frozen


def fixbn_trainer(base=None):
    """A DetectionTrainer that re-freezes BatchNorm after every epoch's ``train()``.

    FixBN (arXiv:2303.06530): train normally for a warm-up phase, then freeze the
    normalisation statistics for the rest of the run. The paper measures it beating
    GroupNorm across most federated settings and being simpler than FedBN -- and, for this
    project, it has a property FedBN does not: **it leaves a single global model**. FedBN
    keeps each vehicle's BatchNorm local, so the checkpoint carries an averaged BatchNorm
    that is no vehicle's and a holdout score on it measures a model that never existed.
    FixBN's statistics are the aggregate's, shared by every client, and the holdout stays
    a measurement of the thing that was trained.

    Why a subclass and not a one-off call before ``train()``: ``_model_train`` runs
    ``self.model.train()`` at the start of each epoch (``engine/trainer.py:701``), which
    would undo an eval() set from outside on the second epoch and leave the first epoch
    frozen and the rest not -- a warm-up phase nobody chose, inside the round.

    ``base`` exists so this composes with another trainer subclass later (true per-step
    FedProx is the one on the list) instead of the two having to be merged.

    MEASURED on a real 2-epoch round of batch_1, against the same round without it:

    ```
    plain   running_mean moved 57/57   running_var 57/57   conv weights 57/58   counters 57/57
    fixbn   running_mean moved  0/57   running_var  0/57   conv weights 57/58   counters  0/57
    ```

    The network keeps learning and the statistics do not move, which is the method. The
    counters do not move either -- nothing re-enters ``train()`` between epochs, so the
    ``eval()`` set at the start of each one holds all the way through it. The
    ``momentum = 0.0`` above is what covers the case where something does.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class FixBNTrainer(base or DetectionTrainer):
        """Ultralytics' detection trainer with the normalisation statistics pinned."""

        _fixbn_logged = False

        def _model_train(self):
            super()._model_train()
            frozen = freeze_batchnorm_stats(self.model)
            if not FixBNTrainer._fixbn_logged:
                FixBNTrainer._fixbn_logged = True
                logger.info(f"[FixBN] froze {frozen} BatchNorm layers: statistics come "
                            f"from the aggregate and do not move this round.")
            if frozen == 0:
                raise RuntimeError(
                    "FixBN was requested but this model has no BatchNorm layers. That "
                    "would run plain local training while the log says FixBN.")

    return FixBNTrainer
