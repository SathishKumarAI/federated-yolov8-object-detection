# FixBN — pin the normalisation statistics after a warm-up — build prompt

`2026-09-26` · branch `feat/fixbn` off `feat/accuracy-program`

## Goal

`get_weights` sends the full `state_dict`, so FedAvg averages BatchNorm running statistics
across vehicles. FedBN's answer is to keep them local, and this repo has that — but it
leaves **no single global model**: the checkpoint carries an averaged BatchNorm that is no
vehicle's, so the holdout scores a model that never existed. FixBN (arXiv:2303.06530)
answers the same problem the other way: after a warm-up phase, every client normalises with
the *shared* statistics and stops updating them. Nothing drifts, there is still one global
model, and the paper measures it beating GroupNorm across most federated settings.

## Hard constraints

- **The warm-up is the method.** `fix_bn_from_round = 1` pins COCO's initial statistics;
  the flag names the round it starts at and it stays on to the end of the run.
- Off by default.
- It must survive `_model_train`, which calls `self.model.train()` at the start of **every
  epoch** (`engine/trainer.py:701`). An `eval()` set from outside the trainer is undone on
  epoch 2, leaving epoch 1 frozen and the rest not — a warm-up phase nobody chose, inside
  the round, invisible in every log. So: a `DetectionTrainer` subclass, passed through
  `YOLO.train(trainer=...)`.
- Do **not** use the `"optimizer_step"` callback route for anything. It is in
  `default_callbacks` and `BaseTrainer.optimizer_step` never fires it; registering one is a
  silent no-op that looks like it works.
- Zero BatchNorm layers found must raise, not proceed. "Plain local training while the log
  says FixBN" is this repo's characteristic failure.

## Deliverables

`my-project/my_project/trainers.py` (new: `freeze_batchnorm_stats`, `fixbn_trainer`), the
`fix_bn` round key, the client's use of it, the `pyproject.toml` declaration, the pipeline
flag `--fix-bn-from-round R`, and `my-project/tests/test_fixbn.py`.

## Definition of done

Both suites green, **and** a real 2-epoch Ultralytics round showing the statistics pinned
while the weights still move. The unit tests use a toy net; `_model_train` is the part that
actually matters, and only a real round exercises it.

## Out of scope

- Replacing BatchNorm with GroupNorm. That changes the architecture and invalidates every
  measured number in the repo; FixBN is the cheap end of the same literature.
- Deciding the warm-up round. That is a measurement, gated behind the noise floor.
