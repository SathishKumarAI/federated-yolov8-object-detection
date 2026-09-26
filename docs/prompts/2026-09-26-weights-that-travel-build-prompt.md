# Send the weights the optimizer produced — build prompt

`2026-09-26` · branch `fix/weights-that-travel` off `feat/accuracy-program`

## Goal

FedAvg must average the weights the clients trained. Today it averages the
**fp16-rounded EMA of the epoch each client's own val split liked best**, because
`YOLO.train()` rebinds `yolo.model` to the module it reloads from `best.pt` when it
returns, and `best.pt` holds only `deepcopy(ema).half()`. Make the client send the live
fp32 module the optimizer stepped, and make the difference visible in the log so it can
never silently return.

## What is already verified, and how

`MEASURED` on the installed `ultralytics 8.4.115`, one epoch on `batch_1` at
`fraction=0.15`, batch 8:

```
weights leaving the client  : 355/355 tensors are exactly fp16-representable
live trained model          :  58/355 tensors are exactly fp16-representable
weights leaving vs live trained model : 354/355 differ, max abs diff 1.190e+02
weights leaving vs trainer EMA (fp32) : 297/355 differ, max abs diff 3.841e-03
ema.updates 14, decay 0.006975, epochs run 1
```

355/355 exactly fp16-representable is the `.half().float()` signature; a live fp32 tensor
is only exactly fp16-representable by coincidence, and 58 of 355 are. The 1.190e+02
maximum is `num_batches_tracked`, which differs by exactly the epoch's 119 batches —
`ModelEMA.update` lerps only floating-point tensors, so **every BN counter that travels
is 0**, in a transport whose docstring advertises sending them.

`READ`, `engine/trainer.py:725,740` · `engine/model.py:828-833` · `nn/tasks.py:1899` ·
`engine/trainer.py:404` · `utils/torch_utils.py:734` · `utils/metrics.py:1007`.

One claim is **not** measured and must not be written as if it were: that `best.pt` comes
from an *earlier* epoch than the last. A 4-epoch probe had monotone fitness
(mAP50-95 0.1407 → 0.1493 → 0.1541 → 0.1687), so `best == last` there, and
`strip_optimizer` erases the epoch field from both files at the end of training, so the
checkpoint cannot be asked afterwards. It is a live risk of the mechanism at
`local_epochs > 1`, not a demonstrated loss. The fix removes it either way.

## Hard constraints

- **Do not** change what `get_weights` / `set_weights` serialise, or the wire order. The
  positional zip against `state_dict().keys()` is load-bearing and `strict=True` is the
  desired failure mode.
- **Do not** set `save=False` to dodge the reload. It changes which weights FedAvg
  receives as a side effect of a speed flag — the exact class of accident this fixes.
- **Do not** silently fall back. If the trainer is not reachable after `train()`, log that
  the fallback was taken and what it means, at warning level.
- The fix must be inert when there is no trainer (the first `set_weights`, and
  `evaluate()`, both run before any `train()` in a fresh client).
- `pipeline/` must not be touched. This is `my-project/` only, per CLAUDE.md rule 3.

## Deliverables

| File | Change |
|---|---|
| `my-project/my_project/client_app.py` | a `trained_model` property that returns the module the optimizer touched, used at send time; both checksums logged each round |
| `my-project/tests/test_client_weight_transport.py` | the regression test, named for the failure it catches |
| `docs/findings/2026-09-26-accuracy-findings.md` | the probe's numbers, and the 1a downgrade |

## Definition of done

```
python -m pytest my-project/tests -q          # 56 today, more after
python -m pytest pipeline/tests -q            # 171, unchanged
```

and a re-run of the probe showing the sent weights are no longer fp16-representable and
no longer differ from `trainer.model`.

## Out of scope

- Choosing EMA *deliberately* as the thing to send. It may later be worth measuring, and
  it is a different decision from this bug; a run-config key for it is not built until
  something wants to measure it.
- The best-round promotion on the server (`holdout --promote`) — already exists.
- Server-side EMA across rounds. That is item 5, its own branch.
