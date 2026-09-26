# Average the aggregate across rounds — build prompt

`2026-09-26` · branch `feat/server-side-ema` off `feat/accuracy-program`

## Goal

The federation ends every round on an essentially unsmoothed model; the centralised
ceiling it is measured against does not. Ultralytics builds a fresh `ModelEMA` per
`train()` with `updates = 0` (`engine/trainer.py:404`) and ramps decay as
`0.9999*(1 - exp(-updates/2000))` (`utils/torch_utils.py:734`): 88 client steps reach
**0.043**, 352 reach **0.161**, the ceiling's 3 168 reach **0.795**. No client-side
setting can fix this — the counter restarts by construction — so the averaging must
happen on the server, where the rounds are.

## Hard constraints

- **Bias-corrected.** An uncorrected EMA returns `d*initial + (1-d)*aggregate` in round 1
  and drags the fleet back toward its starting weights, in the round that moves furthest.
  The run would then read as "smoothing hurts" when what hurt was the bias.
- **Off by default** (`0.0`). Every number this repo has measured was taken without it.
- It replaces the **broadcast as well as the checkpoint**. That makes it a different
  federation rather than a reporting change, and the docstring must say so.
- Dtypes must survive: `num_batches_tracked` is int64 and rides the same positional wire
  format. A float64 in that slot fails later, inside a client's `load_state_dict`.
- A changed tensor list must raise, not silently restart the average.

## Deliverables

`server_app.py` (`_apply_server_ema`, run-config key, strategy kwarg), `pyproject.toml`
declaration, the pipeline flag through `stages`/`runner`/`deploy`/`server`, and
`my-project/tests/test_server_ema.py`.

## Definition of done

Both suites green. The decisive test is that a **constant** aggregate comes back
unchanged every round — an off-by-one in the correction shows up there as drift, and did.

## Out of scope

- Server momentum and server-side Adam. `fedavgm` / `fedadam` already exist in the
  registry; this is the averaging axis, not the optimiser axis.
- Choosing a decay. That is a measurement, and it is gated behind the noise floor.
