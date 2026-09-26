# Freeze the backbone for round 1 — build prompt

`2026-09-26` · branch `feat/head-freeze-round-one` off `feat/accuracy-program`

## Goal

`.load()` transfers 349 of 355 tensors and cannot transfer the three classification
convolutions, so the head starts part-random — 9 of 13 classes warmed from COCO, four
random. Round 1 backpropagates that head's error into a backbone that already knew what a
car looks like, and the damage is measured: the warm-started model scored **0.2582** on
the holdout untrained and **0.2073** after two rounds. Give round 1 a frozen backbone so
the head settles against fixed features, as one run-config key, defaulting to off.

## Inputs

- Prior art, three papers that arrive at the same arrangement independently:
  linear-probe-then-fine-tune (arXiv:2306.03937), FedBABU (arXiv:2106.06042), FedSTO's
  first stage on BDD100K itself (arXiv:2310.17097).
- `freeze` semantics in the installed 8.4.115: `engine/trainer.py:330-352`.

## Hard constraints

- **Round 1 only.** A freeze left on federates a frozen backbone for the whole run, which
  is a different experiment wearing this one's name.
- **Default 0.** Adding the lever must change no number this repo has already measured.
- `freeze=0` must not be passed to Ultralytics at all — 0 is a value, not a silence.
- The lever must reach the federation from the pipeline, not stop at `server_app`; the
  existing test that run-config keys are declared in `pyproject.toml` must stay green.

## Deliverables

| File | Change |
|---|---|
| `my-project/my_project/server_app.py` | `round_config(..., freeze_round1=)` emitting `freeze` on round 1; read from run_config |
| `my-project/my_project/client_app.py` | read `freeze`, pass it when > 0, log it |
| `my-project/pyproject.toml` | declare `freeze_round1 = 0` |
| `pipeline/{stages,runner,deploy,server}.py` | `--freeze-round1 N`, through the run-config string and the dashboard's run form |
| tests | round 1 only; off by default; reaches the federate command |

## Definition of done

Both suites green, and a **measured** check that the freeze does what it claims on a real
`train()` — not that the key appears in a dict.

## Out of scope

- A freeze schedule over several rounds. Nothing has asked for one.
- `freeze` as a list of layer indices. `int` is what the round needs.
