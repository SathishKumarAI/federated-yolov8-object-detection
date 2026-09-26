# One image size, reaching every stage — build prompt

`2026-09-26` · branch `perf/imgsz-as-a-lever` off `feat/accuracy-program`

## Goal

BDD100K frames are 1280×720 and this project trains at 640, where a traffic light, a sign
or a rider is sub-20 px. Published YOLOv8 results on this dataset are **0.470 mAP50 for
yolov8n at 640** against **0.625 for yolov8s at 1024**, and changing the resolution touches
no data. Make it one number that reaches the federation, the clients' own validation, the
centralised baseline and the holdout score.

`Config.imgsz`'s own docstring records why it could not be done before: *"The federation's
size is my-project's DEFAULT_IMAGE_SIZE (640) and is not reachable from here — changing it
would mean editing client_app.py, which this component is not allowed to do."* The client
read a module constant. `_cmd_baseline` and `_cmd_evaluate` already passed `cfg.imgsz` as a
flag, so the federation's own size was the only unreachable piece.

## Hard constraints

- **One number for all four.** A federation at 1024 scored against a ceiling at 640 is not
  a comparison, and that failure would look exactly like a win.
- The profile defaults stand: demo 320, full 640, so nothing already measured moves.
- The batch heuristic must take the image size. Activation memory scales with the pixel
  count; a batch of 16 at 1024 is 2.56× what it was sized for, and the run dies mid-round
  rather than refusing at the start. Never scale below 2.
- The client's own `val()` must use the round's size too — `on_evaluate_config_fn` is
  `fit_config_fn`, so the value is already in that dict.

## Deliverables

`task.py` (`get_optimal_batch_size(imgsz)`), `server_app.py` (`imgsz` in `round_config`,
read from run_config), `client_app.py` (train and val), `pyproject.toml`,
`pipeline/stages.py` (`imgsz_override` + the property), `runner`/`deploy`/`server` flags,
tests on both sides.

## Definition of done

Both suites green, including a test that the same number reaches `_cmd_federate`,
`_cmd_baseline` and `_cmd_evaluate`.

## Out of scope

- Actually running at 1024. That is GPU time and it is the next thing, not this thing.
- `rect`, `multi_scale`, or any other resolution-adjacent Ultralytics argument.
