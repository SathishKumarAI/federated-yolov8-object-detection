# CLAUDE.md — federated-yolov8

Project-scoped rules. The workspace file at `~/coding/CLAUDE.md` still applies.

## What this project is

Federated YOLOv8 over BDD100K driving data: a Flower server aggregating per-client
YOLO training, with each client holding its own shard. `pipeline/` reproduces the
whole flow and visualises a simulated vehicle fleet while it runs.

## Where to look

| Question | File |
|---|---|
| **What is the order of work, and what gates each phase?** | [`docs/PHASED_PLAN.md`](docs/PHASED_PLAN.md) |
| How do I branch, commit, merge? | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Why is mAP low, what do I run next? | [`docs/ML_PLAN.md`](docs/ML_PLAN.md) |
| How do I try another FL algorithm? | [`docs/FL_TECHNIQUES.md`](docs/FL_TECHNIQUES.md) |
| **What does Flower × YOLO actually need, measured?** | [`docs/FEDERATED_DETECTION.md`](docs/FEDERATED_DETECTION.md) |
| How do I run the model live on other machines? | [`docs/REALTIME_NODES.md`](docs/REALTIME_NODES.md) |
| What should I build? | [`docs/BACKLOG_100.md`](docs/BACKLOG_100.md) |
| **Is this actually a federation — did anything really travel?** | [`pipeline/roundtrip.py`](pipeline/roundtrip.py), and `python -m pipeline.verify` |
| **Are the shards' pixels and labels actually what they claim?** | [`docs/DATA_VALIDATION.md`](docs/DATA_VALIDATION.md) |
| **Why is accuracy where it is, and what do the papers say?** | [`docs/findings/2026-09-26-accuracy-findings.md`](docs/findings/2026-09-26-accuracy-findings.md) |
| **What counts as a result, and when does that number go stale?** | [`docs/NOISE_FLOOR.md`](docs/NOISE_FLOOR.md) |
| **Who else has built this, and what do we owe them?** | [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) |
| How do I run any of it? | [`pipeline/README.md`](pipeline/README.md) |
| Where did the last session stop? | [`STATUS.md`](STATUS.md) and `docs/prompts/` |

## Where to change it

Open the one file that owns the thing. Do not read the package to find it.

| Change | File |
|---|---|
| A dashboard panel's look, or any colour, spacing, type | `pipeline/static/app.css` |
| Dashboard markup, a new panel, an element id | `pipeline/static/index.html` |
| Chart axes, ticks, tooltips, sparkline, the noise band | `pipeline/static/js/chart.js` |
| How state reaches the browser, and the polling fallback | `pipeline/static/js/stream.js` |
| Checksum ledger, per-class small multiples, round profile, provenance | `pipeline/static/js/insight.js` |
| The projection form | `pipeline/static/js/simulate.js` |
| The demo/help walkthrough and its steps | `pipeline/static/js/tour.js` |
| A run lever's caveat, or which levers the form offers | `pipeline/measurements.py` `LEVER_NOTES`, `pipeline/server.py` `LEVERS` |
| The fleet grid / vehicle drawer / live polling / run form | `pipeline/static/js/{fleet,drawer,live,control}.js` |
| Label boxes drawn over a frame, the trainer's own pictures | `pipeline/static/js/consumed.js` |
| Which of ultralytics' output pictures are served, and their captions | `pipeline/train_artifacts.py` — `KINDS` |
| An HTTP route or what `/api/state` returns | `pipeline/server.py` |
| Real SuperLink/SuperNode processes instead of the simulator | `pipeline/deploy.py` |
| The state push: the diff format, the sequence, the SSE frames | `pipeline/statestream.py` |
| **A measured number the UI quotes, or its provenance** | `pipeline/measurements.py` — nowhere else |
| What the demo/help walkthrough replays, and what it admits it lacks | `pipeline/demo_run.py` |
| What a configuration would cost, and what it refuses to project | `pipeline/plan.py` |
| The live edge fleet: what a node may report, and its bounds | `pipeline/nodes.py` |
| The edge node itself — camera, inference, heartbeat | `pipeline/edge.py` |
| The live-nodes panel | `pipeline/static/js/edge.js` |
| Which stages exist, what "already done" means, gating | `pipeline/stages.py` |
| How a stage subprocess is run, env, SuperLink handling | `pipeline/runner.py` |
| A path, or an env var a subprocess needs | `pipeline/paths.py` — nowhere else |
| Shard assignment, conditions, partitioning, quantity skew | `pipeline/vehicles.py` |
| What a log line means | `pipeline/logparse.py` |
| Where a round's seconds went | `pipeline/profile.py` |
| The five pass criteria | `pipeline/verify.py` |
| **Whether the weights really made the round trip, and the aggregate is their mean** | `pipeline/roundtrip.py` |
| The shared holdout, and scoring the global model on it | `pipeline/holdout.py` |
| The centralised baseline, and the gap to it | `pipeline/baseline.py` |
| What makes a fleet's shards invalid | `pipeline/validate.py` |
| Decoding sampled images and checking their labels against BDD's own JSON | `pipeline/spotcheck.py` |
| Comparing runs to each other | `pipeline/compare.py` |
| Running a set of configurations and tabling them | `pipeline/experiment.py` |
| The human-facing runbook | `docs/RUNBOOK.md`, `scripts/run_pipeline.{ps1,sh}` |
| Per-vehicle learning maths (divergence, contribution) | `pipeline/vehicle_metrics.py` |
| The run report | `pipeline/report.py` |
| ⚠ Aggregation strategy — **different branch and prompt** | `my-project/my_project/server_app.py` |
| ⚠ Client training loop, checksums it logs — **ditto** | `my-project/my_project/client_app.py` |
| ⚠ Data yaml, batch path resolution, model loading — **ditto** | `my-project/my_project/task.py` |
| ⚠ What happens *inside* a round — FixBN, and per-step FedProx when it lands | `my-project/my_project/trainers.py` |

Full dashboard map, including the rules that keep it split:
[`pipeline/static/README.md`](pipeline/static/README.md).

Two features span files. Change both halves in the same commit:

- a condition profile → `PROFILES` in `pipeline/vehicles.py` **and** `GLYPHS` in
  `pipeline/static/js/util.js`
- a new artifact kind → the code that writes it **and** `.gitignore` **and**
  `test_generated_paths_are_all_gitignored`

## How work gets done here

**plan → prompt → code → verify.** Not optional, and in that order.

| Step | Artifact | Location |
|---|---|---|
| plan | design, with rejected options and why | `docs/superpowers/specs/` |
| prompt | the brief the code is written from | `docs/prompts/` |
| code | the implementation | wherever it belongs |
| verify | a command whose output proves it | recorded in the commit |

Write the prompt **before** the code, not after. A prompt reconstructed afterwards
records what happened rather than what was intended, and the gap between those two is
the part worth keeping. See `docs/prompts/README.md`.

Work on a branch. Do not mix an unverified change into a verified one. `main` stays
green and is only reached through a squash-merged PR — the full discipline, and the
reasons behind each rule, are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Hard rules

1. **Never commit data.** The BDD100K images are 7.6 GB and live in the kagglehub
   cache; shards hardlink to them. Datasets, checkpoints, MLflow stores, Ray temp
   files and reports are all gitignored, and a test asserts those rules still match
   real paths. If a change can generate a new kind of artifact, add its ignore rule
   in the same commit.
2. **Never commit credentials.** Nothing here needs any: kagglehub downloads
   anonymously. No `kaggle.json`, no `.env`, no tokens — not read, not copied, not
   echoed into a log. Hosted trackers (W&B, Comet, Neptune) are rejected for this
   reason, not on quality.
3. **`pipeline/` must not modify `my-project/`.** It invokes its scripts as
   subprocesses and reads its outputs. Enforced by
   `pipeline/tests/test_pipeline.py::test_pipeline_never_writes_into_my_project`.
4. **Assemble before building.** MLflow owns metrics and history; the Ray Dashboard
   owns actor and GPU internals; Ultralytics already ships an MLflow callback. Write
   only what none of them can do. A bespoke dashboard was proposed once and correctly
   rejected as reinventing the wheel.
5. **Fail loudly.** A failed stage halts the chain. This project's history is a
   catalogue of silent no-ops — see below — so nothing may continue past a failure or
   let a no-op look like success.

## Silent failures this project has already shipped

Read these before assuming a green run means a working one.

| | What looked fine | What was happening |
|---|---|---|
| B4 | rounds completed, metrics logged | clients returned the weights they were *sent*; FedAvg averaged its own input |
| B9 | server logged two shard assignments | `FedAvg` shares one `FitIns`, so every client got the **last** `batch_id` and the fleet trained one shard |
| B7 | federation ran to completion | checkpointing silently skipped every round |
| — | run succeeded, checksums identical | shard too small for the batch size: **no optimizer step happened at all** |
| — | `metrics.csv` showed 6 308 examples | only 10 images existed; `num_examples` is FedAvg's weight |
| — | rounds completed, checksums changed every round | clients sent the **fp16-rounded EMA of `best.pt`**, not the weights they trained. `YOLO.train()` rebinds `yolo.model` to a reloaded checkpoint, and the checkpoint holds only `deepcopy(ema).half()`. Measured: 355/355 tensors leaving the client were exactly fp16-representable, against 58/355 in the live model |

**The single most useful signal is the round-over-round aggregate checksum.** Equal
consecutive values mean nothing is being learned, whatever the metrics say. It has a
blind spot, found 2026-09-26: it proves *something* changed, never that the *right
tensors* travelled. The fp16-EMA transport bug above moved the checksum every round.

## Facts about the training stack, measured 2026-08-16

Checked against the installed **ultralytics 8.4.115** in the project venv, not assumed.
Each of these is a lever the [phased plan](docs/PHASED_PLAN.md) pulls; each is also a way
to be wrong quietly.

| | Fact | Why it matters |
|---|---|---|
| 1 | **`optimizer="auto"` is the default and it REPLACES `lr0`** with `0.002·5/(4+nc)` = 5.88e-4, and logs that it did | passing `lr0` without also passing `optimizer` is a **silent no-op**. Every run so far trained with AdamW at 5.88e-4, not SGD at 0.01. This is why PR #53's anneal result must be struck rather than believed |
| 2 | `warmup_epochs` is clamped: `min(warmup_epochs, max(epochs-1, 0))` | at `local_epochs = 1` there is **no warmup at all**; at 4 there really are three of four. Any warmup claim must name its `local_epochs` |
| 3 | `lrf` and `warmup_epochs` are **not** overridden by `auto` — only `lr0`, `momentum`, `warmup_bias_lr` are | `lrf = 0.01` still decays within the round, and each round calls `train()` fresh. Six rounds is six independent anneals |
| 4 | `"optimizer_step"` is in `default_callbacks`, but `BaseTrainer.optimizer_step` **never calls `run_callbacks` for it** | registering that callback is a silent no-op that looks like it works. Override the method and pass `trainer=` to `train()` instead |
| 5 | The dataloader costs **7.93 ms/sample** (5.56 with `mosaic=0`), on the training thread at `workers=0` | ~127 ms per batch of 16, the same order as the GPU step. **This is the 27 % utilisation.** Not decode — `cache="ram"` removes decode and measured *slower*; the cost is mosaic assembly and the warps |
| 6 | `workers>0` does not help on Windows: 0 → 36.2 s, 4 → 34.3 s, 8 → **40.1 s** | no `fork`, so each worker re-imports torch+ultralytics, and a 1–4 epoch round cannot amortise it. The recorded "Ray deadlock" reason did not reproduce; spawn cost is the real one |
| 7 | `plots=True` is the default and cost **1.19×** of a round, per client, per round — into a directory the next round overwrote | fixed: the server sends `plots` and sets it True on the final round only. `exist_ok=True` meant only the last round's pictures ever survived anyway |
| 8 | Peak VRAM at 1 400 images/vehicle is **5 087 MiB of 16 303**, with `num-gpus = 1.0` | clients are serialised on a card that fits three of them |
| 9 | The 13-class head **was random**; COCO transfers only the backbone. Now warm-started for 9 of 13 classes (`warm_start_head`) | untrained holdout mAP50 went **0.0053 → 0.2582**. And it exposed the next problem: round 1 *costs* the warm model 0.066 mAP50 — at 5.88e-4, not at the `lr0` the old note named |
| 10 | `get_weights` sends the **full `state_dict`**, so FedAvg averages BatchNorm running stats across weather conditions | correct for IID clients; this fleet is partitioned by *condition*, which is feature shift — exactly what BN buffers encode. See FedBN in [`docs/FEDERATED_DETECTION.md`](docs/FEDERATED_DETECTION.md) |
| 11 | **Measured seed spread is ±0.0018 mAP50** (n=3, max−min 0.0036, stdev 0.0019; IID, 2 rounds × 1 epoch, same holdout in every arm) | the old **±0.016** was inferred from a *centralised* ceiling anomaly across two different data volumes — never a seed spread, and 8.9× too loose. It has been dismissing real differences, including FedBN's +0.0040. n=3 under-reports, so treat ±0.0018 as a **lower bound**, and re-measure at 6 × 4 non-IID before trusting it there |

**And a measurement trap, learned here.** A first `train()` in a process pays CUDA
context + cuDNN autotune + the AMP check: 34.6 s against 27.1 s warm. Benchmarking arms
in a fixed order gives that entire cost to the first arm of the first repeat. One run
per arm said `plots=False` was worth 1.52×; three said **1.19×**. Interleave, repeat,
and quote the median with its spread.

## Upstream, checked against the live docs 2026-09-02

| | installed | latest | gap |
|---|---|---|---|
| flwr | **1.33.0** | 1.36.0 | 3 minor |
| ultralytics | **8.4.115** | 8.4.138 | 23 patch |
| torch | 2.11.0+cu128 | — | correct for sm_120 |
| opencv-python | 5.0.0.93 | — | already present; webcam capture needs no new dependency |

**Do not upgrade either one to chase these.** Every measured number in this repo was
taken on the installed versions, and the floors in `my-project/pyproject.toml` were
raised deliberately. What follows is what an upgrade would *mean*, so the decision is
informed rather than automatic.

### Flower: this project is written against the legacy API

`flwr` has a **Message API** (since 1.21) that supersedes what this project uses. Not a
rename — a different shape:

| this project | Message API |
|---|---|
| `ServerApp(server_fn=...)` | no `server_fn`; `strategy.start(grid, initial_arrays, num_rounds)` |
| `ClientApp(client_fn=...)` | no `client_fn`; `@app.train()` / `@app.evaluate()` decorators |
| `flwr.server.strategy` | `flwr.serverapp.strategy` |
| `configure_fit` / `aggregate_fit` / `configure_evaluate` / `aggregate_evaluate` | one `Message` carrying a `RecordDict` |
| `FitIns` / `FitRes` / `EvaluateIns` / `EvaluateRes` | `Message` |
| `fraction_fit`, `min_fit_clients` | `fraction_train`, `min_train_nodes` |

`BatchAssignmentMixin` is built entirely out of the left-hand column, so migrating is a
rewrite of `server_app.py` and `client_app.py`, not an edit. It still runs on 1.33 —
that is verified, and it is why the floor is where it is.

**The one reason it may be worth doing anyway:** the B9 bug is *structurally impossible*
in the Message API. FedAvg building one `FitIns` and handing the same object to every
client is what made the whole fleet train one shard; a Message is per-node by
construction. `configure_fit` currently copies the config dict to work around exactly
that, and the copy is load-bearing — see the silent-failures table.

### Flower deployment: the repo's commented block is the OLD spelling

`my-project/pyproject.toml` carries a commented `[tool.flwr.federations.remote-deployment]`
block. **That form is superseded** — flwr migrated federations out of pyproject into
`~/.flwr/config.toml` (the migration notice flwr writes into pyproject on every run says
so). The current spelling:

```toml
# ~/.flwr/config.toml
[superlink.local-deployment]
address  = "127.0.0.1:9093"
insecure = true                 # TLS: replace with certificate paths
```

```bash
flower-superlink --insecure                      # Fleet API on 127.0.0.1:9092
flower-supernode --insecure --superlink 127.0.0.1:9092 \
                 --host 127.0.0.1 --port 9094 \
                 --node-config "partition-id=0 num-partitions=5"
flwr run . local-deployment --stream
```

Ports: **9092** Fleet API (SuperNodes dial in), **9093** Control API (`flwr run` submits
here), **9094+** each SuperNode's own Runtime API — distinct per node when several share
a host. This is the path off the simulation engine and onto real machines.

### Ultralytics 8.4.116–8.4.138

| version | change | why it matters here |
|---|---|---|
| **8.4.130** | tuning's default optimizer changed to **AdamW**, explicitly *"to ensure tuning parameters such as learning rate and momentum actually affect training"* | **upstream hit fact 1 and fixed it in their tuner.** Independent confirmation that `optimizer="auto"` silently discarding `lr0` is a real trap and not a misreading. The trainer default is unchanged, so this repo still must pass `optimizer` explicitly |
| **8.4.137** | channels-last CUDA training auto-enabled on torch ≥1.11 | **no upgrade needed to try it.** 8.4.115 already has the `channels_last` argument, defaulting to `False`; 8.4.137 only flips the default. So it is reachable today as `channels_last=True`. Untested here, and it is not free of risk: it changes tensor memory format, and `get_weights` serialises the whole `state_dict` — the transport that produced the B4 bug. Measure the aggregate checksum, not just the wall clock. Ultralytics does serialise checkpoints as NCHW regardless, so saved weights are unaffected |
| **8.4.129** | BF16 mixed precision (`amp="bf16"`) | Blackwell has the hardware; untested here |
| **8.4.131** | validation forced onto the **unaugmented** pipeline when `split=train` | a correctness fix in the evaluation path this project reports from |
| **8.4.135** | `max_det` auto-matched to dataset object counts | BDD frames are crowded — `max_det=300` is a live ceiling at this scale, worth checking before it silently truncates |
| 8.4.132–8.4.135 | `fraction` gains count-based limits and boundary standardisation | `fraction=1000` for exactly 1 000 images would replace some shard plumbing |

**Nothing about `DetMetrics`, `nt_per_class`, `ap_class_index` or the `optimizer_step`
callback is documented as changed** — the four facts this repo measured against 8.4.115
still hold, but they are measured facts about *one version* and an upgrade re-opens all
of them. Re-run the probes before believing them on 8.4.138.

## Environment traps

- Use the venv on **python.org 3.12**, not conda: Smart App Control blocks
  conda-forge's `_bz2.pyd`. See `docs/ENV_WINDOWS.md`.
- **Never kill python by image name.** `taskkill /F /IM python.exe` matches every
  `python.exe` on the machine. One agent clearing a stale dashboard server that way killed
  a live `flwr run` in another session mid-round, and the only trace it left was a raylet
  `Windows fatal exception: access violation` -- which reads like a Ray bug and was written
  up as one before the real cause surfaced. Kill by PID. Matters whenever two sessions
  share this machine, which is now normal.
- **Windows Application Control blocks `Scripts\flwr.exe`**, measured 2026-09-26: a run
  halted at the federate stage with `[WinError 4551] An Application Control policy has
  blocked this file`. A console script is an unsigned .exe generated on this machine,
  which is what those policies stop — the same mechanism that makes conda unusable here.
  `stages.flwr_launcher()` therefore returns `python -c "from flwr.cli.app import app;
  app()"`, which is what the shim wraps. Do not "fix" this back to the .exe.
- Export `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or flwr
  builds its own runtime env with the CPU-only torch wheel and every client trains on
  CPU at ~5.5x wall clock, silently.
- `flwr run` **rewrites `my-project/pyproject.toml`**, commenting out
  `[tool.flwr.federations]`. Restore it; never commit the rewritten form. The pipeline
  does this automatically after every run.
- Blackwell (`sm_120`) needs torch **cu128**. cu118 has no kernel for it.
- VRAM depends on the profile: a full 6 308-image shard peaks at **15.9 GB of 16.3 GB**,
  the 300-image demo at ~5 GB. `client-resources.num-gpus = 1.0` serialises clients,
  which is required at full scale and leaves real headroom at demo scale.
- The detached SuperLink caches the CWD **and environment** of whichever `flwr run`
  started it. The pipeline kills it before every federation for that reason.
- **`--gpu-fraction 0.33` has no headroom left, and the failure is on the HOST, not the
  card.** It is the fastest setting (1.94×) and it fills 94.9–96.6 % of VRAM with three
  concurrent Ray actors. One run at that setting died mid-round-2 on
  `numpy ... _ArrayMemoryError: Unable to allocate 11.8 MiB` — a full-resolution BDD
  frame — with peak VRAM at 15 751 of 16 303 MiB. Three actors each hold their own
  interpreter, torch, and decoded image buffers. If anything else on the machine wants
  memory, use **0.5** (two clients, still 1.50×). The pipeline halted correctly rather
  than reporting a short run as a finished one: Ray exits **0** after an actor dies, and
  the runner's output inspection is the only thing that catches it.
- Condition partitioning is only real while the condition has images, and the number
  that decides a shard is the **locally reachable** one, not BDD100K's: `overcast
  residential` has 1 419 in all of BDD100K but **685** in the pool on this machine, and
  `parking / tunnel` has **319**. At 1 400 per vehicle those two profiles are 51 % and
  77 % random top-up — measured, see
  [`docs/DATA_VALIDATION.md`](docs/DATA_VALIDATION.md). Asking for more per vehicle
  silently tops up with random images and turns a non-IID run into a nearly-IID one. `--size-skew`
  sharpens this: the fleet total is preserved, so a large skew hands one vehicle several
  times `per_vehicle` and that vehicle is the one whose condition runs dry first.

## Agile, in the way that actually matters here

Small vertical slices that each end in something demonstrable — a passing test, a moved
metric, a screenshot. A slice that cannot be demonstrated is not done, it is in progress.

Every increment leaves three artifacts behind: the **prompt** it was built from, the
**verification** that it works, and an updated **STATUS.md** so the next session starts
from fact rather than archaeology. Experiments live on `exp/` branches and are allowed to
be thrown away — but their *result* gets written down somewhere permanent before the
branch dies.

## Verification

```bash
python -m pytest my-project/tests -q     # 31 tests
python -m pytest pipeline/tests -q       # 130 tests
python -m pipeline.verify                # the five pass criteria against the last run
python -m pipeline.roundtrip             # did the weights travel, and is the aggregate their mean
python -m pipeline.holdout --evaluate    # the global model on data no vehicle saw
```

CI additionally runs an end-to-end federation smoke on CPU and asserts the aggregate
checksum changes between rounds.

<!-- plane-agent-rules:v2 -->
## Issue tracking (Plane, local)

All work across `~/Documents/coding` is tracked in one Plane board.
The `plane` MCP server is registered at user scope, so its tools are available
in every session — no setup needed per repo.

- Workspace `coding`, project `Coding` (identifier `COD`), at <http://localhost:8080/coding/>
- **This repo is the label `repo:federated-yolov8`.** Every work item you create must carry it.
- Also add one `type:` label matching the conventional-commit type you intend to
  use: `type:feat` `type:fix` `type:refactor` `type:perf` `type:docs` `type:test`
  `type:build` `type:chore`.

States, and what each one means here:

| State | Means |
|---|---|
| `Backlog` | Captured, not committed to. Default for anything you file mid-task. |
| `Todo` | Pulled into the current cycle. This week's list. |
| `In Progress` | A branch exists. |
| `In Review` | A PR is open, waiting on CI or a read. |
| `Done` | Squash-merged, branch deleted. |
| `Cancelled` | Decided against. Say why in a comment — that reasoning is the value. |

Rules:

1. **Before starting work, check for an existing work item** for what you are
   about to do. Duplicates are worse than nothing because they split the history
   of a decision. **Two ways to look, and both have a trap** — see "Finding an
   existing item" below. An empty result from a search you got wrong reads
   exactly like an empty board, which is how duplicates get filed.
2. **A found bug outside the current task's scope gets filed, not silently left.**
   File it in `Backlog` with `repo:federated-yolov8`, say in your reply that you filed it.
   This is the mechanism the global CLAUDE.md rule refers to.
3. **Move the item as the branch moves**: `In Progress` when the branch is cut,
   `In Review` when the PR opens, `Done` on squash-merge.
4. **Put the work item id in the PR body** (`COD-12`), not only in the branch name.
5. Do not create Plane *projects*. One project is deliberate — repos are labels
   so a repo can move between `now/`, `shelf/` and `live/` without its tickets
   being migrated.
6. Cycles are weeks. If the user asks "what am I doing this week", read the
   current cycle, not the whole backlog.

### Finding an existing item

This Plane is the **Community edition**. `workitem list` with a `pql` or any
structured filter fails outright:

> PQL and structured filters are not supported on this Plane edition.

So **there is no server-side way to filter by the `repo:` label.** Filter in your
own head instead — list, then read:

```
workitem list  project_id=<COD uuid>  per_page=100
               fields=sequence_id,name,state,labels
```

and keep only the rows whose `labels` contain this repo's label UUID. Get that
UUID once from `label list` (the API returns UUIDs everywhere and accepts nothing
else). The board is small enough that one unfiltered list is cheaper than the
round-trips to avoid it.

`workitem search` also works, but **it matches a contiguous substring of the
title, not a set of words.** Searching `"LM Studio local model"` returns nothing
while `"LM Studio"` returns two items — the first phrase appears in no title.
**Search one distinctive token** (`local_model`, `vault.yaml`, `8787`), never a
sentence, and treat a miss as "my query was too long", not as "no such ticket".

### Useful UUIDs

Every repo shares one project and one set of states, so these are fixed. Only the
`repo:` label differs — look yours up with `label list`.

| Thing | UUID |
|---|---|
| project `Coding` (COD) | `384bb763-72eb-497f-8ddb-142f7c178668` |
| state `Backlog` | `c1497bfa-8446-49f0-aa45-976b0311b82f` |
| state `Todo` | `c074ade8-4a34-4a89-8de3-e7ab61caedf6` |
| state `In Progress` | `824d6862-acf5-4562-82d3-fc1ee7eaadd9` |
| state `In Review` | `25021b28-b089-490e-9628-d4c0fd1a5253` |
| state `Done` | `ede567e7-3e57-405e-ac93-fb04db6bcfff` |
| state `Cancelled` | `85b6f97d-30e3-4cf4-ae58-063a0e239b4f` |

Plane does not replace `STATUS.md`. `STATUS.md` is re-entry context — where you
stopped, the next action, the traps. Plane is the queue. Both, in the same commit
as the work.

<!-- /plane-agent-rules -->
