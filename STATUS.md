# federated-yolov8 — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-09-02

## Where I stopped

Six commits on a new stack above the previous session's, none merged. `main` is
**unchanged since PR #29** and is now ~76 commits behind; merging that stack is still
the largest single item outstanding and still the one step nothing here can do.

This session did not move mAP. It moved what the numbers *mean*: two of the beliefs
the phased plan is built on turned out to be false, and the one that mattered most had
been silently true since the first run.

## `lr0` has never been the learning rate

`optimizer="auto"` is the Ultralytics default and the client never overrode it. `auto`
**replaces `lr0`** with `0.002·5/(4+nc)` = **5.88e-4** and says so:

```
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and
           determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000588, momentum=0.9)
```

Every run this project has ever done trained with **AdamW at 5.88e-4**, not SGD at
0.01. `lrf` and `warmup_epochs` are *not* overridden, so the "six independent anneals"
fact survives; only its `lr0` framing falls.

**This strikes PR #53.** That branch computes `lr0_round` per round and passes it
without setting `optimizer`, so every value was discarded and only its `lrf_round`
applied. Its "−0.0079 mAP50, negative at six of six rounds" is currently the reason
this project believes a global anneal does not help, and **it did not test one.**
Rebase with `optimizer` set explicitly and re-run at `local_epochs = 4`.

The client now warns loudly when handed `lr0` with `optimizer="auto"`, because that
combination is a silent no-op that reads as a learning-rate experiment.

**Checked and clean, but only by luck.** `auto` also picks AdamW vs MuSGD by
`iterations = ceil(len(dataset)/max(batch, nbs)) * epochs`, divisor 64 not 16. Client
1400×4 = 88, centralised ceiling 8400×24 = 3168, both under the 10 000 threshold, so
both arms got the same optimiser at the same LR and **the 84.5 % headline is not
contaminated.** At *full* scale the ceiling is 14 208 iterations, crosses it, and would
train with MuSGD at lr0 = 0.01 against clients on AdamW at 5.88e-4 — a 17× gap, in the
one run this project exists to produce, with nothing warning. Set `optimizer` on both
sides before running that.

## The 27 % utilisation was the dataloader, and it was never decode

Phase 1 closed with "neither of the two suspects in this plan explains it" and three
guesses. It was none of them. Measured on the dataset alone — no model, no GPU:

| | ms/sample |
|---|---|
| stock (`mosaic=1.0`, `erasing=0.4`) | **7.93** |
| `mosaic=0.0` | 5.56 |

At batch 16 that is ~127 ms of CPU per batch on the training thread, the same order as
the GPU step. That also explains `cache="ram"` measuring *slower*: caching removes the
JPEG decode and leaves the mosaic assembly and the warps. And `close_mosaic = 10` fires
at `epoch == epochs − close_mosaic`, negative for a 1–4 epoch round, so **mosaic never
closes inside a federated round.**

`workers > 0` is now measured rather than deduced-moot, same conclusion, different
reason: `workers=0` 36.2 s, `workers=4` 34.3 s, `workers=8` **40.1 s**. Windows has no
`fork`. The recorded "deadlock inside a Ray actor" did not reproduce.

## What shipped: 1.19×, free

`plots=True` is the default and the client never overrode it, so every round drew
`labels.jpg`, `train_batch*.jpg` and the confusion matrix / PR curves — **into a
directory the next round overwrote.** `exist_ok=True` means only the last round's
pictures ever survived, and `train_artifacts.py` says so in its own docstring. Five
rounds of six paid GPU time for files destroyed unread.

The server now sends `plots` and sets it True on the final round only.

| arm (1 epoch, batch_1, 3 interleaved repeats) | median | spread | util | |
|---|---|---|---|---|
| baseline | 27.2 s | 7.6 | 25.4 % | 1.00× |
| `plots=False` | 22.9 s | 0.3 | 31.2 % | **1.19×** |
| `plots=False save=False` | 22.8 s | 0.4 | 29.9 % | 1.20× |
| `plots=False save=False mosaic=0` | 20.4 s | 0.9 | 32.4 % | 1.34× |

**Read that table twice.** One run per arm said `plots=False` was worth **1.52×**. It
is worth **1.19×**. The difference is one cold start — the baseline's repeats were
34.6 / 27.1 / 27.2 s, and the first `train()` in a process pays CUDA context, cuDNN
autotune and the AMP check. Arms run in a fixed order, so the first arm of the first
repeat always eats it. Interleave, repeat, quote the median.

`save=False` buys nothing, so `final_eval`'s second validation pass is not where the
time goes and the EMA-versus-raw-weights question it would have raised does not need
answering. `mosaic=0` is worth a further 1.12× but changes the data path, so it stays a
run-config key at its default until the holdout clears it.

## FedBN is the missing technique, and it should be phase 5 item 0

`get_weights` sends the full `state_dict` — deliberately, and its docstring explains
why: dropping BatchNorm buffers would make the federated model wrong. Correct for IID
clients.

This fleet is partitioned by **condition**. That is *feature* shift, which is exactly
what BN running statistics encode, so FedAvg is averaging a night vehicle's
`running_mean` with a clear-daylight vehicle's and producing statistics that describe
no vehicle's data. FedBN — keep BN local, share the rest — targets that axis directly,
costs a filter on which tensors travel, and was absent from the phase-5 table where
every other entry addresses *parameter*-space drift.

Design note, with the mechanism for true per-step FedProx and the `num_examples`
images-vs-objects question: [`docs/FEDERATED_DETECTION.md`](docs/FEDERATED_DETECTION.md).

## Also landed

- **Per-class AP** on the holdout — scorer, run report, and a dashboard panel. `car` is
  ~90 % of the objects in a 1 000-image holdout, so one averaged mAP is close to a car
  detector's report card. Two Ultralytics traps guarded by tests: `box.ap50` is indexed
  by position in `ap_class_index`, not by class id, and `box.maps` pre-fills absent
  classes with the overall `map` — `train`, 29 instances fleet-wide, would have been
  reported as scoring the fleet average.
- **Holdout fingerprint.** `size` and `seed` describe how the slice was requested; the
  same pair drawn from a val pool that has since grown gives different images and
  identical metadata. (The *fleet* was already content-hashed — `Vehicle.fingerprint`
  and `fleet.meta.json` — despite the phased plan listing it as missing. The leakage
  gate was already a halting stage too.)
- **`fraction_evaluate`** is a run-config key. It was never set, so FedAvg's 1.0
  applied and every client re-scored itself every round: phase 0's 13.8 %. Still 1.0.
- The view-id test now covers every dashboard module, not just `control.js`.

## Later the same day: FedBN, an IID fleet, and the model running live

**The fleet is now random (IID)**, fingerprint `090d345dbb14`, `validate` clean. The
`Config.partition` default moved with it — `_check_fleet` rebuilds whenever the config
disagrees with the data, so a default left at `condition` would have silently
repartitioned the fleet on the next run.

**FedBN is implemented and it engaged** — clients log *"kept 285 local tensors, applied
70 from the aggregate"*, matching the 285 of 355 measured statically. YOLOv8s is
BatchNorm-dense in tensor count (80.3 %) and nearly BatchNorm-free in weight (0.36 %).

| round | FedAvg | FedBN | Δ mAP50 |
|---|---|---|---|
| 1 | 0.1045 | 0.1085 | +0.0040 |
| 2 | 0.1201 | 0.1218 | +0.0017 |

**No measured difference**, both far inside ±0.016 — and that is the *predicted* result,
flagged before the run. Random partitioning gives every client the same input
distribution, so there is no feature shift for a local BatchNorm to preserve. FedBN was
given nothing to do. It is **implemented, tested, and unmeasured on the partition it is
for**; the run that would test it is the same pair on `--partition condition`
(fingerprint `7170c3ee9350`), reported per vehicle, after the seed spread.

Second caveat, structural: under FedBN the saved checkpoint carries the *averaged* BN,
which is no vehicle's, so a holdout score on it measures a model that never existed.
The FedBN column is a lower bound even where the method applies.

**The model now runs live on other machines.** `pipeline/edge.py` per test machine pulls
the current global checkpoint, runs it on a camera, and reports to a new dashboard panel.
Verified end to end: a node downloaded `global_round_6.pt` and ran at **15.07 fps,
36.74 ms/frame on CPU**, frames served as JPEG, `/api/node-frame/..%2f..%2fsecret` → 404.
Nodes cache on the checkpoint's content hash, not its name, because a re-run rewrites
`global_round_1.pt` with different weights. Nothing on a node trains — a camera stream
has no labels. See [`docs/REALTIME_NODES.md`](docs/REALTIME_NODES.md).

The dashboard still binds **loopback by default**; `--host 0.0.0.0` is opt-in and prints
a warning, because `POST /api/run` starts training subprocesses and nothing authenticates.

**Upstream checked** (see CLAUDE.md): flwr 1.33.0 vs 1.36.0, ultralytics 8.4.115 vs
8.4.138. This project is written against Flower's **legacy** API — the Message API
replaces `server_fn`/`client_fn`/`FitIns`/`FitRes` and would make the B9 bug
structurally impossible. Ultralytics **8.4.130 changed their tuner's default optimizer
to AdamW "so that learning rate and momentum actually affect training"** — upstream
independently hitting this repo's fact 1.

## Next action

1. **Merge the stack.** Unchanged and still blocking everything. `gh pr merge` is
   refused by a permission classifier here, so it needs a human or an allowlist entry.
   Bottom-up: `39 → 43 → 44 → 45 → 46 → 47 → 48 → 49 → 50 → 51 → 52`, then this
   session's six. **#53 must not merge and its result must not be carried forward.**
2. **Phase 3, the seed spread.** `python -m pipeline.experiment --preset seeds --seeds
   0,1,2 --yes`. Still the blocking item for every comparison, and this session added
   two more results (mosaic, plots) that want a spread beside them.
3. **Set `optimizer` explicitly, then redo phase 2.** Nothing about learning rate is
   testable until this lands. Then: a lower `lr0` for the warm-started head, and #53's
   anneal re-run at `local_epochs = 4`, which is the configuration its argument is
   about.
4. **FedBN.** Cheapest entry in phase 5 and the only one aimed at this fleet's actual
   non-IID axis. Note it removes the single global model, so the leaderboard must say
   whether it reports per-vehicle BN or BN re-estimated on the holdout.
5. **Re-run the headline at 6 × 4 with the warm-started head**, once 2 and 3 are done.

## Verification

```bash
python -m pytest my-project/tests -q     # 45
python -m pytest pipeline/tests -q       # 148
python -m pipeline.verify                # the four pass criteria against the last run
python -m pipeline.holdout --evaluate    # now prints a per-class table too
```

## Environment (the part that costs an hour if you forget it)

Venv at `C:\Users\PRANAS\venvs\fl_yolov8`, built on python.org 3.12 — *not* conda; Smart
App Control blocks conda-forge's `_bz2.pyd`. See [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md).
Export `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or every
client trains on CPU at 5.5× the wall clock with no error anywhere.

**`--gpu-fraction 0.33` has no headroom, and it bit this session.** It is the fastest
setting and it fills 94.9–96.6 % of VRAM with three concurrent Ray actors. A 2-round run
died mid-round-2 on a **host** allocation — `numpy ... _ArrayMemoryError: Unable to
allocate 11.8 MiB` — with peak VRAM at 15 751 of 16 303 MiB. Use **0.5** (two clients,
still 1.50×) if anything else is running on the machine. The pipeline halted correctly:
**Ray exits 0 after an actor dies**, and the runner's output inspection is the only
thing standing between that and a short run reported as a finished one. The
snapshot-based `pyproject.toml` restore also survived the crash.

**Data: unchanged.** All ten shards hold real BDD100K, hardlinked onto the kagglehub
cache. The fleet on disk is **1 400 images/vehicle, condition-partitioned, seed 0**.
The attribute index (79 863 images) is cached at `pipeline/.state/attributes.json`.

## The result this project exists to produce, unchanged

6 rounds × 4 local epochs × 6 vehicles × 1 400 images, against a budget-matched
centralised ceiling on the same 201 600 image-visits:

| on 1 000 held-out images | federated | centralised | retained |
|---|---|---|---|
| mAP50 | 0.4173 | 0.4936 | **84.5 %** |
| mAP50-95 | 0.2313 | 0.2770 | 83.5 % |

Nothing this session changed that number. It is now known to be a fair comparison —
both arms got the same optimiser at the same learning rate — which was worth checking
and was not guaranteed.

---

## The dashboard, 2026-09-26 — `feat/dashboard-2030`, not merged

A separate branch from the accuracy work, in its own worktree. Five commits, 206
pipeline tests green, no GPU touched. What changed, and where to pick it up.

### What it does now

| | |
|---|---|
| transport | `/api/stream` pushes `/api/state` as a snapshot then numbered diffs, woken by the event bus. One shared snapshot with a 0.35 s TTL serves every tab. Idle: 3 801 B + 1 155 B of patches per six seconds, against 11 403 B for the old 2 s poll. `/api/state` unchanged and still the resync path |
| provenance | `pipeline/measurements.py` is the one place a measured number lives, each with the document that records it. `--check` re-reads those documents and fails on drift — it caught two wrong citations the day it was written |
| the band | the holdout chart draws the run-to-run spread, labelled with its conditions, and **prefers an observed spread** from this machine's own seed repeats when the ledger has any. It has none yet |
| refusals | the Simulate tab projects cost and **refuses** any lever outside its measured range, naming the measurement that would lift the refusal. It projects no mAP at all |
| walkthrough | eleven steps, keyboard-only navigation, `?` to open. With no run it replays `pipeline/demo_run.py`; with one it narrates that |

### The next action

**Merge it, and merge the accuracy branch, in either order — they were written to not
collide.** Expected conflicts are three, all small:

1. `do_POST` in `pipeline/server.py` — the accuracy branch wires the five lever body
   fields; this branch does not touch that block, but both edit the file.
2. `CLAUDE.md` — both add rows to the "Where to change it" table.
3. `pipeline/measurements.py` `noise_floor_map50` cites `docs/NOISE_FLOOR.md`, written here
   because this branch predates the `CLAUDE.md` revision that carries fact 11. **When the
   two disagree, `CLAUDE.md` is right**; that is written into the doc.

The lever controls turn themselves on when the accuracy branch merges: `options.levers` is
derived from `Config.__dataclass_fields__`, so no edit is needed. Until then the form
disables them and says the server has no such field, rather than posting into a void.

### Traps this branch hit

- **A module-scope `setInterval` hangs the node checks.** `stream.js` started a timer at
  import; the JS test runner waits for an event loop that never drains. Timers now start
  inside `connectState`.
- **`f"{1.0:g}"` is `"1"`.** The packing record's key is `"1.0"`, so the wall-clock divisor
  and the hazard note both silently vanished at `gpu_fraction 1.0` — the one setting
  actually in use — and the projection still looked complete. `measurements.packing()`
  matches on the float now.
- **`drawer.js` registers its own `keydown` at import time.** The walkthrough's keyboard
  check picked that handler up first and passed vacuously; it now captures only the
  listeners `wireTour()` itself adds.
- **A browser was never available.** The Chrome DevTools MCP profile stayed locked by
  another session. `pipeline/tests/js/live_render.mjs` renders a whole real payload through
  the real panels under node instead, and it found a live `ReferenceError` that reading the
  code had not.

### Left undone on purpose

Shard composition against what a condition can actually supply. The measurement lives in
`docs/DATA_VALIDATION.md` and `pipeline/spotcheck.py`, neither on `main`, and this machine
has no shards on disk — so it could only have been written, not verified.
