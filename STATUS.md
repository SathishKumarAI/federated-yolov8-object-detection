# federated-yolov8 — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-09-26

## Where I stopped

One integration branch, `feat/accuracy-program`, holding eight commits above the previous
session's six. Everything is merged **into it**; nothing has reached `main` yet. Tests:
**78 my-project + 175 pipeline = 253**, all green.

```
main
 └── fix/experiment-passes-through-the-runners-levers   (6, from 2026-09-02)
      └── feat/accuracy-program                         ← everything below is here
           ├── test/shard-spotcheck  ·  docs/accuracy-findings
           ├── fix/weights-that-travel                  the bug
           ├── feat/head-freeze-round-one               ·  feat/server-side-ema
           ├── feat/fixbn                               ·  perf/imgsz-as-a-lever
           └── fix/flwr-launcher-blocked-by-app-control
```

## The bug: FedAvg was not averaging what the clients trained

`YOLO.train()` does not hand back the module it trained. It reloads a checkpoint
(`engine/model.py:828`) and rebinds `yolo.model` to it, and the checkpoint holds only
`deepcopy(ema).half()` — `"model"` is written as `None` (`engine/trainer.py:725,740`) and
the loader prefers `ckpt["ema"]` (`nn/tasks.py:1899`). So what `get_weights` serialised was
the **fp16-rounded EMA of the epoch each client's own 280-image val split liked best**.

```
weights leaving the client  : 355/355 tensors exactly fp16-representable
live trained model          :  58/355 tensors exactly fp16-representable
num_batches_tracked         : differs by exactly the epoch's 119 batches
```

355/355 is the `.half().float()` signature — a tensor that has been through fp16 is
bit-identical to its own fp16 cast. Every BatchNorm counter that travelled was **0**,
because `ModelEMA.update` lerps only floating-point tensors.

**The round-over-round checksum cannot see this.** Both the EMA and the trained weights
change every round. It proves *something* moved, never that the *right tensors* did — the
one caveat this project's most trusted signal was missing.

Fixed by taking the weights from the trainer (`trained_model`), which removes the fp16
round-trip, the unasked-for EMA, and the per-client best-epoch selection at once. Both
checksums are now logged every round.

## Four levers added, every one off by default

| lever | flag | what it does |
|---|---|---|
| round-1 backbone freeze | `--freeze-round1 10` | measured: also pins the backbone's BN statistics (`trainer.py:701-707`), so round 1 returns a bit-identical backbone from every vehicle and federates **only the head** — LP-FT / FedBABU / FedSTO, arrived at by the library's own implementation detail |
| server-side EMA | `--server-ema 0.7` | bias-corrected average of the aggregate across rounds. The client's EMA restarts at `updates = 0` every round (88 steps → decay 0.043; the ceiling's 3 168 → 0.795) |
| FixBN | `--fix-bn-from-round 3` | pins the **shared** statistics after a warm-up. Unlike FedBN it leaves one global model, so the holdout still measures what was trained. Measured on a real round: `running_mean` 0/57 moved, conv weights 57/58 moved |
| image size | `--imgsz 1024` | one number for the federation, the clients' own val, the centralised baseline and the holdout. The last two already took the flag, so raising it cannot silently become an unfair comparison |

`optimizer` and `strategy` were already run-config keys, so "put adaptivity on the server"
needs **no code** — `--strategy fedadam` with an explicit `optimizer` is two runs.

## What the literature says, and the result that reframes the project

Full reading with every source link:
[`docs/findings/2026-09-26-accuracy-findings.md`](docs/findings/2026-09-26-accuracy-findings.md).

The one to internalise — [arXiv:2509.01868](https://arxiv.org/abs/2509.01868), Sept 2026,
YOLOv8 on **BDD100K**, 8 clients, 10 rounds × 3 local epochs:

| | mAP50 |
|---|---|
| centralised | 61.4 |
| FedAvg | **61.5** |
| FedAsync | 45.7 |

FedAvg matched its ceiling. **84.5 % retention is not a law of federated detection**, and
staleness (FedAsync, −15.8) is what actually destroys detection accuracy — worth remembering
before straggler simulation gets added as "realism". Their 61.4 against this project's
0.4936 ceiling is the other half: absolute accuracy here is capped by **resolution and
budget**, not by federation.

## Environment: two new traps, both cost a run

1. **Application Control blocks `Scripts\flwr.exe`.** `[WinError 4551] An Application
   Control policy has blocked this file`, mid-run, on a machine where nothing changed on
   purpose. A pip console script is an unsigned `.exe` generated locally — the same
   mechanism that makes conda unusable here. `stages.flwr_launcher()` now returns
   `python -c "from flwr.cli.app import app; app()"`. **Do not "fix" it back to the .exe.**
2. **A concurrent agent's `taskkill /F /IM python.exe` killed a federation mid-round.**
   `/IM` matches *every* `python.exe` on the machine, so clearing one stale dashboard
   server took three processes with it, one of them a live `flwr run`. **Never blanket-kill
   an image name here; kill by PID.**

   I first wrote this up as "Ray crashed at `--gpu-fraction 0.5`", on the strength of a
   `Windows fatal exception: access violation` in the raylet in that run's log. **That was
   wrong and the evidence says so:** the same line appears in a later arm that carried on
   training past it, and the run with no violation at all is the one that also completed.
   The line is intermittent Ray-on-Windows noise around `worker.disconnect`, not a cause of
   death. `--gpu-fraction 0.5` is **not** implicated — the documented 0.33 host-memory
   failure is a separate, real measurement and still stands.

## Measured this session, after the code landed

**The floor is ±0.0077, not ±0.0018.** Re-measured at the same conditions on the fixed
transport: seeds 0/1/2 scored **0.2135 / 0.2207 / 0.2289**, mean 0.2210, max−min 0.0154.
Still a lower bound at n=3. Every lever above has to clear 0.0077.

**The old floor's three arms are irreproducible** — 0.1193 / 0.1157 / 0.1186 then against
those three now, no overlap, a systematic ≈ +0.10 that the transport fix does **not**
explain (it measures +0.0042 on a controlled single-client comparison). Unattributed, and
recorded that way. Aug-16 runs at this budget scored 0.2073 and 0.2097 — where today's arms
sit — so the five Sept-2 `random` runs are the outliers.

**And the B4 guard was reading a three-week-old log.** `paths.log_dirs()` did not include
`pipeline/vehicles/logs`, which is where a federation's server log goes (it follows
`FL_AV_DATA_ROOT`, not the package). So `federation_learned()` returned PASS from Sept 2's
checksums for every run since the fleet existed — inert in the direction that passes. Fixed,
with a general test that no federation log may sit outside the searched directories.

## Next action

1. `--imgsz 1024` at 6 × 4. Largest expected effect, touches no data, and the baseline and
   holdout follow the same flag.
3. `--freeze-round1 10`, then `--server-ema 0.7`, then `--fix-bn-from-round` at half the
   run length. One at a time; each has to clear **±0.0077** to count.
4. `--strategy fedadam` / `fedavgm` with an explicit `optimizer`, at a real budget — the
   existing comparison ran 2 × 1, where a server-side optimiser has had two steps.
5. **Per-class AP on the four classes `warm_start_head` could not warm** (`rider`,
   `trailer`, `other person`, `other vehicle`). Free, and it tests *Where to Begin?*'s
   prediction about where the residual loss now sits.
6. **Open the PR for `feat/accuracy-program` and squash-merge it.** `main` is still
   ~80 commits behind.

## Verification

```bash
python -m pytest my-project/tests -q     # 78
python -m pytest pipeline/tests -q       # 175
python -m pipeline.verify                # the four pass criteria against the last run
python -m pipeline.holdout --evaluate    # per-class table included
```

## Environment (the part that costs an hour if you forget it)

Venv at `C:\Users\PRANAS\venvs\fl_yolov8`, built on python.org 3.12 — *not* conda; Smart
App Control blocks conda-forge's `_bz2.pyd`. See
[`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md), which now also carries the `flwr.exe` case.
Export `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or every client
trains on CPU at 5.5× the wall clock with no error anywhere.

**Data: unchanged, and now checked against BDD itself.** Ten shards of real BDD100K,
hardlinked onto the kagglehub cache, random partition, seed 0, 1 400 images/vehicle,
fingerprint `c26b38858636`. `pipeline/spotcheck.py` decoded 100 sampled images and
reconstructed 1 724 boxes from `bdd100k_labels_images_*.json`: **max coordinate error 0.0**.
The condition partition's *labels* are the part that is not sound — `overcast residential`
can supply 685 images locally and `parking / tunnel` 319, so at 1 400/vehicle those two
profiles are 51 % and 77 % random top-up. See
[`docs/DATA_VALIDATION.md`](docs/DATA_VALIDATION.md).

## The result this project exists to produce, unchanged

6 rounds × 4 local epochs × 6 vehicles × 1 400 images, against a budget-matched centralised
ceiling on the same 201 600 image-visits:

| on 1 000 held-out images | federated | centralised | retained |
|---|---|---|---|
| mAP50 | 0.4173 | 0.4936 | **84.5 %** |
| mAP50-95 | 0.2313 | 0.2770 | 83.5 % |

**It was produced by the broken transport.** Both arms were, so the *ratio* is not
obviously wrong — the centralised arm has no federation to break — but the federated number
is the average of fp16 EMAs of per-client best epochs, and it has to be re-run before it
means what it says.

---

## The dashboard, 2026-09-26 — `feat/dashboard-2030`, merged into `feat/accuracy-program`

Built on a separate branch in its own worktree. Six commits, 206
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

**Merged 2026-09-26.** The three predicted conflicts were the only ones, all additive:

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
