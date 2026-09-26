# The phased plan — faster runs, better numbers, in that order

Written 2026-08-16, after the run that produced the project's first real result:
**84.5 % of a budget-matched centralised ceiling** (0.4173 vs 0.4936 mAP50 on the
1 000-image holdout), 3 296 s of GPU time, 82.2 Wh.

That run also produced the two numbers this plan exists to attack:

| Measured | Value | What it means |
|---|---|---|
| Mean GPU utilisation | **27 %** | roughly three quarters of the wall clock is not training |
| Peak VRAM | **5 087 MiB of 16 303** | two more clients fit on the card, unused |
| Run-to-run variance (observed) | **≥ ±0.016 mAP50** | larger than several differences the project wants to call results |

So: the experiment loop is both slow and imprecise. Everything below is ordered by
*what unblocks the next thing*, not by size. Speed first, because every later phase is
paid for in runs, and precision second, because until the noise floor is known no
result is a result.

## Order, and why this order

| Phase | Question it answers | Gate to the next phase |
|---|---|---|
| **0 — Measure** ✅ | Where does the 73 % of non-training wall clock go? | **answered: it does not leave training.** 85.3 % train, 13.8 % evaluate, 0.6 % idle, clients never overlapping |
| **1 — Runtime** ✅ | How many runs per GPU-hour? | **1.94× and 43 % less energy** from `--gpu-fraction 0.33`. The cache lever measured *slower* and was dropped; utilisation is still only 30.9 %, so the remaining idle is not what this plan guessed |
| **2 — Schedule & head** | Why do 24 effective epochs only reach 0.4173? | box/cls/dfl fall *within* a round instead of rising |
| **3 — Evidence** | Is any difference real? | the seed spread is known and printed beside every comparison |
| **4 — Data** | Is the input reproducible and clean? | a fleet is addressable by content hash and reproducible from it |
| **5 — Advanced FL** | Which algorithm suits non-IID driving data? | a leaderboard where the gaps exceed the phase-3 spread |
| **G — GitHub** | Does the repo defend all of the above? | CI is green on Windows *and* Linux, nightly, on `main` |

**Phases 1 and 2 are the ones that change every subsequent number.** Phase 5 is the
interesting one and it is deliberately last: comparing strategies on a slow, noisy,
badly-scheduled trainer measures the trainer.

---

## Phase 0 — measure the round before optimising it — **done**

Backlog 95. `pipeline/profile.py` pairs markers the logs already carry into per-phase
intervals, so the 3 296 s reference run could be profiled after the fact rather than
re-run:

```bash
python -m pipeline.profile --server-log my-project/logs/server.30716.log --json
```

| phase | seconds | share of wall |
|---|---|---|
| **train** | 2 784.1 | **85.3 %** |
| **evaluate** | 450.7 | **13.8 %** |
| construct (model load) | 8.6 | 0.3 % |
| aggregate + checkpoint | 1.7 | 0.05 % |
| weights in + out | 0.6 | 0.02 % |
| unaccounted (Ray, teardown, idle) | 19.9 | 0.6 % |

72 client episodes, **never more than one overlapping**. Wall clock 3 266 s.

**Both worlds are real, and the measurement separates them.**

- Clients *are* serialised — `max_concurrent = 1` across 72 episodes. Lever 1 is
  available in full, and it is mathematically a no-op.
- Orchestration is *not* where the time goes — 99.1 % of the wall clock is inside a
  client doing work, and the GPU still averaged 27 % while it did. The idle is inside
  `train()`, which is the data path, which is levers 2–3.

**What this deletes from phase 1.** Model construction totals 8.6 s across 72 episodes.
Persistent client actors (lever 5) and anything else that removes per-round fixed cost
is capped at **0.3 %** of the run. It was ranked "small, but free"; it is small enough
not to be worth the state-handling. Cut.

**What it adds.** `evaluate` is 13.8 % — six clients re-scoring their own val split
every round, for a number `docs/PHASED_PLAN.md` already calls the flattering one. The
holdout is what the project reports. Evaluating every client every round is 450 s of
GPU time spent on a metric that is not the headline.

---

## Phase 1 — runtime: more runs per GPU-hour

Backlog 89, 90, 91, 92. Five levers, all measured against the same fleet fingerprint,
all required to leave holdout mAP unchanged within the phase-3 spread. Ranked by
expected effect on this hardware.

| # | Lever | Where | Why it should work here | Expected |
|---|---|---|---|---|
| 1 ✅ | **Pack clients concurrently** — `--gpu-fraction` | `pipeline/stages.py`, **not** pyproject: the pipeline overrides it on the CLI because flwr migrates the federation config out of pyproject.toml | peak is 5.1 GB of 16.3 at the 1 400-image profile. Serialisation is a full-scale setting applied at demo scale | **measured 1.50×** at 0.5; 0.33 does not fit |
| 2 ❌ | **`cache="ram"`** in the client's `train()` | `client_app.py` ⚠ | 1 400 images at 640 px is ~2–3 GB of RAM. Removes JPEG decode from the step loop — the prime suspect for 27 % utilisation | **measured 5.9 % _slower_**; utilisation moved 19.3 → 22.7 % and the wall clock got worse |
| 3 ❌ | **Windows dataloader** — `workers=0` today | `client_app.py` ⚠ | the comment is right that spawned workers deadlock inside a Ray actor, but `workers=0` means decode happens on the training thread. With `cache="ram"` this stops mattering; without it, it *is* the bottleneck | **moot.** Lever 2 shows decode is not the bottleneck, so moving it off the training thread cannot be the fix |
| 4 | **Reuse the label cache across rounds** | `pipeline/build_fleet.py`, shard dirs | Ultralytics writes `labels.cache` beside each shard and rescans when it is missing. 6 vehicles × 6 rounds = 36 scans. The cache survives only if the fleet is not rebuilt between rounds — assert that, do not assume it | seconds × 36 |
| ~~5~~ | ~~**Persistent client actors**~~ | — | **Cut by phase 0.** Model construction is 8.6 s across 72 episodes, 0.3 % of the run. A perfect fix here buys three seconds per round | measured, not worth it |
| 6 | **Evaluate fewer clients per round** | `my-project/pyproject.toml` ⚠ | phase 0 found `evaluate` at **13.8 %** of wall clock, spent on the self-reported metric the project already calls flattering. `fraction_evaluate < 1.0`, or evaluate only on the final round | up to 1.16× |

### Levers 1 and 2, measured at the profile the 27 % came from

1 400 images/vehicle at 640 px — the reference run's profile — 6 vehicles × 2 rounds ×
1 epoch, one fleet built once and reused by every arm:

| `--gpu-fraction` | `--cache` | clients at once | wall | mean util | peak VRAM | energy |
|---|---|---|---|---|---|---|
| 1.0 | — | 1 | 562.1 s | 19.3 % | 6 453 MiB | 10.36 Wh |
| 1.0 | ram | 1 | 595.2 s | 22.7 % | 6 313 MiB | 10.67 Wh |
| 0.5 | — | 2 | 394.9 s | 29.7 % | 10 474 MiB | 8.37 Wh |
| **0.33** | — | **3** | **289.2 s** | **30.9 %** | **15 468 MiB (94.9 %)** | **5.92 Wh** |

**Lever 1 is the phase. 1.94× wall clock and 43 % less energy**, and it changes nothing
mathematically — clients are independent within a round. Checksums moved every round in
every arm and all four criteria stayed green. Three clients take 94.9 % of the card, so
0.33 is the floor on this hardware at this profile, not a step on the way to 0.25.

**Lever 2 does not work here. `cache="ram"` was 5.9 % _slower_**, at both profiles —
the demo arm sat inside its controls' spread, the full arm was outside it in the wrong
direction. The hypothesis was reasonable and it is wrong: with the shard cached, mean
utilisation still only reached 22.7 %, so JPEG decode was not what the card was waiting
for. Default stays `""`.

**What that leaves.** The gate was ≥2× and lever 1 alone gives 1.94×. But utilisation
at three concurrent clients is **30.9 %**, barely above the 27 % that started this: the
card is still mostly idle, and neither of the two suspects in this plan explains it.
Whatever the remaining bottleneck is — step overhead at batch sizes this small, the
Python train loop, per-round trainer construction inside each actor — it is not
addressed by anything ranked here, and finding it is a new phase-0-shaped question
rather than another lever to pull.

### Answered, 2026-09-02 — and it was none of those three

Measured on the dataset alone, no model and no GPU, so the number is the dataloader's
own cost: **7.93 ms/sample** stock, **5.56 ms** with `mosaic=0`. At batch 16 that is
~127 ms of CPU per batch, on the training thread, against a GPU step of the same order.
The bottleneck is **the transform stack**, and the guess above did not list it.

It also explains lever 2's failure rather than leaving it as a surprise: `cache="ram"`
removes the JPEG decode and leaves the mosaic assembly and `random_perspective` warps,
which is why utilisation moved 19.3 → 22.7 % and the clock got worse. And
`close_mosaic = 10` fires at `epoch == epochs - close_mosaic`, which is negative for a
1–4 epoch round — **mosaic never closes inside a federated round.**

Lever 3 is now measured rather than deduced-moot, and the answer is the same for a
different reason. One epoch on `batch_1`: `workers=0` 36.2 s, `workers=4` 34.3 s,
`workers=8` **40.1 s**. Windows has no `fork`, so each worker re-imports torch and
ultralytics and a short round cannot amortise it; `mosaic=0` is worth more at
`workers=0` (1.45×) than at `workers=4` (1.21×). The client's stated reason — a
deadlock inside a Ray actor — did not reproduce. Keep `workers=0`, correct the comment.

**And the lever that was in none of the six rows.** `plots=True` is the Ultralytics
default and the client never overrode it, costing **1.19×** of every round, per client,
drawn into a directory the next round overwrote. Shipped: the server sends `plots` and
sets it True on the final round only. `mosaic=0` is worth a further 1.12× and is
holdout-gated because it changes the data path.

Full detail, including the three-repeat medians and why the single-run pass said 1.52×
for something worth 1.19×: [`docs/FEDERATED_DETECTION.md`](FEDERATED_DETECTION.md).

Demo-scale table and the VRAM-to-fraction guide: [`docs/RUNBOOK.md`](RUNBOOK.md) §8.

**The trap this phase must not fall into.** Every one of these can make a run finish
faster *and* train less. Lever 1 changes nothing mathematically — clients are
independent within a round — but levers 2–5 touch the data path. Each is accepted only
if the holdout curve is unchanged and the round-over-round aggregate checksum still
moves. Speed that costs mAP is not speed, it is a shorter run.

**Do not** reach for `torch.compile`, multi-GPU or mixed-resolution training here.
Compile's warmup is paid per process and clients are short-lived; the card is one card;
and mixed resolution changes the result you are trying to hold constant.

Prompt: [`docs/prompts/2026-08-16-phase1-runtime-throughput.md`](prompts/2026-08-16-phase1-runtime-throughput.md).

---

## Phase 2 — the schedule and the head: better mAP per epoch

Backlog 27, 30, 28, 29. ⚠ All of it changes `my-project/`, so: own branch, own prompt.

This is the phase with the largest expected effect on the headline number, and it rests
on three facts checked against the installed ultralytics 8.4.115 rather than assumed:

```
warmup_epochs = 3.0   lr0 = 0.01   lrf = 0.01   cos_lr = False
close_mosaic  = 10    mosaic = 1.0   freeze = None   patience = 100   nbs = 64
optimizer = "auto"    <- 2026-09-02: this line is why the two below are half wrong
```

> **Correction, 2026-09-02.** `optimizer="auto"` is the default and the client never
> overrode it, and `auto` **replaces `lr0`** with `0.002·5/(4+nc)` = 5.88e-4 — logging
> that it has done so. So every run this project has ever done trained with **AdamW at
> 5.88e-4**, not SGD at 0.01, and *`lr0` has never once been the number in effect.*
>
> `lrf` and `warmup_epochs` are **not** overridden — `_setup_scheduler` reads them
> directly — so fact 2 below survives; only its `lr0` framing is wrong.
>
> **This strikes PR #53's result.** That branch varies `lr0_round` per round without
> setting `optimizer`, so every value it computed was discarded and only its
> `lrf_round` applied. Its "−0.0079 mAP50, negative at six of six rounds" is currently
> the reason this project believes a global anneal does not help, and it did not test
> one. Rebase it with `optimizer` set explicitly and re-run.
>
> Fact 1's arithmetic is also wrong in the other direction:
> `_get_warmup_iterations` clamps to `min(warmup_epochs, max(epochs-1, 0))`, so at
> `local_epochs = 1` there is **no warmup at all** — and every measurement in the
> session that produced the numbers below ran at 1. At `local_epochs = 4` there really
> are three warmup epochs of four. Any warmup claim must name its `local_epochs`.

**Fact 1 — three of every four epochs are warmup.** `local_epochs = 4` against
`warmup_epochs = 3.0` means the LR ramp never finishes. The Metrics tab already shows
box, cls **and** dfl rising across the four epochs of every round: each client ends the
round worse than the aggregate it started from, and FedAvg then averages six models
that each went slightly backwards.

**Fact 2 — the LR schedule restarts every round.** `lrf = 0.01` decays the LR to 1 % of
`lr0` *within* the round, then the next round starts a fresh `YOLO.train()` at `lr0`
again. Six rounds is therefore six independent anneals, not one. The fleet never gets
the low-LR consolidation phase that makes the last epochs of a centralised run count —
which is a decent structural explanation for the 15 % gap to the ceiling, separate from
anything about federation.

The fix is a **server-driven schedule**: the round number is already broadcast, so the
client can set `lr0_round = lr0 · f(round / total_rounds)` and `warmup_epochs ≈ 0.1`
for every round after the first. One global anneal, spread across rounds.

### Fact 3, measured — and what it exposed

The head warm start is done (`warm_start_head` in `get_set_model.py`): 9 of BDD's 13
classes are copied out of the COCO head instead of drawn from a distribution. All
numbers below are on the **1 000-image shared holdout**, same fleet, 1 400
images/vehicle, 6 vehicles × 2 rounds × 1 epoch:

| head | untrained | after round 1 | after round 2 |
|---|---|---|---|
| random (what every run so far used) | 0.0053 | 0.1277 | 0.1690 |
| **warm-started** | **0.2582** | 0.1924 | 0.2073 |

Untrained, the warm-started model scores **0.2582 against 0.0053** — 49× — and that is
before a single gradient step. It is also **59 % of what six rounds × four epochs of
federation reached** (0.4334). Round 1 was never spent learning; it was spent
recovering from an initialisation nobody had looked at.

**And the number that matters more.** The warm-started model is *better untrained than
after two rounds of federation*: 0.2582 → 0.1924 → 0.2073. **Round 1 costs it 0.066
mAP50**, and round 2 has not won that back.

This is facts 1 and 2 of this phase, made visible for the first time. With a random
head, "round 1 damages the model" was unobservable — there was nothing to damage. With
a head that already detects, the LR schedule's cost shows up as a number: three epochs
of warmup inside a one-epoch round means the round runs entirely at a ramping LR, and
every round restarts it. The warm start did not create this problem; it made it
measurable, and it makes the rest of phase 2 the highest-value work left in the plan.

**Fact 3 — the 13-class head starts random.** COCO weights transfer the backbone and
discard the head, so round 1 spends its gradients teaching the head what a car is,
while backpropagating noise into good features. Two independent fixes, cheap, and
testable separately:

- `freeze=10` for round 1 only — protects the backbone while the head settles.
- **Warm-start the head from COCO rows.** BDD100K's 13 classes overlap COCO heavily
  (`person`, `car`, `bus`, `truck`, `train`, `motorcycle`, `bicycle`, `traffic light`,
  `stop sign`). Copying the matching output channels out of the COCO head instead of
  initialising them randomly is a few lines and starts the fleet from a detector rather
  than from noise. This is not in the backlog and should be; it is the highest
  expected-value item in the phase.

Also here, because it is nearly free and saves GPU hours: **early stopping on the
shared holdout** (backlog 29). `patience = 100` inside a 4-epoch round does nothing;
the stopping decision belongs to the server, on the holdout curve, between rounds.

Prompt: [`docs/prompts/2026-08-16-phase2-schedule-and-head.md`](prompts/2026-08-16-phase2-schedule-and-head.md).

---

## Phase 3 — evidence: know the noise floor before claiming a result

Backlog 42, 31, 81. This phase produces no feature. It produces the right to say the
word "better".

> **Answered 2026-09-02, and ±0.016 was the wrong instrument.**
>
> Three seeds, everything else identical, IID fleet, 6 × 1 400 images, 2 rounds × 1
> epoch, **the same holdout in every arm** (seed 0, fingerprint `3af571c2c901`, verified
> per arm):
>
> | seed | holdout mAP50 |
> |---|---|
> | 0 | 0.1193 |
> | 1 | 0.1157 |
> | 2 | 0.1186 |
>
> **max − min = 0.0036, half-range ±0.0018, stdev 0.0019 — 8.9× tighter than ±0.016.**
>
> ±0.016 was inferred from the centralised-ceiling anomaly below: two *different data
> volumes* in a *non-federated* setting. It was never a seed spread, and using it as one
> has been dismissing real differences. **It re-opens results previously written off**,
> including FedBN's +0.0040, which exceeds this spread and was recorded as null.
>
> Caveats, because n=3: `max − min` under-reports the true spread, so ±0.0018 is a lower
> bound. Measured at 2 × 1 on IID; the headline runs 6 × 4 non-IID, where partitioning
> adds a second source of variance. Full detail in
> [`FEDERATED_DETECTION.md`](FEDERATED_DETECTION.md).

The known bad case, which is a *different* phenomenon and remains unexplained: a ceiling
trained on 14 000 images for 24 epochs (1.667× the budget) scored **0.4771**, *lower*
than the 8 400-image ceiling's **0.4936**. More data and more compute produced a worse
model. That anomaly is real and worth chasing — it is simply not the federated seed
spread, and must stop being quoted as one.

| Run | Command | Answers |
|---|---|---|
| Seed repeats | `python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes` | what a difference has to exceed to be real |
| Rounds × epochs at constant product | 12×2, 6×4, 3×8 at identical image-visits | client drift, directly. Phase 2 predicts fewer local epochs win |
| Partition control | `--preset partitions` | how much of the result is federation and how much is just data |

**Report the spread everywhere, not the mean.** The Metrics tab already groups repeats;
after this phase every comparison table in this repo gets a ± column, and any row whose
delta is inside the spread is written as "no measured difference" rather than as a
winner.

Prompt: [`docs/prompts/2026-08-16-phase3-evidence.md`](prompts/2026-08-16-phase3-evidence.md).

---

## Phase 4 — data management: reproducible, clean, cheap to rebuild

Backlog 65 (finish), 66, 69, 70, 71, 74, 44, 75, 36, 76.

The fleet is already deterministic and `fleet.meta.json` records partition, α, seed and
sizes. What is missing is that a fleet is not yet **addressable**: there is no hash you
can quote in a paper that pins the exact image list.

| # | Item | Why it earns its place |
|---|---|---|
| 1 | **Content-hash fleet manifest** — exact relative image list per shard, sorted, SHA-256 over it | makes "same fleet" checkable rather than asserted, and lets phase 3 prove two arms shared data |
| 2 | **Leakage gate** — no image in two splits, no image in two shards, holdout disjoint from every shard, promoted from report to *stage failure* | the last session shipped a run where 439 holdout images sat in vehicles' val splits. It was caught by reading, not by the pipeline |
| 3 | **Incremental populate** — relink only what changed | fleet rebuild is currently rmtree-and-relink, which is why it must never run during a federation |
| 4 | **Parquet attribute index** instead of the 6.7 MB JSON | parsed on every invocation of several tools; columnar read is a fraction of it |
| 5 | **Data-quality audit** — empty label files, zero-area boxes, images with no objects, per shard | backlog 44. These silently reweight FedAvg, since `num_examples` counts images not objects |
| 6 | **Per-class histogram per shard, and per-class mAP** | `car` is 55.4 % of objects, `train` has 29 instances fleet-wide. An averaged mAP is a car detector's report card |
| 7 | **Stale-artifact detection** | already bit the project once: the holdout scorer mixed one run's checkpoints with the previous run's |

Dataset versioning (backlog 69): a content-hash manifest, **not** DVC. DVC adds a
remote, a cache and a daemon to a repo whose hard rule is that data is never committed;
the hash is the part that was actually needed.

Prompt: [`docs/prompts/2026-08-16-phase4-data-management.md`](prompts/2026-08-16-phase4-data-management.md).

---

## Phase 5 — advanced FL, now that a comparison can mean something

Backlog 48, 49, 50, 37, 53, 54, 56, 57, 60, 55.

The strategy registry (backlog 47) already makes twelve Flower strategies reachable by
name. What is missing is not plumbing, it is a fair contest. Ordered by expected value
on *this* dataset:

| # | Technique | Why here | Cost |
|---|---|---|---|
| 0 | **FedBN — keep BatchNorm local** | added 2026-09-02, and it should be first. `get_weights` sends the full `state_dict`, so FedAvg averages BN `running_mean`/`running_var` across vehicles. That is right for IID clients; this fleet is partitioned by **condition**, which is *feature* shift — precisely what BN buffers encode. Averaging a night vehicle's statistics with a clear-daylight one describes no vehicle's data. Every other entry in this table addresses drift in *parameter* space; this is the only one aimed at the axis the fleet is actually partitioned on, and it is a filter on which tensors travel | **small** ⚠ |
| 1 | **True FedProx** | today's `fedprox` is a weight-space approximation applied after training, which is honest but is not FedProx. See the mechanism note below — it is reachable, contrary to the earlier claim | medium ⚠ |
| 2 | **Personalised heads** (backlog 37) | share the backbone, keep a per-vehicle 13-class head. Condition-partitioned vehicles is the textbook case for this, and it directly attacks the per-vehicle spread (0.4069 → 0.4754) | medium ⚠ |
| 3 | **FedAvgM** | server momentum, one keyword argument, frequently a free win on non-IID | one run |
| 4 | **FedAdam / FedYogi at a real budget** | the existing comparison ran 2 rounds × 1 epoch, where a server-side optimiser has not had time to help. Rerun at 6 × 4 | 2 runs |
| 5 | **Faulty vehicle, then Krum / trimmed-mean** | robustness claims need something to be robust against. Label noise or random weights from one vehicle | medium |
| 6 | **Client sampling and stragglers** | `fraction_fit < 1.0`, vehicles that miss rounds. This is what a real fleet does | small |
| 7 | **Communication cost** — quantise / sparsify, measure the mAP paid | the only axis where FL has an engineering story beyond privacy | medium |
| 8 | **DP wrappers with a stated ε** | the honest privacy story. FL alone is not privacy: gradients leak | medium |

### The mechanism note — true FedProx *is* reachable

`docs/FL_TECHNIQUES.md` said Ultralytics exposes no per-step hook. Checked against
8.4.115, that is half right and the half that is wrong is the useful half:

- `"optimizer_step"` **is** a key in `ultralytics.utils.callbacks.base.default_callbacks`,
  but `BaseTrainer.optimizer_step` never calls `run_callbacks("optimizer_step")`. The
  callback is dead. Registering one buys nothing and looks like it works.
- `BaseTrainer.optimizer_step` **is** an ordinary method, called from `_do_train` after
  `backward()` and before `zero_grad()`, and `YOLO.train()` accepts a `trainer=` class.

So the proximal term goes in a `DetectionTrainer` subclass that adds
`μ · (w − w_global)` to `p.grad` and then calls `super().optimizer_step()` — a real
per-step FedProx, in about fifteen lines, with the global weights captured at round
start.

Prompt: [`docs/prompts/2026-08-16-phase5-advanced-fl.md`](prompts/2026-08-16-phase5-advanced-fl.md).

---

## Phase G — GitHub management, running alongside everything

Backlog 96, 97, 87, 99, 100. Not a phase in sequence; the thing that keeps the other
phases from rotting.

| Item | What it prevents |
|---|---|
| **CI matrix on Windows + Linux** | every defect in this project's history is a path, CWD or encoding trap, and CI runs on one OS |
| **Nightly smoke on `main`** | `main` is green at merge time and never checked again |
| **Branch protection**: PR required, CI required, squash-only, delete branch on merge | the discipline is currently a document, not a rule |
| **Run bundle as a CI artifact** | config + manifest + metrics, so a number in a PR is downloadable |
| **PR body carries the numbers** | the template exists; the required-evidence block does not |
| **One issue per phase, backlog items as a checklist** | `BACKLOG_100.md` is a good list and a bad tracker — it cannot be assigned, closed or linked from a commit |
| **ADRs for choices already made** | assemble-don't-build, the isolation rule, the partition design. Written once, they stop being re-argued |

Prompt: [`docs/prompts/2026-08-16-github-management.md`](prompts/2026-08-16-github-management.md).

---

## What this plan deliberately does not do

- **No architecture change before phase 3.** yolov8s vs 8m vs yolo11s at fixed epochs is
  backlog 34 and it is a phase-6 question. Swapping the model while the schedule is
  broken and the noise floor is unknown produces a comparison that says nothing.
- **No second dataset** until the first one is hash-addressable.
- **No new dashboard surface.** Assemble before building: MLflow owns metrics history
  and is already wired. It refused the last run — mlflow 3.15 answers a `file://`
  tracking URI with "the filesystem tracking backend is in maintenance mode" — and now
  writes to `sqlite:///pipeline/mlruns/mlflow.db`, with the artifact location named
  explicitly rather than left to resolve against whatever CWD a subprocess had.
- **No hosted tracker.** W&B / Comet / Neptune are rejected on the credentials rule,
  not on quality.
