# Backlog — 100 features

Ordered within each section by value per unit of effort. **P1** = do next, **P2** =
soon, **P3** = when the obvious work runs out. Anything marked ⚠ changes `my-project/`
and therefore needs its own branch and prompt.

---

## A. UI — from "ASCII log" to premium (1–24)

The current dashboard is functional and looks like tooling. These are what make it look
like a product.

| # | Feature | P |
|---|---|---|
| 1 | Design system pass: type scale, 8px spacing grid, consistent radii, one accent — replace ad-hoc inline styles | ✅ 2026-08-06 |
| 2 | Real chart component with axes, gridlines, ticks and hover tooltips (still inline SVG, still no CDN) | ✅ 2026-08-06 |
| 3 | Vehicle cards as a proper fleet grid: condition icon, sparkline, delta chip, status ring | ✅ 2026-08-06 |
| 4 | Skeleton loaders instead of "—" while the first poll lands | ✅ 2026-08-06 |
| 5 | Empty states that say what to do next, not just that data is missing | ✅ 2026-08-06 |
| 6 | Toast notifications for stage transitions and failures | P2 |
| 7 | Animated round transitions so progress is felt, not read | P2 |
| 8 | Dark/light toggle that persists (currently follows OS only) | P2 |
| 9 | Keyboard shortcuts: `r` run, `s` stop, `1..9` select vehicle, `?` help | P2 |
| 10 | Command palette (⌘K) for stages, vehicles, reports | P3 |
| 11 | Responsive layout that works at 1280px and on a tablet | P2 |
| 12 | Per-vehicle detail drawer: full curve, config, shard composition, sample images | ✅ 2026-08-06 |
| 13 | Sample image strip per vehicle — see what "rain / fog" actually looks like | ✅ 2026-08-06 |
| 14 | Live map/schematic of the fleet with vehicles lighting up as they train | P3 |
| 15 | Round timeline scrubber: drag back through the run's history | P2 |
| 16 | Compare two runs side by side in the browser | P2 |
| 17 | Confusion-matrix / per-class mAP panel | P2 |
| 18 | Weight-flow sankey: global → clients → aggregate, widths by contribution | P3 |
| 19 | GPU panel: temperature, clocks, fan, and a cost estimate in local currency | P3 |
| 20 | Log viewer: level filter, search, regex, jump-to-error, copy-line | P2 |
| 21 | Progressive disclosure — collapse advanced panels by default | P2 |
| 22 | Accessibility pass: focus rings, ARIA on charts, contrast ≥ 4.5:1, reduced-motion | ✅ 2026-08-06 |
| 23 | Export any chart as PNG/SVG | P3 |
| 24 | Server-side render of the first paint so the page is useful before JS runs | P3 |

## B. ML research (25–46)

| # | Feature | P |
|---|---|---|
| 25 | **Centralised baseline run** — FL numbers mean nothing without the ceiling | ✅ 2026-08-06 |
| 26 | **Shared holdout set** no vehicle trains on, for honest global evaluation | ✅ 2026-08-06 |
| 27 | Freeze the backbone for round 1 while the random 13-class head settles ⚠ | ✅ 2026-09-26 — `--freeze-round1 N`, and it pins the backbone's BN statistics too |
| 28 | Report mAP50-95 everywhere mAP50 is reported | P1 |
| 29 | Early stopping on the global model, on the shared holdout | P2 |
| 30 | LR schedule tuned for short local rounds (warmup is a third of a 4-epoch round) ⚠ | P1 |
| 31 | `(rounds, local_epochs)` sweep at constant product — measures client drift | P1 |
| 32 | Per-class metrics: which of the 13 classes the fleet is failing | P2 |
| 33 | Per-condition evaluation matrix: every vehicle's model on every condition | P2 |
| 34 | Model comparison at fixed epochs: yolov8s vs 8m vs yolo11s | P2 |
| 35 | Augmentation study — mosaic helps big shards, hurts small ones | P2 |
| 36 | Class-imbalance handling; BDD100K is dominated by `car` | P2 |
| 37 | Personalisation: keep a per-vehicle head, share the backbone | P3 |
| 38 | Knowledge distillation from the centralised model into the federated one | P3 |
| 39 | Checkpoint averaging / SWA across rounds | P3 |
| 40 | Confidence calibration of the global model | P3 |
| 41 | Failure-case gallery: worst predictions per condition | P2 |
| 42 | Statistical significance across seeds — one run is an anecdote | ✅ 2026-09-26 — `experiment --preset seeds`; the floor is ±0.0077, see `docs/NOISE_FLOOR.md` |
| 43 | Learning-curve extrapolation to predict the value of more rounds | P3 |
| 44 | Data-quality audit: mislabelled / empty-label images per shard | P2 |
| 45 | Active learning: pick the next images for a vehicle by uncertainty | P3 |
| 46 | Continual learning: new condition arrives without forgetting the old | P3 |

## C. Federated learning techniques (47–62)

See [`FL_TECHNIQUES.md`](FL_TECHNIQUES.md) — Flower already ships 24 strategies.

| # | Feature | P |
|---|---|---|
| 47 | **Strategy plugin architecture** — mixin + registry, so any Flower strategy works ⚠ | ✅ 2026-08-06 |
| 48 | True FedProx proximal term rather than the weight-space approximation ⚠ | P1 |
| 49 | FedAdam / FedYogi / FedAdagrad comparison at fixed everything else | P1 |
| 50 | FedAvgM (server momentum) — cheap, often free gains on non-IID | P2 |
| 51 | Strategy leaderboard in the UI and the report | P2 |
| 52 | μ sweep for FedProx {0.001, 0.01, 0.1, 1.0} | P2 |
| 53 | Simulated faulty vehicle (label noise / random weights) — a prerequisite for robustness claims | P2 |
| 54 | Krum / Bulyan / trimmed-mean robustness runs against #53 | P2 |
| 55 | Differential privacy wrappers with a stated ε | P2 |
| 56 | Client sampling: `fraction_fit < 1.0` with many vehicles | P2 |
| 57 | Stragglers and dropouts — vehicles that miss rounds | P2 |
| 58 | Asynchronous FL | P3 |
| 59 | Hierarchical FL: vehicles → regional aggregator → global | P3 |
| 60 | Communication-efficiency: quantise / sparsify updates, measure the mAP cost | P2 |
| 61 | Secure aggregation | P3 |
| 62 | Gradient-leakage demonstration — shows *why* DP is needed | P3 |

## D. Data engineering (63–79)

| # | Feature | P |
|---|---|---|
| 63 | **Partition strategies as plugins**: condition, random, mixed, dirichlet(α), by-size, by-class | ✅ 2026-08-06 |
| 64 | Dirichlet partitioning with α — the standard non-IID knob in FL papers | ✅ 2026-08-06 |
| 65 | Manifest per fleet: exact image list, hashes, seed, partition — reproducibility | ✅ 2026-08-06 |
| 66 | Shard validation: every image has a label, no duplicates, no cross-shard leakage | P1 |
| 67 | Condition-supply guard: refuse a fleet whose per-vehicle size exceeds the rarest condition | P1 |
| 68 | Streaming/lazy shard materialisation for fleets larger than disk | P2 |
| 69 | Dataset versioning (DVC or a content-hash manifest) | P2 |
| 70 | Incremental populate: only link what changed | P2 |
| 71 | Parquet/arrow index of image attributes instead of a 6.7 MB JSON | P3 |
| 72 | Second dataset (nuScenes / Cityscapes) behind the same interface | P3 |
| 73 | Synthetic condition augmentation: rain/fog applied to clear images | P3 |
| 74 | Train/val/test split integrity test — no image in two splits | P1 |
| 75 | Per-shard class histogram, surfaced in the UI | P2 |
| 76 | Automatic detection of stale run artifacts (already bit us once) | P2 |
| 77 | Data card per fleet, in the ML-documentation sense | P3 |
| 78 | Image dedup by perceptual hash across shards | P3 |
| 79 | Label-quality scoring, to weight or drop bad shards | P3 |

## E. Observability & reporting (80–88)

| # | Feature | P |
|---|---|---|
| 80 | MLflow logging wired for real (module exists, nothing calls it in anger) | ✅ 2026-08-16 — sqlite backend; the sink was never called by anything |
| 81 | Run comparison report: N runs, one table, deltas | P1 |
| 82 | Alerting: notify on failure or on a plateau | P2 |
| 83 | Cost accounting per run — kWh, and money at a configured tariff | P2 |
| 84 | Trace/timeline view of a round: who trained when, and the gaps | P2 |
| 85 | Live per-epoch streaming, not just per-round | P2 |
| 86 | Report diffing between two runs | P2 |
| 87 | Export the whole run as a reproducible bundle (config + manifest + metrics) | P2 |
| 88 | Prometheus endpoint for external dashboards | P3 |

## F. Performance & optimisation (89–95)

| # | Feature | P |
|---|---|---|
| 89 | Measure real per-client VRAM and pack clients concurrently where they fit | ✅ 2026-08-16 — `--gpu-fraction 0.33`, 1.94×, 43 % less energy |
| 90 | AMP / channels-last / `torch.compile` benchmark at fixed accuracy | P2 |
| 91 | Cache the dataset scan across rounds — Ultralytics rescans every time | P2 — but 95 caps all per-round fixed cost at 0.3 % |
| 92 | Persistent client actors so the model is not reloaded per round | P2 |
| 93 | Multi-GPU / multi-node once the fleet outgrows one card | P3 |
| 94 | Mixed-resolution training: small images early, full later | P3 |
| 95 | Profile the round to find the non-training overhead | ✅ 2026-08-16 — `pipeline/profile.py`; 99.1 % of wall clock is inside a client |

## G. Reliability, process, docs (96–100)

| # | Feature | P |
|---|---|---|
| 96 | CI matrix: Windows + Linux, so the CWD/path traps are caught | P1 |
| 97 | Nightly scheduled smoke run against `main` | P2 |
| 98 | Resume a run from a checkpoint after a crash | P2 |
| 99 | One-command reproduction of any past run from its report bundle | P2 |
| 100 | Architecture decision records for the choices already made (assemble-don't-build, isolation rule, partition design) | P2 |

## H. Added after 2026-08-16 (101–105)

Found while writing [`PHASED_PLAN.md`](PHASED_PLAN.md), each checked against the
installed ultralytics 8.4.115 rather than assumed.

| # | Feature | P |
|---|---|---|
| 101 | **Warm-start the 13-class head from the COCO head rows** for the nine classes BDD100K shares with COCO, instead of random init ⚠ | ✅ 2026-08-16 — untrained holdout mAP50 0.0053 → 0.2582 |
| 102 | **Server-driven LR schedule across rounds** — today `lrf` anneals *within* each round and the next round restarts at `lr0`, so the fleet never anneals globally ⚠ | ❌ 2026-08-16 — built and measured −0.0079 mAP50, negative at 6 of 6 rounds. Untested at `local_epochs = 4` |
| 103 | **`cache="ram"` and the Windows dataloader path** — decode currently runs on the training thread (`workers=0`), the prime suspect behind 27 % GPU utilisation ⚠ | ❌ 2026-08-16 — 5.9 % *slower*; decode is not the bottleneck, which makes the dataloader half moot |
| 104 | **True FedProx via a `DetectionTrainer.optimizer_step` override.** Note: the `"optimizer_step"` *callback* is registered but never fired — using it would be a silent no-op ⚠ | P1 |
| 105 | **Per-round profiling** — where the 73 % of non-training wall clock goes, before optimising any of it | ✅ 2026-08-16 — duplicate of 95 |

## I. Added after 2026-08-17 (106)

| # | Feature | P |
|---|---|---|
| 106 | **Delete `Research_docs/installations/requirements_history/`** — 832 of the repo's 885 Dependabot alerts come from four *byte-identical* copies of one 2024-06-21 `pip freeze`, none of them installable or referenced | ✅ 2026-09-26 |

### 106, before anyone tries to fix it by upgrading something

GitHub reports **885 open Dependabot alerts, 65 critical**, and the number is
misleading in a way that matters:

| manifest | alerts |
|---|---|
| `Research_docs/installations/requirements_history/requirements_20240621155143_new.txt` | 208 |
| `…_20240621155143_old.txt` | 208 |
| `…_20240621161334_old.txt` | 208 |
| `…_20240621161625_old.txt` | 208 |
| `Research_docs/installations/cuda_test_file/requirements.txt` | 53 |

```bash
$ md5sum Research_docs/installations/requirements_history/*.txt
4611da35399aaf8da2fd6ae9e2009603 *requirements_20240621155143_new.txt
4611da35399aaf8da2fd6ae9e2009603 *requirements_20240621155143_old.txt
4611da35399aaf8da2fd6ae9e2009603 *requirements_20240621161334_old.txt
4611da35399aaf8da2fd6ae9e2009603 *requirements_20240621161625_old.txt
```

**All four are the same file.** So there are 261 distinct alerts, not 885 — one
snapshot counted four times, plus a CUDA smoke test's requirements.

They are a `pip freeze` of a conda environment from **2024-06-21**, full of lines like
`absl-py @ file:///C:/b/abs_5babsu7y5x/croot/absl-py_1666362945682/work`. Those paths
existed on one machine two years ago; the files cannot be installed anywhere, by
anyone, and nothing in the repo references them.

**The fix is `git rm`, not an upgrade.** Nothing this project actually installs is
flagged: `my-project/pyproject.toml` and `pipeline/requirements.txt` have **zero**
alerts between them. The dependency floors there were raised deliberately and are
recorded in `docs/ENGINEERING_NOTES.md`.

Worth doing because 885 alerts is 885 alerts: a real one arriving in a manifest that
matters would land in a list nobody reads. Keep `cuda_test_file/requirements.txt` and
its 53 alerts — that one is a live file — or pin it, but triage it on its own.

---

## If you only do ten

**1, 2, 3, 12, 13** (the UI stops looking like a log), **25, 26** (results become
meaningful), **47** (every FL technique unlocks at once), **63/64** (partitioning becomes
research-grade), **89** (the GPU stops idling between serialised clients).

**Done 2026-08-06:** 1–5, 12, 13, 22 (dashboard rebuilt and split by concern), 25, 26
(shared holdout + centralised baseline — the federation finally has a scale), 47
(strategy registry: 12 Flower strategies reachable), 63, 64 (partition registry +
Dirichlet α), 65 in part (`fleet.meta.json` records partition, α, seed, per-vehicle
count and holdout size; the exact image list and hashes are still to do).

**Found 2026-08-06, not previously in this list:** my-project's loggers use
CWD-relative paths configured at import, so importing `server_app` writes an empty
`logs/server.<pid>.log` wherever the process happens to be standing — and an empty
one looked newer than a real run's, which made `verify` report zero rounds after a
successful six-round federation. ⚠ own branch. See STATUS.md, next-session item 1.

**Next ten, in order** (rewritten 2026-08-17, after 80, 89, 95, 101 and 105 landed and
102 and 103 were measured and rejected): 42 (seeds — one run is an anecdote, and three
of this session's results sat inside the unmeasured spread), 31 (rounds × epochs at
constant product), 27 (⚠ backbone freeze — but the head is no longer random, so its
motivation is weaker than when it was written), 28 (mAP50-95 everywhere), 32
(per-class), 33 (per-condition matrix), 96 (CI matrix on Windows + Linux), 106 (delete
the dead requirements snapshots), 66 (leakage gate as a stage failure), 65 (finish the
content-hash fleet manifest).

**Still open from the phase-2 work, and now the most valuable single item:** the
learning-rate *level* for a warm-started head. Round 1 costs the warm-started model
0.066 mAP50 at `lr0 = 0.01`. Item 102 assumed the cause was the schedule *restarting*
and measured no improvement; the level itself was never varied.
