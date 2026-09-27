# Session log — 2026-09-26, what actually travels

The session started as "why is mAP low" and became an audit of the weight transport. Four
separate defects were found in what moves between the server and the vehicles, none of
which any existing check could see. One row per branch, because each is independently
reviewable.

## The ask

Five things, in the user's order: show the previous session's results; **improve accuracy
without touching the data**, find the bug, and read what other people have measured; save
the findings and implement on one integration branch with sub-branches, spin up the live
server, and put the UI work on a separate agent; then show the UI, clear the Dependabot
alerts, cost the remaining GPU work, and **establish whether the simulation is real at
all** — whether weights genuinely travel between the master and the nodes; finally, a page
that explains to a layman what a weight *is*, built by another agent, plus "complete the
plan work, make it faster, no bugs".

## What shipped

| PR | Branch | What it is | CI |
|---|---|---|---|
| [#64](https://github.com/SathishKumarAI/federated-yolov8-object-detection/pull/64) | `feat/accuracy-program` | the transport fix, four levers, the dashboard rebuild. 29 commits, squash-merged as `0184a646f` | ✅ merged |
| [#65](https://github.com/SathishKumarAI/federated-yolov8-object-detection/pull/65) | `docs/status-after-the-merge` | STATUS still said `main` had nothing | ✅ merged |
| [#63](https://github.com/SathishKumarAI/federated-yolov8-object-detection/pull/63) | — | **closed, not merged**: contained in #64's squash, `git diff` added zero files. Its ±0.0018 result is superseded, said so on the PR | — |
| [#66](https://github.com/SathishKumarAI/federated-yolov8-object-detection/pull/66) | `feat/proof-and-plumbing` | the round-trip proof, two payload fixes, the Weights page, Dependabot, `fraction_evaluate` | ✅ green |

Findings: [`findings/2026-09-26-accuracy-findings.md`](../findings/2026-09-26-accuracy-findings.md).
Prior art and credits: [`RELATED_WORK.md`](../RELATED_WORK.md).
Noise floor: [`NOISE_FLOOR.md`](../NOISE_FLOOR.md).
Prompts: [weights-that-travel](2026-09-26-weights-that-travel-build-prompt.md),
[head-freeze](2026-09-26-head-freeze-round-one-build-prompt.md),
[server-ema](2026-09-26-server-side-ema-build-prompt.md),
[fixbn](2026-09-26-fixbn-build-prompt.md),
[imgsz](2026-09-26-imgsz-as-a-lever-build-prompt.md).

## The bug the session was asked to find

`YOLO.train()` does not hand back the module it trained. It reloads a checkpoint
(`engine/model.py:828`) and rebinds `yolo.model` to it, and that checkpoint holds only
`deepcopy(ema).half()` — `"model"` is written as `None` (`engine/trainer.py:725,740`) and
the loader prefers `ckpt["ema"]` (`nn/tasks.py:1899`). So `get_weights` serialised the
**fp16-rounded EMA of whichever epoch each client's own 280-image val split liked best**,
and FedAvg averaged that.

```
weights leaving the client  : 355/355 tensors exactly fp16-representable
live trained model          :  58/355 tensors exactly fp16-representable
```

A tensor that has been through `.half().float()` is bit-identical to its own fp16 cast;
355 of 355 is not a coincidence. **Measured worth of the fix: +0.0042 mAP50**, by scoring
both candidate weight sources from one real round on the same holdout. Real, and small —
which matters, because a federation number moved by +0.10 the same day and was *not*
attributed to it.

## Things found that nobody was looking for

| | What it looked like | What was happening |
|---|---|---|
| **the B4 guard** | `VERIFY: PASS`, four criteria, every run | it was reading a **three-week-old** server log. `paths.log_dirs()` never included `pipeline/vehicles/logs`, where a federation's log actually goes — it follows `FL_AV_DATA_ROOT`, not the package. Inert in the direction that passes. Found because two reports from different days and different fleets quoted bit-identical 16-digit checksums |
| **integer buffers** | `get_weights` sends the full `state_dict`, buffers included | FedAvg's `aggregate_inplace` scales by `num_examples/total` with `np.multiply(x, f, out=x)`; for int64 that truncates, so `89 × 0.5` written back into int64 is **0**, for any fleet above one. Clients logged 89, the server saved 0 |
| **the fix for that** | counters restored | `inplace=False` returns *every* array as float64, so the float-only checksum began counting the 57 counters clients exclude — published `4643.564230` against a weighted mean of `-429.435242`, a difference of exactly 57 × 89. Caught by the new Weights page rendering the identity |
| **`fraction_evaluate`** | a documented 13.8 % saving | inert twice: never sent by the pipeline, and floored at `min_evaluate_clients = min_clients` = the whole fleet. A run at 0.34 still performed 12 self-evaluations |
| **the noise floor** | ±0.0018, quoted everywhere | **±0.0077**, re-measured. And the old figure's three arms are irreproducible on current code — 0.1193/0.1157/0.1186 then against 0.2135/0.2207/0.2289 now, no overlap |
| **`Scripts\flwr.exe`** | a run halting mid-stage | `[WinError 4551] An Application Control policy has blocked this file`. A pip console script is an unsigned .exe generated locally — the same mechanism that makes conda unusable here |
| **a concurrent agent** | `Windows fatal exception: access violation` in the raylet | `taskkill /F /IM python.exe` from another session killed a live federation. I wrote it up as a Ray bug first; the evidence said otherwise |
| **933 Dependabot alerts** | a security problem | four **byte-identical** copies of one 2024 `pip freeze`, 57 lines of `file:///C:/b/...` paths. Zero alerts in anything this project installs |
| **the head's class biases** | a table on the new page | `rider`, `trailer`, `other person`, `other vehicle` carry identical values — the four classes COCO cannot warm, visible as numbers, confirming what *Where to Begin?* predicts |

## What the literature changed

[arXiv:2509.01868](https://arxiv.org/abs/2509.01868) ran YOLOv8 on **BDD100K**, 8 clients,
10 rounds × 3 local epochs: **FedAvg 61.5 against centralised 61.4 mAP50**. FedAvg matched
its ceiling. So "84.5 % retention" is not a law of federated detection, and their
FedAsync arm (−15.8) says **staleness**, not heterogeneity, is what destroys detection
accuracy. Their absolute 61.4 against this project's 0.4936 says the rest is resolution
and budget: this trains at `imgsz=640` on 1280×720 frames.

[UltraFlwr](https://github.com/KCL-BMEIS/UltraFlwr) is the closest prior art — Flower ×
Ultralytics YOLO with partial aggregation — and its README independently calls out "the
problem of loading the YOLO state dict", which is exactly the seam both of this project's
transport bugs lived in. AGPL-3.0: ideas borrowable, code would relicense this repo.

## Corrections made during the session, kept rather than tidied away

- **Frozen layers' BatchNorm statistics do not drift.** I wrote that they did;
  `_model_train` puts them in `eval()` (`trainer.py:701-707`). Measured: 0 of 27 backbone
  `running_var` moved at `freeze=10`.
- **`num_batches_tracked` climbs whatever `momentum` is** — a test failed on my assumption
  that `momentum=0` stops it.
- **My first server-EMA bias correction over-divided** from round 2 on; the "a constant
  aggregate comes back unchanged" test caught it.
- **My round-trip checker failed a healthy run** (tolerance 1e-6 against 1.13e-06 of real
  accumulation), and the reasoning behind the tolerance was wrong too: a dropped client is
  a **2.3e-04** signal, not an obvious one, because every vehicle starts each round from
  the same model. The window is ~200×, not orders of magnitude.
- **It also reported "1 client" for a six-client run** — Ray serves all six from one actor
  at `--gpu-fraction 1.0`, so a positional read of the log folded the fleet into one.
- **"Ray crashed at `--gpu-fraction 0.5`"** was wrong and is struck; it was the taskkill.
- `Config().imgsz == 320`, not 640 — `Config()` is the demo profile.
- A `git commit --amend` clobbered a message with a stale file's contents; restored.

## Still open

1. **No lever has been shown to move mAP.** `--freeze-round1`, `--server-ema`,
   `--fix-bn-from-round`, `--imgsz` are implemented, verified to do what they claim on a
   real `train()`, and unmeasured against the holdout. Each needs a run clearing ±0.0077.
2. **The +0.10 shift is unattributed.** Reproduced across three seeds; the transport fix
   accounts for +0.0042. Settling it means deliberately re-running the old transport.
3. `Research_docs/installations/cuda_test_file/requirements.txt` still raises alerts —
   kept on purpose per backlog 106; clearing them is a call about the repo's security
   surface.
4. Two agent worktrees remain under `.claude/worktrees/`; their content is merged.

## What the remaining work costs

| run | image-visits | wall clock |
|---|---|---|
| one noise-floor arm (2 × 1) | 16,800 | ~8.5 min measured |
| 6 × 4 @ 640 — one lever | 201,600 | 55 min |
| 40 × 1 @ 640 | 336,000 | 92 min |
| 6 × 4 @ 1024 | 201,600 | **refused** — outside what was measured |

≈ **5 GPU-hours** for one run of each lever plus a `fedadam` arm; ≈ **3 h** more to repeat
the winner across three seeds. The 1024 run cannot be costed honestly — scaling 640's
number by the 2.56× pixel ratio gives ~2.3 h, but that is arithmetic, not a measurement,
which is why `pipeline/plan.py` refuses to print it.
