# Where the accuracy is going — findings, 2026-09-26

Two questions, answered in that order: **what in this repo's own process is costing
mAP**, and **what has already been measured by other people on this exact dataset**.
No data was changed to produce any of it.

Status of each claim is marked. `READ` means verified by reading the installed
`ultralytics 8.4.115` source in the project venv, with file and line. `MEASURED` means a
command was run and its output is quoted. `PAPER` means someone else's number, cited.
Nothing here is `ASSUMED`.

---

## Part 1 — the bug: what FedAvg receives is not what the clients trained

`READ`, three files, and it is one mechanism with three consequences.

```python
# engine/trainer.py:725,740  -- what save_model writes
ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
...
"model": None,   # resume and final checkpoints derive from EMA
"ema": ema,

# engine/model.py:828-833  -- what YOLO.train() leaves behind
ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
self.model, self.ckpt = load_checkpoint(ckpt)

# nn/tasks.py:1899  -- which tensors load_checkpoint picks
candidate = ckpt.get("ema") or ckpt.get("model")
```

`client_app.py`'s `model` property reads `self.yolo.model` — which is correct, and is the
fix that killed the B4 bug. But **after `train()` that attribute is no longer the module
the optimizer touched.** It is:

> the **fp16-rounded EMA** of the epoch that scored best on **that client's own
> 280-image val split**.

So `get_weights(self.model)` serialises that, and FedAvg averages it.

### `MEASURED`, and the fp16 signature is bit-exact

One epoch on `batch_1` at `fraction=0.15`, batch 8, the client's own load path:

```
weights leaving the client  : 355/355 tensors are exactly fp16-representable
live trained model          :  58/355 tensors are exactly fp16-representable
weights leaving vs live trained model : 354/355 differ, max abs diff 1.190e+02
weights leaving vs trainer EMA (fp32) : 297/355 differ, max abs diff 3.841e-03
ema.updates 14, decay 0.006975, epochs run 1
```

A tensor that has been through `.half().float()` is bit-identical to its own fp16 cast.
355 of 355 is that signature; a live fp32 model manages 58 of 355 by coincidence. The
1.190e+02 maximum is not a weight — it is `num_batches_tracked`, differing by exactly the
epoch's 119 batches, because `ModelEMA.update` lerps only floating-point tensors. **Every
BatchNorm counter that travelled was 0**, in the transport whose docstring advertises
sending them. The float-tensor disagreement is the 3.841e-03 against the live EMA, which
is fp16 spacing at the magnitude of the head's class biases.

After the fix, on the same probe, through the client's own property:

```
sent == trainer.model exactly: True
sent  fp16-representable: 1/298        (old path: 298/298)
num_batches_tracked  sent=120  old path=1
```

**One claim in the table below is not measured.** That `best.pt` comes from an *earlier*
epoch than the last was not observed: a 4-epoch probe had monotone fitness
(mAP50-95 0.1407 → 0.1493 → 0.1541 → 0.1687), so `best == last` there, and
`strip_optimizer` erases the epoch field from both checkpoints at the end of training, so
the file cannot be asked afterwards. It is a live risk of the mechanism at
`local_epochs > 1`, not a demonstrated loss — written as a risk, and removed by the same
fix either way.

| | consequence | why it costs mAP | size |
|---|---|---|---|
| **a** | **`best.pt`, not the last epoch.** `DetMetrics.fitness` is `[0,0,0,1]·[P,R,mAP50,mAP50-95]` (`utils/metrics.py:1007`) — pure mAP50-95 on the client's own val | at `local_epochs = 4` each vehicle returns a *different* epoch, chosen by a noisy 280-image split. "4 local epochs" is not what happens; FedAvg averages models trained for different lengths. **Invisible at `local_epochs = 1`**, where `best == last` — which is why every 2×1 probe in this repo missed it and the 6×4 headline did not | largest |
| **b** | **fp16 round-trip per client per round.** `.half()` on save, `.float()` on load | at \|w\| ≈ 0.05 the fp16 spacing is ~3e-5; a round's AdamW update at 5.88e-4 over 88 steps random-walks ~5e-3, so ~0.6 % of each round's update is discarded. Second-order — and free to remove. It also rounds `running_var` | small |
| **c** | **EMA is rebuilt every round.** `trainer.py:404` constructs `ModelEMA(self.model)` with `updates=0`; decay is `0.9999·(1 − exp(−updates/2000))` (`utils/torch_utils.py:734`) | 88 updates gives d = **0.043**, 352 gives **0.161**, the centralised ceiling's 3 168 reaches **0.795**. The fleet ships an essentially unsmoothed endpoint each round while the ceiling ends on a genuinely averaged model. A real asymmetry in the 84.5 % comparison, modest in size, and fixable **server-side** without touching a client | medium |

### The harness trap this also exposes

With `save=False` there is no checkpoint to reload, so `yolo.model` stays the **live fp32
trained module**. The `plots=False save=False` arm in
[`FEDERATED_DETECTION.md`](../FEDERATED_DETECTION.md) therefore changed *which weights
FedAvg would receive* while being reported as a pure speed knob. It measured 22.8 s
against 22.9 s and was dropped as "buys nothing" — correct about wall clock, and silent
about semantics. This is the same shape as every entry in CLAUDE.md's silent-failures
table: the number was right, the thing it was a number *about* was not what anyone
thought.

### Why the round-over-round checksum could not catch it

The repo's most trusted signal is "equal consecutive aggregate checksums mean nothing is
learned". Both the EMA and the trained weights change every round, so the checksum moves
either way. It proves *something* is learned, never that the *right tensors* travelled.

## Part 2 — adaptivity is on the side that gets reset

`READ`. `train()` is called fresh every round, so AdamW's first and second moments are
destroyed and re-initialised **once per round per client** — 6 times in the headline, 40
times in the long run. Client-side adaptivity plus a per-round reset is the worst of both
arrangements: the bias-corrected cold start is paid every round and the curvature
estimate never accumulates.

The literature's arrangement is the opposite: clients run plain SGD, the **server** runs
Adam / Yogi / momentum on the aggregate — [Reddi et al., *Adaptive Federated
Optimization*](https://arxiv.org/abs/2003.00295). `fedadam`, `fedyogi` and `fedavgm` are
already registered here and have never been run at a real budget: the existing comparison
was 2 rounds × 1 epoch, where a server-side optimiser has had two steps.

---

## Part 3 — what other people measured, and the result that reframes the project

### FedAvg matched centralised on BDD100K

`PAPER`. [*Enabling Federated Object Detection for Connected Autonomous Vehicles: A
Deployment-Oriented Evaluation*](https://arxiv.org/abs/2509.01868) (arXiv:2509.01868,
September 2026). YOLOv5 / YOLOv8 / YOLOv11 / Deformable DETR on KITTI, **BDD100K**,
nuScenes. 8 clients, 10 rounds × 3 local epochs, label-skew partition:

| BDD100K, YOLOv8 | mAP50 |
|---|---|
| centralised | 61.4 |
| **FedAvg** | **61.5** |
| FedProx | 61.5 |
| FedAsync | 45.7 |

Two things follow, and both matter more than any strategy choice:

1. **84.5 % retention is not a law of federated detection.** FedAvg matched its ceiling
   on this dataset. The gap here is this project's schedule and transport, not federation.
2. **Staleness is what actually destroys detection accuracy**, not heterogeneity —
   FedAsync lost 15.8 points. Worth remembering before `fraction_fit < 1.0` and
   straggler simulation are added as "realism".

### The absolute number is limited by resolution, not by FL

`PAPER`. A published BDD100K → YOLOv8 pipeline reports **YOLOv8n @ 640 = 0.470 mAP50**
and **YOLOv8s @ 1024 = 0.625** ([FiftyOne + Ultralytics
walkthrough](https://aftabgazali001.medium.com/from-bdd100k-to-yolov8-a-practical-traffic-safety-detection-pipeline-fiftyone-ultralytics-256ec1fd1437)).

This project trains at `imgsz = 640` on 1280×720 frames. `traffic light`, `traffic sign`
and `rider` are sub-20 px after that downscale. **Changing `imgsz` changes no data**, and
it is the largest single lever on the absolute number available without touching a pixel.

### Prior art mapped onto this repo's open items

| technique | paper | what it says here |
|---|---|---|
| **Pre-trained init closes the non-IID gap** | [Where to Begin? (ICLR 2023)](https://arxiv.org/abs/2206.15387) | a pretrained start reduces *both* data and system heterogeneity penalties, and makes the server-optimiser choice matter less. This repo's 0.0053 → 0.2582 warm-start jump is that effect. **Testable prediction:** the residual loss now concentrates in the four classes `warm_start_head` could not warm — `rider`, `trailer`, `other person`, `other vehicle`. Check per-class AP before buying a new algorithm |
| **Fit the head first, then fine-tune** | [Guiding the Last Layer in FL with Pre-Trained Models](https://arxiv.org/abs/2306.03937) | linear-probe-then-fine-tune converges faster and generalises better than full fine-tuning from round 1. The `freeze=10`-on-round-1 idea in [`PHASED_PLAN.md`](../PHASED_PLAN.md) is this, and the paper's point is that the *ordering* is the win |
| **Do not train the head federatedly at all** | [FedBABU](https://arxiv.org/abs/2106.06042) | freeze the classifier, aggregate the body only; the advantage grows with heterogeneity. Detector analogue: freeze the `cv3` class convolutions for the first N rounds after the COCO warm start |
| **Backbone-only stage, then orthogonality-regularised full training** | [FedSTO / SSFOD](https://arxiv.org/abs/2310.17097) | *on BDD100K, Cityscapes and SODA10M.* Stage 1 refines the backbone selectively to avoid overfitting; reaches near-centralised with 20–30 % of labels. Third independent paper arriving at freeze-then-unfreeze |
| **BatchNorm is the non-IID failure point** | [Non-IID Quagmire](https://arxiv.org/pdf/1910.00189), [FedBN](https://openreview.net/forum?id=6YEQUn0QICG), [SiloBN in FedDrive](https://arxiv.org/abs/2202.13670), [Rethinking Normalization in FL](https://arxiv.org/pdf/2210.03277) | FedDrive's finding matches this repo's exactly: keeping BN local helps under **domain shift** and much less under **label skew**. The null FedBN result on an IID fleet is the predicted result, not a disappointment |
| **FixBN — the entry missing from phase 5** | [Making BatchNorm Great in Federated Deep Learning](https://arxiv.org/abs/2303.06530) | **freeze BN statistics after a warm-up phase, keep the layers live.** Beats GroupNorm in most FL settings, simpler than FedBN, no communication cost — and it leaves a *single* global model, so unlike FedBN it does not invalidate the holdout scorer. Cheapest unexplored item available |
| **Server-side EMA / SWA of the aggregate** | [FedSWA](https://arxiv.org/pdf/2507.20016), [FedEMA for autonomous driving](https://arxiv.org/html/2505.00318v1) | the server fuses this round's aggregate with a running EMA of previous rounds. Directly answers finding **1c**, costs one tensor list on the server, and `holdout --promote` already provides the measurement |
| **Aggregate predictions, not only weights** | [Cross-domain Federated Object Detection (FedOD)](https://arxiv.org/pdf/2206.14996) | multi-teacher distillation plus weighted box fusion — the detection-specific aggregation family. Heavy; parked |
| **Imbalance-aware aggregation** | [Survey on Class Imbalance in FL](https://arxiv.org/pdf/2303.11673), [cwFedAvg](https://arxiv.org/html/2406.07800) | the open `num_examples` images-vs-objects question is a recognised axis. Unsettled in detection; per-class weighting exists in classification |

---

## Part 4 — the ordered programme

Expected mAP per GPU-hour, on this machine, **without changing any data**. Item 0 gates
1–7: if the probe finds the two state dicts identical, Part 1 is wrong and gets struck
rather than built on.

| # | change | status | why |
|---|---|---|---|
| **0** | **prove Part 1** — checksum `trainer.model` against `yolo.model` after one `train()` | **done**, `fix/weights-that-travel`. Confirmed bit-exactly; numbers above | decided whether the rest of this document was real |
| **1** | **send the trained fp32 weights** — from the trainer, not the reloaded checkpoint | **done**, same branch. `trained_model` property; both checksums logged every round | removed the best-epoch lottery, the fp16 rounding, and the hidden `save=False` semantics change together |
| **2** | **`imgsz` as one number for federation, clients' val, baseline and holdout** | **done**, `perf/imgsz-as-a-lever`. `--imgsz PX`; **not yet run at 1024** | biggest absolute-mAP lever that touches no data. `_cmd_baseline` and `_cmd_evaluate` already took the flag, so raising it cannot silently become an unfair win |
| **3** | **`local_epochs = 1`, rounds up** | reachable already; a run, not a change | second reason now: the warmup clamp *and* the best-epoch lottery both vanish at 1 epoch |
| **4** | **head schedule** — `freeze=10` on round 1, then full | **done**, `feat/head-freeze-round-one`. `--freeze-round1 10`, default 0 | LP-FT, FedBABU and FedSTO independently. Measured: it also pins the backbone's BN statistics, so round 1 federates only the head |
| **5** | **server-side EMA of the aggregate** | **done**, `feat/server-side-ema`. `--server-ema D`, bias-corrected, default 0 | answers 1c without touching a client |
| **6** | **explicit `optimizer` on clients, adaptivity on the server** | **no code needed** — `optimizer` and `strategy` are already run-config keys. Two runs | the side that is reset every round should not be the adaptive one |
| **7** | **FixBN** as the first BN experiment, before FedBN | **done**, `feat/fixbn`. `--fix-bn-from-round R`, default 0 | keeps one global model, so unlike FedBN the holdout stays meaningful |
| **8** | **per-class AP on the four un-warmed classes** | reachable already (`holdout --evaluate` prints it) | tests *Where to Begin?*'s prediction; says whether to spend on the head or the aggregation |

**Every lever above defaults to off.** Nothing this document changed moves a number that
has already been measured, which is deliberate: the levers and the measurements are
separate pieces of work, and mixing them would make the first comparison uninterpretable.

⚠ touches `my-project/`, so a separate branch and prompt each, per
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).

**One correction to carry forward.** The ±0.0018 noise floor was measured at 2 × 1 on an
IID fleet. Items 1 and 2 change the transport and the input resolution, so the floor must
be re-measured after item 1 lands — otherwise the first real win is indistinguishable
from a change in variance.

---

## Sources

Papers, in the order they appear above:

- Reddi et al., *Adaptive Federated Optimization* (FedOpt / FedAdam / FedYogi) — <https://arxiv.org/abs/2003.00295>
- *Enabling Federated Object Detection for Connected Autonomous Vehicles: A Deployment-Oriented Evaluation* — <https://arxiv.org/abs/2509.01868>
- *From BDD100K to YOLOv8: a practical traffic-safety detection pipeline* — <https://aftabgazali001.medium.com/from-bdd100k-to-yolov8-a-practical-traffic-safety-detection-pipeline-fiftyone-ultralytics-256ec1fd1437>
- Nguyen et al., *Where to Begin? On the Impact of Pre-Training and Initialization in Federated Learning* — <https://arxiv.org/abs/2206.15387>
- Legate et al., *Guiding The Last Layer in Federated Learning with Pre-Trained Models* — <https://arxiv.org/abs/2306.03937>
- Oh et al., *FedBABU: Toward Enhanced Representation for Federated Image Classification* — <https://arxiv.org/abs/2106.06042>
- Kim et al., *Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection* (FedSTO) — <https://arxiv.org/abs/2310.17097>
- Hsieh et al., *The Non-IID Data Quagmire of Decentralized Machine Learning* — <https://arxiv.org/pdf/1910.00189>
- Li et al., *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization* — <https://openreview.net/forum?id=6YEQUn0QICG>
- Fantauzzo et al., *FedDrive: Generalizing Federated Learning to Semantic Segmentation in Autonomous Driving* (SiloBN) — <https://arxiv.org/abs/2202.13670>
- Du et al., *Rethinking Normalization Methods in Federated Learning* — <https://arxiv.org/pdf/2210.03277>
- Wang et al., *Making Batch Normalization Great in Federated Deep Learning* (FixBN) — <https://arxiv.org/abs/2303.06530>
- *FedSWA: Improving Generalization in Federated Learning ... via Momentum-Based Stochastic Controlled Weight Averaging* — <https://arxiv.org/pdf/2507.20016>
- *FedEMA: Federated Exponential Moving Averaging ... in Autonomous Driving* — <https://arxiv.org/html/2505.00318v1>
- Yu et al., *Cross-domain Federated Object Detection* (FedOD) — <https://arxiv.org/pdf/2206.14996>
- *A Survey on Class Imbalance in Federated Learning* — <https://arxiv.org/pdf/2303.11673>
- *Class-Wise Federated Averaging for Efficient Personalization* (cwFedAvg) — <https://arxiv.org/html/2406.07800>

Source read locally, for Part 1 and 2 — `ultralytics 8.4.115` in
`C:\Users\PRANAS\venvs\fl_yolov8`:

| claim | file:line |
|---|---|
| checkpoints hold only the fp16 EMA | `engine/trainer.py:725`, `:740` |
| `train()` reloads `best.pt` afterwards | `engine/model.py:828-833` |
| the loader prefers `ckpt["ema"]` | `nn/tasks.py:1899` |
| EMA rebuilt per round, `updates = 0` | `engine/trainer.py:404` |
| EMA decay ramp `0.9999·(1 − exp(−u/2000))` | `utils/torch_utils.py:734` |
| fitness is mAP50-95 alone | `utils/metrics.py:1007-1010` |
