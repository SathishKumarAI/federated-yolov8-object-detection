# Related work, and what this project owes it

Other people have built federated YOLO before, and at least one of them hit the same wall
this project spent a session on. This page names them, says what each one contributes that
is worth borrowing, and is explicit about what was taken: **ideas and measurements, read
from papers and public repositories. No code from any of these projects is in this one.**

That distinction has a licence consequence, so it is stated plainly rather than implied —
see [Licences](#licences) at the foot.

---

## The closest prior art: UltraFlwr

**[KCL-BMEIS/UltraFlwr](https://github.com/KCL-BMEIS/UltraFlwr)** · MICCAI-AMAI'25 ·
[arXiv:2503.15161](https://arxiv.org/abs/2503.15161) · AGPL-3.0

Flower × Ultralytics YOLO, for surgical object detection. The same pairing as this
project, arrived at independently, and the only other public codebase solving the same
integration problem.

| what they have | what it means here |
|---|---|
| **YOLO-PA — partial aggregation.** A custom Flower strategy that aggregates the backbone and neck and keeps the **head local** per client | This is `docs/PHASED_PLAN.md`'s "personalised heads" (phase 5, item 2) with a working implementation and a name. It is also the mirror image of `--freeze-round1`: that freezes the backbone and federates only the head for one round; YOLO-PA federates only the backbone, permanently |
| **Communication cost as the motivation**, not just accuracy — transmit fewer parameters | The head is where this argument is weakest for YOLOv8s and worth checking before copying the idea: this project measured its BatchNorm tensors at **80.3 % of tensor count but 0.36 % of values**, so "fewer tensors" and "fewer bytes" are very different claims. Measure the split before quoting a saving |
| **"The problem of loading the YOLO state dict"** is called out in their own README as a documented difficulty | Independent confirmation that the transport is the hard part of this pairing, not the strategy. This project's B4 bug (clients returning the weights they were sent) and the 2026-09-26 fp16-EMA bug were both in exactly that seam |
| Random, equally sized partitioning; they list better splits as future work | This project's `pipeline/vehicles.py` already does condition, dirichlet, mixed and size-skew partitioning, with a content-hash fingerprint per shard. That is the one axis where this repo is ahead |

**Not adopted, and why:** YOLO-PA is a real candidate for phase 5, but it cannot be judged
here until a difference bigger than the measured **±0.0077 mAP50** noise floor is what
"better" means. It also removes the single global model — the same caveat this project
already records for FedBN — so the holdout scorer would need to say whether it reports
per-client heads or a re-estimated one.

---

## The result that reframed this project's headline

**[Enabling Federated Object Detection for Connected Autonomous Vehicles: A
Deployment-Oriented Evaluation](https://arxiv.org/abs/2509.01868)** (September 2026)

YOLOv5 / YOLOv8 / YOLOv11 / Deformable DETR on KITTI, **BDD100K** and nuScenes, 8 clients,
10 rounds × 3 local epochs. On BDD100K with YOLOv8 they measured **FedAvg 61.5 against
centralised 61.4 mAP50** — federation matching its ceiling — and **FedAsync 45.7**.

Two things this project took from it, both recorded in
[`docs/findings/2026-09-26-accuracy-findings.md`](findings/2026-09-26-accuracy-findings.md):

1. "84.5 % retention" is not a law of federated detection. The gap was this project's
   schedule and transport, not federation itself.
2. **Staleness**, not heterogeneity, is what destroys detection accuracy — worth
   remembering before straggler simulation is added here as "realism".

---

## Techniques borrowed as ideas, each measured by someone else

| technique | source | where it landed here |
|---|---|---|
| Freeze-then-fine-tune, and why the ordering matters | [Guiding the Last Layer in FL with Pre-Trained Models](https://arxiv.org/abs/2306.03937) | `--freeze-round1` |
| Freeze the classifier, aggregate the body | [FedBABU](https://arxiv.org/abs/2106.06042) | same lever; also the argument for YOLO-PA above |
| Backbone-only first stage, on BDD100K itself | [FedSTO / SSFOD](https://arxiv.org/abs/2310.17097) | same lever, third independent confirmation |
| Pin BatchNorm statistics after a warm-up | [FixBN](https://arxiv.org/abs/2303.06530) | `--fix-bn-from-round`, and `pipeline/../trainers.py` |
| Keep BatchNorm local per client | [FedBN](https://openreview.net/forum?id=6YEQUn0QICG), [SiloBN in FedDrive](https://arxiv.org/abs/2202.13670) | `--local-bn`, already present |
| Average the aggregate across rounds | [FedSWA](https://arxiv.org/pdf/2507.20016), [FedEMA](https://arxiv.org/html/2505.00318v1) | `--server-ema` |
| Adaptivity belongs on the server | [Reddi et al., Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) | `--strategy fedadam` / `fedyogi` / `fedavgm`, already registered |
| Pre-trained init closes the non-IID gap | [Where to Begin? (ICLR 2023)](https://arxiv.org/abs/2206.15387) | `warm_start_head`, and the prediction that the residual loss sits in the four classes COCO could not warm |
| Aggregate predictions, not only weights | [FedOD](https://arxiv.org/pdf/2206.14996) | considered, parked — heavier than anything else on the list |

---

## Foundations

- **[Flower](https://github.com/adap/flower)** (Apache-2.0) — the federation runtime. This
  project is written against its **legacy** `ServerApp(server_fn=...)` API; the caveats of
  that choice, and what the Message API would change, are in
  [`../CLAUDE.md`](../CLAUDE.md).
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** (AGPL-3.0) — the
  detector, used as an installed dependency. Every behavioural claim this project makes
  about it was verified by reading the installed **8.4.115** source, with file and line.
- **[BDD100K](https://github.com/bdd100k/bdd100k)** (BSD-3-Clause toolkit; dataset under
  its own terms) — the data. Downloaded through kagglehub; never committed here.
- **[McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized
  Data](https://arxiv.org/abs/1602.05629)** — FedAvg itself. The weighted mean this
  project's [`pipeline/roundtrip.py`](../pipeline/roundtrip.py) checks the server against
  is the arithmetic from this paper.

## Licences

**No code from any project on this page has been copied into this one.** What was taken is
what papers and public repositories are for: a technique, a measurement, or a warning.

That matters most for **UltraFlwr (AGPL-3.0)**: it is the closest relative and the most
tempting to borrow from, and copying any of it would make this repository AGPL too. If
YOLO-PA is implemented here, it should be written from the paper's description — the way
FixBN and the server-side EMA in this repo were — and this page should record that it was.

Ultralytics is also AGPL-3.0, and is used as a dependency rather than vendored, which is
the ordinary use it is licensed for.
