# Data validation — the shards, checked against BDD100K itself

Measured 2026-09-25 against the fleet on disk (`random`, seed 0, 10 × 1 400,
fingerprint `c26b38858636`). Every number here came from a command whose output is
quoted; nothing is inferred from reading the builder.

**Result: the image data is sound. The `condition` partition's *labels* are not,
for two of the eight profiles.** The pixels, the boxes and the links are all exactly
what they claim to be; what is not true is the claim that vehicle 6 holds overcast
residential driving.

## What was run

| | command | outcome |
|---|---|---|
| structure, whole fleet | `python -m pipeline.validate` | `OK — labels present and usable, slices disjoint, splits separate, holdout intact.` |
| pixels + labels, sampled | `python -m pipeline.spotcheck -n 40 --seed 0` | `40/40 clean, 769 boxes checked against BDD's own annotation` |
| second, independent sample | `python -m pipeline.spotcheck -n 60 --seed 1` | `60/60 clean, 955 boxes checked against BDD's own annotation` |

100 images across all 10 shards, 28 train : 12 val in the first sample, **1 724 boxes**
reconstructed from `bdd100k_labels_images_*.json` and compared coordinate by
coordinate.

## What the spot-check proves that `validate.py` cannot

`pipeline/validate.py` says so itself — *"Cheap: no image decoding"*. It checks that a
label exists, not that it is the **right** label. The two are linked by filename alone,
and nothing had ever checked that link against the source.

| check | result over 100 images |
|---|---|
| decodes fully (catches truncated JPEGs) | 100/100 |
| 1280×720 RGB | 100/100 — no resized or greyscale copy anywhere |
| hardlinked to the kagglehub original, same inode | 100/100, `nlink` 3–7 |
| every row 5 fields, class id 0–12, coordinates in [0,1], non-zero area, inside frame | 100/100 |
| **box count equals BDD's own annotation for that image** | 100/100 |
| **every BDD box reproduces in the shard label, max coordinate error** | 1 724/1 724, **0.0** |
| sampled image found inside the holdout | 0 |

`nlink` 3–7 is the expected shape: one link in `my-project/batch`, one in the fleet
shard, one in the pooled baseline set, plus the kagglehub original, plus one per extra
fleet that also drew the image. Median file size 57 410 bytes — the shards cost no disk.

Max coordinate error **exactly 0.0** across 1 724 boxes is stronger than it looks: it
means the conversion is not merely close, it is the same arithmetic, and a label
swapped between two images would show up as a near-total mismatch rather than a
rounding drift.

## Two things that looked like data bugs and were not

Both were bugs in the *checker*, found and fixed before believing either. Recorded
because the next person to write this check will hit them in the same order.

| what it reported | what was actually true |
|---|---|
| "no original in the kagglehub cache" for **every train image** | the Kaggle mirror nests `images/100k/train/` into `trainA/`, `trainB/`, … — a flat listing finds **1 160 of 70 000**. `stages.py` already documents this ("a flat count reads low even when the download is complete"); the checker had to be told. Index the pool recursively |
| "33 boxes in the shard, 32 in BDD's annotation" | BDD's category strings are **`bike`** and **`motor`**, not `bicycle` and `motorcycle`, and `lane` / `drivable area` carry `poly2d` and never become a box. The shard's conversion is right; the reconstruction was wrong |

The second one matters beyond this file: any future comparison against raw BDD —
class-balance analysis, a re-conversion, a second dataset source — needs the same two
aliases or it will silently under-count bicycles and motorcycles, the two rarest
classes that still have enough instances to score.

## The real finding: two conditions cannot fill a 1 400-image shard

The condition partition draws from images actually present in `my-project/batch`, not
from all of BDD100K, and `_picker` **tops up with random unused images** when a
condition runs short rather than failing. Measured against the live pool
(41 036 train / 6 541 val, before `assign()` subtracts the 1 000-image holdout):

| profile | train images available | shard at 1 400 | top-up |
|---|---|---|---|
| night | 16 475 | pure | — |
| daytime city | 12 761 | pure | — |
| highway | 10 195 | pure | — |
| snow | 3 258 | pure | — |
| rain / fog | 3 037 | pure | — |
| dawn / dusk | 2 973 | pure | — |
| **overcast residential** | **685** | 685 real | **715 random — 51 %** |
| **parking / tunnel** | **319** | 319 real | **1 081 random — 77 %** |

The last condition fleet's own stats (`pipeline/.state/dataset_stats.json`,
fingerprint `510346f679aa`) confirm it from the other direction: vehicle 8
"parking / tunnel" holds **195 parking-lot images of 1 400** — 86 % of its shard is
city street, highway and residential, which is to say it is an IID vehicle wearing a
non-IID label. Vehicle 6 "overcast residential" holds 738 residential of 1 400.

**Consequences, in the order they bite:**

1. A 6-vehicle condition run is honestly non-IID — profiles 1–6 include overcast
   residential at 51 % top-up, so five of six vehicles are clean and one is half-diluted.
   An **8-vehicle** run adds the 77 % one.
2. `--size-skew` makes it worse in the way already documented: the largest vehicle
   is the one whose condition runs dry first.
3. This is the measurement to quote when FedBN is finally tested on
   `--partition condition`. Two of its clients have far less feature shift than their
   labels claim, so a null result there is **not** evidence that FedBN does nothing.

CLAUDE.md quotes 1 419 for overcast residential "in all of BDD100K". That is the whole
dataset; **685** is what is reachable locally, and the reachable number is the one that
decides a shard.

## Also, incidentally

The local pool is **47 577 distinct images** in `my-project/batch` — 41 036 train +
6 541 val, from which `assign()` removes the 1 000 held out — against an attribute
index of **79 863**. Nothing is broken by that — the partitioners read the directory
listing, not the index, so they can only ever pick a file that exists — but the two
numbers get quoted interchangeably and only the smaller one is real.

## Reproducing

```bash
python -m pipeline.validate                            # structure, all 16 800
python -m pipeline.spotcheck -n 40 --seed 0            # pixels + labels vs BDD
python -m pipeline.spotcheck -n 200 --seed 7 --json report.json
```

Exit code is non-zero if any sampled image fails. The BDD scan is ~8 s regardless of
sample size; decoding dominates past a few hundred images. It repairs nothing, for the
reason `validate.py` gives: a checker that fixes data hides the bug that produced it.
