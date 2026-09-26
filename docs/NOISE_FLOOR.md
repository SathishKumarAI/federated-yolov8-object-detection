# The noise floor — what counts as a result

The dashboard draws this number as a band on the holdout curve, and the projection panel
quotes it as "the smallest difference this run could resolve". Both need one place that
says what it is, how it was obtained, and when it stops being true. This is that place;
`pipeline/measurements.py` reads the value from here and
`python -m pipeline.measurements --check` fails if the two stop agreeing.

## The number

**±0.0018 mAP50**, measured 2026-09-26.

| | |
|---|---|
| conditions | 2 rounds × 1 local epoch, **IID** fleet, the same holdout in every arm |
| repeats | n = 3 |
| max − min | 0.0036 |
| stdev | 0.0019 |
| status | **lower bound.** n = 3 under-reports a spread |

## What it replaces, and why the old number was wrong

The figure this project used before was **±0.016**, and it was never a seed spread. It
was inferred from a *centralised* ceiling anomaly across two different data volumes — a
14 000-image ceiling scoring 0.4771 against an 8 400-image ceiling's 0.4936 — which is a
statement about data volume, not about run-to-run variance.

It was **8.9× too loose**, and a band that wide dismisses real differences as noise.
FedBN's +0.0040 was one of them.

## When this number stops being true

Two changes landed on 2026-09-26 that reopen it, and neither is cosmetic:

1. **The transport changed.** Until that day every client returned the fp16-rounded EMA
   of its own `best.pt` rather than the weights it had trained, so FedAvg averaged that.
   Variance measured under the old transport is variance of a different system.
2. **Resolution became a lever.** `imgsz` now propagates to the federation, the clients'
   validation, the centralised baseline and the holdout together. Changing it changes the
   scale of everything the floor was measured on.

So ±0.0018 is **measured at 2 × 1 on an IID fleet, under the old transport**, and the
honest reading of it is a lower bound on a system that has since moved. It must be
re-measured at 6 × 4 on a non-IID fleet before any ranking close to it is trusted.

## How the UI is allowed to use it

- Draw it as a band, centred on the first scored round, labelled with its conditions.
- Never as a tolerance that makes a difference "significant" — a delta *outside* the band
  is not thereby a result either; it is a difference worth repeating.
- Prefer an **observed** spread when the run ledger actually holds seed repeats:
  `pipeline/measurements.observed_spread()` finds runs identical except for `seed` and
  reports their real max − min with its n. A spread from this machine's own runs beats a
  recorded one, and when they disagree the recorded one is the stale half.

## Provenance note

This document exists because the branch that carries the dashboard predates the
`CLAUDE.md` revision that records the measurement. The canonical home is **`CLAUDE.md`,
"Facts about the training stack", fact 11**; if the two disagree, `CLAUDE.md` is right and
this file and `pipeline/measurements.py` are stale.
