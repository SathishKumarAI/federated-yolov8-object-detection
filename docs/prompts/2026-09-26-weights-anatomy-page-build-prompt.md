# A page that shows what actually travels — build prompt

`2026-09-26` · branch `feat/weights-anatomy` off `main`

## Goal

One new page in the existing dashboard that explains, to someone with no ML background,
**what actually travels between the server and the vehicles** — what a "weight" is, what
the numbers mean, and what information they carry. A real visualisation of real tensors:
live during a run, and a replayable demo when nothing is running. It joins the dashboard
that exists; it does not start a parallel one.

## What it must explain, and show

1. **What is in the model at all.** The state a round exchanges, broken down by kind —
   convolution kernels, BatchNorm scale/shift, BatchNorm running statistics, the
   detection head's class biases — with counts, share of tensors, share of values and
   bytes. Surface the counter-intuitive pair: BatchNorm is about four fifths of the
   tensor *count* and a third of a percent of the *values*.
2. **What one weight actually is.** A real tensor, rendered. The first convolution is 32
   kernels of 3×3×3 — 32 little RGB images, the one place where "weights" become
   something a person can literally see. The learned BatchNorm statistics as a
   distribution. The head's per-class bias, connected to the 13 class names, because that
   is where "this model knows what a car is" becomes concrete.
3. **What travels each round, and in which direction.** Down: the server's aggregate.
   Up: each vehicle's trained arrays plus one integer, `num_examples`, which is FedAvg's
   weight. Bytes on the wire per round per vehicle, and at the configured fleet size.
   State plainly that **no image ever moves** — and that weights are not raw data but are
   not nothing either. Gradient leakage is a real research area; say so rather than
   overclaiming privacy.
4. **The averaging, visibly.** For a finished run: each vehicle's returned checksum, its
   `num_examples`, the weighted mean, and the server's published aggregate side by side.
   When they agree, the reader has *seen* that the federation is real.
5. **Live.** While a run is in progress the page updates through the existing SSE stream.
   No second transport, no polling loop.
6. **Demo.** With no run and no GPU it replays recorded state the way the walkthrough
   already does, and says clearly that it is a replay.

## Hard constraints

- **Read real checkpoints.** `my-project/checkpoints/global_*.pt`. Load torch **lazily,
  inside the function that needs it** — the pipeline test job installs `pytest` and
  `pyyaml` only, and a module-scope `import torch` fails the whole suite on a bare
  interpreter. This is not hypothetical; it broke CI today.
- **Never fabricate.** No placeholder tensors, no "example" numbers presented as
  measurements. If no checkpoint exists, the panel says there is none — not zero.
- **Do not re-implement `roundtrip.py`'s arithmetic.** Forward its verdict.
- No GPU training, no `flwr run`, no `pipeline.runner`: the card is in use. CPU only;
  reading checkpoint files is fine.
- Never kill a process by image name. Kill the PID you started, or nothing.
- Keep files small and single-purpose. A new measured *constant* goes through
  `pipeline/measurements.py` so it carries provenance.
- Accessibility: keyboard reachable, visible focus, real contrast,
  `prefers-reduced-motion` respected, no meaning carried by colour alone. Large tensors
  must not blow up the page — decimate sensibly and say that you did.
- Loopback only. Any new route is read-only.

## Deliverables

| File | What |
|---|---|
| `pipeline/anatomy.py` | reads the newest global checkpoint; returns the whole payload |
| `pipeline/static/js/anatomy.js` | the panels, static half and live half |
| `pipeline/static/index.html` | the tab and its section |
| `pipeline/static/app.css` | the kernel grid and the page's prose style |
| `pipeline/server.py` | `GET /api/anatomy` |
| `pipeline/static/js/{main,live,tour}.js` | wiring, the live snapshot, one walkthrough step |
| `pipeline/tests/` | Python and a node render check |
| `pipeline/static/README.md`, `CLAUDE.md`, `pipeline/docs_index.py` | the change→file maps |

## Definition of done

```
python -m pytest pipeline/tests -q      # green, on the venv AND on a bare interpreter
python -m pipeline.anatomy              # a real checkpoint, read out
python -m pipeline.measurements --check  # every record still matches its document
```
plus the server exercised over HTTP on a port that is not the running dashboard's, and
the page rendered — in a browser if one is available, otherwise through node the way
`pipeline/tests/js/live_render.mjs` does.

## Out of scope

- Any change to `my-project/`. The page reads its outputs; it never writes there.
- Any new dependency. Stdlib, torch (lazily), and the dashboard's own modules.
- Re-deriving the FedAvg identity, the log parsing, or the checksum arithmetic. Those
  already exist and having two of any of them is how they come to disagree.
