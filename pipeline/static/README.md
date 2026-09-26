# The dashboard, by file

Open the one file that owns the thing you are changing. Nothing here needs a build
step: the server reads these files off disk on every request, so an edit is live on
reload. ES modules, no bundler, no CDN, no network at runtime.

| Change | File | Size |
|---|---|---|
| Anything visual — colour, spacing, type, a component's look | `app.css` | tokens at the top, components below |
| Page structure, a new panel, an element's `id` | `index.html` | markup only, no styles, no script |
| Axes, ticks, tooltips, sparkline, progress ring | `js/chart.js` | the only file that draws SVG plots |
| The run form, launch/stop, the stage table | `js/control.js` | |
| How state ARRIVES — the event stream, the diff apply, the fallback | `js/stream.js` | |
| The heartbeat, GPU readouts, criteria, reports, log stream | `js/live.js` | |
| Checksum ledger, per-class small multiples, round profile, provenance | `js/insight.js` | the panels that make a number harder to believe |
| The Simulate tab: the projection form and its table | `js/simulate.js` | the maths is server-side in `plan.py` |
| The walkthrough: its steps, its keyboard, the replay | `js/tour.js` | the recording is `pipeline/demo_run.py` |
| The fleet grid, the comparison and divergence charts | `js/fleet.js` | |
| The per-vehicle drawer | `js/drawer.js` | |
| The Data tab: counts, mixes, the shard table | `js/data.js` | |
| The Weights tab: what travels, the rendered kernels, the FedAvg identity | `js/anatomy.js` | the numbers come from `pipeline/anatomy.py` |
| Label boxes over a frame, and the trainer's own pictures | `js/consumed.js` | the only file that draws over an image |
| Helpers, colours, condition glyphs | `js/util.js` | |
| What the views share | `js/state.js` | one object, documented per field |
| Wiring and startup | `js/main.js` | ~20 lines; if it grows, something is in the wrong file |

## Rules that keep it that way

1. **`index.html` holds no CSS and no JavaScript.** A style belongs in `app.css`, a
   behaviour in `js/`. Layout-only inline `style` on a grid container is the single
   exception, and it is still a smell.
2. **Colour is a token.** `var(--accent)`, never a hex literal, except in the chart
   palette in `util.js` where a series needs a stable per-index colour.
3. **One concern per module, and modules import downward:** `main` → views
   (`control`, `live`, `fleet`, `drawer`) → primitives (`chart`, `util`, `state`).
   `fleet` and `drawer` reference each other on purpose — ES module live bindings
   handle the cycle, because both are called only after load.
4. **Everything the page shows comes from `/api/state`,** which the server derives
   from files on disk. That is why a run launched from the CLI lights up the same
   panels as one launched from the form. `/api/stream` pushes that same value as a
   snapshot then numbered diffs — a latency and bandwidth optimisation, never a
   second source of truth, and `stream.js` refetches `/api/state` whole if a
   sequence number ever skips.
5. **A view is handed a whole snapshot.** `stream.js` applies the diffs; no view
   ever sees a patch. A panel that needed to know about the transport would be a
   panel that breaks when the fallback poll kicks in.
6. **A new condition profile touches two files** — `PROFILES` in
   `pipeline/vehicles.py` and `GLYPHS` in `js/util.js`. Add both in the same commit.
7. **A value the server does not have renders as `unknown()`, never as a dash.** A
   dash reads as data, and `0` reads as a measurement. `unknown()` says "not
   measured" and carries the reason.
8. **A number on screen cites where it was measured.** The record lives in
   `pipeline/measurements.py`, whose `--check` mode re-reads the document it names.
   A panel that prints a figure with no row there is a bug. The Weights tab is the one
   place a figure may have no row: everything on it is computed from a named file at
   request time, which is a stronger provenance than a recorded constant — so the panel
   prints the file, its size and its mtime instead of a citation, and says it is doing so.

## Server routes it depends on

| Route | Serves |
|---|---|
| `GET /` | `index.html` |
| `GET /static/...` | this directory, guarded against traversal |
| `GET /api/state` | everything the panels render, in one response |
| `GET /api/stream` | SSE: that same state as a snapshot then numbered diffs |
| `GET /api/events` | SSE: log lines, stage transitions, signals |
| `GET /api/measurements` | every recorded number the page cites, with its source |
| `GET /api/profile` | seconds per phase for the last run, plus the verdict |
| `GET /api/simulate?...` | a projection for an arbitrary configuration; read-only |
| `GET /api/demo` | a recorded run in /api/state's shape, its sources, and its gaps |
| `GET /api/anatomy` | the newest global checkpoint, read out: tensors by kind, the first convolution's kernels, the head's class biases, bytes on the wire, and `roundtrip.py`'s verdict |
| `GET /api/vehicle/<vid>` | shard composition and sample image names |
| `GET /api/shard-image/<vid>/<name>` | one image out of that vehicle's shard |
| `GET /api/shard-labels/<vid>/<name>` | that frame's label rows, normalised, for the overlay |
| `GET /api/train-artifacts` | which of the trainer's own pictures exist, per vehicle |
| `GET /api/train-artifact/<vid>/<name>` | one of them; `<name>` must be in `train_artifacts.KINDS` |
| `POST /api/run`, `POST /api/stop` | start and stop the one allowed run |

The label overlay is an SVG on the unit square laid over the same `<img>` — the frame
is never sent twice and nothing renders boxes server-side. That only works while the
image element is not cropped, which is why `.strip .ovfig img` overrides
`object-fit:cover`; `pipeline/tests/test_pipeline.py::test_label_overlay_boxes_land_where_the_label_file_says`
runs the conversion under node and fails if a box moves.
