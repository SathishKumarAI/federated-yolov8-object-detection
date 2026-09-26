# Build prompt — modernise the dashboard: push, provenance, projection, walkthrough

Received before the code, as the rule here requires. Recorded verbatim in substance, with
the two amendments that arrived mid-work marked as such, and a closing note on what was
deliberately not done — that gap is the part worth keeping.

Branch: `feat/dashboard-2030`, off `main` at `d9072300b`, in an isolated worktree.

## Goal

The dashboard should reflect state faster, make every number on it traceable, explain the
things this project's history says are easy to misread, let someone project a run's cost
before spending GPU hours, and teach a newcomer how to read it with no run in progress
and no GPU.

## Hard constraints, stated as prohibitions

- **Do not touch** `my-project/**`, `pipeline/holdout.py`, `pipeline/runner.py`,
  `pipeline/baseline.py`. A second agent is changing the ML side concurrently; anything
  written there conflicts and is thrown away.
- **Do not run GPU training**, `flwr run`, or `python -m pipeline.runner`. CPU only.
- **Do not commit data**, checkpoints or images. A new kind of generated artifact gets its
  ignore rule in the same commit.
- **Never render a placeholder, a fabricated number, or a fake "live" value.** Where a
  value is unknown the panel says so explicitly rather than showing zero or a dash that
  reads like data.
- **Do not widen the mutating surface.** The server binds loopback and `POST /api/run`
  starts training subprocesses with nothing authenticating them. Any new mutating route
  needs the same treatment and a note in the commit body.
- Keep the split: one concern per file, no giant `app.js`. Update
  `pipeline/static/README.md`'s `change → file` table in the same commit as any file
  added, and keep the view-id test passing for every module.
- Accessibility is not optional: keyboard reachable, visible focus, real contrast,
  `prefers-reduced-motion` respected, no meaning carried by colour alone.

## Deliverables, in priority order

1. **Replace polling with push.** Server-Sent Events with a *diff* payload — only what
   changed since the client's last sequence number, a full snapshot on connect and on
   reconnect. Keep `/api/state` working. No busy-waiting. Stdlib only; SSE is the default
   because it survives proxies and needs no dependency, and a websocket needs justifying
   in the commit body.
2. **Truth-dense display.** Every number a real measured value with its provenance
   reachable — which run, which round, which fingerprint, and a `±` where the repo knows
   the spread.
3. **Contextual visualisations that explain, not decorate.** Per-round mAP with the
   measured noise floor as a band so a delta inside it reads as noise; per-class AP small
   multiples, including which classes could not be warm-started from COCO; per-vehicle
   divergence and contribution; where a round's seconds went; the aggregate checksum,
   prominently, because equal consecutive values mean nothing is being learned; shard
   composition against what a condition can actually supply.
4. **Model simulation from the frontend.** Rounds, local epochs, vehicles,
   images-per-vehicle, partition, strategy, image size → a *projection* of wall clock,
   VRAM, image-visits and expected effect, computed from what this repo has already
   measured. Clearly labelled a projection, naming which measurement each part rests on,
   and **refusing to project outside the ranges that were actually measured** rather than
   extrapolating silently. It must not start GPU training.
5. **Demo / help mode.** A guided walkthrough that works with no run and no GPU: replays a
   recorded run (or a small committed fixture), explains each panel in order with what it
   is for and the one mistake it prevents. Reachable from the UI, keyboard-navigable,
   dismissible.

## Amendment received mid-work (coordinator)

Five run levers landed on the ML side — `freeze_round1`, `server_ema`,
`fix_bn_from_round`, `imgsz`, `local_bn`. Surface them in the run form and the projection
panel, with these caveats visible: FixBN from round 1 pins COCO's initial statistics and
is **not** the method; FedBN leaves **no single global model**, so a holdout score then
measures a model that never existed. Also:

- **The measured noise floor is ±0.0018 mAP50** (n = 3, 2 rounds × 1 epoch, IID), not the
  ±0.016 older docs quote. Treat it as a lower bound, note that the transport fix and the
  resolution lever reopen it, read it from a run if possible, and label it with the
  conditions it was measured under.
- **A bug fixed on 2026-09-26 changes what past numbers mean.** Until then every client
  returned the fp16-rounded EMA of its own `best.pt` rather than the weights it trained,
  and FedAvg averaged that. Runs either side of the fix must not be plotted as one series;
  the new lever fields in `report.json` are a usable discriminator.
- `Scripts\flwr.exe` is blocked by Windows Application Control (`WinError 4551`); the
  pipeline launches flwr via `python -c "from flwr.cli.app import app; app()"`.
- Ray crashed the pipeline process at `--gpu-fraction 0.5` with a Windows access
  violation; 1.0 completed. 0.5 is no longer known-safe and 0.33 has no headroom at all.

## Definition of done

```
python -m pytest pipeline/tests -q        # green, plus a test per new behaviour
python -m pipeline.measurements --check   # every cited value still in its document
```

The load-bearing test is an SSE diff test that fails if a change is dropped between
sequence numbers. Start the server on `--port 8788` and exercise it with real requests and
a real event stream; quote the output. Measure, do not infer.

## Out of scope, deliberately

- Any change to the aggregation strategy, the client training loop, or the data.
- Any new runtime dependency, build step, bundler or CDN.
- Plumbing a run lever through to a subprocess: that path runs through
  `pipeline/runner.py`, which is the other branch's.
- Predicting an mAP for an unrun configuration. One configuration's end-to-end result is
  recorded; extrapolating from it would be a fabricated number carrying a measured one's
  confidence, which is the failure the rest of this brief exists to prevent.

## What was not delivered, and why

- **Browser verification.** The Chrome DevTools MCP profile was locked by another session
  for the whole of this work, so the page was never loaded in a real browser. Substituted:
  `pipeline/tests/js/live_render.mjs` renders a whole `/api/state` payload through the
  real panels under node and asserts what each one shows. It caught a genuine
  `ReferenceError` in `renderHoldoutProvenance` that no amount of reading had found.
- **Shard composition against what a condition can supply** (deliverable 3, last item).
  The measurement lives in `docs/DATA_VALIDATION.md` and `pipeline/spotcheck.py`, neither
  of which is on `main`, and this machine has no shards on disk — so it could not have
  been verified, only written. Left undone rather than shipped unmeasured.
