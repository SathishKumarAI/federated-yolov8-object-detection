# Running the model live, on five machines

Written 2026-09-02. The federation trains a model; this runs it. One central machine
holds the dashboard and the checkpoints, and up to five test machines each point a
camera at the world and report what the current global model sees.

**Nothing on a test node trains.** A camera stream has no ground-truth labels, and
training on the model's own predictions is a confirmation loop that reinforces its
mistakes with no held-out check to catch the drift. Training stays on the labelled
BDD100K shards, where there is a holdout. The nodes are the demonstration, not a data
source.

## What each piece is

| Piece | File | Runs on |
|---|---|---|
| Dashboard, checkpoint server, node registry | `pipeline/server.py`, `pipeline/nodes.py` | the central machine |
| The edge node: camera → model → telemetry | `pipeline/edge.py` | each test machine |
| The panel | `pipeline/static/js/edge.js` | the browser |

## Start it

**Central machine.** Loopback by default, which is deliberate: `POST /api/run` starts
training subprocesses from a request body and **nothing here authenticates**. Binding
wider is a per-launch decision on a network you control, and the server prints a
warning when you make it.

```bash
python -m pipeline.server                 # 127.0.0.1:8800, nodes must be local
python -m pipeline.server --host 0.0.0.0  # nodes on other machines can report in
```

**Each test machine**, pointed at the central one:

```bash
python -m pipeline.edge --id cam-1 --server http://192.168.1.10:8800
python -m pipeline.edge --id cam-2 --server http://192.168.1.10:8800 --camera 1
```

No camera on a machine? `--source synthetic` drives every other step — model download,
inference, telemetry, frame upload — with a generated frame. It is honest about what it
proves: the detector finds nothing in a moving rectangle, so this tests the plumbing and
says nothing about detection quality.

`--device cpu` is the default on purpose: a test node should not compete for the card
the federation is training on. Measured on this machine, CPU inference at 640px ran
**15.1 fps at 36.7 ms** per frame — comfortably real-time for a webcam.

## What the panel shows, and the column that matters

Each node reports at ~1 Hz and uploads a frame once a second. The panel shows fps,
latency, device, detections and per-class counts — and **the round each node is on**.

That last one is the point. A node keeps running the checkpoint it downloaded until it
notices a newer one, so during a federation the fleet is visibly split across rounds.
The boxes drawn on a node's frame belong to *that node's* round, not to whatever the
server has finished aggregating. Without the round on screen, a stale node's detections
get read as the current model's.

Nodes cache on the checkpoint's **content hash**, not its round number: re-running a
federation rewrites `global_round_1.pt` with different weights under the same name, and
a node keyed on the name would serve the previous run's model forever without ever
looking stale.

## The routes

| Route | Serves |
|---|---|
| `GET /api/model` | newest checkpoint: name, round, size, sha256 |
| `GET /api/model-file` | its bytes, ~22 MB |
| `POST /api/node/<id>` | one heartbeat: telemetry, and optionally a base64 JPEG |
| `GET /api/nodes` | every node, online flag, fleet totals |
| `GET /api/node-frame/<id>` | that node's newest frame, `no-store` |

`global_last.pt` is deliberately never served: it duplicates the highest round under a
name that carries no round number, so a node caching by name would never see it change.

## The trust boundary, since this one accepts data from other machines

`pipeline/nodes.py` is the only part of this project that takes input from off-machine,
and it is written as if the network is not trusted even while the default binding says
it is:

- **Frames live in memory and never touch disk.** One per node, the previous dropped.
  No path to traverse, no directory to fill, nothing to clean up when a node vanishes.
- **Node ids are refused, not sanitised** — `^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$`. A
  silently renamed node is two rows in the listing and one confused operator.
- **Every field is bounded before storage**: strings truncated, numbers clamped, the
  open-ended `counts` map capped at 32 keys. A request body over 1 MB is refused with
  413 before it is read as JSON.
- **The registry is capped** at 64 nodes, so a caller generating a fresh id per POST
  cannot grow it forever. An existing node can still report after the cap is reached.
- **A node that stops reporting goes offline rather than disappearing.** Vanishing would
  read as "never existed"; the failure an operator cares about is "it was there and it
  stopped".

There is still **no authentication**. On a LAN you control that is a reasonable trade;
across anything else it is not, and adding a shared token to the heartbeat route is the
smallest honest fix.

## Where this goes next

This runs the model live. It does **not** make the five machines real Flower *clients* —
they infer, they do not train. Training over real machines is the Deployment Engine, and
`pipeline/deploy.py` now drives it:

```bash
python -m pipeline.deploy --nodes 2 --rounds 2 --epochs 1   # all local
python -m pipeline.deploy --dry-run                          # print, start nothing
python -m pipeline.deploy --external-only --superlink-host 0.0.0.0
```

`--external-only` starts just the SuperLink and prints the exact command each real
machine runs, which is the whole difference between one host and five.

**`--nodes` defaults to 2, and that default is the interesting part.** Under simulation,
Ray places clients with `client-resources.num-gpus`, so `--gpu-fraction 0.33` caps the
fleet at three concurrent clients on one card. A SuperNode is an ordinary OS process:
nothing schedules it and nothing caps its VRAM. Each carries its own CUDA context
(~300–500 MiB before any weights) plus a full training footprint, and the measured
demo-scale peak is already 12–15 GB for two or three *scheduled* clients. Six SuperNodes
on one 16 GB card will not fit — it fails as out-of-memory, not as slowness.

Underlying commands, for reference:

```bash
flower-superlink --insecure                        # Fleet API 9092, Control API 9093
flower-supernode --insecure --superlink <host>:9092 \
                 --host 0.0.0.0 --port 9094 \
                 --node-config "partition-id=0 num-partitions=5"
flwr run . local-deployment --stream
```

with the federation declared in `~/.flwr/config.toml` as `[superlink.local-deployment]`
carrying `address` and `insecure`. Note the commented `[tool.flwr.federations.*]` block
in `my-project/pyproject.toml` is the **old** spelling — flwr migrated federations out
of pyproject, which is what the migration notice it writes on every run has been saying.

Do that on this one machine as five processes first. It is the same code path as five
hosts, and it separates "does the deployment engine work" from "is the network right".
