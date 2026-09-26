"""The two dashboards: a control view to launch runs, a live view to watch them.

Scope is deliberately narrow. MLflow owns metrics storage, history and run
comparison; the Ray Dashboard owns actor and GPU internals. This server only does
what neither can: start a run from a form, and narrate a fleet of vehicles while it
trains. It stores nothing — every number here is read from a running subprocess,
from my-project's logs, or from nvidia-smi.

Stdlib only, loopback only, no build step.

    python -m pipeline.server           # then open http://127.0.0.1:8800
"""
from __future__ import annotations

import base64
import binascii
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from urllib.parse import parse_qs, unquote

from . import (baseline, dataset_stats, demo_run, docs_index, gpu, holdout, ledger,
               logparse, measurements, nodes, paths, plan, profile as profiler, stages,
               statestream, train_artifacts, vehicle_metrics, vehicles, verify)
from .runner import Run
from .stages import Config

STATIC = Path(__file__).resolve().parent / "static"
HISTORY_LIMIT = 500

#: How stale the shared snapshot may be. One snapshot serves every open tab and both
#: `/api/state` and `/api/stream`, so this is the whole cost of the push loop -- it is
#: not multiplied by the number of browsers watching.
SNAPSHOT_TTL = 0.35
#: Longest a stream sits idle before it sends a keep-alive. A bus event (a log line, a
#: stage transition, a checksum) wakes it sooner, so this is the ceiling on latency for
#: things that leave no trace on the bus -- the GPU sampler, mostly.
STREAM_TICK = 1.0


def safe_child(root: Path, rel: str) -> Path | None:
    """``root/rel`` if it is a real file genuinely under ``root``, else None.

    One guard for every route that maps a URL onto a path. Written once because two
    copies of a traversal check is one copy too many: `/reports/../../secret` and
    `/api/shard-image/1/../../../secret` are the same bug.
    """
    target = (root / rel).resolve()
    if not target.is_file() or root.resolve() not in target.parents:
        return None
    return target


class Broadcaster:
    """Fan one Run's event queue out to every connected browser."""

    def __init__(self):
        self.subscribers: list["list"] = []
        self.history: list[dict] = []
        self.lock = threading.Lock()
        # Bumped whenever anything happens, so a state stream can wait on real events
        # instead of spinning on a timer. A revision counter rather than an Event:
        # with an Event, the first waiter to clear it steals the wake-up from every
        # other open tab, and the second tab would lag by a whole STREAM_TICK.
        self.revision = 0
        self.changed = threading.Condition(self.lock)

    def publish(self, ev: dict) -> None:
        with self.changed:
            self.history.append(ev)
            del self.history[:-HISTORY_LIMIT]
            for q in self.subscribers:
                q.append(ev)
            self.revision += 1
            self.changed.notify_all()

    def wait(self, seen: int, timeout: float) -> int:
        """Block until the revision moves past ``seen``, or ``timeout``. Returns it."""
        with self.changed:
            self.changed.wait_for(lambda: self.revision != seen, timeout)
            return self.revision

    def subscribe(self) -> list:
        q: list = []
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: list) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


class State:
    """Everything the dashboards need, and the one run allowed at a time."""

    def __init__(self):
        self.bus = Broadcaster()
        self.run: Run | None = None
        self.thread: threading.Thread | None = None
        self.idle_sampler = gpu.Sampler(interval=3.0).start()
        self._snap: dict | None = None
        self._snap_at = 0.0
        self._snap_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg: Config, chain, confirm: bool, ray_address: str | None) -> bool:
        if self.busy:
            return False
        run = Run(cfg, confirm_all=confirm, ray_address=ray_address)
        self.run = run

        def pump():
            while True:
                ev = run.events.get()
                self.bus.publish(ev)
                if ev.get("kind") == "run_end":
                    return

        threading.Thread(target=pump, daemon=True).start()
        self.thread = threading.Thread(target=run.execute, args=(chain,), daemon=True)
        self.thread.start()
        return True

    def live(self) -> dict:
        """Run state read from disk.

        Derived from logs and metrics rather than from this server's event bus, so a
        run launched from the CLI -- or one that started before this server did --
        still shows up. The event bus stays for the low-latency log stream.
        """
        checksums = logparse.aggregate_checksums()
        metrics_path = verify.metrics_csv()
        rows = logparse.read_metrics_csv(metrics_path)

        per_vehicle: dict[str, dict] = {}
        current = None
        noop = 0
        for f in logparse.current_run_logs("client*.log"):
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for ev in logparse.parse_text(text):
                if ev.kind == "training_start":
                    current = str(ev.value)
                    v = per_vehicle.setdefault(current, {"rounds": 0, "received": None,
                                                         "sent": None, "device": None})
                    v["rounds"] += 1
                elif ev.kind == "no_optimizer_step":
                    noop += 1
                elif current:
                    v = per_vehicle.setdefault(current, {"rounds": 0, "received": None,
                                                         "sent": None, "device": None})
                    if ev.kind == "client_received_checksum":
                        v["received"] = ev.value
                    elif ev.kind == "client_sent_checksum":
                        v["sent"] = ev.value
                    elif ev.kind == "device":
                        v["device"] = ev.value

        ok, criteria = verify.check()
        evaluated = [r for r in rows if r.get("stage") == "evaluate"]
        return {
            "checksums": checksums,
            "rounds_done": len(checksums),
            "metrics": rows,
            "map50": [r.get("mAP50") for r in evaluated],
            "loss": [r.get("loss") for r in evaluated],
            "per_vehicle": per_vehicle,
            "training_now": current if self.busy else None,
            "no_optimizer_steps": noop,
            "criteria": criteria,
            "criteria_ok": ok,
            "checkpoints": sorted(p.name for p in (paths.PROJECT / "checkpoints").glob("global_*.pt")),
            "learning": vehicle_metrics.summary(),
            # The only metric measured on data no client trained on, and the
            # centralised ceiling it is worth comparing against.
            "holdout": holdout.curve(),
            "baseline": baseline.gap(),
        }

    def reports(self) -> list[dict]:
        out = []
        if paths.REPORTS.is_dir():
            for d in sorted(paths.REPORTS.iterdir(), reverse=True)[:12]:
                if (d / "report.html").exists():
                    out.append({"name": d.name, "url": f"/reports/{d.name}/report.html"})
        return out

    def snapshot(self, cfg: Config) -> dict:
        live = self.run.sampler.telemetry if self.busy and self.run else self.idle_sampler.telemetry
        latest = live.latest
        return {
            "busy": self.busy,
            "current": self.run.current if self.run else None,
            "config": cfg.to_dict(),
            "stages": stages.snapshot(cfg),
            "fleet": vehicles.load_fleet(),
            "gpu": {
                "util_pct": latest.util_pct if latest else None,
                "mem_used_mib": latest.mem_used_mib if latest else None,
                "mem_ceiling_mib": gpu.VRAM_CEILING_MIB,
                "power_w": latest.power_w if latest else None,
                "temp_c": latest.temp_c if latest else None,
                "history": [{"util": s.util_pct, "power": s.power_w, "mem": s.mem_used_mib}
                            for s in live.samples[-120:]],
                **live.summary(),
            },
            "links": {"mlflow": "http://127.0.0.1:5000", "ray": "http://127.0.0.1:8265"},
            "options": {"partitions": list(vehicles.PARTITIONS),
                        "strategies": list(stages.STRATEGIES)},
            "results": [r.__dict__ for r in (self.run.results if self.run else [])],
            "live": self.live(),
            "reports": self.reports(),
        }

    def current(self, cfg: Config) -> dict:
        """The snapshot, recomputed at most every SNAPSHOT_TTL seconds.

        Shared, and the sharing is the point: `snapshot()` reads every client log, the
        metrics CSV, the holdout curve and the baseline off disk, and the push loop
        below asks for it far more often than the old two-second poll did. One tab or
        five, that work happens once. Callers get the *same object* back within the
        TTL, which is also how a stream recognises "nothing to send" for free.
        """
        with self._snap_lock:
            now = time.monotonic()
            if self._snap is None or now - self._snap_at > SNAPSHOT_TTL:
                self._snap = self.snapshot(cfg)
                self._snap_at = now
            return self._snap


STATE = State()
CONFIG = Config()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):        # the browser polls; the console stays readable
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            # Kept as the whole truth, and as what a stream falls back to: the browser
            # refetches it if a sequence number ever skips.
            return self._json(STATE.current(CONFIG))
        if self.path == "/api/stream":
            return self._state_stream()
        if self.path == "/api/events":
            return self._sse()
        if self.path.startswith("/static/"):
            return self._static()
        if self.path.split("?")[0] == "/api/data":
            # Seconds to compute over 14 000 label files, so it is cached against the
            # fleet fingerprint and never computed inside the 2-second poll.
            return self._json(dataset_stats.cached(force="refresh=1" in self.path))
        if self.path.split("?")[0] == "/api/ledger":
            rows = ledger.load()
            return self._json({"runs": rows, "approaches": ledger.by_approach(rows)})
        if self.path.split("?")[0] == "/api/docs":
            return self._json(docs_index.index())
        if self.path.split("?")[0] == "/api/plan":
            return self._json(plan.plan(CONFIG))
        if self.path.split("?")[0] == "/api/demo":
            # A recorded run in the shape /api/state returns, so the real panels render
            # it and the walkthrough needs no views of its own.
            return self._json(demo_run.payload())
        if self.path.split("?")[0] == "/api/simulate":
            # Read-only and side-effect free: it starts nothing, writes nothing, and
            # touches no global. A GET on purpose, so it adds nothing to the mutating
            # surface that POST /api/run already is.
            return self._simulate()
        if self.path.split("?")[0] == "/api/measurements":
            # Static and tiny: the provenance every panel cites. Deliberately NOT part
            # of /api/state -- it never changes while the server runs, so putting it in
            # the diff stream would ship it once and then never again, which is fine,
            # and putting it in the snapshot would ship it to every reconnect.
            return self._json(measurements.table())
        if self.path.split("?")[0] == "/api/profile":
            # Seconds per phase, parsed out of the logs the run already wrote. Costs a
            # full read of every client log, so it is fetched when someone looks at the
            # panel rather than on every tick of the state stream.
            got = profiler.profile()
            # The verdict is the point: a phase breakdown that leaves the reader to
            # decide between "clients serialised" and "dataloader starving the GPU"
            # has not made the measurement useful.
            if "error" not in got:
                got = {**got, "verdict": profiler.verdict(got)}
            return self._json(got)
        if self.path.startswith("/api/vehicle/"):
            return self._vehicle()
        if self.path.split("?")[0] == "/api/train-artifacts":
            return self._json(train_artifacts.listing())
        if self.path.startswith("/api/train-artifact/"):
            return self._train_artifact()
        if self.path.startswith("/api/shard-labels/"):
            return self._shard_labels()
        if self.path.startswith("/api/shard-image/"):
            return self._shard_image()
        if self.path.startswith("/reports/"):
            return self._report_file()
        if self.path.split("?")[0] == "/api/nodes":
            return self._json(nodes.listing())
        if self.path.split("?")[0] == "/api/model":
            return self._json(nodes.latest_model())
        if self.path.split("?")[0] == "/api/model-file":
            got = nodes.model_bytes()
            if got is None:
                return self._json({"error": "no global checkpoint yet"}, 404)
            data, name = got
            return self._send(200, data, "application/octet-stream")
        if self.path.startswith("/api/node-frame/"):
            return self._node_frame()
        self._json({"error": "not found"}, 404)

    def _node_heartbeat(self, body: dict) -> None:
        """One report from an edge node running the global model on a camera."""
        node_id = unquote(self.path.split("?")[0].rsplit("/", 1)[-1])
        frame = None
        raw = body.pop("frame", None)
        if raw:
            try:
                frame = base64.b64decode(raw, validate=True)
            except (binascii.Error, ValueError) as e:
                # Refused, not dropped: a node silently losing its picture while its
                # telemetry keeps arriving looks like a camera fault on the far end.
                return self._json({"error": f"frame is not valid base64: {e}"}, 400)
        try:
            record = nodes.heartbeat(node_id, body, frame)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"ok": True, "id": record["id"], "frames": record["frames"]})

    def _node_frame(self) -> None:
        node_id = unquote(self.path.split("?")[0].rsplit("/", 1)[-1])
        data = nodes.frame(node_id)
        if data is None:
            return self._json({"error": "no frame for that node"}, 404)
        # No caching: the whole value of this image is that it is the newest one, and a
        # browser holding the first frame forever would show a live fleet as frozen.
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    #: Only what the dashboard is made of. An unknown suffix is a 404, not an
    #: octet-stream download, so a stray file here cannot be exfiltrated by URL.
    STATIC_TYPES = {
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def _static(self) -> None:
        """Serve the dashboard's own CSS and JS modules, straight off disk.

        Read per request on purpose: an edit is live on reload, which is what makes
        a no-build-step page worth having.
        """
        target = safe_child(STATIC, self.path[len("/static/"):].split("?")[0])
        if target is None or target.suffix not in self.STATIC_TYPES:
            return self._json({"error": "not found"}, 404)
        self._send(200, target.read_bytes(), self.STATIC_TYPES[target.suffix])

    #: What a projection query may set, and how to read it. Anything else in the query
    #: string is ignored rather than guessed at -- the panel and this table are the
    #: contract, and a mistyped parameter silently changing nothing is better than one
    #: silently being interpreted.
    SIM_FIELDS = {
        "vehicles": ("n_vehicles", int), "rounds": ("rounds", int),
        "epochs": ("local_epochs", int), "per_vehicle": ("per_vehicle_override", int),
        "seed": ("seed", int), "partition": ("partition", str),
        "strategy": ("strategy", str), "alpha": ("alpha", float),
        "size_skew": ("size_skew", float), "gpu_fraction": ("gpu_fraction", float),
        "profile": ("profile", str),
    }

    def _simulate(self) -> None:
        """Project an arbitrary configuration. Changes nothing, starts nothing."""
        query = parse_qs(self.path.partition("?")[2])
        fields: dict = {}
        for name, (attr, cast) in self.SIM_FIELDS.items():
            raw = (query.get(name) or [None])[0]
            if raw in (None, ""):
                continue
            try:
                fields[attr] = cast(raw)
            except (TypeError, ValueError):
                return self._json({"error": f"{name}={raw!r} is not a {cast.__name__}"}, 400)
        # Validated here rather than three layers down, and refused rather than silently
        # corrected: a projection of a configuration the run form would reject is a
        # projection of something that cannot happen.
        if fields.get("partition", Config.partition) not in vehicles.PARTITIONS:
            return self._json({"error": f"unknown partition {fields['partition']!r}"}, 400)
        if fields.get("strategy", Config.strategy) not in stages.STRATEGIES:
            return self._json({"error": f"unknown strategy {fields['strategy']!r}"}, 400)
        if not 0 < fields.get("gpu_fraction", 1.0) <= 1:
            return self._json({"error": "gpu_fraction must be in (0, 1]"}, 400)
        try:
            cfg = Config(**fields)
        except TypeError as e:
            return self._json({"error": str(e)}, 400)
        raw_imgsz = (query.get("imgsz") or [None])[0]
        try:
            imgsz = int(raw_imgsz) if raw_imgsz else None
        except ValueError:
            return self._json({"error": f"imgsz={raw_imgsz!r} is not an integer"}, 400)
        self._json(plan.project(cfg, imgsz))

    def _vehicle(self) -> None:
        """Shard composition for one vehicle, for the detail drawer."""
        try:
            vid = int(self.path.rsplit("/", 1)[-1].split("?")[0])
        except ValueError:
            return self._json({"error": "vehicle id must be an integer"}, 400)
        self._json(vehicles.composition(vid))

    def _shard_image(self) -> None:
        """One image out of one vehicle's shard: what its condition looks like."""
        rel = self.path[len("/api/shard-image/"):].split("?")[0]
        vid, _, name = rel.partition("/")
        if not vid.isdigit():
            return self._json({"error": "vehicle id must be an integer"}, 400)
        root = paths.VEHICLE_BATCHES / f"batch_{int(vid)}" / "images" / "train"
        target = safe_child(root, name)
        if target is None:
            return self._json({"error": "not found"}, 404)
        ctype = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
        self._send(200, target.read_bytes(), ctype)

    def _shard_labels(self) -> None:
        """The label rows for one shard image, so the browser can draw them on it.

        Sent as normalised numbers rather than as a rendered image: the overlay is an
        SVG over the same `<img>`, so there is no second copy of the frame on the wire
        and no server-side image library in the dependency list.
        """
        rel = self.path[len("/api/shard-labels/"):].split("?")[0]
        vid, _, name = rel.partition("/")
        if not vid.isdigit():
            return self._json({"error": "vehicle id must be an integer"}, 400)
        self._json({"boxes": dataset_stats.boxes(int(vid), name),
                    "class_names": dataset_stats.class_names()})

    def _train_artifact(self) -> None:
        """One picture ultralytics drew during this vehicle's last round.

        The filename is checked against `train_artifacts.KINDS` rather than joined and
        guarded: an allowlist cannot be traversed out of, and it also means a future
        ultralytics release cannot quietly publish a new file over HTTP.
        """
        rel = self.path[len("/api/train-artifact/"):].split("?")[0]
        vid, _, name = rel.partition("/")
        if not vid.isdigit():
            return self._json({"error": "vehicle id must be an integer"}, 400)
        target = train_artifacts.artifact(int(vid), name)
        if target is None:
            return self._json({"error": "not found"}, 404)
        ctype = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
        self._send(200, target.read_bytes(), ctype)

    def _report_file(self) -> None:
        """Serve a generated report, refusing anything outside the reports dir."""
        rel = self.path[len("/reports/"):].split("?")[0]
        target = safe_child(paths.REPORTS, rel)
        if target is None:
            return self._json({"error": "not found"}, 404)
        ctype = "text/html; charset=utf-8" if target.suffix == ".html" else "text/plain; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = STATE.bus.subscribe()
        try:
            for ev in list(STATE.bus.history):     # replay so a late tab is not blank
                self._event(ev)
            while True:
                if q:
                    self._event(q.pop(0))
                else:
                    # Comment frame doubles as a keep-alive and as the signal that
                    # tells us the browser has gone away (raises on write).
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            STATE.bus.unsubscribe(q)

    def _event(self, ev: dict) -> None:
        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.flush()

    def _state_stream(self) -> None:
        """Push `/api/state` as a full snapshot then a numbered stream of diffs.

        Separate from `/api/events` rather than multiplexed onto it, because the two
        have opposite shapes: the event bus is an append-only narration that must be
        replayed to a late tab, and this is one value whose *history is worthless* --
        a tab that connects now wants the state now, not every intermediate.

        Latency comes from the bus: a log line, a stage transition or a checksum wakes
        this loop immediately. STREAM_TICK is the ceiling for the things that leave no
        event behind, and an unchanged tick sends a comment frame, not a payload.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # Nagle would hold a small frame waiting for the next one, which is precisely
        # the latency this route exists to remove.
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.end_headers()

        seq = statestream.Sequence()
        seen = STATE.bus.revision
        try:
            self.wfile.write(seq.open(STATE.current(CONFIG)))
            self.wfile.flush()
            while True:
                seen = STATE.bus.wait(seen, STREAM_TICK)
                payload = seq.update(STATE.current(CONFIG))
                # The comment frame is the keep-alive AND the liveness probe: writing
                # to a browser that has gone away raises, which is the only way this
                # thread learns to exit.
                self.wfile.write(payload if payload is not None else b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    #: Ceiling on a request body. Every other POST here is a small JSON config; the
    #: node heartbeat carries a base64 JPEG, which is the only reason this is not tiny.
    #: Read bounded rather than trusting Content-Length: this is the one route that
    #: accepts data from another machine.
    MAX_BODY = 1024 * 1024

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > self.MAX_BODY:
            return self._json({"error": f"body too large ({length} > {self.MAX_BODY})"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._json({"error": f"malformed JSON: {e}"}, 400)
        if not isinstance(body, dict):
            return self._json({"error": "body must be a JSON object"}, 400)

        if self.path.startswith("/api/node/"):
            return self._node_heartbeat(body)

        if self.path == "/api/run":
            global CONFIG
            # Built locally and only adopted once every check has passed. Assigning
            # first meant a rejected request still replaced the config the stage table
            # previews, so /api/plan then described a run the server had refused to
            # start -- and the next valid POST hid it by overwriting.
            # `or 1.0` would read gpu_fraction=0 as "unset" and silently serialise the
            # run the caller asked to pack -- the guard below would never see the zero.
            raw_fraction = body.get("gpu_fraction")
            cfg = Config(
                profile=body.get("profile", "demo"),
                n_vehicles=int(body.get("vehicles", 6)),
                rounds=int(body.get("rounds", 2)),
                local_epochs=int(body.get("epochs", 1)),
                seed=int(body.get("seed", 0)),
                partition=body.get("partition", Config.partition),
                local_bn=bool(body.get("local_bn", False)),
                alpha=float(body.get("alpha", 0.5) or 0.5),
                size_skew=float(body.get("size_skew", 0.0) or 0.0),
                gpu_fraction=float(raw_fraction) if raw_fraction not in (None, "") else 1.0,
                cache=body.get("cache", "") or "",
                strategy=body.get("strategy", "fedavg"),
                proximal_mu=float(body.get("proximal_mu", 0.0) or 0.0),
                ray_address=body.get("ray_address") or None,
            )
            if cfg.strategy not in stages.STRATEGIES:
                return self._json({"error": f"unknown strategy {cfg.strategy!r}; "
                                            f"known: {', '.join(stages.STRATEGIES)}"}, 400)
            if cfg.partition not in vehicles.PARTITIONS:
                # Rejected here rather than three subprocesses later, where it would
                # surface as a stage failure with the real cause buried in a log.
                return self._json({"error": f"unknown partition {cfg.partition!r}; "
                                            f"known: {', '.join(vehicles.PARTITIONS)}"}, 400)
            if cfg.size_skew < 0:
                # Same reason: caught here, not as a ValueError inside build_fleet's
                # subprocess three stages later.
                return self._json({"error": "size_skew must be >= 0, "
                                            f"got {cfg.size_skew}"}, 400)
            if cfg.cache not in ("", "ram", "disk"):
                return self._json({"error": f"unknown cache {cfg.cache!r}; "
                                            "known: '', 'ram', 'disk'"}, 400)
            if not 0 < cfg.gpu_fraction <= 1:
                # Ray takes a fraction above 1 and then places no client at all, so the
                # federation hangs waiting for clients that cannot be scheduled.
                return self._json({"error": "gpu_fraction must be in (0, 1], "
                                            f"got {cfg.gpu_fraction}"}, 400)
            try:
                chain = stages.resolve(body.get("stages"), body.get("skip"))
            except SystemExit as e:
                return self._json({"error": str(e)}, 400)
            CONFIG = cfg
            started = STATE.start(CONFIG, chain, bool(body.get("confirm")),
                                  body.get("ray_address") or None)
            return self._json({"started": started, "busy": STATE.busy},
                              200 if started else 409)

        if self.path == "/api/stop":
            if STATE.run:
                STATE.run.stop()
            return self._json({"stopping": True})

        self._json({"error": "not found"}, 404)


def serve(port: int = 8800, host: str = "127.0.0.1") -> None:
    # Loopback by default, and that default is load-bearing: POST /api/run starts a
    # subprocess chain from a request body, so anything that can reach this port can
    # make this machine train. Binding wider is an explicit choice, made per launch,
    # on a trusted network -- there is no authentication here.
    srv = ThreadingHTTPServer((host, port), Handler)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  ! bound to {host}:{port}, not loopback.\n"
              f"  ! /api/run starts training subprocesses and NOTHING here authenticates.\n"
              f"  ! Use this only on a network you control.\n")
    print(f"control + live dashboards : http://{host}:{port}")
    print(f"MLflow (metrics, history) : http://127.0.0.1:5000   [mlflow ui --port 5000]")
    print(f"Ray (actors, GPU internals): http://127.0.0.1:8265  [ray start --head]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to let edge nodes on other machines report in. "
                         "Read the warning it prints before you do")
    args = ap.parse_args()
    serve(args.port, args.host)
