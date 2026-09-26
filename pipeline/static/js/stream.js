// The state transport: one EventSource, a full snapshot, then numbered diffs.
//
// This file owns *how* state arrives and nothing about what it means. It hands whole
// snapshots to one callback, so every view still reads the same shape it read when the
// page polled — see pipeline/statestream.py for the wire format, which is mirrored by
// applyPatch below and checked against the Python half by tests/js/state_patch.mjs.
//
// Three things it must get right, each of which was a real bug class in the poll:
//   * a gap in the sequence is a resync, never a silently stale panel;
//   * a server that cannot stream at all must still show data, so polling remains as
//     an explicitly-labelled fallback;
//   * the page has to SAY which of the two it is using, because "the number stopped
//     moving" and "the transport stopped" look identical on screen.
import { $ } from "./util.js";

const POLL_FALLBACK_MS = 2000;

/** Apply one `statestream` op. The twin of `statestream.apply`; keep them together. */
export function applyPatch(old, op) {
  if (!op) return old;
  if (op[0] === "=") return op[1];
  if (op[0] === "-") return null;
  const out = (old && typeof old === "object" && !Array.isArray(old)) ? Object.assign({}, old) : {};
  for (const [key, sub] of Object.entries(op[1])) {
    if (sub[0] === "-") delete out[key];
    else out[key] = applyPatch(out[key], sub);
  }
  return out;
}

export const transport = {
  mode: "connecting",   // connecting | push | polling
  seq: 0,
  patches: 0,
  resyncs: 0,
  at: null,             // ms timestamp of the last state actually applied
};

let snap = null;
let polling = false;
let held = false;
let onState = () => {};

/** Everything the views got last. Read-only as far as they are concerned. */
export const lastState = () => snap;

function deliver(next, label) {
  snap = next;
  transport.mode = label;
  transport.at = Date.now();
  // While held, the stream keeps its own copy current but the views are not touched:
  // the walkthrough is showing a recorded run through the same panels, and a live patch
  // arriving underneath it would replace the recording mid-sentence.
  if (!held) onState(snap);
  renderTransport();
}

/**
 * Stop delivering to the views without disconnecting.
 *
 * Releasing delivers the newest state immediately, so nothing is missed -- a diff
 * stream has no backlog to replay, which is exactly why holding it is safe.
 */
export function hold(on) {
  held = !!on;
  if (!held && snap) onState(snap);
  renderTransport();
}

/** Say out loud how this page is getting its numbers, and how fresh they are. */
export function renderTransport() {
  const el = $("transport");
  if (!el) return;
  const age = transport.at == null ? null : (Date.now() - transport.at) / 1000;
  const word = held ? "held"
    : { connecting: "connecting", push: "push", polling: "polling" }[transport.mode];
  el.className = "lamp " + (held ? "l-warn" : transport.mode === "push" ? "l-ok"
    : transport.mode === "polling" ? "l-warn" : "l-idle");
  el.textContent = word + (!held && transport.mode === "push" ? ` · seq ${transport.seq}` : "");
  const detail = held
    ? "The walkthrough is showing a recorded run through these panels. The live stream "
      + "is still connected and will take over the moment the replay stops."
    : transport.mode === "push"
    ? `Server-Sent Events on /api/stream. ${transport.patches} diff(s) applied, ` +
      `${transport.resyncs} resync(s). Last update ${age == null ? "never" : age.toFixed(1) + " s ago"}.`
    : transport.mode === "polling"
      ? `The event stream did not connect, so this page is falling back to GET /api/state ` +
        `every ${POLL_FALLBACK_MS / 1000} s. Numbers can be that stale.`
      : "Opening /api/stream.";
  // Both, so the explanation is not mouse-only: the lamp is focusable.
  el.title = detail;
  el.setAttribute("aria-label", "state transport: " + el.textContent + ". " + detail);
}

async function resync(why) {
  transport.resyncs++;
  try {
    deliver(await (await fetch("/api/state")).json(), transport.mode);
    console.warn("state resynced:", why);
  } catch {
    /* the server is restarting; the stream's own reconnect will pick it up */
  }
}

async function pollOnce() {
  try {
    deliver(await (await fetch("/api/state")).json(), "polling");
  } catch {
    /* next tick */
  }
  setTimeout(pollOnce, POLL_FALLBACK_MS);
}

/**
 * Start the transport. `handler` is called with a whole state snapshot every time one
 * arrives — on connect, on every diff, and on every poll if it comes to that.
 */
export function connectState(handler) {
  onState = handler;
  renderTransport();
  // The freshness in the lamp's explanation is only true while something recomputes
  // it. Started here rather than at module scope so importing this file has no side
  // effect — the node check in tests/js would otherwise never exit.
  setInterval(renderTransport, 1000);

  let es;
  try {
    es = new EventSource("/api/stream");
  } catch {
    polling = true;
    pollOnce();
    return;
  }

  es.addEventListener("snapshot", (m) => {
    const d = JSON.parse(m.data);
    transport.seq = d.seq;
    deliver(d.state, "push");
  });

  es.addEventListener("patch", (m) => {
    const d = JSON.parse(m.data);
    if (snap == null) return resync("a patch arrived before any snapshot");
    // The sequence is per connection and strictly consecutive, so a gap cannot happen
    // without a bug. Checked anyway: the failure it would otherwise cause is a panel
    // that keeps rendering, keeps looking live, and is wrong — which is the exact
    // shape of every entry in this project's silent-failures table.
    if (d.seq !== transport.seq + 1) return resync(`seq ${transport.seq} -> ${d.seq}`);
    transport.seq = d.seq;
    transport.patches++;
    deliver(applyPatch(snap, d.patch), "push");
  });

  es.onerror = () => {
    // EventSource reconnects on its own and the server replies with a fresh snapshot,
    // so a dropped connection needs no handling here. What needs handling is a server
    // that never streams at all: then nothing has ever arrived, and polling is the
    // only way this page shows a number.
    if (snap == null && !polling) {
      polling = true;
      pollOnce();
    }
    renderTransport();
  };
}
