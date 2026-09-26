// Live view: the heartbeat, GPU readouts, criteria, reports, log stream.
//
// State arrives from stream.js as whole snapshots; this file renders them. The panels
// whose job is to make a number harder to believe live in insight.js.
import { $, esc, unknown } from "./util.js";
import { lineChart } from "./chart.js";
import { state } from "./state.js";
import { renderStages, renderOptions } from "./control.js";
import { renderFleet } from "./fleet.js";
import { renderNowTraining } from "./consumed.js";
import { renderPerClass, renderChecksumLedger, measured, record,
         observedSpread } from "./insight.js";

//: The trainer's own pictures cost an /api/train-artifacts round trip and a directory
//: listing, so they are not re-read at the rate state now arrives. A vehicle change
//: refreshes them immediately; otherwise this is the ceiling.
const ARTIFACT_MS = 2000;
let artifactsAt = 0;
let artifactsKey = "";

/** Render one whole state snapshot. Called by the stream on every push. */
export function applyState(s) {
  if (!s) return;
  $("skew").hidden = !!s.live;
  state.cfg = s.config || {};
  state.fleet = s.fleet || [];
  state.learning = s.live && s.live.learning;
  renderStages(s.stages);
  renderOptions(s.options);
  renderGpu(s.gpu);
  renderLive(s.live, s.config);

  const key = `${s.live && s.live.training_now}|${s.busy}`;
  if (key !== artifactsKey || Date.now() - artifactsAt > ARTIFACT_MS) {
    artifactsKey = key;
    artifactsAt = Date.now();
    renderNowTraining($("nowTraining"), s.live && s.live.training_now, nowLabel(s), s.busy);
  } else {
    nowLabel(s);
  }

  renderReports(s.reports);
  renderFleet();
  document.body.dataset.loaded = "1";
  $("runState").textContent = s.busy ? (s.current || "running") : "idle";
  $("runState").className = "lamp " + (s.busy ? "l-run" : "l-idle");
  $("launch").disabled = s.busy;
  $("stop").disabled = !s.busy;
  $("mlflowLink").href = s.links.mlflow;
  $("rayLink").href = s.links.ray;
}

/** "vehicle 3 · rain / fog", and the same string into the panel's own subtitle. */
function nowLabel(s) {
  const who = s.live && s.live.training_now;
  const v = who == null ? null : (s.fleet || []).find(x => String(x.vid) === String(who));
  const label = who == null ? (s.busy ? "between vehicles" : "idle")
    : `vehicle ${who}${v && v.condition ? " · " + v.condition : ""}`;
  $("nowTrainingWho").textContent = label;
  return label;
}

function renderLive(L, cfg) {
  if (!L) return;
  const evald = (L.map50 || []).filter(v => v != null);
  const losses = (L.loss || []).filter(v => v != null);
  // The server's own config only describes runs launched from the form. A CLI run
  // has its own round count, and printing "4/2" against it is a lie, so the total
  // is shown only while it is still credible.
  const planned = cfg && cfg.rounds != null && cfg.rounds >= L.rounds_done ? `/${cfg.rounds}` : "";
  $("rRound").textContent    = `${L.rounds_done}${planned}`;
  $("rVehicles").textContent = Object.keys(L.per_vehicle || {}).length || "0";
  // A dash here reads as "zero" at a glance, and a self-evaluated mAP is not the
  // honest number anyway -- the holdout panel below is. So the absence says so.
  $("rMap").innerHTML  = evald.length ? evald[evald.length - 1].toFixed(4)
    : unknown("no client has finished an evaluation round yet");
  $("rLoss").innerHTML = losses.length ? losses[losses.length - 1].toFixed(4)
    : unknown("no client has reported a loss yet");
  $("rCkpt").textContent     = (L.checkpoints || []).length;
  $("rBar").style.width = cfg && cfg.rounds
    ? Math.min(100, 100 * L.rounds_done / cfg.rounds) + "%" : "0";

  for (const [vid, v] of Object.entries(L.per_vehicle || {})) {
    state.vehicles[vid] = Object.assign({}, state.vehicles[vid] || {}, v,
      { training: String(L.training_now) === String(vid) });
  }

  $("criteria").innerHTML = (L.criteria || []).map(c => {
    const cls = c.startsWith("[PASS]") ? "s-ok" : c.startsWith("[WARN]") ? "s-needs_confirm" : "s-failed";
    const word = c.slice(1, c.indexOf("]"));
    return `<li><span class="lamp ${cls}">${word}</span>` +
           `<span style="color:var(--ink-2)">${esc(c.slice(c.indexOf("]") + 2))}</span></li>`;
  }).join("") || '<li class="hint">No run has produced criteria yet.</li>';

  lineChart("mapChart", {
    series: [{ label: "mAP50", color: "var(--accent)", values: evald, area: true }],
    aria: "Evaluated mAP50 by round", yFmt: v => v.toFixed(3),
  });
  $("mapNote").innerHTML = evald.length
    ? "By round: " + evald.map(v => v.toFixed(4)).join(" → ") +
      (evald.length > 1 ? ` <b>${evald[evald.length - 1] >= evald[0] ? "+" : ""}` +
        `${(evald[evald.length - 1] - evald[0]).toFixed(4)} overall</b>` : "")
    : "Waiting for the first evaluation. The federate stage produces this.";
  if (L.no_optimizer_steps) {
    $("mapNote").innerHTML += ` <span class="warn">${L.no_optimizer_steps} round(s) took no ` +
      `optimizer step — the weights could not change.</span>`;
  }

  if ((L.checksums || []).length !== state.checksums.length) {
    state.checksums = (L.checksums || []).slice();
  }
  drawHeartbeat();
  renderHoldout(L.holdout, L.baseline);
}

/** The one metric no client could have flattered: the shared holdout.
 *
 * The band is the measured run-to-run spread, centred on the FIRST scored round: a
 * later round inside it is indistinguishable from where the run started, whatever the
 * curve looks like. Drawing it is the difference between "round 6 is better" and
 * "round 6 is better by more than this measurement can resolve".
 */
function renderHoldout(holdout, baseline) {
  const rows = (holdout && holdout.rounds) || [];
  // An observed spread from this machine's own seed repeats beats a recorded one: the
  // recorded figure was measured at 2x1 on an IID fleet under the old weight transport,
  // and both of those have since changed.
  const obs = observedSpread();
  const floor = obs ? obs.half_spread : measured("noise_floor_map50");
  const anchor = rows.length ? rows[0].mAP50 : null;
  lineChart("holdoutChart", {
    series: [{ label: "holdout mAP50", color: "var(--ok)", values: rows.map(r => r.mAP50), area: true },
             { label: "mAP50-95", color: "var(--dim)", values: rows.map(r => r["mAP50-95"]), dashed: true }],
    aria: "Global model mAP50 on the shared holdout by round", yFmt: v => v.toFixed(3),
    band: (floor != null && anchor != null && rows.length > 1)
      ? { lo: anchor - floor, hi: anchor + floor,
          label: `±${floor} ${obs ? "observed" : "recorded"} spread, around round ` +
                 `${rows[0].round}` }
      : null,
  });
  const best = rows.length ? Math.max(...rows.map(r => r.mAP50)) : null;
  $("hMap").innerHTML = best == null
    ? unknown("no global checkpoint has been scored on the holdout yet")
    : best.toFixed(4);
  $("hMapSpread").textContent = best == null || floor == null ? "" : `±${floor}`;
  $("hCeiling").innerHTML = baseline && baseline.retained
    ? (100 * baseline.retained).toFixed(0) + "%"
    : unknown("no centralised baseline has been trained, so this number has no scale");
  renderHoldoutProvenance(holdout, baseline, rows, floor);

  if (!rows.length) {
    $("holdoutNote").innerHTML = "No holdout evaluation yet. Run the " +
      "<code>holdout</code> stage to carve the set, then <code>evaluate</code> to score " +
      "every global checkpoint on it. Until then, every number on this page was measured " +
      "by a client on its own split.";
    return;
  }
  const size = (holdout.holdout && holdout.holdout.size) || "?";
  $("holdoutNote").innerHTML =
    `${rows.length} checkpoint(s) scored on ${size} images no vehicle trained on: ` +
    rows.map(r => r.mAP50.toFixed(4)).join(" → ") +
    (baseline && baseline.centralised_mAP50
      ? `. Centralised ceiling on the same images: <b>${baseline.centralised_mAP50.toFixed(4)}</b>, ` +
        `so the federation retains <b>${(100 * baseline.retained).toFixed(1)}%</b> of it ` +
        `(gap ${baseline.gap.toFixed(4)}).` +
        (baseline.matched === false
          ? ` <span class="warn">That ceiling had ${baseline.budget_ratio}x the ` +
            `image-visits, so the retention is a lower bound.</span>`
          : "")
      : ". No centralised baseline yet, so this number still has no scale — run the " +
        "<code>baseline</code> stage.");
  renderPerClass(rows);
}

/** Which run, which holdout, which fingerprint — and where the ± comes from. */
function renderHoldoutProvenance(holdout, baseline, rows, floor) {
  const meta = (holdout && holdout.holdout) || {};
  const fr = record("noise_floor_map50");
  const bits = [
    `holdout: ${meta.size != null ? meta.size + " images" : "size not recorded"}` +
      `, seed ${meta.seed != null ? meta.seed : "not recorded"}`,
    `fleet fingerprint: ${meta.fleet_fingerprint || meta.fingerprint || "not recorded"}`,
    `rounds scored: ${rows.length}`,
    baseline && baseline.centralised_mAP50 != null
      ? `ceiling budget ratio: ${baseline.budget_ratio != null ? baseline.budget_ratio + "x" : "not recorded"}`
      : "ceiling: none trained",
  ];
  $("holdoutProvenance").innerHTML =
    `<span class="prov">${bits.map(esc).join(" · ")}</span>` +
    (floor == null ? ""
      : obs
        ? `<br><span class="prov">± is <b>observed on this machine</b>: ${obs.n} runs ` +
          `identical except for the seed, spread ${obs.spread} mAP50 (` +
          obs.runs.map(r => `seed ${esc(String(r.seed))} → ${r.mAP50.toFixed(4)}`).join(", ") +
          `). That beats the recorded figure, which was measured on a system this repo ` +
          `has since changed.</span>`
        : `<br><span class="prov">± is the recorded run-to-run spread of ${floor} mAP50 ` +
          `from <code>${esc(fr ? fr.source : "?")}</code> (${esc(fr ? fr.measured_on : "?")}), ` +
          `a <b>${esc(fr ? fr.bound || "point" : "?")}</b> bound, ` +
          `<b>${esc(fr ? fr.confidence || "measured" : "?")}</b> at ` +
          `${esc(fr && fr.conditions ? fr.conditions : "conditions not recorded")}. ` +
          `It is stale: ${esc(fr && fr.reopened_by ? fr.reopened_by : "")} ` +
          `No seed repeats are on this machine yet, so nothing better is available.</span>`);
}

/** The signature panel: the one number whose stillness invalidates every other one. */
export function drawHeartbeat() {
  const v = state.checksums;
  renderChecksumLedger(v);
  lineChart("heartChart", {
    series: [{ label: "checksum", color: "var(--accent)", values: v, area: true }],
    aria: "Aggregate weight checksum by round",
    yFmt: n => Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(2),
  });

  const lamp = $("heartLamp");
  if (!v.length) {
    lamp.className = "lamp l-idle";
    lamp.textContent = "waiting";
    $("heartValue").innerHTML = unknown("no round has been aggregated yet");
    $("heartSub").textContent = "No aggregate yet. The federate stage produces one per round.";
    return;
  }
  const stuck = new Set(v.map(n => n.toFixed(6))).size !== v.length;
  const moving = v.length > 1 && !stuck;
  lamp.className = "lamp " + (moving ? "l-ok" : stuck ? "l-bad" : "l-warn");
  lamp.textContent = moving ? "moving" : stuck ? "stuck" : "one round";
  $("heartValue").textContent = v[v.length - 1].toFixed(2);
  $("heartSub").innerHTML = stuck
    ? '<span class="warn">Two rounds aggregated to the same weights. Nothing is being learned.</span>'
    : v.length > 1
      ? `Δ ${(v[v.length - 1] - v[v.length - 2]).toFixed(2)} since the previous round, ` +
        `over ${v.length} rounds.`
      : "One round so far — two are needed before movement can be claimed.";
}

function renderGpu(g) {
  if (!g) return;
  // nvidia-smi absent is a different fact from a GPU reading zero, and this project
  // has a machine where the difference decides whether a run is on CPU at 5.5x.
  const nosmi = unknown("nvidia-smi returned nothing -- no GPU, or no driver on PATH");
  const num = (x, dp) => x == null ? nosmi : (dp == null ? String(x) : x.toFixed(dp));
  $("gUtil").innerHTML   = num(g.util_pct);
  $("gMem").innerHTML    = g.mem_used_mib == null ? nosmi : String(Math.round(g.mem_used_mib));
  $("gPower").innerHTML  = num(g.power_w);
  $("gEnergy").innerHTML = g.energy_wh == null ? nosmi : g.energy_wh.toFixed(2);
  $("gTemp").innerHTML   = num(g.temp_c);
  $("gUtilBar").style.width = (g.util_pct || 0) + "%";
  const pct = g.mem_used_mib ? 100 * g.mem_used_mib / (g.mem_ceiling_mib || 16303) : 0;
  $("gMemBar").style.width = Math.min(100, pct) + "%";
  $("gMemBar").style.background = pct > 92 ? "var(--bad)" : pct > 75 ? "var(--warn)" : "var(--accent)";

  const hist = g.history || [];
  const maxP = Math.max.apply(null, [1].concat(hist.map(h => h.power || 0)));
  lineChart("gpuChart", {
    series: [{ label: "util %", color: "var(--accent)", values: hist.map(h => h.util || 0), area: true },
             { label: "power W", color: "var(--warn)", values: hist.map(h => h.power || 0), dashed: true }],
    xLabel: i => "t-" + (hist.length - i), zero: true, yFmt: v => v.toFixed(0),
    aria: `GPU utilisation and power, last ${hist.length} samples, peak power ${maxP.toFixed(0)} watts`,
  });
}

function renderReports(list) {
  $("reports").innerHTML = (list && list.length)
    ? list.map(r => `<li><a class="ext" href="${r.url}" target="_blank" rel="noopener">` +
        `${esc(r.name)} ↗</a></li>`).join("")
    : '<li class="hint">No reports yet. One is written at the end of every run.</li>';
}

// ---- event stream: the low-latency half, for the log and stage transitions ---
const SIGNAL_HINTS = ["checksum", "batch_id", "Run finished", "optimizer step", "cuda", "mAP"];
const looksStructured = (line) => SIGNAL_HINTS.some(h => line.includes(h));

export function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.kind === "log") {
      if ($("onlySignals").checked && !looksStructured(ev.line)) return;
      const pre = $("log");
      pre.insertAdjacentHTML("beforeend", `<b>${esc(ev.stage)}</b> ${esc(ev.line)}\n`);
      while (pre.childNodes.length > 600) pre.removeChild(pre.firstChild);
      pre.scrollTop = pre.scrollHeight;
    } else if (ev.kind === "signal") {
      onSignal(ev);
    } else if (ev.kind === "stage") {
      state.current = ev.status === "running" ? ev.stage : null;
      addTimeline(ev);
    } else if (ev.kind === "run_end") {
      addTimeline({ stage: "run", status: ev.ok ? "ok" : "failed", seconds: ev.seconds,
                    detail: `energy ${(ev.gpu && ev.gpu.energy_wh) ?? 0} Wh` });
    }
  };
}

function onSignal(ev) {
  if (ev.signal === "aggregate_checksum") {
    state.checksums.push(ev.value);
    drawHeartbeat();
  }
  if (ev.signal === "training_start") {
    Object.values(state.vehicles).forEach(v => { v.training = false; });
    state.vehicles[ev.value] = Object.assign({}, state.vehicles[ev.value] || {}, { training: true });
    renderFleet();
  }
  if (ev.signal === "client_received_checksum" || ev.signal === "client_sent_checksum") {
    const vid = Object.keys(state.vehicles).find(k => state.vehicles[k].training) || "?";
    const key = ev.signal === "client_received_checksum" ? "received" : "sent";
    state.vehicles[vid] = Object.assign({}, state.vehicles[vid] || {}, { [key]: ev.value });
  }
  if (ev.signal === "no_optimizer_step") {
    $("heartSub").innerHTML = '<span class="warn">No optimizer step was possible this round — ' +
      'the shard is too small for the batch size, so the weights cannot change.</span>';
  }
}

function addTimeline(ev) {
  $("timelineEmpty").hidden = true;
  $("timeline").insertAdjacentHTML("beforeend",
    `<tr><td>${esc(ev.stage)}</td><td><span class="lamp s-${esc(ev.status)}">${esc(ev.status)}</span></td>` +
    `<td class="num">${ev.seconds ?? ""}</td><td class="num">${(ev.gpu && ev.gpu.energy_wh) ?? ""}</td>` +
    `<td style="color:var(--dim);font-size:var(--t-sm)">${esc(ev.detail || "")}</td></tr>`);
}
