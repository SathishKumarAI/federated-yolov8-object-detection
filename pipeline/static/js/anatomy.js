// The Weights tab: what actually travels between the server and the vehicles.
//
// Two halves, two sources, on purpose:
//
//   * the static half is `/api/anatomy` -- a real checkpoint, read by pipeline/anatomy.py
//     at request time. Fetched on first sight of the tab and again whenever a round lands,
//     never on a state tick: a cache miss there is a 22 MB torch load.
//   * the live half is the ordinary `/api/state` snapshot every other panel renders, so a
//     run in progress updates through the existing event stream and the walkthrough's
//     recorded replay lands here exactly like real data.
//
// Nothing in this file invents a number. Where the server says it has no checkpoint, the
// panel says so; where a run's logs cannot support the round-trip identity, it says which
// half is missing. A zero here would read as a measurement.
import { $, esc, unknown, empty, PALETTE } from "./util.js";
import { barChart } from "./chart.js";
import { cite } from "./insight.js";

let an = null;              // the last /api/anatomy payload
let fetchedRounds = -1;     // rounds_done the static half was fetched at
let seq = 0;                // request token, so a slow answer cannot win

const MIB = 1024 * 1024;
const mib = (b) => (b / MIB).toFixed(1);
//: Bytes a person can read, and never "0.0 MiB" for 456 bytes: a rounded-away quantity
//: reads as nothing at all, which on this page is the one thing it must not do.
const size = (b) => b >= 1024 * MIB ? `${(b / (1024 * MIB)).toFixed(2)} GiB`
  : b >= MIB ? `${mib(b)} MiB`
  : b >= 1024 ? `${(b / 1024).toFixed(1)} KiB` : `${num(b)} B`;
const num = (n) => Number(n).toLocaleString();
const pct = (x, total) => (100 * x / total).toFixed(x / total < 0.01 ? 3 : 1) + "%";

/**
 * The checkpoint-derived half.
 *
 * Tokened rather than locked. An in-flight guard looked simpler and dropped the refetch
 * for a round that landed while the previous one was still loading -- leaving the tab a
 * round behind with nothing on screen saying so. A token keeps the newest answer and
 * discards a slow one instead.
 */
export async function loadAnatomy() {
  const mine = ++seq;
  let got;
  try {
    got = await (await fetch("/api/anatomy")).json();
  } catch {
    got = { available: false, reason: "the server did not answer /api/anatomy" };
  }
  if (mine !== seq) return;          // a newer request is out; its answer is the truth
  an = got;
  renderAnatomy();
}

/**
 * The live half, plus the trigger for the other one.
 *
 * Called with every snapshot. The refetch is keyed on the round count rather than on a
 * timer: a checkpoint is written once per round, so that is exactly when anything here
 * can have changed.
 */
export function renderExchange(s) {
  if (!s) return;
  const done = (s.live && s.live.rounds_done) || 0;
  if (!s.demo && done !== fetchedRounds) {
    fetchedRounds = done;
    loadAnatomy();
  }
  // A replay is a recorded /api/state, and state is all it is: the recording carries no
  // checkpoint, so the panels above keep showing this machine's own file. Two different
  // runs on one screen is exactly the kind of thing that has to be said out loud.
  const note = $("anReplayNote");
  if (note) {
    note.hidden = !s.demo;
    note.innerHTML = s.demo
      ? `<b>A recorded run is being replayed below.</b> The recording is a state ` +
        `snapshot and contains no model, so everything on this panel and the ones after ` +
        `it is still read from the checkpoint on <em>this</em> machine — the two are not ` +
        `from the same run.`
      : "";
  }
  renderLiveVehicles(s);
}

// --------------------------------------------------------------- the static half
function renderAnatomy() {
  if (!an) return;
  if (!an.available) {
    $("anWhen").textContent = "no checkpoint";
    $("anTotals").innerHTML = "";
    $("anProvenance").innerHTML = "";
    $("anKinds").innerHTML = empty("No global checkpoint on disk.", esc(an.reason));
    $("anShares").innerHTML = "";
    $("anSharesLegend").innerHTML = "";
    $("anKindsNote").innerHTML = "";
    $("anKernels").innerHTML = "";
    $("anConvWhich").textContent = "";
    $("anConvNote").innerHTML = esc(an.reason);
    barChart("anRunningMean", { items: [] });
    barChart("anRunningVar", { items: [] });
    $("anRunningMeanCap").textContent = "";
    $("anRunningVarCap").textContent = "";
    $("anBnNote").innerHTML = "";
    $("anHead").innerHTML = "";
    $("anHeadNote").innerHTML = "";
    $("anWire").innerHTML = "";
    $("anWireNote").innerHTML = "";
    renderRoundtrip();
    return;
  }
  const m = an.model, c = an.checkpoint, x = an.exchange;

  $("anWhen").textContent = `${c.dir}/${c.name}`;
  $("anTotals").innerHTML =
    readout(num(m.tensors), "arrays in one exchange") +
    readout(num(m.values), "numbers in the model") +
    readout(mib(x.down_per_vehicle), "one direction", "MiB") +
    readout("0", "images that move");

  const dtypes = Object.entries(m.dtypes)
    .map(([d, n]) => `${n} × <code>${esc(d.replace("torch.", ""))}</code>`).join(", ");
  $("anProvenance").innerHTML =
    `Read from <code>${esc(c.dir)}/${esc(c.name)}</code> — ${mib(c.bytes)} MiB on disk, ` +
    `written ${esc(new Date(c.mtime * 1000).toLocaleString())} by ultralytics ` +
    `${esc(m.ultralytics_version)}, from its <code>${esc(m.slot)}</code> entry` +
    (m.ema_slot_empty ? " (its <code>ema</code> entry is empty)" : "") + `. ` +
    `Tensors in the file: ${dtypes} — <b>the file is half precision</b>, because ` +
    `ultralytics halves every checkpoint it writes. What the transport carries is the ` +
    `live model, whose floats are 32-bit, so the file is a lossy record of the exchange ` +
    `and the byte counts here say which of the two they mean. ` +
    `${cite("fp16_transport_tensors")} · ${cite("state_tensors")}. ` +
    `<span class="prov">Every other figure on this tab is computed from that file when ` +
    `you ask for it, not quoted from a table.</span>`;

  renderKinds(m);
  renderKernels(m.first_conv);
  renderBatchNorm(m.batchnorm);
  renderHead(m.head);
  renderWire(x, m);
  renderRoundtrip();
}

const readout = (v, k, unit) =>
  `<div class="readout"><span class="v">${esc(v)}</span>` +
  (unit ? `<span class="u">${esc(unit)}</span>` : "") +
  `<div class="k">${esc(k)}</div></div>`;

function renderKinds(m) {
  const rows = m.kinds;
  $("anKinds").innerHTML =
    '<table><caption class="sr-only">Model tensors by kind</caption><thead><tr>' +
    '<th scope="col">Kind</th><th scope="col">Arrays</th><th scope="col">Numbers</th>' +
    '<th scope="col">Bytes on the wire</th><th scope="col">What it is</th></tr></thead><tbody>' +
    rows.map(r =>
      `<tr><td><b>${esc(r.label)}</b><br><code>${esc(r.example)}</code></td>` +
      `<td class="num">${num(r.tensors)}<br><span class="dim">${pct(r.tensors, m.tensors)}</span></td>` +
      `<td class="num">${num(r.values)}<br><span class="dim">${pct(r.values, m.values)}</span></td>` +
      `<td class="num">${size(r.wire_bytes)}</td>` +
      `<td style="color:var(--ink-2)">${esc(r.what)}</td></tr>`).join("") +
    '</tbody></table>';

  const colour = (i) => PALETTE[i % PALETTE.length];
  const bar = (label, key, total) =>
    `<div class="stackrow"><span class="lbl">${esc(label)}</span><span class="stack">` +
    rows.map((r, i) => `<i style="width:${(100 * r[key] / total).toFixed(3)}%;` +
      `background:${colour(i)}"><title>${esc(r.label)}: ${pct(r[key], total)}</title></i>`).join("") +
    `</span></div>`;
  // Short labels on purpose: the row's label column is one line, and the counts are in
  // the readouts above and the table beside it. A truncated "11,160,720 numb…" is worse
  // than no number at all.
  $("anShares").innerHTML =
    bar("share of the arrays", "tensors", m.tensors) +
    bar("share of the numbers", "values", m.values);
  $("anSharesLegend").innerHTML = rows.map((r, i) =>
    `<span><i style="background:${colour(i)}"></i>${esc(r.label)}</span>`).join("");

  const bn = rows.filter(r => r.key.startsWith("bn_"));
  const bnT = bn.reduce((a, r) => a + r.tensors, 0);
  const bnV = bn.reduce((a, r) => a + r.values, 0);
  const conv = rows.find(r => r.key === "conv");
  $("anKindsNote").innerHTML =
    `The two bars are the same model measured two ways, and they disagree completely. ` +
    `<b>Normalisation is ${num(bnT)} of the ${num(m.tensors)} arrays ` +
    `(${pct(bnT, m.tensors)}) and ${pct(bnV, m.values)} of the numbers.</b> ` +
    (conv ? `The ${num(conv.tensors)} convolution arrays are ${pct(conv.values, m.values)} ` +
            `of them. ` : "") +
    `So "most of what travels" depends entirely on whether you are counting parcels or ` +
    `contents — and the averaging happens per number, which is why the second bar is the ` +
    `one that decides what the aggregate looks like. The first is why a bug in the ` +
    `BatchNorm half can hide: it touches four fifths of the arrays and almost none of ` +
    `the model.`;
}

/**
 * The first convolution as 32 little RGB images.
 *
 * Each kernel is scaled to its own range, which is what makes a weak filter visible next
 * to a strong one — and also what makes their brightness incomparable, so the range each
 * one actually spans is in its tooltip and in the text alternative below. Colour is not
 * carrying a claim here: the colour IS the weight, three input channels at a time.
 */
function renderKernels(fc) {
  if (!fc) { $("anKernels").innerHTML = ""; return; }
  const convs = an.model.kinds.find(r => r.key === "conv");
  const deeper = convs ? convs.tensors - 1 : 0;
  const [out, cin, kh, kw] = fc.shape;
  $("anConvWhich").textContent = `${out} kernels of ${cin}×${kh}×${kw}`;
  $("anKernels").innerHTML = fc.kernels.map((k, i) => {
    const cells = k.rgb.map(([r, g, b]) =>
      `<i style="background:rgb(${Math.round(r * 255)},${Math.round(g * 255)},` +
      `${Math.round(b * 255)})"></i>`).join("");
    return `<figure class="kern" title="kernel ${i}: ${k.min} to ${k.max}">` +
      `<span class="grid3" style="grid-template-columns:repeat(${kw},1fr)">${cells}</span>` +
      `<figcaption>${i}</figcaption></figure>`;
  }).join("");
  $("anKernels").setAttribute("aria-label",
    `${out} kernels of the first convolution, each a ${kh} by ${kw} colour image; ` +
    `the numeric ranges are listed below the picture.`);

  const ranges = fc.kernels
    .map((k, i) => `<code>${i}</code> ${k.min} … ${k.max}`).join(" · ");
  $("anConvNote").innerHTML =
    `<code>${esc(fc.key)}</code>, ${num(fc.values)} numbers — ${out} filters, each ` +
    `${cin}×${kh}×${kw}. Every one of them is applied to the raw frame; what comes out is ` +
    `what the rest of the model has to work with. <b>Each square is scaled to its own ` +
    `range</b>, so a pale kernel is not a weak one — the ranges are: ` +
    `<details><summary>all ${out} kernel ranges</summary><div class="prov">${ranges}</div></details>` +
    `This is the only layer rendered. The next convolution's inputs are not colours but ` +
    `32 abstract channels, and painting three of them as red, green and blue would be a ` +
    `decoration rather than a measurement — so the other ${num(deeper)} convolutions ` +
    `are summarised in the table above instead.`;
}

function renderBatchNorm(bn) {
  if (!bn) return;
  const hist = (id, capId, h, label, fmtEdge) => {
    if (!h) { barChart(id, { items: [] }); $(capId).textContent = ""; return; }
    const items = h.counts.map((n, i) => ({
      label: fmtEdge(h.edges[i]), value: n,
      color: "var(--accent)",
    }));
    barChart(id, { items, labelWidth: 78, rowHeight: 16, fmt: v => num(v),
                   aria: `${label}: ${h.counts.reduce((a, b) => a + b, 0)} values between ` +
                         `${fmtEdge(h.lo)} and ${fmtEdge(h.hi)}` });
    $(capId).innerHTML = `${esc(label)} — ${num(h.counts.reduce((a, b) => a + b, 0))} ` +
      `values, ${esc(fmtEdge(h.lo))} to ${esc(fmtEdge(h.hi))}, ${h.counts.length} bins. ` +
      `Row labels are the low edge of each bin.`;
  };
  hist("anRunningMean", "anRunningMeanCap", bn.running_mean, "running mean",
       v => v.toFixed(2));
  hist("anRunningVar", "anRunningVarCap", bn.running_var, "running variance",
       v => v.toFixed(3));

  $("anBnNote").innerHTML =
    `${num(bn.layers)} normalisation layers, each contributing five arrays: a scale, a ` +
    `shift, a running mean, a running variance and a step counter. The first two are ` +
    `learned; the middle two are <em>measured</em> from the images that went past, which ` +
    `is why averaging them across vehicles partitioned by weather is averaging six ` +
    `different descriptions of what a road looks like. ` +
    (bn.counters_all_zero
      ? `<span class="warn">Every step counter in this file is 0</span> — ` +
        `<code>ModelEMA.update</code> interpolates only floating-point tensors, so a ` +
        `checkpoint written through ultralytics' EMA path carries no batch counts at all. ` +
        `That is a property of the file, not proof about the exchange: the counters are ` +
        `what the aggregate checksum deliberately excludes, because they climb on their ` +
        `own whatever the weights do.`
      : `Step counters in this file: ${bn.counter_values.map(esc).join(", ")}. They climb ` +
        `on their own, which is why the checksum this project trusts sums floating-point ` +
        `tensors only.`);
}

function renderHead(h) {
  if (!h || !h.scales.length) { $("anHead").innerHTML = ""; return; }
  const cold = new Set(h.cold_start);
  $("anHead").innerHTML =
    '<table><caption class="sr-only">Detection head class bias per scale</caption><thead><tr>' +
    '<th scope="col">Class</th>' +
    h.scales.map((s, i) => `<th scope="col">scale ${i + 1}</th>`).join("") +
    '<th scope="col">warm start</th></tr></thead><tbody>' +
    h.classes.map((name, ci) =>
      `<tr><td>${esc(name)}</td>` +
      h.scales.map(s => `<td class="num">${s.values[ci].toFixed(4)}</td>`).join("") +
      `<td>${cold.has(name)
        ? '<span class="lamp s-needs_confirm">cold</span>'
        : '<span class="lamp s-ok">from COCO</span>'}</td></tr>`).join("") +
    '</tbody></table>';

  // Which of the cold-start classes still hold the identical stored number. They were
  // initialised to one shared value, so an exact match means this class has not moved off
  // its starting point at the precision the file keeps.
  const groups = h.scales.map(s => {
    const vals = Object.values(s.cold_values);
    const counts = {};
    vals.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return { key: s.key, shared: Number(top[1]), value: top[0],
             names: Object.keys(s.cold_values).filter(n => String(s.cold_values[n]) === top[0]) };
  });
  // The two ends of one scale, so the sentence compares something. Reading the first and
  // last CLASS instead compared "person" against "other vehicle", which is a cold-start
  // class still sitting on its initialisation -- a real number making a false point.
  const first = h.scales[0].values.map((v, i) => ({ name: h.classes[i], value: v }));
  const keenest = first.reduce((a, b) => (b.value > a.value ? b : a));
  const wariest = first.reduce((a, b) => (b.value < a.value ? b : a));
  $("anHeadNote").innerHTML =
    `Every number here is negative, and that is the design: before it sees anything, the ` +
    `model assumes nothing is there. The least reluctant class at scale 1 is ` +
    `<b>${esc(keenest.name)}</b> at ${keenest.value.toFixed(2)} and the most reluctant is ` +
    `<b>${esc(wariest.name)}</b> at ${wariest.value.toFixed(2)} — a model that needs more ` +
    `evidence for the second than for the first, before a single pixel. ` +
    `The ${h.cold_start.length} classes marked <b>cold</b> are the ones COCO had no row ` +
    `for, so <code>warm_start_head</code> gave them one shared starting value. ` +
    groups.map((g, i) => `At scale ${i + 1}, ${g.shared} of them still hold exactly ` +
      `<code>${esc(g.value)}</code> (${g.names.map(esc).join(", ")}).`).join(" ") +
    ` A class sitting on its initialisation is a class the fleet has not taught anything ` +
    `yet. ` +
    `<span class="prov">Class names come from <code>pipeline/measurements.py</code>, not ` +
    `from the checkpoint: this architecture declares no names, so the file's own list is ` +
    `“0”…“12” and a table named from it would be labelled by index.</span>`;
}

function renderWire(x, m) {
  $("anWireWhen").textContent =
    `${x.vehicles} vehicles × ${x.rounds} rounds, as configured`;
  const row = (dir, what, bytes) =>
    `<tr><td><b>${esc(dir)}</b></td><td style="color:var(--ink-2)">${what}</td>` +
    `<td class="num">${num(bytes)}</td><td class="num">${size(bytes)}</td></tr>`;
  $("anWire").innerHTML =
    '<table><caption class="sr-only">Bytes per round per vehicle</caption><thead><tr>' +
    '<th scope="col">Direction</th><th scope="col">Payload</th>' +
    '<th scope="col">Bytes</th><th scope="col"></th></tr></thead><tbody>' +
    row("server → vehicle", `the aggregate: ${num(x.arrays)} arrays, 32-bit floats and ` +
        `64-bit counters`, x.down_per_vehicle) +
    row("vehicle → server", `the same ${num(x.arrays)} arrays after local training, plus ` +
        `<code>num_examples</code> — one integer, and FedAvg's weight`, x.up_per_vehicle) +
    row("one round, whole fleet", `${x.vehicles} vehicles, both directions`, x.per_round) +
    row("the whole run", `${x.vehicles} × ${x.rounds} rounds`, x.total) +
    row("images", "never — not one frame leaves a vehicle", 0) +
    '</tbody></table>';
  $("anWireNote").innerHTML =
    `${esc(x.excluded)} So these are floors, not totals: a real gRPC stream adds framing ` +
    `around each of the ${num(x.arrays)} arrays. ` +
    `<b>No image ever moves</b>, and that is the entire reason to federate: ` +
    `${num(x.vehicles)} vehicles' worth of driving footage stays on the vehicles, and ` +
    `${mib(x.total)} MiB of numbers moves instead. ` +
    `<b>That is not the same as privacy.</b> Weights are not raw data, but they are not ` +
    `nothing either — reconstructing training inputs from gradients or from weight ` +
    `updates is an active research area, and this project implements no defence against ` +
    `it: no secure aggregation, no differential privacy, no clipping. The honest claim is ` +
    `“the images stayed put”, not “nothing about them left”.`;
}

/** The identity, from pipeline/roundtrip.py. Its arithmetic, not a second copy of it. */
function renderRoundtrip() {
  const rt = (an && an.roundtrip) || null;
  if (!rt) { $("anRoundtrip").innerHTML = ""; return; }
  if (!rt.available) {
    $("anRoundtrip").innerHTML = empty("The identity cannot be checked here.",
      esc(rt.reason));
    $("anRoundtripNote").innerHTML =
      `Run <code>python -m pipeline.roundtrip</code> for the same answer from a ` +
      `terminal. It is deliberately not computed on this page: one implementation of an ` +
      `identity is enough, and the second copy is always the one that is quietly wrong.`;
    return;
  }
  $("anRoundtrip").innerHTML =
    '<table><caption class="sr-only">FedAvg identity per round</caption><thead><tr>' +
    '<th scope="col">Round</th><th scope="col">Vehicles</th>' +
    '<th scope="col">Images each (FedAvg\'s weights)</th>' +
    '<th scope="col">Weighted mean of what they sent</th>' +
    '<th scope="col">What the server published</th>' +
    '<th scope="col">Relative difference</th><th scope="col">Verdict</th></tr></thead><tbody>' +
    rt.rounds.map(r => {
      const agrees = r.deviation != null && r.deviation < rt.tolerance;
      const total = (r.num_examples || []).reduce((a, b) => a + b, 0);
      return `<tr><td class="num">${r.round}</td><td class="num">${r.clients}</td>` +
        `<td class="num">${(r.num_examples || []).map(num).join(" + ")}` +
        (total ? ` = ${num(total)}` : "") + `</td>` +
        `<td class="num">${r.weighted_mean == null ? unknown("no num_examples in the log")
          : r.weighted_mean.toFixed(6)}</td>` +
        `<td class="num">${r.aggregate.toFixed(6)}</td>` +
        `<td class="num">${r.deviation == null ? "—" : r.deviation.toExponential(1)}</td>` +
        `<td><span class="lamp ${agrees ? "s-ok" : "s-failed"}">` +
        `${agrees ? "is the mean" : "mismatch"}</span></td></tr>`;
    }).join("") +
    '</tbody></table>';
  $("anRoundtripNote").innerHTML =
    `Tolerance ${rt.tolerance.toExponential(0)} relative: summing eleven million floats ` +
    `six times does not associate, so the identity holds to parts per million rather ` +
    `than exactly. A dropped or double-counted vehicle moves the mean by percent. ` +
    `<span class="prov">Computed by <code>pipeline/roundtrip.py</code> from the run's own ` +
    `logs — the same code as <code>python -m pipeline.roundtrip</code>, not a copy of its ` +
    `arithmetic.</span>`;
}

// ----------------------------------------------------------------- the live half
/** Who sent what this run, from the same snapshot every other panel gets. */
function renderLiveVehicles(s) {
  const L = s.live || {};
  const per = L.per_vehicle || {};
  const ids = Object.keys(per);
  const cond = (vid) => {
    const v = (s.fleet || []).find(f => String(f.vid) === String(vid));
    return v && v.condition;
  };
  $("anLiveWhen").textContent = s.demo ? "a recorded replay"
    : s.busy ? `round ${L.rounds_done + 1}, in progress` : `${L.rounds_done || 0} round(s) done`;

  if (!ids.length) {
    $("anLive").innerHTML = empty("No vehicle has reported weights yet.",
      "The federate stage logs one 'Received' and one 'Sending back' line per vehicle " +
      "per round; this table is those lines.");
  } else {
    $("anLive").innerHTML =
      '<table><caption class="sr-only">Per-vehicle weight checksums</caption><thead><tr>' +
      '<th scope="col">Vehicle</th><th scope="col">Condition</th>' +
      '<th scope="col">Rounds</th><th scope="col">Checksum received</th>' +
      '<th scope="col">Checksum sent back</th><th scope="col">Did it change?</th>' +
      '</tr></thead><tbody>' +
      ids.sort((a, b) => Number(a) - Number(b)).map(vid => {
        const v = per[vid];
        const both = v.received != null && v.sent != null;
        const same = both && v.received.toFixed(6) === v.sent.toFixed(6);
        return `<tr><td><b>${esc(vid)}</b>${v.training
            ? ' <span class="lamp l-run">training</span>' : ""}</td>` +
          `<td>${esc(cond(vid) || "—")}</td><td class="num">${esc(String(v.rounds))}</td>` +
          `<td class="num">${v.received == null
            ? unknown("no 'Received weights' line for this vehicle in this run's logs")
            : v.received.toFixed(6)}</td>` +
          `<td class="num">${v.sent == null
            ? unknown("no 'Sending back weights' line yet — it appears when the round ends")
            : v.sent.toFixed(6)}</td>` +
          `<td>${!both ? '<span class="dim">—</span>'
            : `<span class="lamp ${same ? "s-failed" : "s-ok"}">` +
              `${same ? "identical" : "trained"}</span>`}</td></tr>`;
      }).join("") + '</tbody></table>';
  }

  $("anLiveNote").innerHTML =
    (s.demo
      ? `<b>This is a replay.</b> A recorded run is being fed through these panels, and ` +
        `every checksum it did not record says “not measured” rather than showing a ` +
        `number. `
      : "") +
    `A vehicle whose two checksums are <b>identical</b> returned the weights it was ` +
    `handed — the failure this project shipped twice, where the rounds completed and ` +
    `FedAvg averaged its own input. The numbers are read from the client log lines, not ` +
    `computed here. ` +
    `<span class="prov">One per vehicle per round; only the latest of each is kept, ` +
    `which is why the Rounds column can exceed one while a single pair of checksums ` +
    `is shown.</span>`;
}
