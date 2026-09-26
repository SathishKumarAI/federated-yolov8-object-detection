// Simulate: what a configuration would cost, before it costs a GPU hour.
//
// The arithmetic is all server-side in `pipeline/plan.py::project`, because the measured
// constants live in `pipeline/measurements.py` and a second copy of them in JavaScript
// is a second copy that will disagree. This file is the form and the table.
//
// Two rules it exists to enforce, and both are visible on screen rather than only true
// in the code:
//
//   * every projected number names the measurement it rests on;
//   * a lever pushed outside the range that measurement covers is REFUSED, with the
//     measurement that would lift the refusal — never extrapolated into a plausible
//     number that carries a measured one's confidence.
//
// It launches nothing. The only thing it can do to the rest of the page is fill in the
// run form on the Control tab, which still needs a human to press Launch.
import { $, esc, empty } from "./util.js";

const FIELDS = ["simVehicles", "simPerVehicle", "simRounds", "simEpochs", "simPartition",
                "simStrategy", "simImgsz", "simGpuFraction"];

let timer = null;
let last = null;

const query = () => new URLSearchParams({
  vehicles: $("simVehicles").value,
  per_vehicle: $("simPerVehicle").value,
  rounds: $("simRounds").value,
  epochs: $("simEpochs").value,
  partition: $("simPartition").value,
  strategy: $("simStrategy").value,
  imgsz: $("simImgsz").value,
  gpu_fraction: $("simGpuFraction").value,
  profile: "full",
}).toString();

export async function loadSimulation() {
  let pr;
  try {
    const r = await fetch("/api/simulate?" + query());
    pr = await r.json();
    if (!r.ok) {
      $("simBody").innerHTML = empty("The server refused that configuration.",
        pr.error || "No reason given.");
      return;
    }
  } catch {
    $("simBody").innerHTML = empty("The server did not answer.", "Is it still running?");
    return;
  }
  last = pr;
  render(pr);
}

/** Debounced so dragging a number input does not fire a request per keystroke. */
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(loadSimulation, 180);
}

const num = (v) => typeof v === "number"
  ? (Number.isInteger(v) ? v.toLocaleString() : String(v)) : String(v);

function value(row) {
  if (row.status === "refused") return '<span class="refused">refused</span>';
  if (Array.isArray(row.value)) return `${num(row.value[0])} – ${num(row.value[1])}`;
  return num(row.value);
}

function render(pr) {
  const b = pr.budget;
  const hhmm = (s) => s < 90 ? `${s} s`
    : s < 5400 ? `${Math.round(s / 60)} min`
    : `${(s / 3600).toFixed(1)} h`;
  const clock = pr.projections.find(p => p.name === "wall clock");

  $("simBody").innerHTML =
    `<div class="panel sim-banner"><h2>This is a projection, not a prediction</h2>` +
      `<p class="hint" style="margin-top:0">${esc(pr.accuracy_note)}</p></div>` +

    `<div class="panel"><h2>Budget<span class="n">counted, not estimated</span></h2>` +
      `<div class="sum">${b.vehicles} vehicles <em>×</em> ` +
      `${b.images_per_vehicle.toLocaleString()} images <em>×</em> ${b.rounds} rounds ` +
      `<em>×</em> ${b.local_epochs} local epochs <em>=</em> ` +
      `<b>${b.image_visits.toLocaleString()}</b> image-visits</div>` +
      (clock && clock.status !== "refused"
        ? `<p class="hint">Roughly <b>${hhmm(clock.value)}</b> of GPU time. A matched ` +
          `centralised ceiling would be ${b.pooled_images.toLocaleString()} pooled ` +
          `images for ${b.centralised_epochs_to_match} epochs — the same ` +
          `${b.image_visits.toLocaleString()} visits, which is the only way the two ` +
          `numbers are comparable.</p>`
        : `<p class="hint warn">The wall clock cannot be projected for this ` +
          `configuration; see the refusals below.</p>`) + `</div>` +

    `<div class="panel"><h2>Projected<span class="n">each from a measurement, named</span></h2>` +
      `<table><thead><tr><th scope="col">Quantity</th><th scope="col">Projection</th>` +
      `<th scope="col">Rests on</th><th scope="col">How, and where it can be wrong</th>` +
      `</tr></thead><tbody>` +
      pr.projections.map(row =>
        `<tr class="${row.status === "refused" ? "refused-row" : ""}">` +
        `<td><b>${esc(row.name)}</b></td>` +
        `<td class="num">${value(row)} <span class="dim">${esc(row.unit)}</span></td>` +
        `<td>${row.rests_on.length
          ? row.rests_on.map(id => `<code>${esc(id)}</code>`).join("<br>")
          : '<span class="dim">nothing — this is arithmetic on the inputs</span>'}</td>` +
        `<td style="color:var(--ink-2)">${esc(row.how)}</td></tr>`).join("") +
      `</tbody></table></div>` +

    (pr.refusals.length
      ? `<div class="panel"><h2>Refused<span class="n">and what would lift each one</span></h2>` +
        `<ul class="plain">` + pr.refusals.map(r =>
          `<li><span class="lamp s-failed">refused</span><span style="color:var(--ink-2)">` +
          `<b>${esc(r.lever)} = ${esc(String(r.asked))}</b> — measured only at ` +
          `${esc(r.measured)}, in <code>${esc(r.source)}</code>.<br>` +
          `To project it: ${esc(r.needed)}.</span></li>`).join("") +
        `</ul><p class="hint">Extrapolating past a measurement is how a number with no ` +
        `evidence behind it ends up on a slide. The refusal is the answer.</p></div>`
      : "") +

    (pr.vram
      ? `<div class="panel"><h2>Will it fit on the card` +
          `<span class="n">16 303 MiB</span></h2>` +
        `<div class="readouts">` +
          `<div class="readout"><span class="v">${pr.vram.clients_on_the_card}</span>` +
            `<div class="k">client(s) on the card at once</div></div>` +
          `<div class="readout"><span class="v">${pr.vram.total_lo_mib.toLocaleString()}` +
            (pr.vram.total_hi_mib !== pr.vram.total_lo_mib
              ? `–${pr.vram.total_hi_mib.toLocaleString()}` : "") +
            `</span><span class="u">MiB</span><div class="k">projected peak</div></div>` +
          `<div class="readout"><span class="lamp ${pr.vram.fits ? "s-ok"
            : pr.vram.may_not_fit ? "s-needs_confirm" : "s-failed"}">` +
            `${pr.vram.fits ? "fits" : pr.vram.may_not_fit ? "may not fit" : "will not fit"}` +
            `</span><div class="k">against the ceiling</div></div>` +
        `</div><p class="hint">${esc(pr.vram.how)}. Packing clients is the fastest ` +
        `setting measured and also the one that killed a run: at 0.33 the card is ` +
        `94.9–96.6 % full and the failure was a <em>host</em> allocation, not a CUDA ` +
        `one.</p></div>`
      : "") +

    `<div class="panel"><h2>Against the one run whose result is recorded</h2>` +
      `<p class="sum">${pr.reference.holdout_mAP50} <em>mAP50 on the holdout,</em> ` +
      `<b>${(100 * pr.reference.retained).toFixed(1)}%</b> <em>of its matched ceiling</em></p>` +
      (pr.differs_from_reference.length
        ? `<p class="hint">This configuration differs from it in ` +
          `<b>${pr.differs_from_reference.length}</b> setting(s): ` +
          pr.differs_from_reference.map(d => `<code>${esc(d)}</code>`).join(", ") +
          `. Each difference is a reason the recorded number does not transfer, and a ` +
          `comparison that changes more than one of them cannot say which one did it.</p>`
        : `<p class="hint ok">Identical to the recorded run, so that result is what to ` +
          `expect — within the noise floor above.</p>`) +
      `<p class="prov">Recorded in <code>${esc(pr.reference.source)}</code>: ` +
      `${pr.reference.vehicles} vehicles × ${pr.reference.per_vehicle.toLocaleString()} ` +
      `images × ${pr.reference.rounds} rounds × ${pr.reference.local_epochs} epochs, ` +
      `${pr.reference.seconds.toLocaleString()} s, ${pr.reference.wh} Wh.</p></div>` +

    (pr.imgsz_reachable ? "" :
      `<div class="panel"><h2>One of these is not a setting</h2>` +
      `<p class="hint warn">${esc(pr.imgsz_note)}</p></div>`) +

    (pr.warnings.length
      ? `<div class="panel"><h2>Before you spend the hours</h2><ul class="plain">` +
        pr.warnings.map(w => `<li><span class="lamp s-needs_confirm">check</span>` +
          `<span style="color:var(--ink-2)">${esc(w)}</span></li>`).join("") + `</ul></div>`
      : "") +

    `<div class="panel"><h2>Take it to the run form</h2>` +
      `<button id="simAdopt">Copy into the Control tab</button>` +
      `<p class="hint">Fills the run form in. It does <b>not</b> launch anything — ` +
      `starting a run is a separate, deliberate press, and <code>imgsz</code> is not ` +
      `copied because the federation's resolution is not settable from here.</p></div>`;

  $("simAdopt").onclick = adopt;
}

/** Push the simulated configuration into the real run form. Launches nothing. */
function adopt() {
  if (!last) return;
  const c = last.config;
  const set = (id, v) => { const el = $(id); if (el) el.value = String(v); };
  set("vehicles", c.n_vehicles);
  set("rounds", c.rounds);
  set("epochs", c.local_epochs);
  set("seed", c.seed);
  set("partition", c.partition);
  set("strategy", c.strategy);
  set("alpha", c.alpha);
  set("sizeSkew", c.size_skew);
  $("simAdopt").textContent = "Copied — the Control tab is set, nothing has started";
}

export function wireSimulate() {
  FIELDS.forEach(id => {
    const el = $(id);
    if (!el) return;
    el.oninput = schedule;
    el.onchange = schedule;
  });
}
