// Panels whose job is to stop a number being believed too easily.
//
// Everything here exists because some reading of this dashboard was once wrong in a way
// the dashboard could have prevented:
//
//   * the checksum ledger, because two equal consecutive aggregates invalidate every
//     other number on the page and a chart makes "equal" look like "flat";
//   * the per-class table, because `car` is over half the objects and the four classes
//     COCO could not warm-start are where the residual loss is predicted to be;
//   * the round profile, because "27 % GPU utilisation" is compatible with two worlds
//     that have opposite fixes;
//   * the provenance table, because a number with no source on screen is
//     indistinguishable from a number somebody typed.
//
// Numbers here are read, never derived into something new. The one computation is a
// subtraction (Δ between rounds), and it is labelled as one.
import { $, esc, fmt, unknown, empty } from "./util.js";
import { lineChart, barChart, sparkline } from "./chart.js";

let facts = null;           // /api/measurements, fetched once

/** The measurement table. Fetched once — it cannot change while the server runs. */
export async function loadMeasurements() {
  if (facts) return facts;
  try {
    facts = await (await fetch("/api/measurements")).json();
  } catch {
    facts = { records: [], lever_notes: {},
              classes: { all: [], warm_started: [], unwarmed: [] } };
  }
  return facts;
}

/** The spread across this machine's own seed repeats, or null when there are none. */
export const observedSpread = () => (facts && facts.observed_spread) || null;

export const record = (id) => (facts && facts.records.find(r => r.id === id)) || null;
export const measured = (id) => { const r = record(id); return r ? r.value : null; };

/** What a run lever does and how it is easy to misread. From pipeline/measurements.py. */
export const leverNote = (name) =>
  (facts && facts.lever_notes && facts.lever_notes[name])
    ? esc(facts.lever_notes[name]).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    : null;

/** Cite a measurement inline: the value, the file, and the date. */
export function cite(id) {
  const r = record(id);
  if (!r) return unknown("no measurement with that id is recorded");
  return `<span class="cite" title="${esc(r.note)}">${esc(r.source)} · ${esc(r.measured_on)}</span>`;
}

// ------------------------------------------------------------- checksum ledger
/**
 * One row per round, with the Δ spelled out.
 *
 * The chart above it can show a visually flat line for a real change and a visually
 * moving one for rounding noise; the subtraction cannot. `stuck` is decided on the
 * printed precision, the same way `pipeline/verify.py` decides it, so the panel and
 * the pass criteria cannot disagree.
 */
export function renderChecksumLedger(checksums) {
  const v = checksums || [];
  const body = $("heartLedger");
  if (!body) return;
  if (!v.length) {
    body.innerHTML = '<tr><td colspan="4" class="hint">No aggregate yet. Each round of ' +
      'the <code>federate</code> stage writes one line to the server log.</td></tr>';
    return;
  }
  body.innerHTML = v.map((x, i) => {
    const prev = i ? v[i - 1] : null;
    const d = prev == null ? null : x - prev;
    const same = prev != null && x.toFixed(6) === prev.toFixed(6);
    const word = prev == null ? "first" : same ? "not moving" : "moved";
    const cls = prev == null ? "s-pending" : same ? "s-failed" : "s-ok";
    return `<tr><td class="num">${i + 1}</td><td class="num">${x.toFixed(6)}</td>` +
      `<td class="num">${d == null ? '<span class="dim">—</span>' :
        (d >= 0 ? "+" : "") + d.toFixed(6)}</td>` +
      `<td><span class="lamp ${cls}">${word}</span></td></tr>`;
  }).join("");
}

// ------------------------------------------------------------------- per class
/**
 * AP50 per class on the holdout, with the two things the averaged number hides:
 * how many instances the class actually has, and whether COCO could warm-start it.
 *
 * The sparkline is the small multiple — one class's AP across every scored round, on
 * its own scale, so a class that never moves is visible next to one that does. That is
 * the shape of `docs/findings/2026-09-26-accuracy-findings.md`'s prediction: if a
 * pretrained start is what closed the non-IID gap, the loss that is left lives in the
 * four classes the warm start could not reach.
 */
export function renderPerClass(rows) {
  const last = rows.length ? rows[rows.length - 1].per_class : null;
  $("perClassPanel").hidden = !(last && last.length);
  if (!last || !last.length) return;

  const unwarmed = new Set((facts && facts.classes.unwarmed) || []);
  const first = {};
  for (const c of (rows[0].per_class || [])) first[c.class_id] = c;
  const totalInst = last.reduce((a, c) => a + (c.instances || 0), 0) || 1;
  const sorted = last.slice().sort((a, b) => (b.instances || 0) - (a.instances || 0));

  // One class's AP across every round that was scored. A round that did not report
  // this class is a hole, not a zero, so it is dropped rather than plotted at 0.
  const history = (id) => rows.map(r => (r.per_class || []).find(c => c.class_id === id))
                              .filter(Boolean).map(c => c.AP50);

  $("perClassWhen").textContent = rows.length > 1
    ? `round ${rows[rows.length - 1].round}, change since round ${rows[0].round}`
    : `round ${rows[rows.length - 1].round}`;

  $("perClass").innerHTML = '<div class="classes">' + sorted.map(c => {
    const was = first[c.class_id];
    const d = (was && rows.length > 1) ? c.AP50 - was.AP50 : null;
    const share = 100 * (c.instances || 0) / totalInst;
    const curve = history(c.class_id);
    const lo = Math.min(...curve), hi = Math.max(...curve);
    const cold = unwarmed.has(c.name);
    return `<div class="cls${cold ? " cold" : ""}">` +
      `<div class="nm">${esc(c.name)}` +
      (cold ? '<abbr title="Not warm-started: COCO has no matching head row, so this ' +
              'class began from a distribution. docs/PHASED_PLAN.md predicts the ' +
              'residual loss concentrates here.">cold start</abbr>' : "") + `</div>` +
      `<div class="ap"><i style="width:${(100 * c.AP50).toFixed(1)}%"></i>` +
      `<span>${c.AP50.toFixed(3)}</span></div>` +
      (curve.length > 1 ? sparkline(curve, lo, hi, cold ? "var(--warn)" : "var(--accent)")
                        : '<span class="dim spark-none">one round</span>') +
      `<div class="meta">${c.instances == null ? unknown("the scorer did not report an " +
        "instance count for this class") : c.instances.toLocaleString() + " inst"}` +
      ` · ${share.toFixed(1)}%` +
      (d == null ? "" : ` · <span class="${d >= 0 ? "ok" : "warn"}">${d >= 0 ? "+" : ""}` +
        `${d.toFixed(3)}</span>`) + `</div></div>`;
  }).join("") + "</div>";

  const top = sorted[0];
  const thin = sorted.filter(c => (c.instances || 0) < 100);
  const cold = sorted.filter(c => unwarmed.has(c.name));
  const coldMean = cold.length ? cold.reduce((a, c) => a + c.AP50, 0) / cold.length : null;
  const warmMean = sorted.length - cold.length
    ? sorted.filter(c => !unwarmed.has(c.name)).reduce((a, c) => a + c.AP50, 0) /
      (sorted.length - cold.length) : null;

  $("perClassNote").innerHTML =
    `<b>${esc(top.name)}</b> alone is ${(100 * (top.instances || 0) / totalInst).toFixed(1)}% ` +
    `of the objects scored here, so the single mAP50 above is mostly its number.` +
    (coldMean != null && warmMean != null
      ? ` Mean AP50 is <b>${warmMean.toFixed(3)}</b> over the ${sorted.length - cold.length} ` +
        `warm-started classes present and <b>${coldMean.toFixed(3)}</b> over the ` +
        `${cold.length} that started cold — the prediction to check before buying a new ` +
        `aggregation algorithm.`
      : "") +
    (thin.length
      ? ` ${thin.length} class(es) have under 100 instances (${thin.map(c => esc(c.name)).join(", ")}); ` +
        `their AP moves a lot for reasons that are not the model.`
      : "") +
    ` Classes with no instances in the holdout are omitted rather than scored — ` +
    `Ultralytics' <code>maps</code> would have given them the fleet average.` +
    `<br><span class="prov">Instance counts and AP come from ` +
    `<code>pipeline/holdout.py</code>'s own scoring run; the cold-start list is ` +
    `<code>pipeline/measurements.py</code>, checked against the document it cites.</span>`;
}

// --------------------------------------------------------------- round profile
/** Seconds per phase for the last federated run, from the timestamps in its logs. */
export async function loadProfile() {
  const note = $("profileNote");
  note.textContent = "Reading the logs…";
  let p;
  try {
    p = await (await fetch("/api/profile")).json();
  } catch {
    note.innerHTML = '<span class="warn">The server did not answer.</span>';
    return;
  }
  if (p.error) {
    barChart("profileChart", { items: [] });
    $("profileVerdict").innerHTML = "";
    note.innerHTML = `Nothing to profile: ${esc(p.error)} Run the ` +
      `<code>federate</code> stage, then come back — this reads the logs it wrote ` +
      `rather than instrumenting anything.`;
    return;
  }

  const order = ["train", "evaluate", "weights_in", "weights_out", "construct",
                 "aggregate", "checkpoint"];
  const items = order.filter(k => p.phases[k]).map(k => ({
    label: k, value: p.phases[k],
    color: k === "train" ? "var(--accent)" : "var(--dim)",
  }));
  items.push({ label: "unaccounted", value: p.unaccounted_s, color: "var(--warn)" });
  barChart("profileChart", {
    items, labelWidth: 110, fmt: v => v.toFixed(1) + " s",
    aria: `Seconds per phase over a ${p.wall_s} second run`,
  });

  $("profileVerdict").innerHTML = (p.verdict || []).map(line => {
    const word = line.slice(1, line.indexOf("]"));
    const bad = word.startsWith("AROUND") || word === "SERIALISED";
    return `<li><span class="lamp ${bad ? "s-needs_confirm" : "s-ok"}">${esc(word)}</span>` +
      `<span style="color:var(--ink-2)">${esc(line.slice(line.indexOf("]") + 2))}</span></li>`;
  }).join("");

  note.innerHTML =
    `${esc(p.server_log.split(/[\\/]/).pop())} — ${p.client_logs.length} client log(s), ` +
    `${p.episodes} episode(s), wall ${p.wall_s.toFixed(0)} s, at most ` +
    `<b>${p.max_concurrent}</b> client(s) on the card at once, ` +
    `<b>${(100 * p.train_share).toFixed(0)}%</b> of it inside <code>train()</code>. ` +
    `Phases are unions of intervals, so overlapping episodes are not double-counted, ` +
    `and whatever the timestamps do not explain is named <em>unaccounted</em> rather ` +
    `than modelled. <span class="prov">Measured for this run by ` +
    `<code>pipeline/profile.py</code>; nothing was added to <code>my-project/</code> ` +
    `to make it possible.</span>`;
}

// ------------------------------------------------------------------ provenance
/** The measurement table itself: what the rest of the page is allowed to claim. */
export async function renderProvenance() {
  await loadMeasurements();
  const el = $("provenance");
  if (!el) return;
  if (!facts.records.length) {
    el.innerHTML = empty("No measurement table.",
      "Is this page talking to an older server? Restart python -m pipeline.server.");
    return;
  }
  const show = (r) => typeof r.value === "object"
    ? Object.entries(r.value).map(([k, v]) => `${k}: ${v}`).join(", ")
    : (Math.abs(r.value) < 0.01 && r.value !== 0 ? r.value.toExponential(2) : fmt(r.value, 4));

  el.innerHTML =
    '<table><thead><tr><th scope="col">Measurement</th><th scope="col">Value</th>' +
    '<th scope="col">Recorded in</th><th scope="col">What it means, and how it misleads</th>' +
    '</tr></thead><tbody>' +
    facts.records.map(r =>
      `<tr><td><b>${esc(r.label)}</b><br><code>${esc(r.id)}</code></td>` +
      `<td class="num">${esc(show(r))} <span class="dim">${esc(r.unit)}</span>` +
      (r.bound ? `<br><span class="warn">${esc(r.bound)} bound, ${esc(r.confidence)}</span>` : "") +
      `</td>` +
      `<td><code>${esc(r.source)}</code><br><span class="dim">${esc(r.measured_on)}</span></td>` +
      `<td style="color:var(--ink-2)">${esc(r.note)}</td></tr>`).join("") +
    '</tbody></table>' +
    `<p class="hint">Classes: ${facts.classes.all.length} in the label set, ` +
    `${facts.classes.warm_started.length} warm-started from COCO. The other ` +
    `${facts.classes.unwarmed.length} began cold — ` +
    `<b>${facts.classes.unwarmed.map(esc).join(", ")}</b>. ` +
    `${esc(facts.classes.note)}</p>`;
}
