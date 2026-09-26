// Metrics view: every run as one comparable record — what was tried, how much data
// it used, how long it took, what it cost, and what it produced.
//
// The per-epoch panel is the point of this tab. Round-level metrics say a vehicle
// improved; the epochs inside the round say whether it was still learning when the
// round ended or had stopped and started drifting from the others. That is the
// difference between "run more rounds" and "run longer rounds", and nothing on the
// other tabs could tell them apart.
import { $, esc, empty, PALETTE } from "./util.js";
import { lineChart, barChart, scatterChart } from "./chart.js";

let ledger = null;
let selected = null;
let sort = { key: "run", dir: -1 };

export async function loadMetrics(force) {
  if (ledger && !force) return render();
  $("metricsBody").innerHTML = '<div class="skel-block" style="min-height:300px"></div>';
  try {
    ledger = await (await fetch("/api/ledger")).json();
  } catch {
    $("metricsBody").innerHTML = empty("The server did not answer.", "Is it still running?");
    return;
  }
  render();
}

const min = (s) => (s / 60).toFixed(1);
const num = (v, d = 4) => v == null ? "—" : v.toFixed(d);

function runsTable(runs) {
  const cols = [["run", "run"], ["approach", "approach"], ["visits", "image-visits"],
                ["minutes", "min"], ["wh", "Wh"], ["holdout", "holdout mAP50"],
                ["retained", "of ceiling"]];
  const rows = runs.map(r => ({
    r,
    run: r.run, approach: r.approach,
    visits: r.data.image_visits, minutes: r.time.seconds / 60,
    wh: r.cost.energy_wh || 0,
    holdout: r.result.holdout_mAP50 == null ? -1 : r.result.holdout_mAP50,
    retained: r.result.retained == null ? -1 : r.result.retained,
  }));
  rows.sort((a, b) => {
    const x = a[sort.key], y = b[sort.key];
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sort.dir;
  });
  return '<table><thead><tr>' + cols.map(([k, t]) =>
      `<th scope="col"><button class="sortbtn" data-key="${k}">` +
      `${t}${sort.key === k ? (sort.dir > 0 ? " ▲" : " ▼") : ""}</button></th>`).join("") +
    '</tr></thead><tbody>' + rows.map(({ r, ...v }) => {
      const res = r.result;
      const flag = res.ceiling_suspect
        ? '<span class="chip down" title="the federation scored above its own ceiling: that ceiling is stale or was never matched">suspect</span>'
        : (res.retained != null && res.budget_matched === false
          ? '<span class="chip flat" title="the ceiling had a different budget, so this is a bound">bound</span>' : "");
      const old_transport = r.era && r.era.transport === "ema-fp16-best";
      const eraChip = r.era
        ? `<span class="chip ${old_transport ? "flat" : "up"}" title="${esc(r.era.note)} ` +
          `Evidence: ${esc(r.era.evidence)}.">${old_transport ? "old transport" : "fp32"}</span>`
        : "";
      return `<tr class="runrow" data-run="${esc(r.run)}" data-sel="${selected === r.run}">` +
        `<td><button class="linky" data-run="${esc(r.run)}">${esc(r.run)}</button></td>` +
        `<td style="font-size:var(--t-sm)">${esc(r.approach)} ${eraChip}</td>` +
        `<td class="num">${v.visits.toLocaleString()}</td>` +
        `<td class="num">${min(r.time.seconds)}</td>` +
        `<td class="num">${(r.cost.energy_wh || 0).toFixed(1)}</td>` +
        `<td class="num"><b>${num(res.holdout_mAP50)}</b></td>` +
        `<td class="num">${res.retained == null ? "—" :
          (100 * res.retained).toFixed(1) + "%"} ${flag}</td></tr>`;
    }).join("") + "</tbody></table>";
}

function detail(r) {
  const res = r.result, L = r.learning;
  const vids = L.trained.length ? L.trained : Object.keys(L.epochs);
  const epochVid = vids.includes(selectedVehicle) ? selectedVehicle : vids[0];

  return `<div class="panel"><h2>${esc(r.run)}<span class="n">${esc(r.approach)}</span></h2>` +
    `<div class="readouts">` +
      `<div class="readout"><span class="v">${num(res.holdout_mAP50)}</span>` +
        `<div class="k">holdout mAP50</div></div>` +
      `<div class="readout"><span class="v">${num(res.holdout_mAP50_95)}</span>` +
        `<div class="k">holdout mAP50-95</div></div>` +
      `<div class="readout"><span class="v">${num(res.self_mAP50)}</span>` +
        `<div class="k">self-evaluated</div></div>` +
      `<div class="readout"><span class="v">${r.data.image_visits.toLocaleString()}</span>` +
        `<div class="k">image-visits</div></div>` +
      `<div class="readout"><span class="v">${min(r.time.seconds)}</span><span class="u">min</span>` +
        `<div class="k">wall clock</div></div>` +
      `<div class="readout"><span class="v">${(r.cost.energy_wh || 0).toFixed(1)}</span>` +
        `<span class="u">Wh</span><div class="k">energy</div></div>` +
      `<div class="readout"><span class="v">${r.cost.wh_per_point ?? "—"}</span>` +
        `<div class="k">Wh per mAP point</div></div>` +
      `<div class="readout"><span class="v">${r.time.seconds_per_kvisit ?? "—"}</span>` +
        `<span class="u">s</span><div class="k">per 1k image-visits</div></div>` +
    `</div></div>` +

    `<div class="grid two" style="grid-template-columns:1fr 1fr">` +
      `<div class="panel"><h2>Global model, by round<span class="n">on the shared holdout</span></h2>` +
        `<figure><svg class="chart" id="mRoundChart" viewBox="0 0 900 220" role="img" tabindex="0"></svg></figure>` +
        `<p class="hint">${res.holdout_curve.length
          ? "By round: " + res.holdout_curve.map(v => v.toFixed(4)).join(" → ")
          : "This run has no holdout curve; it predates the holdout or never ran evaluate."}</p></div>` +

      `<div class="panel"><h2>Where the time went</h2>` +
        `<figure><svg class="chart" id="mStageChart" viewBox="0 0 900 200" role="img"></svg></figure>` +
        `<p class="hint">Stage seconds for this run. The federation dominates; everything ` +
        `else is preparation and measurement.</p></div>` +
    `</div>` +

    `<div class="panel"><h2>Per-vehicle learning<span class="n">each on its own split</span></h2>` +
      `<figure><svg class="chart" id="mVehicleChart" viewBox="0 0 900 240" role="img" tabindex="0"></svg></figure>` +
      `<div class="legend" id="mVehicleLegend"></div></div>` +

    `<div class="panel"><h2>Inside a round — per epoch` +
      `<span class="n">${vids.length ? "vehicle " + esc(String(epochVid)) : "no data"}</span></h2>` +
      (vids.length ? `<div class="legend" id="mEpochPick">` + vids.map((v, i) =>
        `<button type="button" data-vid="${esc(String(v))}" aria-pressed="${String(v) === String(epochVid)}">` +
        `<i style="background:${PALETTE[i % PALETTE.length]}"></i>v${esc(String(v))}</button>`).join("") + `</div>` : "") +
      `<figure><svg class="chart" id="mEpochChart" viewBox="0 0 900 220" role="img" tabindex="0"></svg></figure>` +
      `<div class="legend"><span><i style="background:#5ad1e6"></i>box</span>` +
      `<span><i style="background:#f0a92b"></i>cls</span>` +
      `<span><i style="background:#4ec9a0"></i>dfl</span></div>` +
      `<p class="hint" id="mEpochNote"></p></div>`;
}

let selectedVehicle = null;

function drawDetail(r) {
  const res = r.result, L = r.learning;
  lineChart("mRoundChart", {
    series: [{ label: "holdout mAP50", color: "var(--ok)", values: res.holdout_curve, area: true }],
    aria: `holdout mAP50 by round for ${r.run}`, yFmt: v => v.toFixed(3),
  });

  barChart("mStageChart", {
    items: r.time.per_stage.filter(s => s.seconds > 0).map(s => ({
      label: s.name, value: s.seconds,
      color: s.status === "failed" ? "var(--bad)" : "var(--accent)",
    })),
    fmt: v => v >= 60 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(1)} s`,
    aria: "seconds per stage",
  });

  const vids = L.trained.length ? L.trained : Object.keys(L.per_vehicle_rounds);
  lineChart("mVehicleChart", {
    series: vids.map((vid, i) => ({
      key: vid, label: "v" + vid, color: PALETTE[i % PALETTE.length],
      values: (L.per_vehicle_rounds[vid] || []).map(x => x.mAP50),
    })),
    aria: "per-vehicle mAP50 by round", yFmt: v => v.toFixed(3),
  });
  $("mVehicleLegend").innerHTML = vids.map((vid, i) =>
    `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>v${esc(String(vid))} · ` +
    `${esc(L.conditions[vid] || "?")}</span>`).join("");

  const epochVid = vids.includes(selectedVehicle) ? selectedVehicle : vids[0];
  const rows = (L.epochs || {})[epochVid] || [];
  lineChart("mEpochChart", {
    series: [{ label: "box", color: "#5ad1e6", values: rows.map(r2 => r2.box) },
             { label: "cls", color: "#f0a92b", values: rows.map(r2 => r2.cls) },
             { label: "dfl", color: "#4ec9a0", values: rows.map(r2 => r2.dfl) }],
    xLabel: i => "e" + (i + 1), aria: `per-epoch losses for vehicle ${epochVid}`,
  });
  const note = $("mEpochNote");
  if (note) {
    if (!rows.length) {
      note.textContent = "No per-epoch rows for this vehicle. Ultralytics keeps only the " +
        "most recent round's results.csv, so this is empty for older runs.";
    } else {
      const box = rows.map(x => x.box).filter(v => v != null);
      const first = box[0], last = box[box.length - 1];
      const delta = last - first;                    // positive means the loss ROSE
      const half = box.slice(Math.floor(box.length / 2));
      const flat = half.length > 1 && Math.abs(half[0] - half[half.length - 1]) < 0.005;

      // Three different findings, and they call for three different actions. Reading
      // "still falling" off a loss that rose is how a diverging client gets mistaken
      // for one that needs longer rounds.
      let verdict;
      if (delta > 0.005) {
        verdict = `<span class="warn">Box loss <b>rose</b> across the round. The client ` +
          `is being pulled away from a better model than it ended with — warmup is a ` +
          `third of a short round, so the schedule never leaves warmup. Fewer local ` +
          `epochs, or a schedule tuned for short rounds.</span>`;
      } else if (flat) {
        verdict = `<span class="warn">Flat across the second half: this vehicle stopped ` +
          `learning before the round ended. Longer rounds buy nothing here — more ` +
          `rounds might.</span>`;
      } else {
        verdict = `Still falling at the end of the round, so the round cut training ` +
          `short: more local epochs would have kept helping this vehicle, at the cost ` +
          `of more drift from the others.`;
      }
      note.innerHTML = `${rows.length} epochs in the last round. Box loss ` +
        `${first?.toFixed(3)} → ${last?.toFixed(3)} ` +
        `(${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(3)}). ${verdict}`;
    }
  }

  const pick = $("mEpochPick");
  if (pick) pick.querySelectorAll("button").forEach(b => b.onclick = () => {
    selectedVehicle = b.dataset.vid;
    render();
  });
}

function render() {
  const d = ledger;
  if (!d) return;
  const runs = d.runs || [];
  if (!runs.length) {
    $("metricsBody").innerHTML = empty("No runs recorded yet.",
      "Every run writes a report to <code>pipeline/reports/</code>; this reads them.");
    return;
  }
  const sel = runs.find(r => r.run === selected) || runs[runs.length - 1];
  selected = sel.run;
  const scored = runs.filter(r => r.result.holdout_mAP50 != null && r.cost.energy_wh);

  $("metricsBody").innerHTML =
    `<div class="panel"><h2>Runs<span class="n">${runs.length} recorded · click one</span></h2>` +
      runsTable(runs) +
      `<p class="hint">image-visits is vehicles × images × rounds × local epochs — the ` +
      `unit that makes two runs comparable. <b>bound</b> means the ceiling had a ` +
      `different budget; <b>suspect</b> means the federation scored above its own ` +
      `ceiling, which means that ceiling was stale.</p></div>` +

    (d.approaches && d.approaches.some(a => a.n > 1)
      ? `<div class="panel"><h2>By approach<span class="n">repeats grouped</span></h2>` +
        `<table><thead><tr><th scope="col">Approach</th><th scope="col">Runs</th>` +
        `<th scope="col">Mean holdout</th><th scope="col">Best</th>` +
        `<th scope="col">Spread</th></tr></thead><tbody>` +
        d.approaches.map(a => `<tr><td style="font-size:var(--t-sm)">${esc(a.approach)}</td>` +
          `<td class="num">${a.n}</td><td class="num">${num(a.mean)}</td>` +
          `<td class="num">${num(a.best)}</td>` +
          `<td class="num">${a.spread == null ? "—" : num(a.spread)}</td></tr>`).join("") +
        `</tbody></table><p class="hint">Spread across repeats of the same approach is ` +
        `the noise floor. A difference between two approaches smaller than this says ` +
        `nothing.</p></div>`
      : "") +

    (scored.length > 1
      ? `<div class="panel"><h2>What the energy bought<span class="n">one dot per run</span></h2>` +
        `<figure><svg class="chart" id="mScatter" viewBox="0 0 900 260" role="img"></svg></figure>` +
        `<p class="hint">Up and to the left is better: more mAP for fewer watt-hours. ` +
        `Runs at different scales are not really comparable here — the point is the ` +
        `shape of the trade, not a ranking.</p></div>`
      : "") +

    detail(sel);

  drawDetail(sel);
  if (scored.length > 1) {
    scatterChart("mScatter", {
      points: scored.map((r, i) => ({
        x: r.cost.energy_wh, y: r.result.holdout_mAP50,
        label: `${r.run} — ${r.approach}`,
        color: PALETTE[i % PALETTE.length],
        r: 5 + Math.min(9, Math.log10(Math.max(10, r.data.image_visits))),
      })),
      xName: "energy (Wh)", yName: "holdout mAP50",
      xFmt: v => v.toFixed(0), yFmt: v => v.toFixed(3),
      aria: "energy against holdout mAP50, dot size by image-visits",
    });
  }

  document.querySelectorAll("#metricsBody .sortbtn").forEach(b => b.onclick = () => {
    const key = b.dataset.key;
    sort = { key, dir: sort.key === key ? -sort.dir : 1 };
    render();
  });
  document.querySelectorAll("#metricsBody .runrow, #metricsBody .linky[data-run]")
    .forEach(el => el.onclick = () => {
      selected = el.dataset.run;
      selectedVehicle = null;
      render();
    });
}
