// Control view: the run form, the launch/stop buttons, the stage table.
import { $, esc } from "./util.js";
import { loadMeasurements, leverNote } from "./insight.js";

//: lever body field -> how to read it out of the form. The names are the `Config` field
//: names the server advertises in `options.levers`, so the form cannot offer a lever the
//: running server has no field for -- a body key it has never heard of is dropped in
//: silence, which is the shape of no-op this project keeps shipping.
const LEVERS = {
  freeze_round1: () => +$("freezeRound1").value,
  server_ema: () => +$("serverEma").value,
  fix_bn_from_round: () => +$("fixBnFromRound").value,
  imgsz: () => +$("imgsz").value,
  local_bn: () => $("localBn").checked,
};

let supported = null;         // options.levers, once the server has said

export function config() {
  const body = {
    profile: $("profile").value,
    vehicles: +$("vehicles").value,
    rounds: +$("rounds").value,
    epochs: +$("epochs").value,
    seed: +$("seed").value,
    partition: $("partition").value,
    alpha: +$("alpha").value,
    size_skew: +$("sizeSkew").value,
    strategy: $("strategy").value,
    proximal_mu: +$("mu").value,
    confirm: $("confirm").checked,
  };
  // Only levers this server implements, and only when they are actually set: sending
  // `server_ema: 0` to a server that ignores the field and to one that implements it
  // must mean the same thing, and it does -- off.
  for (const [name, read] of Object.entries(LEVERS)) {
    if (supported && supported[name] === false) continue;
    const v = read();
    if (v) body[name] = v;
  }
  return body;
}

export function estimate() {
  const c = config();
  const per = c.profile === "demo" ? 300 : 6308;
  // ~919 s measured for 2 clients x 2 rounds x 1 epoch on 6308 images, serialised.
  const secs = (per / 6308) * 230 * c.vehicles * c.rounds * c.epochs;
  const m = Math.max(1, Math.round(secs / 60));
  $("estimate").textContent =
    `${c.vehicles} vehicles × ${per} images × ${c.rounds} rounds × ${c.epochs} epochs — ` +
    `roughly ${m} min of GPU time.`;
  const dirichlet = c.partition === "dirichlet";
  $("alphaWrap").hidden = !dirichlet;
  $("alphaNote").hidden = !dirichlet;
  $("sizeSkewNote").hidden = !c.size_skew;
  $("muWrap").hidden = c.strategy !== "fedprox";
  warnAboutLevers(c);
}

/**
 * The two ways these levers are quietly not the method they are named after.
 *
 * FixBN at round 1 pins COCO's initial statistics; the warm-up IS the method, so R=1 is
 * the one value that looks like the technique and is not it. FedBN leaves no single
 * global model, so a holdout score on the checkpoint measures a model that never existed
 * on any vehicle -- which is the honest global metric quietly becoming meaningless.
 */
function warnAboutLevers(c) {
  const out = [];
  const fix = +$("fixBnFromRound").value;
  if (fix === 1) {
    out.push("<b>FixBN from round 1 is not FixBN.</b> It pins COCO's initial BatchNorm " +
      `statistics; the warm-up is the method. For ${c.rounds} rounds, try ` +
      `${Math.max(2, Math.round(c.rounds / 2))}.`);
  }
  if (fix > c.rounds) {
    out.push(`<b>FixBN from round ${fix} never triggers</b> in a ${c.rounds}-round run.`);
  }
  if ($("localBn").checked) {
    out.push("<b>FedBN leaves no single global model.</b> Scoring the saved checkpoint " +
      "on the holdout then measures a model that never existed on any vehicle — the " +
      "one number on this dashboard that is comparable between runs stops being that.");
  }
  const imgsz = +$("imgsz").value;
  if (imgsz && imgsz !== 640) {
    out.push(`<b>${imgsz} px has never been timed here.</b> The Simulate tab refuses to ` +
      "project its wall clock rather than guessing, and it changes no data — it changes " +
      "how much of a sub-20 px traffic light survives the downscale.");
  }
  $("leverWarnings").innerHTML = out.length
    ? out.map(w => `<p class="hint warn">${w}</p>`).join("") : "";
}

/** Fill the strategy and partition menus from what the server actually registered,
 *  so a name the backend does not know cannot be picked here. */
export function renderOptions(options) {
  if (!options) return;
  const fill = (id, values) => {
    const el = $(id);
    if (!el || el.dataset.filled === String(values.length)) return;
    const keep = el.value;
    el.innerHTML = values.map(v => `<option value="${v}">${v}</option>`).join("");
    el.value = values.includes(keep) ? keep : values[0];
    el.dataset.filled = String(values.length);
  };
  fill("strategy", options.strategies || []);
  renderLevers(options.levers);
  estimate();
}

/**
 * Enable only the levers this server can apply, and say so for the rest.
 *
 * The plumbing for these lives on the ML side. A dashboard that offered a control the
 * running server has no `Config` field for would post it into a void and report success,
 * which is indistinguishable on screen from the lever working.
 */
function renderLevers(levers) {
  if (!levers) return;
  supported = levers;
  const missing = Object.keys(levers).filter(k => !levers[k]);
  $("leverSupport").textContent = missing.length
    ? `${Object.keys(levers).length - missing.length} of ${Object.keys(levers).length} available`
    : "all available";
  document.querySelectorAll("#view-control .lever").forEach(box => {
    const name = box.dataset.lever;
    const ok = levers[name] !== false;
    box.dataset.unsupported = String(!ok);
    box.querySelectorAll("input,select").forEach(el => {
      el.disabled = !ok;
      el.title = ok ? "" : `This server has no ${name} field, so the lever cannot be `
        + `applied. It is not sent.`;
    });
  });
}

/** The caveat for every lever, from the measurement table -- never retyped here. */
async function renderLeverNotes() {
  await loadMeasurements();
  $("leverNotes").innerHTML = Object.keys(LEVERS).map(name => {
    const note = leverNote(name);
    return note ? `<p class="hint lever-note" data-lever="${esc(name)}">` +
      `<b>${esc(name)}</b> — ${note}</p>` : "";
  }).join("");
}

export function renderStages(rows) {
  if (!rows) return;
  $("stageCount").textContent = `${rows.filter(r => !r.satisfied).length} of ${rows.length} will run`;
  $("stageTable").innerHTML = rows.map(r => {
    const cls = r.satisfied ? "s-skipped" : (r.gated ? "s-needs_confirm" : "s-pending");
    const word = r.satisfied ? "skip" : (r.gated ? "gated" : "will run");
    return `<tr><td><b>${esc(r.name)}</b><br>` +
      `<span style="color:var(--dim);font-size:var(--t-sm)">${esc(r.title)}</span></td>` +
      `<td><span class="lamp ${cls}">${word}</span></td>` +
      `<td class="num">${esc(r.est)}</td>` +
      `<td style="color:var(--dim);font-size:var(--t-sm)">${esc(r.detail)}</td></tr>`;
  }).join("");
}

export function wireControl(onLaunched) {
  ["profile", "vehicles", "rounds", "epochs", "partition", "sizeSkew", "strategy",
   "freezeRound1", "serverEma", "fixBnFromRound", "imgsz", "localBn"].forEach(id => {
    $(id).oninput = estimate;
    $(id).onchange = estimate;
  });
  estimate();
  renderLeverNotes();

  $("launch").onclick = async () => {
    $("launch").disabled = true;
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config()),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      $("estimate").innerHTML = `<span class="warn">${esc(body.error ||
        "A run is already in flight. Stop it before launching another.")}</span>`;
      $("launch").disabled = false;
      return;
    }
    onLaunched();
  };
  $("stop").onclick = () => fetch("/api/stop", { method: "POST" });
}
