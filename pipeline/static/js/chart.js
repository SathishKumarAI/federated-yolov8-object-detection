// Charts: axes, gridlines, ticks, hover and keyboard readout, in inline SVG.
//
// A library is not an option here — the page is served straight off disk by
// http.server with no build step and must work with no network. Four line charts
// is less code than a bundler config would be.
//
// Everything visual is a CSS variable, so app.css controls the look.
import { $, esc } from "./util.js";

const NICE = [1, 2, 2.5, 5, 10];

/** Round tick values a person would have chosen: 1, 2, 2.5, 5 times a power of ten. */
export function ticks(lo, hi, count) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (hi === lo) hi = lo + 1;
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = (NICE.find(n => raw / mag <= n) || 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(+v.toPrecision(12));
  }
  return out.length > 1 ? out : [lo, hi];
}

/**
 * Draw a line chart into an <svg class="chart"> element.
 *
 * spec = {
 *   series: [{label, color, values:[y|null], area?, dashed?, key?}],
 *   xLabel?: i => string, yFmt?: v => string, zero?: bool, focus?: key, aria?: string,
 *   band?: {lo, hi, label}   a horizontal region drawn BEHIND the series
 * }
 *
 * `band` exists for one reason: this project's measured run-to-run spread is wide
 * enough to explain most of the deltas it has celebrated. A curve that stays inside
 * the band has not moved, and that is a fact about the measurement rather than about
 * the model -- so it is drawn, not left to the reader's arithmetic.
 */
export function lineChart(id, spec) {
  const svg = $(id);
  if (!svg) return;
  const series = (spec.series || []).filter(s => s.values.some(v => v != null));
  const box = svg.viewBox.baseVal, W = box.width || 900, H = box.height || 220;
  const P = { l: 52, r: 14, t: 12, b: 26 };
  if (!series.length) {
    svg.innerHTML = "";
    svg._spec = null;
    svg.setAttribute("aria-label", (spec.aria || "chart") + ": no data yet");
    return;
  }

  const all = series.flatMap(s => s.values.filter(v => v != null));
  let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
  // The band has to be inside the visible range or it silently says nothing.
  if (spec.band) { lo = Math.min(lo, spec.band.lo); hi = Math.max(hi, spec.band.hi); }
  if (spec.zero) lo = Math.min(0, lo);
  const pad = (hi - lo) * 0.12 || Math.abs(hi || 1) * 0.12;
  lo -= pad; hi += pad;
  const n = Math.max.apply(null, series.map(s => s.values.length));
  const yt = ticks(lo, hi, 4);
  lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);

  const x = i => P.l + (n <= 1 ? (W - P.l - P.r) / 2 : i * (W - P.l - P.r) / (n - 1));
  const y = v => H - P.b - ((v - lo) / (hi - lo || 1)) * (H - P.t - P.b);
  const yFmt = spec.yFmt || (v => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(3)));
  const xLabel = spec.xLabel || (i => "r" + (i + 1));

  let g = "";
  if (spec.band) {
    const top = Math.min(y(spec.band.hi), y(spec.band.lo));
    const h = Math.abs(y(spec.band.lo) - y(spec.band.hi));
    g += `<rect x="${P.l}" y="${top}" width="${W - P.l - P.r}" height="${Math.max(1, h)}" ` +
         `fill="var(--dim)" opacity="0.14"/>` +
         `<line x1="${P.l}" y1="${y((spec.band.lo + spec.band.hi) / 2)}" x2="${W - P.r}" ` +
         `y2="${y((spec.band.lo + spec.band.hi) / 2)}" stroke="var(--dim)" ` +
         `stroke-dasharray="2 4"/>`;
    if (spec.band.label) {
      g += `<text x="${W - P.r - 4}" y="${top - 4}" text-anchor="end" fill="var(--dim)" ` +
           `font-size="10" font-family="var(--mono)">${esc(spec.band.label)}</text>`;
    }
  }
  for (const v of yt) {
    g += `<line x1="${P.l}" y1="${y(v)}" x2="${W - P.r}" y2="${y(v)}" stroke="var(--line-soft)"/>` +
         `<text x="${P.l - 8}" y="${y(v) + 3.5}" text-anchor="end" fill="var(--dim)" ` +
         `font-size="10" font-family="var(--mono)">${esc(yFmt(v))}</text>`;
  }
  const every = Math.ceil(n / 12);
  for (let i = 0; i < n; i += every) {
    g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle" fill="var(--dim)" ` +
         `font-size="10" font-family="var(--mono)">${esc(xLabel(i))}</text>`;
  }
  g += `<line x1="${P.l}" y1="${H - P.b}" x2="${W - P.r}" y2="${H - P.b}" stroke="var(--line)"/>`;

  for (const s of series) {
    const pts = s.values.map((v, i) => v == null ? null : `${x(i)},${y(v)}`).filter(Boolean);
    if (!pts.length) continue;
    const dim = spec.focus && s.key && spec.focus !== s.key;
    if (s.area && pts.length > 1) {
      g += `<polygon points="${x(0)},${H - P.b} ${pts.join(" ")} ${x(s.values.length - 1)},${H - P.b}" ` +
           `fill="${s.color}" opacity="0.08"/>`;
    }
    g += `<polyline points="${pts.join(" ")}" fill="none" stroke="${s.color}" stroke-width="2.2" ` +
         `stroke-linejoin="round" stroke-linecap="round" opacity="${dim ? 0.22 : 1}"` +
         (s.dashed ? ' stroke-dasharray="4 3"' : "") + "/>";
    if (n <= 40) {
      g += s.values.map((v, i) => v == null ? "" :
        `<circle cx="${x(i)}" cy="${y(v)}" r="3.2" fill="var(--panel)" stroke="${s.color}" ` +
        `stroke-width="2" opacity="${dim ? 0.22 : 1}"/>`).join("");
    }
  }
  g += `<line class="cursor" y1="${P.t}" y2="${H - P.b}" stroke="var(--dim)" ` +
       `stroke-dasharray="3 3" opacity="0"/>`;

  svg.innerHTML = g;
  const last = series[0].values.filter(v => v != null).slice(-1)[0];
  svg.setAttribute("aria-label", `${spec.aria || id}. ${n} points, latest ` +
    `${last == null ? "none" : yFmt(last)}, range ${yFmt(lo)} to ${yFmt(hi)}.` +
    (spec.band ? ` Shaded band ${yFmt(spec.band.lo)} to ${yFmt(spec.band.hi)}: ` +
                 `${spec.band.label || "indistinguishable from the first point"}.` : ""));
  svg._spec = { series, n, x, y, yFmt, xLabel, P, W, H };
}

/** Sparkline for a card: no axes, shared scale passed in so cards compare. */
export function sparkline(values, lo, hi, colour) {
  const W = 240, H = 40, p = 4, span = (hi - lo) || 1;
  if (!values.length) return `<svg class="spark" viewBox="0 0 ${W} ${H}" aria-hidden="true"></svg>`;
  const x = i => p + (values.length === 1 ? (W - 2 * p) / 2 : i * (W - 2 * p) / (values.length - 1));
  const y = v => H - p - ((v - lo) / span) * (H - 2 * p);
  const pts = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">` +
    `<polygon points="${x(0)},${H} ${pts} ${x(values.length - 1)},${H}" fill="${colour}" opacity="0.12"/>` +
    `<polyline points="${pts}" fill="none" stroke="${colour}" stroke-width="2" ` +
    `stroke-linejoin="round" vector-effect="non-scaling-stroke"/>` +
    `<circle cx="${x(values.length - 1)}" cy="${y(values[values.length - 1])}" r="2.6" fill="${colour}"/></svg>`;
}

/** Progress ring: rounds this vehicle has trained, out of the run's total. */
export function ring(fraction, active) {
  const r = 11, c = 2 * Math.PI * r, done = Math.max(0, Math.min(1, fraction || 0));
  return `<svg class="ring" width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">` +
    `<circle cx="14" cy="14" r="${r}" fill="none" stroke="var(--line)" stroke-width="2.5"/>` +
    `<circle cx="14" cy="14" r="${r}" fill="none" stroke="${active ? "var(--accent)" : "var(--ok)"}" ` +
    `stroke-width="2.5" stroke-linecap="round" stroke-dasharray="${c * done} ${c}" ` +
    `transform="rotate(-90 14 14)"/></svg>`;
}

// ---- one cursor and one tooltip, shared by every chart on the page ----------
let cursorIndex = null;

function showCursor(svg, i, cx, cy) {
  const s = svg._spec;
  if (!s) return;
  cursorIndex = i;
  const line = svg.querySelector(".cursor");
  if (line) {
    line.setAttribute("x1", s.x(i));
    line.setAttribute("x2", s.x(i));
    line.setAttribute("opacity", "1");
  }
  const rows = s.series.filter(se => se.values[i] != null)
                       .map(se => `${se.label}  ${s.yFmt(se.values[i])}`);
  if (!rows.length) return hideCursor(svg);
  const tip = $("tip");
  tip.innerHTML = `<b>${esc(s.xLabel(i))}</b>\n${esc(rows.join("\n"))}`;
  tip.hidden = false;
  const r = svg.getBoundingClientRect();
  const px = cx != null ? cx : r.left + (s.x(i) / s.W) * r.width;
  const py = cy != null ? cy : r.top + r.height / 2;
  tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 12, px + 14) + "px";
  tip.style.top = Math.max(8, py - tip.offsetHeight - 12) + "px";
}

function hideCursor(svg) {
  $("tip").hidden = true;
  const line = svg && svg.querySelector(".cursor");
  if (line) line.setAttribute("opacity", "0");
}

/** Wire hover and arrow-key readout once, for every current and future chart. */
export function enableChartCursor() {
  document.addEventListener("pointermove", e => {
    const svg = e.target.closest && e.target.closest("svg.chart");
    if (svg && svg._spec) {
      const s = svg._spec;
      const rect = svg.getBoundingClientRect();
      const vx = (e.clientX - rect.left) / rect.width * s.W;
      let i = Math.round((vx - s.P.l) / ((s.W - s.P.l - s.P.r) || 1) * (s.n - 1));
      i = Math.max(0, Math.min(s.n - 1, i));
      showCursor(svg, i, e.clientX, e.clientY);
    } else if (!$("tip").hidden) {
      $("tip").hidden = true;
      document.querySelectorAll("svg.chart .cursor").forEach(l => l.setAttribute("opacity", "0"));
    }
  });
  // Keyboard equivalent: a focused chart is steppable, so the values are not
  // mouse-only. This is what makes role="img" on a chart honest.
  document.addEventListener("keydown", e => {
    const svg = document.activeElement;
    if (!svg || !svg.classList || !svg.classList.contains("chart") || !svg._spec) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const next = (cursorIndex == null ? 0 : cursorIndex) + (e.key === "ArrowRight" ? 1 : -1);
    showCursor(svg, Math.max(0, Math.min(svg._spec.n - 1, next)));
  });
  document.addEventListener("focusout", e => {
    if (e.target.classList && e.target.classList.contains("chart")) hideCursor(e.target);
  });
}


/** Horizontal bars with a value axis. For stage timings and anything else where the
 *  category is a name rather than a position on a scale. */
export function barChart(id, spec) {
  const svg = $(id);
  if (!svg) return;
  const items = (spec.items || []).filter(i => i.value != null);
  const box = svg.viewBox.baseVal, W = box.width || 900;
  if (!items.length) { svg.innerHTML = ""; svg._spec = null; return; }

  const rowH = spec.rowHeight || 26, pad = { l: spec.labelWidth || 120, r: 60, t: 6, b: 18 };
  const H = pad.t + pad.b + items.length * rowH;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const max = Math.max.apply(null, items.map(i => i.value)) || 1;
  const x = v => (v / max) * (W - pad.l - pad.r);
  const fmt = spec.fmt || (v => v.toFixed(1));

  svg.innerHTML = items.map((it, i) => {
    const y = pad.t + i * rowH;
    return `<text x="${pad.l - 8}" y="${y + rowH / 2 + 3.5}" text-anchor="end" ` +
      `fill="var(--dim)" font-size="11" font-family="var(--mono)">${esc(it.label)}</text>` +
      `<rect x="${pad.l}" y="${y + 4}" width="${Math.max(1, x(it.value))}" ` +
      `height="${rowH - 10}" rx="3" fill="${it.color || "var(--accent)"}" opacity="0.85">` +
      `<title>${esc(it.label)}: ${esc(fmt(it.value))}</title></rect>` +
      `<text x="${pad.l + x(it.value) + 8}" y="${y + rowH / 2 + 3.5}" fill="var(--ink-2)" ` +
      `font-size="11" font-family="var(--mono)">${esc(fmt(it.value))}</text>`;
  }).join("");
  svg.setAttribute("aria-label", (spec.aria || id) + ". " +
    items.map(i => `${i.label} ${fmt(i.value)}`).join(", "));
}

/** Scatter, for "what did this cost buy": one dot per run, labelled on hover. */
export function scatterChart(id, spec) {
  const svg = $(id);
  if (!svg) return;
  const pts = (spec.points || []).filter(p => p.x != null && p.y != null);
  const box = svg.viewBox.baseVal, W = box.width || 900, H = box.height || 260;
  const P = { l: 56, r: 16, t: 14, b: 34 };
  if (!pts.length) { svg.innerHTML = ""; svg._spec = null; return; }

  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const padX = (x1 - x0) * 0.12 || Math.abs(x1 || 1) * 0.12;
  const padY = (y1 - y0) * 0.12 || Math.abs(y1 || 1) * 0.12;
  x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
  const X = v => P.l + (v - x0) / (x1 - x0 || 1) * (W - P.l - P.r);
  const Y = v => H - P.b - (v - y0) / (y1 - y0 || 1) * (H - P.t - P.b);
  const xf = spec.xFmt || (v => v.toFixed(0)), yf = spec.yFmt || (v => v.toFixed(3));

  let g = "";
  for (const v of ticks(y0, y1, 4)) {
    g += `<line x1="${P.l}" y1="${Y(v)}" x2="${W - P.r}" y2="${Y(v)}" stroke="var(--line-soft)"/>` +
         `<text x="${P.l - 8}" y="${Y(v) + 3.5}" text-anchor="end" fill="var(--dim)" ` +
         `font-size="10" font-family="var(--mono)">${esc(yf(v))}</text>`;
  }
  for (const v of ticks(x0, x1, 5)) {
    g += `<text x="${X(v)}" y="${H - 12}" text-anchor="middle" fill="var(--dim)" ` +
         `font-size="10" font-family="var(--mono)">${esc(xf(v))}</text>`;
  }
  g += pts.map(p =>
    `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.r || 6}" fill="${p.color || "var(--accent)"}" ` +
    `fill-opacity="0.5" stroke="${p.color || "var(--accent)"}" stroke-width="1.6">` +
    `<title>${esc(p.label || "")}
${esc(spec.xName || "x")} ${esc(xf(p.x))}
` +
    `${esc(spec.yName || "y")} ${esc(yf(p.y))}</title></circle>`).join("");
  g += `<text x="${(W + P.l) / 2}" y="${H - 1}" text-anchor="middle" fill="var(--dim)" ` +
       `font-size="10" font-family="var(--mono)">${esc(spec.xName || "")}</text>`;
  svg.innerHTML = g;
  svg._spec = null;
  svg.setAttribute("aria-label", (spec.aria || id) + `. ${pts.length} runs plotted.`);
}
