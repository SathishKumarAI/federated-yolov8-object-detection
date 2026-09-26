// Small shared helpers. Nothing here knows about the API or the DOM layout.
export const $ = (id) => document.getElementById(id);

export const esc = (s) => String(s).replace(/[&<>"]/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

/** Numbers a person reads: three decimals, exponent only when it would not fit. */
export const fmt = (x, d) =>
  x == null ? "—" : (Math.abs(x) >= 1e5 ? x.toExponential(2) : x.toFixed(d == null ? 3 : d));

export const PALETTE = ["#5ad1e6", "#f0a92b", "#4ec9a0", "#e07ab0", "#a98bf5", "#ff6a5e",
                        "#7fd4a0", "#f5915e", "#9fb0c4", "#69a5ff"];

/**
 * A value the server does not have.
 *
 * Deliberately not `0` and not a bare dash: both read as data. "6 308 examples" was
 * once printed for a shard holding 10 images, and a dash where a number belongs is the
 * same bug wearing different clothes. Whatever is not measured says so, and says why.
 */
export const unknown = (why) =>
  `<span class="unk" title="${esc(why)}" tabindex="0" aria-label="not measured: ${esc(why)}">not measured</span>`;

/** Empty state: says what is missing, then which command produces it. */
export const empty = (title, next) =>
  `<div class="empty"><p class="t">${title}</p><p class="n">${next}</p></div>`;

// Condition glyphs, line art drawn from the vehicle's own world. Keys match the
// labels in pipeline/vehicles.py PROFILES; add both together.
const GLYPHS = {
  "daytime city": '<circle cx="8" cy="8" r="3.2"/><path d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3.1 3.1l1.1 1.1M11.8 11.8l1.1 1.1M12.9 3.1l-1.1 1.1M4.2 11.8l-1.1 1.1"/>',
  "night": '<path d="M13 9.6A5.6 5.6 0 1 1 6.6 3a4.6 4.6 0 0 0 6.4 6.6z"/>',
  "rain / fog": '<path d="M4.5 8.5a2.7 2.7 0 0 1 .3-5.4 3.6 3.6 0 0 1 6.8 1.1 2.3 2.3 0 0 1-.4 4.3z"/><path d="M5.5 11v2M8 11.5v2.5M10.5 11v2"/>',
  "highway": '<path d="M2 14.5 6.4 1.5M14 14.5 9.6 1.5"/><path d="M8 3v2M8 7v2M8 11v2"/>',
  "dawn / dusk": '<path d="M1.5 12.5h13"/><path d="M4.2 12.5a3.8 3.8 0 0 1 7.6 0"/><path d="M8 3v2M3.4 5.4l1.3 1.3M12.6 5.4l-1.3 1.3"/>',
  "overcast residential": '<path d="M2.5 6.5 8 2l5.5 4.5"/><path d="M4.5 7.5v6h7v-6"/><path d="M7 13.5v-3h2v3"/>',
  "snow": '<path d="M8 1.5v13M2.2 4.8l11.6 6.4M13.8 4.8 2.2 11.2"/>',
  "parking / tunnel": '<path d="M2.5 14V8a5.5 5.5 0 0 1 11 0v6"/><path d="M6 14V8.5a2 2 0 0 1 4 0V14"/>',
  "random mix": '<path d="M1.5 4.5h3l7 7h3M11.5 2.5l3 2-3 2M1.5 11.5h3l2-2M11.5 13.5l3-2-3-2"/>',
  "dirichlet": '<path d="M8 1.5 14.5 14H1.5z"/><path d="M5.4 9.6h5.2"/>',
};

export function glyph(condition) {
  const key = Object.keys(GLYPHS).find(k => (condition || "").startsWith(k)) || "random mix";
  return '<svg class="glyph" width="18" height="18" viewBox="0 0 16 16" fill="none" aria-hidden="true" ' +
         'stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">' +
         GLYPHS[key] + '</svg>';
}
