// A whole /api/state snapshot, rendered by the real panels, against a fake DOM.
//
// This is the closest thing to loading the page that runs without a browser, and it
// exists because "the code looks right" is not evidence: one `$("typo")` in a render
// path throws, the pass dies silently, and the page just stops updating. The module-load
// check catches a module that will not parse; this catches a module that parses and then
// falls over on real data.
//
// Both fixtures are written by the test that runs this, from pipeline/demo_run.py and
// pipeline/measurements.py, so the payload is the server's own rather than a guess at it.
import { readFileSync } from "node:fs";

const read = (name) => JSON.parse(readFileSync(new URL(name, import.meta.url)));
const STATE = read("./state.json");
const FACTS = read("./facts.json");

const els = new Map();
const make = (id) => ({
  id, innerHTML: "", textContent: "", value: "", checked: false, hidden: false,
  className: "", title: "", disabled: false, href: "",
  style: {}, dataset: {}, attrs: {},
  classList: { contains: () => false, add() {}, remove() {} },
  viewBox: { baseVal: { width: 900, height: 220 } },
  setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k] ?? null; },
  removeAttribute(k) { delete this.attrs[k]; },
  querySelector: () => null, querySelectorAll: () => [],
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 220 }),
  insertAdjacentHTML() {}, appendChild() {}, removeChild() {}, remove() {},
  addEventListener() {}, focus() {}, closest() { return this; },
  childNodes: [], scrollTop: 0, scrollHeight: 0,
});
const el = (id) => { if (!els.has(id)) els.set(id, make(id)); return els.get(id); };

globalThis.document = {
  getElementById: el, querySelector: () => el("stub"), querySelectorAll: () => [],
  createElement: () => make("x"), addEventListener() {}, body: el("body"),
};
globalThis.window = { innerWidth: 1440, matchMedia: () => ({ matches: true }) };
globalThis.performance = { now: () => 0 };
globalThis.fetch = async (url) => ({
  ok: true,
  json: async () => String(url).includes("measurements") ? FACTS : { vehicles: [], pairs: [] },
});

const { loadMeasurements } = await import("./insight.js");
const { applyState } = await import("./live.js");

await loadMeasurements();
applyState(STATE);                 // throws on any missing id or bad field access

// The heartbeat: two real aggregates, both moving, and the delta printed.
const ledger = el("heartLedger").innerHTML;
console.assert((ledger.match(/<tr>/g) || []).length === 2, `expected 2 ledger rows: ${ledger}`);
console.assert(ledger.includes(">moved<"), "two differing aggregates must read as moved");
console.assert(ledger.includes("-1614.373832"), `the delta is missing: ${ledger}`);
console.assert(el("heartValue").textContent.includes("-2646.91"),
  `the headline checksum is wrong: ${el("heartValue").textContent}`);

// Round progress, and the two values this run has no measurement for.
console.assert(el("rRound").textContent === "2/2", `rRound: ${el("rRound").textContent}`);
console.assert(el("rMap").innerHTML.includes("not measured"),
  "no client evaluation was recorded; that must not read as a number");
console.assert(el("gUtil").innerHTML.includes("not measured"),
  "absent GPU telemetry must not read as an idle card");
console.assert(el("gEnergy").innerHTML.includes("not measured"),
  "absent energy must not read as 0.00 Wh");

// The honest metric, and the fact that it has no scale.
console.assert(el("hMap").innerHTML.includes("0.2073"),
  `the best holdout round is wrong: ${el("hMap").innerHTML}`);
console.assert(el("hCeiling").innerHTML.includes("not measured"),
  "with no baseline, 'of the ceiling' must say so rather than show a percentage");
console.assert(el("holdoutNote").innerHTML.includes("1000 images"),
  `the holdout size is missing: ${el("holdoutNote").innerHTML}`);
console.assert(el("holdoutNote").innerHTML.includes("no scale"),
  "a holdout number with no ceiling must say it has no scale");

// The band and its provenance: measured, and labelled with its conditions.
const prov = el("holdoutProvenance").innerHTML;
console.assert(prov.includes("0.0018"), `the noise floor is not cited: ${prov}`);
console.assert(prov.includes("NOISE_FLOOR.md"), "the band must name its source document");
console.assert(prov.includes("IID"), "the band must name the conditions it was measured at");
console.assert(prov.includes("holdout: 1000 images"), `provenance line: ${prov}`);

// The per-class panel stays hidden rather than showing thirteen zeros.
console.assert(el("perClassPanel").hidden === true,
  "no per-class data was recorded; the panel must hide, not show zeros");

// Criteria came through, including the warning that there is no ceiling.
console.assert(el("criteria").innerHTML.includes("PASS"), "the criteria did not render");
console.assert(el("criteria").innerHTML.includes("WARN"), "the ceiling warning is missing");

// And the fleet: six real conditions, 1 400 images each.
console.assert(el("fleet").innerHTML.includes("rain / fog"),
  `the fleet grid did not render its conditions: ${el("fleet").innerHTML.slice(0, 200)}`);

console.log("live render OK — a whole snapshot through the real panels");
process.exit(0);
