// The checksum ledger and the per-class panel, against a fake DOM.
//
// The ledger is the one panel whose verdict overrides every other number on the page,
// and the thing it must never do is call two equal consecutive aggregates "moved".
// `stuck` is decided on six printed decimals — the same precision pipeline/verify.py
// uses — so a difference far below what anyone can read must still count as equal.
const els = {};
const el = (id) => (els[id] ||= { id, innerHTML: "", textContent: "", hidden: false,
                                  className: "", style: {}, setAttribute() {} });
globalThis.document = { getElementById: el, querySelectorAll: () => [] };
globalThis.fetch = async () => { throw new Error("offline on purpose"); };

const { renderChecksumLedger, renderPerClass, loadMeasurements } = await import("./insight.js");

// ---- no rounds yet: an explanation, never an empty table --------------------
renderChecksumLedger([]);
console.assert(el("heartLedger").innerHTML.includes("No aggregate yet"),
  "an empty ledger must say why it is empty");

// ---- a fleet that is learning ----------------------------------------------
renderChecksumLedger([-1032.5395936965942, -2646.913425683975, -3110.25]);
let html = el("heartLedger").innerHTML;
console.assert((html.match(/>moved</g) || []).length === 2, `expected 2 'moved', got: ${html}`);
console.assert(html.includes(">first<"), "round 1 has nothing to compare against");
console.assert(!html.includes("not moving"), "a moving fleet must not be flagged stuck");
console.assert(html.includes("-1614.373832"), `the delta is not printed: ${html}`);

// ---- the B4 bug: FedAvg averaging its own input -----------------------------
renderChecksumLedger([647.578, 647.578]);
html = el("heartLedger").innerHTML;
console.assert(html.includes("not moving"), "two equal aggregates must be called out");
console.assert(html.includes("s-failed"), "and marked as a failure, not a warning");

// A difference below the printed precision is still not movement.
renderChecksumLedger([647.5780000001, 647.5780000002]);
console.assert(el("heartLedger").innerHTML.includes("not moving"),
  "a change below six decimals is rounding, not learning");

// ---- per class: the cold-start four are marked, and a hole is not a zero ----
await loadMeasurements();          // fetch throws, so the table is empty on purpose
el("perClass").innerHTML = "";
const rows = [
  { round: 1, per_class: [
      { class_id: 2, name: "car", AP50: 0.61, instances: 9000 },
      { class_id: 1, name: "rider", AP50: 0.02, instances: 41 }] },
  { round: 2, per_class: [
      { class_id: 2, name: "car", AP50: 0.66, instances: 9000 },
      { class_id: 1, name: "rider", AP50: 0.02, instances: null }] },
];
renderPerClass(rows);
html = el("perClass").innerHTML;
console.assert(html.includes("car") && html.includes("rider"), "both classes render");
console.assert(html.includes("not measured"),
  "a null instance count must say so, not print 0");
console.assert(!/>0<|>0 inst</.test(html), "nothing may print a bare zero for a hole");
console.assert(html.includes("+0.050"), `car's delta is missing: ${html}`);
console.assert(!el("perClassPanel").hidden, "the panel must be shown when there is data");

// A holdout with no per-class breakdown hides the panel rather than showing an
// empty one — the panel's whole claim is about which classes are weak.
renderPerClass([{ round: 1, per_class: [] }]);
console.assert(el("perClassPanel").hidden, "no per-class data means no per-class panel");

console.log("insight panels OK");
