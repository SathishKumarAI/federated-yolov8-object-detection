// The Weights tab's panels, against a fake DOM.
//
// Two payloads, because they are the two states that matter:
//
//   * a whole one, through every panel -- one `$("typo")` in a render path throws, the
//     pass dies, and the symptom is a tab that renders its headings and nothing else;
//   * an unavailable one, which must produce the server's reason. A checkpoint-less
//     checkout showing "0 arrays, 0 numbers, 0 MiB" would be this repo's signature bug
//     shipped inside the page built to explain it.
//
// The payload here is HAND-BUILT and small on purpose: the Python side owns whether a
// real checkpoint's numbers add up (test_a_real_checkpoints_numbers_add_up), and this
// side owns whether they reach the screen. Its shape is `pipeline/anatomy.payload()`.
import { readFileSync } from "node:fs";

const FACTS = JSON.parse(readFileSync(new URL("./facts.json", import.meta.url)));

const els = new Map();
const make = (id) => ({
  id, innerHTML: "", textContent: "", hidden: false, className: "", title: "",
  style: {}, dataset: {}, attrs: {},
  viewBox: { baseVal: { width: 900, height: 220 } },
  setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k] ?? null; },
  removeAttribute(k) { delete this.attrs[k]; },
  querySelector: () => null, querySelectorAll: () => [],
  addEventListener() {}, focus() {}, closest() { return this; },
});
const el = (id) => { if (!els.has(id)) els.set(id, make(id)); return els.get(id); };
globalThis.document = {
  getElementById: el, querySelector: () => el("stub"), querySelectorAll: () => [],
  createElement: () => make("x"), addEventListener() {}, body: el("body"),
};
globalThis.window = { innerWidth: 1440, matchMedia: () => ({ matches: true }) };

const PAYLOAD = {
  available: true,
  checkpoint: { name: "global_round_2.pt", bytes: 22486245, mtime: 1790000000,
                dir: "my-project/checkpoints" },
  model: {
    tensors: 355, values: 11160720, file_tensor_bytes: 22321782, wire_bytes: 44643108,
    dtypes: { "torch.float16": 298, "torch.int64": 57 },
    slot: "model", ema_slot_empty: true, ultralytics_version: "8.4.115",
    written: "2026-09-26T13:50:10", names_are_indices: true,
    kinds: [
      { key: "bn_counter", label: "BatchNorm step counters", what: "One integer per layer.",
        tensors: 57, values: 57, file_bytes: 456, wire_bytes: 456,
        example: "model.0.bn.num_batches_tracked", shape: [] },
      { key: "bn_stats", label: "BatchNorm running statistics", what: "Measured, not learned.",
        tensors: 114, values: 20032, file_bytes: 40064, wire_bytes: 80128,
        example: "model.0.bn.running_mean", shape: [32] },
      { key: "bn_affine", label: "BatchNorm scale and shift", what: "Two per channel.",
        tensors: 114, values: 20032, file_bytes: 40064, wire_bytes: 80128,
        example: "model.0.bn.weight", shape: [32] },
      { key: "conv", label: "Convolution kernels", what: "The learned filters.",
        tensors: 64, values: 11120368, file_bytes: 22240736, wire_bytes: 44481472,
        example: "model.0.conv.weight", shape: [32, 3, 3, 3] },
      { key: "head_cls_bias", label: "Detection head class biases", what: "One per class.",
        tensors: 3, values: 39, file_bytes: 78, wire_bytes: 156,
        example: "model.22.cv3.0.2.bias", shape: [13] },
      { key: "head_box_bias", label: "Detection head box biases", what: "For the boxes.",
        tensors: 3, values: 192, file_bytes: 384, wire_bytes: 768,
        example: "model.22.cv2.0.2.bias", shape: [64] },
    ],
    first_conv: {
      key: "model.0.conv.weight", shape: [2, 3, 3, 3], values: 54,
      kernels: [
        { min: -0.19763, max: 0.15515,
          rgb: Array.from({ length: 9 }, (_, i) => [i / 9, 0.5, 1 - i / 9]) },
        { min: -0.4868, max: 0.4468,
          rgb: Array.from({ length: 9 }, () => [0.1, 0.2, 0.3]) },
      ],
    },
    batchnorm: {
      layers: 57,
      running_mean: { lo: -4.73, hi: 4.21, counts: [3, 161, 7059, 97, 5],
                      edges: [-4.73, -2.94, -1.15, 0.64, 2.43, 4.21] },
      running_var: { lo: 0.0002, hi: 5.875, counts: [9165, 626, 109, 38, 17],
                     edges: [0.0002, 1.17, 2.35, 3.52, 4.7, 5.875] },
      counter_values: [0], counters_all_zero: true,
    },
    head: {
      classes: FACTS.classes.all,
      cold_start: FACTS.classes.unwarmed,
      scales: [
        { key: "model.22.cv3.0.2.bias",
          values: FACTS.classes.all.map((n, i) =>
            FACTS.classes.unwarmed.includes(n) ? -9.7265625 : -6.5 - i / 10),
          cold_values: Object.fromEntries(
            FACTS.classes.unwarmed.map(n => [n, -9.7265625])) },
      ],
    },
  },
  exchange: { down_per_vehicle: 44643108, up_per_vehicle: 44643116,
              per_round: 535717344, total: 3214304064, vehicles: 6, rounds: 6,
              arrays: 355, images_moved: 0,
              excluded: "gRPC framing, per-array protobuf headers and TLS." },
  roundtrip: {
    available: true, ok: true, tolerance: 1e-6,
    rounds: [
      { round: 1, aggregate: -1032.539594, weighted_mean: -1032.539594,
        deviation: 1.2e-12, clients: 2, num_examples: [1400, 700],
        sent: [-1000.1, -1097.4], received: [500.0, 500.0] },
      { round: 2, aggregate: -2646.913426, weighted_mean: -2600.0,
        deviation: 1.8e-2, clients: 2, num_examples: [1400, 700],
        sent: [-2500.0, -2800.0], received: [-1032.539594, -1032.539594] },
    ],
  },
};

const STATE = {
  demo: false, busy: true,
  fleet: [{ vid: 1, condition: "rain / fog" }, { vid: 2, condition: "night" }],
  live: {
    rounds_done: 1,
    per_vehicle: {
      1: { rounds: 2, received: -1032.5395936, sent: -2000.125, device: "cuda:0",
           training: true },
      2: { rounds: 2, received: -1032.5395936, sent: -1032.5395936, device: "cuda:0" },
      3: { rounds: 1, received: null, sent: null, device: null },
    },
    training_now: "1",
  },
};

let payload = PAYLOAD;
globalThis.fetch = async (url) => ({
  ok: true,
  json: async () => String(url).includes("measurements") ? FACTS : payload,
});

const { loadMeasurements } = await import("./insight.js");
const { loadAnatomy, renderExchange } = await import("./anatomy.js");
await loadMeasurements();

// ---------------------------------------------------------------- a whole payload
await loadAnatomy();

const totals = el("anTotals").innerHTML;
console.assert(totals.includes("355"), `the array count is missing: ${totals}`);
console.assert(totals.includes("11,160,720"), `the value count is missing: ${totals}`);
console.assert(/images that move/.test(totals), "the page must say images do not move");

const prov = el("anProvenance").innerHTML;
console.assert(prov.includes("global_round_2.pt"), `the file is not named: ${prov}`);
console.assert(prov.includes("float16"), "the file's precision must be stated");
console.assert(prov.includes("8.4.115"), "the writer's version must be stated");
console.assert(prov.includes("ema</code> entry is empty"),
  "an empty ema slot is a measured fact about the file and must be said");

// The counter-intuitive pair, computed from the payload rather than typed into it.
const kindsNote = el("anKindsNote").innerHTML;
console.assert(kindsNote.includes("285 of the 355"), `BatchNorm array count: ${kindsNote}`);
console.assert(kindsNote.includes("80.3%"), `BatchNorm share of arrays: ${kindsNote}`);
console.assert(kindsNote.includes("0.359%"), `BatchNorm share of numbers: ${kindsNote}`);
console.assert(el("anKinds").innerHTML.includes("Convolution kernels"),
  "the breakdown table did not render");
console.assert((el("anShares").innerHTML.match(/<i /g) || []).length === 12,
  "two stacked bars of six kinds each");

// The kernels: one square per kernel, nine cells each, and a text alternative.
const kern = el("anKernels").innerHTML;
console.assert((kern.match(/<figure class="kern"/g) || []).length === 2,
  `expected 2 kernels rendered: ${kern.slice(0, 120)}`);
console.assert((kern.match(/background:rgb\(/g) || []).length === 18,
  "nine coloured cells per kernel");
console.assert(el("anKernels").getAttribute("aria-label").includes("colour image"),
  "the picture needs a label for a reader who cannot see it");
console.assert(el("anConvNote").innerHTML.includes("-0.19763"),
  "the real weight range must be readable as text, not only as colour");
console.assert(el("anConvNote").innerHTML.includes("scaled to its own"),
  "per-kernel scaling changes what the picture means and must be disclosed");

// BatchNorm: both distributions, and the zeroed counters said out loud.
console.assert(el("anRunningMean").innerHTML.includes("<rect"), "no running-mean histogram");
console.assert(el("anRunningVar").innerHTML.includes("<rect"), "no running-variance histogram");
console.assert(el("anRunningMeanCap").innerHTML.includes("7,325") ||
  el("anRunningMeanCap").innerHTML.includes("values"), "the histogram needs its count");
console.assert(el("anBnNote").innerHTML.includes("counter in this file is 0"),
  `the zeroed counters must be reported: ${el("anBnNote").innerHTML}`);

// The head: thirteen names, and the cold-start four marked as more than a colour.
const head = el("anHead").innerHTML;
for (const name of FACTS.classes.all) {
  console.assert(head.includes(name), `class ${name} is missing from the head table`);
}
console.assert((head.match(/>cold</g) || []).length === FACTS.classes.unwarmed.length,
  "every cold-start class must be marked in words, not by colour alone");
console.assert(el("anHeadNote").innerHTML.includes("still hold exactly"),
  "the shared initialisation is the point of this panel");
console.assert(el("anHeadNote").innerHTML.includes("measurements.py"),
  "the class names' provenance must be stated: they are not from the checkpoint");

// The wire, and the honest limit of what it proves.
const wire = el("anWire").innerHTML;
console.assert(wire.includes("num_examples"), "FedAvg's weight travels and must be named");
console.assert(wire.includes("42.6"), `one direction in MiB: ${wire}`);
console.assert(el("anWireNote").innerHTML.includes("not the same as privacy"),
  "overclaiming privacy is the one thing this panel must not do");
console.assert(/gradients|reconstruct/.test(el("anWireNote").innerHTML),
  "gradient leakage is real and must be named rather than implied away");

// The identity: agreement and disagreement must read differently.
const rt = el("anRoundtrip").innerHTML;
console.assert(rt.includes("is the mean"), "a round inside tolerance must say so");
console.assert(rt.includes("mismatch"), "a round outside tolerance must NOT read as a pass");
console.assert(rt.includes("1,400") && rt.includes("2,100"),
  `the FedAvg weights and their total must be shown: ${rt}`);
console.assert(el("anRoundtripNote").innerHTML.includes("roundtrip.py"),
  "the arithmetic belongs to roundtrip.py and the page must say so");

// ------------------------------------------------------------------- the live half
renderExchange(STATE);
const live = el("anLive").innerHTML;
console.assert(live.includes("rain / fog"), "the live table names the vehicle's condition");
console.assert(live.includes(">trained<"), "a vehicle whose checksum moved reads as trained");
console.assert(live.includes(">identical<"),
  "a vehicle that returned what it was sent is the B4 failure and must be flagged");
console.assert(live.includes("not measured"),
  "a vehicle with no checksum line yet must not render as a number");
console.assert(el("anLiveNote").innerHTML.includes("averaged its own input"),
  "the note must say what identical checksums mean");

// A replay says so, in the panel, not only in the banner.
renderExchange({ ...STATE, demo: true });
console.assert(el("anLiveNote").innerHTML.includes("This is a replay"),
  "a recorded run must be labelled as one");
console.assert(el("anLiveWhen").textContent.includes("replay"),
  "the panel's own subtitle must say it is a replay");
console.assert(el("anReplayNote").hidden === false &&
  el("anReplayNote").innerHTML.includes("contains no model"),
  "a replay carries no checkpoint, so the model on screen is a different run's and " +
  "the page must say so");
renderExchange(STATE);
console.assert(el("anReplayNote").hidden === true,
  "and the warning must go away again when the replay stops");

// ----------------------------------------------------------------- no checkpoint
payload = { available: false,
            reason: "no global_*.pt in checkpoints/ -- run the federate stage",
            roundtrip: { available: false, reason: "no run log records an aggregation" } };
await loadAnatomy();
console.assert(el("anKinds").innerHTML.includes("run the federate stage"),
  `the reason must be shown: ${el("anKinds").innerHTML}`);
console.assert(el("anTotals").innerHTML === "",
  `absent numbers must be absent, not zero: ${el("anTotals").innerHTML}`);
console.assert(!/\b0 MiB\b/.test(el("anWire").innerHTML + el("anTotals").innerHTML),
  "nothing here may render as a zero measurement");
console.assert(el("anWhen").textContent === "no checkpoint",
  "the panel heading must say there is no file");
console.assert(el("anRoundtrip").innerHTML.includes("no run log records an aggregation"),
  "the identity's own reason must survive too");

console.log("anatomy panels OK — a whole payload, a replay, and a missing checkpoint");
process.exit(0);
