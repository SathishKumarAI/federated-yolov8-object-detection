// The run form's levers: what it posts, and what it refuses to post.
//
// The failure this guards is the quietest one available. A body field the running server
// has never heard of is dropped without a word, so a form that offered `server_ema` to a
// server with no such Config field would report a started run and silently train without
// it. The control has to be disabled and say so, and the field has to be left out.
const els = new Map();
const boxes = [];
const make = (id) => ({
  id, value: "0", checked: false, disabled: false, title: "", innerHTML: "",
  textContent: "", hidden: false, dataset: {}, style: {},
  setAttribute() {}, getAttribute: () => null,
  querySelectorAll: () => [], closest() { return this; },
});
const el = (id) => { if (!els.has(id)) els.set(id, make(id)); return els.get(id); };

// One box per lever, the way index.html wraps them.
for (const name of ["freeze_round1", "server_ema", "fix_bn_from_round", "imgsz", "local_bn"]) {
  const inputs = [];
  boxes.push({ dataset: { lever: name }, inputs,
               querySelectorAll: () => inputs });
}
const byLever = (name) => boxes.find(b => b.dataset.lever === name);
byLever("freeze_round1").inputs.push(el("freezeRound1"));
byLever("server_ema").inputs.push(el("serverEma"));
byLever("fix_bn_from_round").inputs.push(el("fixBnFromRound"));
byLever("imgsz").inputs.push(el("imgsz"));
byLever("local_bn").inputs.push(el("localBn"));

globalThis.document = {
  getElementById: el,
  querySelectorAll: (sel) => sel.includes(".lever") ? boxes : [],
  addEventListener() {},
};
globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({ records: [], observed_spread: null,
                       lever_notes: { server_ema: "A bias-corrected EMA **of the aggregate**." },
                       classes: { all: [], warm_started: [], unwarmed: [] } }),
});

const { config, renderOptions } = await import("./control.js");

// Defaults the form starts at: every lever off, and an off lever is not in the body.
el("profile").value = "full";
el("vehicles").value = "6";
el("rounds").value = "6";
el("epochs").value = "4";
el("seed").value = "0";
el("partition").value = "condition";
el("alpha").value = "0.5";
el("sizeSkew").value = "0";
el("strategy").value = "fedavg";
el("mu").value = "0";

let body = config();
for (const k of ["freeze_round1", "server_ema", "fix_bn_from_round", "imgsz", "local_bn"]) {
  console.assert(!(k in body), `an unset lever must not be in the body, found ${k}`);
}

// A server that implements all of them: set levers travel.
renderOptions({ strategies: ["fedavg"], levers: {
  freeze_round1: true, server_ema: true, fix_bn_from_round: true, imgsz: true, local_bn: true } });
el("freezeRound1").value = "10";
el("serverEma").value = "0.9";
el("fixBnFromRound").value = "3";
el("imgsz").value = "1024";
el("localBn").checked = true;
body = config();
console.assert(body.freeze_round1 === 10, `freeze_round1: ${body.freeze_round1}`);
console.assert(body.server_ema === 0.9, `server_ema: ${body.server_ema}`);
console.assert(body.fix_bn_from_round === 3, `fix_bn_from_round: ${body.fix_bn_from_round}`);
console.assert(body.imgsz === 1024, `imgsz: ${body.imgsz}`);
console.assert(body.local_bn === true, `local_bn: ${body.local_bn}`);
console.assert(boxes.every(b => b.dataset.unsupported === "false"),
  "every lever should be marked supported");

// A server that implements none of them: nothing is posted, and every control says why.
renderOptions({ strategies: ["fedavg"], levers: {
  freeze_round1: false, server_ema: false, fix_bn_from_round: false, imgsz: false,
  local_bn: false } });
body = config();
for (const k of ["freeze_round1", "server_ema", "fix_bn_from_round", "imgsz", "local_bn"]) {
  console.assert(!(k in body), `${k} was posted to a server that cannot apply it`);
}
console.assert(boxes.every(b => b.dataset.unsupported === "true"),
  "an unimplemented lever must be marked, not silently accepted");
console.assert(el("serverEma").disabled === true, "an unimplemented lever must be disabled");
console.assert(el("serverEma").title.includes("not sent"),
  `the control must say it is not sent: ${el("serverEma").title}`);
console.assert(el("leverSupport").textContent.includes("0 of 5"),
  `the header must count them: ${el("leverSupport").textContent}`);

// The two readings that are wrong in a way the value alone cannot show.
renderOptions({ strategies: ["fedavg"], levers: {
  freeze_round1: true, server_ema: true, fix_bn_from_round: true, imgsz: true, local_bn: true } });
el("fixBnFromRound").value = "1";
el("localBn").checked = true;
el("imgsz").value = "1024";
config();
const { estimate } = await import("./control.js");
estimate();
const warned = el("leverWarnings").innerHTML;
console.assert(warned.includes("FixBN from round 1 is not FixBN"),
  `R=1 must be called out: ${warned}`);
console.assert(warned.includes("no single global model"),
  "FedBN must say it invalidates the holdout score");
console.assert(warned.includes("never been timed"),
  "an untimed resolution must say so");

el("fixBnFromRound").value = "99";
estimate();
console.assert(el("leverWarnings").innerHTML.includes("never triggers"),
  "FixBN past the last round must say it never fires");

console.log("run levers OK");
process.exit(0);
