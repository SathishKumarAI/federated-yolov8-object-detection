// Every dashboard module, imported, against a DOM that answers but holds nothing.
//
// There is no build step here, so nothing checks that a module parses or that its
// imports resolve until a browser loads it — and a module that throws at import time
// takes the whole page with it, silently: the panels simply never fill in. This is the
// cheapest check that catches it, and it runs wherever node does.
//
// The list comes from the directory rather than from a literal, so a module added
// without a check of its own is still at least required to load.
//
// It deliberately does NOT assert what anything renders. The panels' behaviour is
// checked by the other files here; this one answers "does the page come up at all".
import { readdirSync } from "node:fs";

const els = new Map();
const make = (id) => ({
  id, innerHTML: "", textContent: "", value: "", checked: false, hidden: false,
  className: "", title: "", disabled: false, href: "",
  style: {}, dataset: {}, classList: { contains: () => false, add() {}, remove() {} },
  viewBox: { baseVal: { width: 900, height: 220 } },
  setAttribute() {}, getAttribute: () => null, removeAttribute() {},
  querySelector: () => null, querySelectorAll: () => [],
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 220 }),
  insertAdjacentHTML() {}, appendChild() {}, removeChild() {}, remove() {},
  addEventListener() {}, focus() {}, childNodes: [], scrollTop: 0, scrollHeight: 0,
});
const el = (id) => { if (!els.has(id)) els.set(id, make(id)); return els.get(id); };

globalThis.document = {
  getElementById: el,
  querySelector: () => el("stub"),
  querySelectorAll: () => [],
  createElement: () => make("created"),
  createRange: () => ({ selectNodeContents() {} }),
  addEventListener() {},
  body: el("body"),
};
globalThis.window = {
  innerWidth: 1440,
  getSelection: () => ({ removeAllRanges() {}, addRange() {} }),
  matchMedia: () => ({ matches: false, addEventListener() {} }),
};
globalThis.performance = { now: () => 0 };
globalThis.EventSource = class { addEventListener() {} close() {} };
globalThis.fetch = async () => ({ json: async () => ({}), ok: true });
// node ships its own read-only `navigator`, so the clipboard is grafted onto it
// rather than assigned over it.
Object.defineProperty(globalThis.navigator, "clipboard",
  { value: { writeText: async () => {} }, configurable: true });

const names = readdirSync(new URL(".", import.meta.url))
  .filter(f => f.endsWith(".js") && f !== "main.js")
  .sort();
console.assert(names.length >= 14, `only found ${names.length} modules; is the copy wrong?`);

let loaded = 0;
for (const name of names) {
  try {
    await import("./" + name);
    loaded++;
  } catch (e) {
    console.assert(false, `${name} failed to import: ${e && e.stack}`);
  }
}
console.assert(loaded === names.length, `${loaded} of ${names.length} modules imported`);

// main.js last and separately: it is the only module with side effects at import, so a
// failure here means the page wires itself up wrong rather than merely parsing wrong.
// It also starts timers on purpose, which is why this file exits explicitly instead of
// waiting for an event loop that will never drain.
try {
  await import("./main.js");
} catch (e) {
  console.assert(false, `main.js failed to wire the page up: ${e && e.stack}`);
}

console.log(`modules load OK — ${loaded} modules + main.js`);
process.exit(0);
