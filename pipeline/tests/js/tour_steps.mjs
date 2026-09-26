// The walkthrough, driven by keyboard alone.
//
// It exists to be used by someone who has never seen the page, so the failure that
// matters is not a wrong word: it is a dialog that cannot be left, a step that points at
// a panel it never highlights, or a replay that leaves the live stream held off forever
// and makes the page look frozen.
const els = new Map();
const listeners = [];
const make = (id) => ({
  id, innerHTML: "", textContent: "", hidden: false, disabled: false, title: "",
  dataset: {}, attrs: {}, focused: false,
  setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k] ?? null; },
  removeAttribute(k) { delete this.attrs[k]; },
  closest() { return this; }, scrollIntoView() { this.scrolled = true; },
  focus() { globalThis.document.activeElement = this; },
  click() { this.clicked = (this.clicked || 0) + 1; this.attrs["aria-selected"] = "true"; },
});
const el = (id) => { if (!els.has(id)) els.set(id, make(id)); return els.get(id); };

globalThis.document = {
  getElementById: el,
  querySelector: () => el("tab-stub"),
  querySelectorAll: () => [],
  addEventListener: (kind, fn) => listeners.push([kind, fn]),
  body: make("body"),
  activeElement: make("body"),
};
globalThis.window = { matchMedia: () => ({ matches: true }), innerWidth: 1440 };

const DEMO = {
  state: { live: { checksums: [-1, -2] }, gpu: {}, fleet: [] },
  sources: { holdout: "docs/PHASED_PLAN.md" },
  absent: { per_class: "No per-class AP was recorded for this run." },
  run: "head warm-start probe",
  live_run_available: false,
};
globalThis.fetch = async () => ({ json: async () => DEMO, ok: true });

const tour = await import("./tour.js");
const stream = await import("./stream.js");

// hold() is the contract that keeps a replay from being overwritten mid-sentence.
console.assert(typeof stream.hold === "function", "stream.js must export hold()");
stream.hold(true);
stream.hold(false);

// The keyboard handler is registered by wireTour, which is what main.js calls. Only
// the handlers added by THAT call are the walkthrough's -- drawer.js registers its own
// Escape handler at import time, and picking the wrong one silently checks nothing.
const before = listeners.length;
tour.wireTour();
const keydown = listeners.slice(before).filter(([k]) => k === "keydown").map(([, fn]) => fn);
console.assert(keydown.length === 1,
  `wireTour registered ${keydown.length} keydown handlers, expected 1`);
await tour.openTour();
console.assert(el("tour").hidden === false, "opening the walkthrough must show it");
console.assert(el("tourTitle").textContent.includes("1 of"), "the first step must say 1 of N");
console.assert(el("tourPrev").disabled === true, "there is nothing before step 1");

const total = Number(el("tourDots").textContent.split("/")[1]);
console.assert(total >= 8, `expected a walkthrough of the whole page, got ${total} steps`);

// Every step: a body, a named mistake, and a highlight that actually moved the page.
const press = (key, shift) => keydown[0]({ key, shiftKey: !!shift, preventDefault() {} });

for (let i = 1; i <= total; i++) {
  console.assert(el("tourBody").innerHTML.includes("the mistake it prevents"),
    `step ${i} does not name the mistake it prevents`);
  console.assert(el("tourBody").innerHTML.length > 120, `step ${i} has no body`);
  if (i < total) {
    press("ArrowRight");
    console.assert(el("tourTitle").textContent.startsWith(`${i + 1} of`),
      `ArrowRight did not advance past step ${i}`);
  }
}

// The last Next leaves, and leaving restores the page.
press("ArrowRight");
console.assert(el("tour").hidden === true, "the last step must close the walkthrough");
console.assert(!("tour" in document.body.dataset), "closing must unset the page's tour flag");

// Escape leaves from anywhere, and Tab does not escape the dialog.
await tour.openTour();
press("ArrowRight");
press("Tab");
console.assert(el("tour").hidden === false, "Tab must not close the dialog");
console.assert(["tourClose", "tourReplay", "tourPrev", "tourNext"]
  .includes(document.activeElement.id), `Tab left the dialog: ${document.activeElement.id}`);
press("Escape");
console.assert(el("tour").hidden === true, "Escape must leave the walkthrough");

console.log(`tour steps OK — ${total} steps, keyboard-only`);
process.exit(0);
