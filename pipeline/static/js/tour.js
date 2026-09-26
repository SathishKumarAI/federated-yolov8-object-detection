// The guided walkthrough: what each panel is for, and the one mistake it prevents.
//
// It works on a clean checkout with no GPU and no data, because a dashboard whose panels
// are all empty teaches nobody anything. If a real run is on disk it narrates that; if
// not it replays the recording served by `pipeline/demo_run.py`, whose every number is
// transcribed from a document in this repo.
//
// Each step names the mistake its panel exists to prevent, because "what does this show"
// is the less useful question. Most of those mistakes are in CLAUDE.md's silent-failures
// table, and the panels were built in response to them.
import { $, esc } from "./util.js";
import { applyState } from "./live.js";
import { hold, lastState } from "./stream.js";

//: panel id -> {title, for, mistake}. `for` is what it shows; `mistake` is the reading
//: it stops. A step whose mistake is vague is a step that should not exist.
const STEPS = [
  { title: "Where the numbers come from", panel: null,
    body: "Everything on this page is derived from files on disk — the server stores " +
      "nothing. That is why a run launched from a terminal lights up the same panels as " +
      "one launched from the form, and why restarting the server loses nothing.",
    mistake: "Believing the dashboard is the record. It is a reader; the logs, the " +
      "metrics CSV and the checkpoints are the record." },

  { title: "The heartbeat: the aggregate checksum", panel: "heartLedger",
    body: "One checksum of the averaged global weights per round, read from the " +
      "server's own log lines, with the round-to-round difference spelled out.",
    mistake: "Trusting a rising mAP. Twice in this project the rounds completed, the " +
      "metrics looked healthy, and the clients were returning the weights they had been " +
      "sent — FedAvg averaging its own input. Equal consecutive checksums are the only " +
      "signal that catches it, and a chart can make a real change look flat, so the " +
      "subtraction is printed." },

  { title: "Round progress", panel: "rRound",
    body: "Rounds completed, vehicles heard from, the latest self-evaluated mAP and " +
      "loss, and how many global checkpoints exist.",
    mistake: "Reading the client-reported mAP as the model's accuracy. Each client " +
      "scores itself on its own slice, so a night-driving vehicle's number says how " +
      "well it does at night. Scroll to the holdout panel for a number that is " +
      "comparable with anything." },

  { title: "GPU", panel: "gUtil",
    body: "Utilisation, VRAM against the card's 16 303 MiB ceiling, power, energy " +
      "integrated over the run, and temperature — sampled from nvidia-smi.",
    mistake: "Reading an absent reading as a zero. A missing nvidia-smi and an idle " +
      "card look identical on a gauge, and on this machine the difference is whether " +
      "flwr built its own CPU-only environment and every client is training at 5.5× " +
      "the wall clock with no error anywhere. Absent says 'not measured'." },

  { title: "The fleet", panel: "fleet",
    body: "One card per vehicle: the condition it was given, the mAP it reaches on its " +
      "own slice, and how far that is from where it started. Open a card for its " +
      "curves, its weight movement and what its shard actually contains.",
    mistake: "Assuming the label is the data. A condition shard tops up with unrelated " +
      "images once the condition runs out — 'overcast residential' has 1 419 images in " +
      "all of BDD100K — so a fleet asked for more than that per vehicle is closer to " +
      "IID than its labels say. The drawer counts what the shard holds." },

  { title: "The honest global metric", panel: "holdoutChart",
    body: "The global model scored on a holdout no vehicle trained or self-evaluated " +
      "on, per round, with the recorded run-to-run spread drawn as a band around the " +
      "first scored round.",
    mistake: "Calling a delta a result. Any movement inside the band is " +
      "indistinguishable from re-running the same configuration with a different seed. " +
      "This is the only number on the page comparable between two runs." },

  { title: "Per class, on the holdout", panel: "perClassPanel",
    body: "AP50 per class with its instance count, a sparkline across rounds, and a " +
      "marker on the four classes COCO could not warm-start.",
    mistake: "Reading one averaged mAP. `car` is 55.4 % of BDD100K's objects, so a " +
      "detector that found nothing else would still look respectable. If this panel is " +
      "hidden right now, the run being shown has no per-class breakdown recorded — the " +
      "panel hides rather than showing thirteen zeros, because zeros there would read " +
      "as 'the model detects nothing'." },

  { title: "Pass criteria", panel: "criteria",
    body: "The same four checks `python -m pipeline.verify` runs, against the logs on " +
      "disk — so a run started from a terminal lights them up too.",
    mistake: "Treating a finished run as a passing one. A stage can exit 0 and have " +
      "done nothing: Ray exits 0 after an actor dies, and checkpointing once skipped " +
      "every round while the federation ran to completion." },

  { title: "Where the round's seconds went", panel: "profilePanel",
    body: "Seconds per phase, parsed out of timestamps the logs already contain, with a " +
      "verdict on which of two worlds the run was in.",
    mistake: "Acting on mean utilisation. 27 % is equally consistent with clients " +
      "serialised on one card and with a dataloader starving the GPU inside train(), " +
      "and the two have opposite fixes. The overlap count and the share spent inside " +
      "train() tell them apart; the mean cannot." },

  { title: "Simulate, before spending the hours", panel: null, tab: "simulate",
    body: "Project wall clock, energy, image-visits and VRAM for a configuration you " +
      "have not run. Every row names the measurement it rests on.",
    mistake: "Extrapolating quietly. A lever outside the range this project measured is " +
      "refused, with the measurement that would lift the refusal — and no accuracy is " +
      "projected at all, because exactly one configuration's end-to-end result is " +
      "recorded here." },

  { title: "Every number's provenance", panel: "provenance", tab: "docs",
    body: "The measurement table itself: each recorded value, the document that records " +
      "it, the date, and how it misleads. `python -m pipeline.measurements --check` " +
      "re-reads those documents and fails on drift.",
    mistake: "Quoting a number whose source has moved on. A figure on this page with no " +
      "row in that table is a bug." },
];

let step = 0;
let demo = null;
let replaying = false;
let opener = null;

export async function openTour() {
  opener = document.activeElement;
  if (demo == null) {
    try {
      demo = await (await fetch("/api/demo")).json();
    } catch {
      demo = { state: null, sources: {}, absent: {}, live_run_available: false, run: "" };
    }
  }
  step = 0;
  $("tour").hidden = false;
  document.body.dataset.tour = "1";
  render();
  $("tourNext").focus();
}

export function closeTour() {
  $("tour").hidden = true;
  delete document.body.dataset.tour;
  clearHighlight();
  if (replaying) stopReplay();
  if (opener && opener.focus) opener.focus();
}

/** Feed the recorded run through the real panels, and hold the live stream off. */
function startReplay() {
  if (!demo || !demo.state) return;
  replaying = true;
  hold(true);
  applyState(demo.state);
  $("demoBanner").hidden = false;
  $("demoBannerText").innerHTML =
    `<b>Replaying a recorded run.</b> ${esc(demo.run)}. Every number on the page is ` +
    `transcribed from a document in this repository, and several panels say ` +
    `<em>not measured</em> because for this run nobody measured that.`;
}

function stopReplay() {
  replaying = false;
  $("demoBanner").hidden = true;
  hold(false);
}

function clearHighlight() {
  document.querySelectorAll("[data-tour-lit]").forEach(e => {
    e.removeAttribute("data-tour-lit");
  });
}

/** Put the step's panel on screen and ring it. */
function highlight(id) {
  clearHighlight();
  if (!id) return;
  const el = $(id);
  if (!el) return;
  const panel = el.closest ? (el.closest(".panel") || el) : el;
  panel.setAttribute("data-tour-lit", "1");
  panel.scrollIntoView({ block: "center",
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto" : "smooth" });
}

function render() {
  const s = STEPS[step];
  const tab = document.querySelector(`.tab[data-view=${s.tab || "live"}]`);
  if (tab && tab.getAttribute("aria-selected") !== "true") tab.click();

  $("tourTitle").textContent = `${step + 1} of ${STEPS.length} — ${s.title}`;
  $("tourBody").innerHTML =
    `<p>${esc(s.body)}</p>` +
    `<p class="mistake"><span class="lbl">the mistake it prevents</span>${esc(s.mistake)}</p>` +
    (s.panel && demo && demo.absent && absentFor(s.panel)
      ? `<p class="mistake"><span class="lbl">why it looks empty right now</span>` +
        `${esc(absentFor(s.panel))}</p>`
      : "");
  $("tourPrev").disabled = step === 0;
  $("tourNext").textContent = step === STEPS.length - 1 ? "Done" : "Next";
  $("tourDots").textContent = `${step + 1}/${STEPS.length}`;
  highlight(s.panel);
}

/** The recorded run's own explanation of a panel that has nothing to show. */
function absentFor(panel) {
  if (!replaying) return null;
  const map = { perClassPanel: "per_class", holdoutChart: "baseline", gUtil: "gpu",
                rRound: "metrics" };
  return demo.absent[map[panel]] || null;
}

const move = (d) => {
  const next = step + d;
  if (next < 0) return;
  if (next >= STEPS.length) return closeTour();
  step = next;
  render();
};

export function wireTour() {
  $("helpBtn").onclick = openTour;
  $("tourNext").onclick = () => move(1);
  $("tourPrev").onclick = () => move(-1);
  $("tourClose").onclick = closeTour;
  $("tourReplay").onclick = () => {
    if (replaying) { stopReplay(); $("tourReplay").textContent = "Replay a recorded run"; }
    else { startReplay(); $("tourReplay").textContent = "Stop replaying"; }
  };

  // Keyboard is the whole navigation, not an alternative to it: arrows step, Escape
  // leaves, and Tab is trapped inside the dialog so focus cannot wander into a page
  // the reader cannot see past the scrim.
  document.addEventListener("keydown", (e) => {
    if ($("tour").hidden) {
      if (e.key === "?" && !/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName)) {
        e.preventDefault();
        openTour();
      }
      return;
    }
    if (e.key === "Escape") { e.preventDefault(); return closeTour(); }
    if (e.key === "ArrowRight") { e.preventDefault(); return move(1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); return move(-1); }
    if (e.key !== "Tab") return;
    const stops = ["tourClose", "tourReplay", "tourPrev", "tourNext"].map($).filter(Boolean);
    const i = stops.indexOf(document.activeElement);
    e.preventDefault();
    stops[(i + (e.shiftKey ? -1 : 1) + stops.length) % stops.length].focus();
  });
}

/** Live data beats a recording. Offered, never forced -- pressing nothing changes nothing. */
export async function offerTour() {
  const state = lastState();
  const empty = !state || !(state.live && state.live.checksums
                            && state.live.checksums.length);
  $("helpBtn").textContent = empty ? "Demo & help" : "Help";
  $("helpBtn").title = empty
    ? "No run has produced an aggregate yet, so the walkthrough can replay a recorded "
      + "one. Keyboard: ? to open."
    : "A guided walkthrough of this page, narrating the run you already have. "
      + "Keyboard: ? to open.";
}
