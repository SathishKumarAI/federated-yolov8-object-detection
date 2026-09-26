// Entry point: wires the tabs, the control form, the chart cursor, and starts
// the two data sources (one SSE stream for state diffs, one for the run log).
import { $ } from "./util.js";
import { enableChartCursor } from "./chart.js";
import { wireControl } from "./control.js";
import { applyState, connectEvents, drawHeartbeat } from "./live.js";
import { connectState } from "./stream.js";
import { renderFleet } from "./fleet.js";
import { loadData } from "./data.js";
import { loadPlan } from "./plan.js";
import { loadDocs } from "./docs.js";
import { loadMetrics } from "./metrics.js";
import { loadAnatomy } from "./anatomy.js";
import { pollEdge } from "./edge.js";
import { loadMeasurements, loadProfile, renderProvenance } from "./insight.js";
import { loadSimulation, wireSimulate } from "./simulate.js";
import { wireTour, offerTour } from "./tour.js";

const VIEWS = ["control", "live", "data", "weights", "metrics", "plan", "simulate", "docs"];

function showView(tab) {
  const want = tab.dataset.view;
  document.querySelectorAll(".tab").forEach(x => x.setAttribute("aria-selected", String(x === tab)));
  VIEWS.forEach(v => { $("view-" + v).hidden = v !== want; });
  // Charts sized while hidden come out wrong; redraw whatever just became visible.
  drawHeartbeat();
  renderFleet();
  // Data and Plan are fetched on first sight rather than on every poll: one costs
  // seconds of label reading, the other is only interesting when someone looks.
  if (want === "data") loadData(false);
  if (want === "plan") loadPlan();
  if (want === "simulate") loadSimulation();
  if (want === "metrics") loadMetrics(true);
  // The checkpoint read is a 22 MB torch load on a cache miss, so it happens on
  // first sight of the tab. After that the state stream refetches it per round.
  if (want === "weights") loadAnatomy();
  if (want === "docs") { renderProvenance(); loadDocs(); }
  // The profile reads every client log end to end, so it is loaded on first sight of
  // the Live tab and then only when asked -- never on a state tick.
  if (want === "live" && !profileLoaded) { profileLoaded = true; loadProfile(); }
}
let profileLoaded = false;
document.querySelectorAll(".tab").forEach(t => { t.onclick = () => showView(t); });
$("profileRefresh").onclick = () => loadProfile();

wireControl(() => showView(document.querySelector('.tab[data-view=live]')));
wireSimulate();
wireTour();
enableChartCursor();
connectEvents();
// The measurement table first: every panel that prints a +/- or cites a source reads
// it, and a panel that rendered before it arrived would quietly print nothing where a
// provenance line belongs.
loadMeasurements().then(() => connectState((s) => { applyState(s); offerTour(); }));
// Its own loop, not part of poll(): edge nodes are live whether or not a federation
// is running, and /api/nodes must not be coupled to the run-state snapshot.
pollEdge();
