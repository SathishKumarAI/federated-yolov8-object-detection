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
import { pollEdge } from "./edge.js";

const VIEWS = ["control", "live", "data", "metrics", "plan", "docs"];

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
  if (want === "metrics") loadMetrics(true);
  if (want === "docs") loadDocs();
}
document.querySelectorAll(".tab").forEach(t => { t.onclick = () => showView(t); });

wireControl(() => showView(document.querySelector('.tab[data-view=live]')));
enableChartCursor();
connectEvents();
// State arrives as a snapshot then numbered diffs; polling is only the fallback
// stream.js falls back to when /api/stream never connects.
connectState(applyState);
// Its own loop, not part of poll(): edge nodes are live whether or not a federation
// is running, and /api/nodes must not be coupled to the run-state snapshot.
pollEdge();
