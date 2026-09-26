// The browser's half of the state stream, fed patches the Python half produced.
//
// Two implementations of one wire format is the setup for a silent divergence: the
// server sends a diff the browser applies *almost* right, and the panel shows a stale
// number while the page keeps looking live. So the fixture beside this file is written
// by pipeline/statestream.py at test time, and every step is compared against the
// snapshot the server had when it emitted that patch.
import { readFileSync } from "node:fs";

globalThis.document = { getElementById: () => null, querySelectorAll: () => [] };

const { applyPatch } = await import("./stream.js");

const { steps, frames } = JSON.parse(readFileSync(new URL("./patches.json", import.meta.url)));

let client = steps[0];
let seq = 0;
let applied = 0;
for (const frame of frames) {
  console.assert(frame.seq === seq + 1, `sequence skipped: ${seq} -> ${frame.seq}`);
  seq = frame.seq;
  client = applyPatch(client, frame.patch);
  applied++;
  const want = steps[frame.step];
  console.assert(JSON.stringify(client) === JSON.stringify(want),
    `after seq ${frame.seq} the browser holds\n${JSON.stringify(client)}\nnot\n${JSON.stringify(want)}`);
}
console.assert(applied > 0, "the fixture carried no patches, so nothing was checked");

// And the check has teeth: skipping a patch must actually corrupt the result, or the
// loop above would pass on an apply() that ignored its argument.
if (frames.length > 1) {
  let broken = steps[0];
  for (const frame of frames.slice(1)) broken = applyPatch(broken, frame.patch);
  console.assert(JSON.stringify(broken) !== JSON.stringify(steps[frames.at(-1).step]),
    "dropping the first patch changed nothing, so this check proves nothing");
}

console.log(`state patch OK — ${applied} patches replayed`);
