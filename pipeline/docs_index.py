"""What each program in this package does, read from the programs themselves.

A hand-written "here is what the modules do" page starts accurate and rots. This
reads the module docstrings, the stage chain and the docs directory at request time,
so the page cannot describe a module that has changed underneath it.

What it adds on top of the docstrings is the part a docstring cannot know: which of
this project's numbers a module contributes to, and which dashboard tab shows its
output.

    python -m pipeline.docs_index
"""
from __future__ import annotations

import argparse
import importlib
import json

from . import paths, stages

#: module -> what it contributes and where its output surfaces. The docstring says
#: what a module does; this says why the project has it.
ROLES: dict[str, dict] = {
    "runner": {
        "command": "python -m pipeline.runner --all --yes",
        "contributes": "Sequences the stages, halts on the first failure, streams the "
                       "log, and writes the run report. Nothing continues past a "
                       "failed stage — this project shipped silent no-ops before it did.",
        "tab": "Live",
    },
    "stages": {
        "command": "python -m pipeline.runner --list",
        "contributes": "Defines what exists, what counts as already done, and what "
                       "costs enough to need confirmation. The 'skip' decisions are "
                       "what make a re-run cheap.",
        "tab": "Control, Plan",
    },
    "vehicles": {
        "command": "python -m pipeline.build_fleet --vehicles 6 --per-vehicle 1400",
        "contributes": "Gives each vehicle a disjoint, condition-biased slice. The "
                       "divergence between vehicles is what makes the run federated "
                       "rather than merely distributed.",
        "tab": "Data, Live",
    },
    "holdout": {
        "command": "python -m pipeline.holdout --build --evaluate",
        "contributes": "The only honest global metric. Every other number is a client "
                       "scoring itself on its own distribution, so only this one can "
                       "be compared between runs.",
        "tab": "Live, Data",
    },
    "baseline": {
        "command": "python -m pipeline.baseline --rounds 6 --local-epochs 4",
        "contributes": "The ceiling the federated number is measured against, at a "
                       "matched budget. Without it 0.4334 mAP50 has no scale.",
        "tab": "Live, Plan",
    },
    "validate": {
        "command": "python -m pipeline.validate",
        "contributes": "Six ways a fleet can be quietly wrong: missing labels, "
                       "cross-shard leakage, train/val leakage, holdout containment. "
                       "Nothing measured on bad shards is worth reading.",
        "tab": "Data",
    },
    "verify": {
        "command": "python -m pipeline.verify",
        "contributes": "The four pass criteria, chief among them that the aggregate "
                       "checksum moves between rounds. Equal consecutive values mean "
                       "nothing is being learned, whatever the metrics say.",
        "tab": "Live",
    },
    "logparse": {
        "command": "—",
        "contributes": "Turns log lines into facts: checksums, which vehicle trained, "
                       "rounds that took no optimizer step. Every live panel is "
                       "downstream of this, which is why it reads one run's log.",
        "tab": "Live",
    },
    "vehicle_metrics": {
        "command": "—",
        "contributes": "Per-vehicle curves, divergence from the fleet mean, and each "
                       "vehicle's share of the aggregate — num_examples is FedAvg's "
                       "weight, so share is influence.",
        "tab": "Live",
    },
    "dataset_stats": {
        "command": "python -m pipeline.dataset_stats",
        "contributes": "Counts the classes and conditions actually present. car is "
                       "55% of BDD100K; a detector that predicted nothing else would "
                       "still look respectable on averaged mAP.",
        "tab": "Data",
    },
    "plan": {
        "command": "python -m pipeline.plan --project --vehicles 6 --rounds 6 --epochs 4",
        "contributes": "The budget arithmetic both sides of a comparison must agree "
                       "on -- its absence is why a centralised ceiling once ran with "
                       "1.667x the federation's image-visits -- and the projection "
                       "behind the Simulate tab, which refuses to extrapolate past the "
                       "measurements it rests on rather than inventing a number.",
        "tab": "Plan, Simulate",
    },
    "measurements": {
        "command": "python -m pipeline.measurements --check",
        "contributes": "Every number this project measured once and now quotes, with "
                       "the document that records it. --check re-reads those documents "
                       "and fails on drift, so the dashboard cannot cite a file that "
                       "no longer says that.",
        "tab": "all",
    },
    "demo_run": {
        "command": "python -m pipeline.demo_run",
        "contributes": "One recorded run, replayable with no GPU and no data, in the "
                       "shape /api/state returns -- so the walkthrough teaches with the "
                       "real panels. Every number is transcribed from a document here, "
                       "and the fields nobody measured are absent with a written reason.",
        "tab": "Live",
    },
    "statestream": {
        "command": "python -m pipeline.statestream",
        "contributes": "The diff format behind the dashboard's push stream. An idle "
                       "page costs a keep-alive comment instead of the whole snapshot "
                       "every two seconds, and a dropped change is a test failure "
                       "rather than a stale panel.",
        "tab": "all",
    },
    "ledger": {
        "command": "python -m pipeline.ledger",
        "contributes": "Every run as one comparable record: approach, data used, wall "
                       "clock, energy, and what it produced — including the per-epoch "
                       "losses, where a plateau shows up long before the round-level "
                       "metric admits it.",
        "tab": "Metrics",
    },
    "compare": {
        "command": "python -m pipeline.compare --last 10",
        "contributes": "Runs side by side, holdout number first, with a warning when "
                       "more than one setting differs or the fleets were not the same.",
        "tab": "—",
    },
    "experiment": {
        "command": "python -m pipeline.experiment --preset seeds --yes",
        "contributes": "One command per question: seeds, strategies, partitions, "
                       "alpha. Every arm changes exactly one setting and ends scored "
                       "on the same holdout.",
        "tab": "—",
    },
    "report": {
        "command": "python -m pipeline.report",
        "contributes": "Self-contained HTML, Markdown and JSON per run: inputs, what "
                       "was learned, what it cost, what it produced.",
        "tab": "Live",
    },
    "gpu": {
        "command": "—",
        "contributes": "Samples nvidia-smi and integrates power into watt-hours, so a "
                       "run's cost is measured rather than estimated.",
        "tab": "Live, Plan",
    },
    "profile": {
        "command": "python -m pipeline.profile --json",
        "contributes": "Where a round's seconds went, from the timestamps already in "
                       "the logs. A mean utilisation figure cannot distinguish "
                       "serialised clients from a starved dataloader; the overlap "
                       "count and the share spent inside train() can.",
        "tab": "—",
    },
    "paths": {
        "command": "—",
        "contributes": "One place that knows the layout and the environment every "
                       "subprocess needs — the UTF-8 setting, the PATH entry, and the "
                       "flag without which flwr silently trains on CPU.",
        "tab": "—",
    },
    "server": {
        "command": "python -m pipeline.server",
        "contributes": "Serves the four tabs and the API behind them. Stores nothing: "
                       "every number is read from disk, so a run launched from the CLI "
                       "lights up the same panels.",
        "tab": "all",
    },
}

#: The tabs, and the question each one exists to answer.
TABS = [
    {"name": "Control", "answers": "What am I about to run, and which stages will "
                                   "actually do work?",
     "reads": "stages.snapshot, and POST /api/run to launch"},
    {"name": "Live", "answers": "Is it learning? The heartbeat is the aggregate "
                                "checksum; the honest metric is the holdout curve.",
     "reads": "logparse, vehicle_metrics, holdout, baseline, gpu — all off disk"},
    {"name": "Data", "answers": "What is the fleet actually training on, and does the "
                                "holdout look like it?",
     "reads": "dataset_stats, cached against the fleet fingerprint"},
    {"name": "Plan", "answers": "What will this configuration cost, and what would a "
                                "fair comparison need?",
     "reads": "plan.budget, plan.commands, stages.snapshot"},
    {"name": "Metrics", "answers": "Which approach was worth it? What did each run "
                                   "cost, and was the model still learning when the "
                                   "round ended?",
     "reads": "ledger, from the reports each run already wrote"},
    {"name": "Simulate", "answers": "What would a configuration cost before it costs a "
                                    "GPU hour, and which of my questions can this "
                                    "project's measurements not answer at all?",
     "reads": "plan.project over measurements.RECORDS; refuses to extrapolate"},
    {"name": "Docs", "answers": "What is each program for, and which number does it "
                                "contribute to?",
     "reads": "this module, from the packages' own docstrings"},
]


def module_docs() -> list[dict]:
    """Every program in the package, described by its own docstring."""
    out = []
    for name, role in ROLES.items():
        try:
            mod = importlib.import_module(f".{name}", __package__)
            doc = (mod.__doc__ or "").strip()
        except Exception as e:                      # a broken module must still list
            doc = f"(could not import: {type(e).__name__}: {e})"
        lines = doc.splitlines()
        out.append({
            "module": f"pipeline/{name}.py",
            "summary": lines[0] if lines else "",
            "doc": doc,
            **role,
        })
    return out


def chain() -> list[dict]:
    """The stage chain, in order, with what each one runs."""
    return [{"name": s.name, "title": s.title, "gated": s.gated, "est": s.est,
             "cwd": str(s.cwd.name)} for s in stages.STAGES]


def documents() -> list[dict]:
    """The markdown in the repo, with its first heading and its size."""
    out = []
    for path in sorted(list(paths.REPO.glob("*.md")) + list((paths.REPO / "docs").glob("*.md"))):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        heading = next((l.lstrip("# ").strip() for l in text.splitlines()
                        if l.startswith("#")), path.stem)
        out.append({"path": path.relative_to(paths.REPO).as_posix(),
                    "heading": heading,
                    "lines": text.count("\n") + 1,
                    "bytes": len(text)})
    return out


def index() -> dict:
    return {"modules": module_docs(), "chain": chain(), "tabs": TABS,
            "documents": documents()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    d = index()
    if args.json:
        print(json.dumps(d, indent=1))
        return 0

    print("stage chain:")
    print("  " + " -> ".join(s["name"] + ("*" if s["gated"] else "") for s in d["chain"]))
    print("  (* needs confirmation: it costs real time or GPU)\n")
    print("programs:")
    for m in d["modules"]:
        print(f"\n  {m['module']}")
        print(f"    {m['summary']}")
        print(f"    contributes: {m['contributes']}")
        print(f"    seen in: {m['tab']}   run: {m['command']}")
    print(f"\n{len(d['documents'])} markdown documents in the repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
