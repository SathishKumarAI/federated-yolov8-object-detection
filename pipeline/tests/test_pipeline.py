"""Tests for the logic that can be wrong quietly.

The GPU stages are not unit-tested — `pipeline.verify` is what asserts those, and it
is the same four criteria the CI simulation-smoke job uses.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline import gpu, logparse, paths, report, stages, vehicles
from pipeline.stages import Config

REPO = paths.REPO


# --------------------------------------------------------------- log parsing
CAPTURED = """
2026-08-05 - INFO - [Server] Aggregated parameters with checksum: -1032.5395936965942
2026-08-05 - INFO - [Server] Aggregated parameters with checksum: -2646.913425683975
2026-08-05 - INFO - [Client] Received weights with checksum: 698.6960870027542
2026-08-05 - INFO - [Client] Sending back weights with checksum: 679.3249893784523
2026-08-05 - INFO - [Client] Starting local training with batch_id=9, local_epochs=1
2026-08-05 - INFO - [Client] Initializing FlowerClient with model=models/yolov8s.pt on cuda:0
"""


def test_parses_negative_checksums():
    """Real aggregates are negative; a naive \\d+ pattern silently drops the sign."""
    vals = [e.value for e in logparse.parse_text(CAPTURED) if e.kind == "aggregate_checksum"]
    assert vals == [-1032.5395936965942, -2646.913425683975]


def test_detects_a_federation_that_did_not_learn(tmp_path):
    (tmp_path / "server.1.log").write_text(
        "Aggregated parameters with checksum: 647.578\n"
        "Aggregated parameters with checksum: 647.578\n")
    learned, detail = logparse.federation_learned(tmp_path)
    assert learned is False
    assert "did not change" in detail


def test_detects_a_federation_that_did_learn(tmp_path):
    (tmp_path / "server.1.log").write_text(
        "Aggregated parameters with checksum: 1.0\n"
        "Aggregated parameters with checksum: 2.0\n")
    assert logparse.federation_learned(tmp_path)[0] is True


def test_single_round_is_not_called_learning(tmp_path):
    (tmp_path / "server.1.log").write_text("Aggregated parameters with checksum: 1.0\n")
    learned, detail = logparse.federation_learned(tmp_path)
    assert learned is False and "need >=2" in detail


def test_no_optimizer_step_warning_is_parsed():
    line = ("[Client] batch=16 over 10 images x 1 epoch(s) gives 1 batch(es), "
            "fewer than the 4 needed for one optimizer step.")
    ev = logparse.parse_line(line)
    assert ev and ev.kind == "no_optimizer_step" and ev.value == 4


# ------------------------------------------------------------------ vehicles
def _index(n=400):
    return {f"img{i}.jpg": {"timeofday": "night" if i % 2 else "daytime",
                            "scene": "city street", "weather": "clear"} for i in range(n)}


def test_vehicles_get_condition_biased_slices():
    idx = _index()
    vs = vehicles.assign(2, 50, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=10, seed=1)
    assert [v.condition for v in vs] == ["daytime city", "night"]
    assert all(idx[n]["timeofday"] == "daytime" for n in vs[0].train)
    assert all(idx[n]["timeofday"] == "night" for n in vs[1].train)


def test_vehicle_slices_are_disjoint():
    """Overlap would train one image on two vehicles per round and flatter the aggregate."""
    idx = _index()
    vs = vehicles.assign(4, 30, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=5, seed=3)
    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen))


def test_vehicle_assignment_is_deterministic():
    idx = _index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=5, seed=7)
    assert [v.train for v in vehicles.assign(3, 20, **kw)] == \
           [v.train for v in vehicles.assign(3, 20, **kw)]


def test_short_condition_tops_up_rather_than_starving_a_vehicle():
    idx = {f"n{i}.jpg": {"timeofday": "night", "scene": "x", "weather": "clear"} for i in range(5)}
    idx.update({f"d{i}.jpg": {"timeofday": "daytime", "scene": "city street", "weather": "clear"}
                for i in range(100)})
    vs = vehicles.assign(2, 40, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=2, seed=0)
    night = next(v for v in vs if v.condition == "night")
    assert night.n_train == 40, "vehicle should be topped up, not left tiny"


# ----------------------------------------------------------------------- gpu
def test_energy_integration():
    t = gpu.Telemetry()
    t.add(gpu.Sample(0.0, 50, 0, 16303, 100.0, 60))
    t.add(gpu.Sample(36.0, 50, 0, 16303, 100.0, 60))
    assert t.energy_wh == pytest.approx(1.0)


def test_energy_uses_the_midpoint_not_a_left_sum():
    t = gpu.Telemetry()
    t.add(gpu.Sample(0.0, 0, 0, 16303, 0.0, 40))
    t.add(gpu.Sample(36.0, 0, 0, 16303, 200.0, 40))
    assert t.energy_wh == pytest.approx(1.0)   # a left-hand sum would give 0.0


def test_gpu_summary_reports_vram_against_the_measured_ceiling():
    t = gpu.Telemetry()
    t.add(gpu.Sample(0.0, 90, 15900, 16303, 250.0, 70))
    s = t.summary()
    assert s["peak_mem_mib"] == 15900 and 97 < s["peak_mem_pct"] < 98


# -------------------------------------------------------------------- stages
def test_every_expensive_stage_is_gated():
    """A browser tab must not be able to start a multi-hour GPU job unprompted."""
    gated = {s.name for s in stages.STAGES if s.gated}
    assert {"dataset", "sanity", "federate"} <= gated


def test_federate_runs_against_the_vehicle_root_not_my_project():
    fed = stages.BY_NAME["federate"]
    assert fed.data_root == paths.VEHICLE_ROOT


def test_snapshot_survives_a_broken_check(monkeypatch):
    boom = stages.Stage("boom", "Boom", False,
                        lambda c: (_ for _ in ()).throw(RuntimeError("nope")),
                        lambda c: ["true"])
    monkeypatch.setattr(stages, "STAGES", [boom])
    row = stages.snapshot(Config())[0]
    assert row["satisfied"] is False and "check failed" in row["detail"]


def test_profile_controls_size_and_resolution():
    assert Config(profile="demo").per_vehicle == 300
    assert Config(profile="demo").imgsz == 320
    assert Config(profile="full").per_vehicle == 6308
    assert Config(profile="full").imgsz == 640


# ------------------------------------------------------------------- reports
def _fixture_report():
    return {"generated": "now", "host": {"platform": "test", "python": "3.12"},
            "config": {"profile": "demo"}, "fleet": [{"vid": 1, "condition": "night",
                                                      "n_train": 300, "n_val": 60}],
            "per_vehicle": {}, "checksums": [1.0, 2.0], "learned": True,
            "learned_detail": "moved", "metrics": [{"round": 1, "stage": "evaluate", "mAP50": 0.3}],
            "gpu": {"energy_wh": 1.5, "peak_mem_mib": 15900, "peak_power_w": 250,
                    "mean_util_pct": 90}, "stages": [{"name": "federate", "status": "ok",
                                                      "seconds": 12.3}],
            "checkpoints": ["global_last.pt"]}


def test_report_renders_both_formats(tmp_path):
    h, m = report.write(_fixture_report(), tmp_path)
    assert h.exists() and m.exists()
    assert "run report" in h.read_text(encoding="utf-8")
    assert "| 1 | night | 300 | 60 |" in m.read_text(encoding="utf-8")


def test_report_html_is_self_contained(tmp_path):
    """No CDN, no external fetches -- it has to survive being emailed."""
    h, _ = report.write(_fixture_report(), tmp_path)
    text = h.read_text(encoding="utf-8")
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', text)


def test_report_states_plainly_when_the_federation_did_not_learn(tmp_path):
    data = _fixture_report() | {"learned": False, "learned_detail": "identical"}
    h, m = report.write(data, tmp_path)
    assert "did NOT learn" in h.read_text(encoding="utf-8")
    assert "DID NOT LEARN" in m.read_text(encoding="utf-8")


# ----------------------------------------------------------------- isolation
def test_pipeline_never_writes_into_my_project():
    """The isolation guarantee, enforced rather than promised.

    Scans this package for writes whose target is my-project. The pipeline may only
    reach my-project by running its scripts as subprocesses.
    """
    offenders = []
    write_call = re.compile(r"\b(write_text|write_bytes|mkdir|unlink|rmtree|copy2|os\.link)\b")
    for py in sorted(Path(__file__).resolve().parents[1].glob("*.py")):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if write_call.search(line) and re.search(r"\bPROJECT\b", line):
                offenders.append(f"{py.name}:{i}: {line.strip()}")
    assert not offenders, "pipeline writes into my-project:\n" + "\n".join(offenders)


def test_vehicle_shards_live_outside_my_project():
    assert paths.PROJECT not in paths.VEHICLE_ROOT.parents
    assert paths.VEHICLE_ROOT.is_relative_to(paths.HERE)


def test_generated_paths_are_all_gitignored():
    """Dataset, checkpoints, MLflow store and reports must be uncommittable.

    Needs a real git repository, which a source tarball or a container image built
    without `.git` is not. Skipped there rather than failed: this guards what may be
    committed, and where nothing can be committed there is nothing to guard.
    """
    import subprocess

    # Two separate ways to not be able to ask: no repository, and no git. The second
    # is not hypothetical -- `python:3.12-slim` ships without git, so a bind-mounted
    # checkout has the `.git` but not the binary, and the test died on FileNotFoundError
    # from deep inside subprocess rather than saying what was missing.
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("no git repository or no git binary; nothing here can be committed")
    targets = [paths.VEHICLE_ROOT / "batch" / "batch_1" / "images" / "x.jpg",
               paths.MLFLOW_STORE / "0" / "meta.yaml",
               paths.REPORTS / "20260805" / "report.html",
               paths.STATE / "attributes.json",
               # The holdout and the pooled baseline set are hardlinks onto the
               # 7.6 GB kagglehub cache, same as the shards.
               paths.VEHICLE_ROOT / "holdout" / "images" / "val" / "x.jpg",
               paths.VEHICLE_ROOT / "pooled" / "images" / "train" / "x.jpg",
               paths.STATE / "baseline_runs" / "centralised" / "weights" / "best.pt",
               paths.STATE / "holdout_metrics.json",
               # Ultralytics' AMP check downloads a small model into the CWD the
               # first time train() runs, so one lands in the repo root after any
               # baseline or sanity run.
               REPO / "yolo26n.pt",
               REPO / "models" / "yolov8s.pt"]
    for t in targets:
        rel = t.relative_to(REPO).as_posix()
        out = subprocess.run(["git", "check-ignore", rel], cwd=REPO,
                             capture_output=True, text=True)
        assert out.returncode == 0, f"{rel} is NOT gitignored"


# ------------------------------------------- exit codes that lie
def test_crash_in_output_is_detected_despite_exit_zero():
    """flwr exits 0 having printed 'Simulation Runtime crashed'. Believe the output."""
    lines = ["INFO: starting", "ERROR: Simulation Runtime crashed.", "ERROR: Exit Code: 700"]
    assert stages.scan_for_crash(lines, stages.CRASH_MARKERS) == "Simulation Runtime crashed"


def test_clean_output_is_not_flagged_as_a_crash():
    lines = ["INFO: Run finished 2 round(s) in 55.6s", "aggregate checksum: 1.0"]
    assert stages.scan_for_crash(lines, stages.CRASH_MARKERS) is None


def test_federate_stage_scans_for_crashes():
    assert stages.BY_NAME["federate"].crash_markers, "federate must not trust its exit code"


@pytest.fixture
def _flwr_launcher(monkeypatch):
    """`_cmd_federate` resolves the flwr launcher, and raises when there is none.

    These two tests are about the *arguments* it builds, not about where the binary
    lives — `test_the_flwr_launcher_never_runs_the_blocked_shim` covers that. So
    they failed on any machine without flwr installed, which includes the pipeline CI
    job by design: it installs pytest and pyyaml only, because the pipeline package
    imports torch and ultralytics lazily and its tests then run in under a second.
    """
    monkeypatch.setattr(stages, "flwr_launcher", lambda: ["/nonexistent/python", "-c", "x"])


def test_init_args_are_omitted_when_attaching_to_an_existing_ray_cluster(_flwr_launcher):
    """Ray: 'When connecting to an existing cluster, num_cpus and num_gpus must not
    be provided.' Passing them anyway crashes the simulation at startup."""
    attached = " ".join(stages._cmd_federate(Config(ray_address="127.0.0.1:6379")))
    standalone = " ".join(stages._cmd_federate(Config()))
    assert "init-args" not in attached
    assert "init-args-num-gpus=1" in standalone


def test_supernodes_track_the_vehicle_count(_flwr_launcher):
    """Otherwise the run hangs forever waiting for clients that never arrive."""
    cmd = " ".join(stages._cmd_federate(Config(n_vehicles=6)))
    assert "num-supernodes=6" in cmd and "min_clients=6" in cmd


def test_clients_stay_serialised_unless_the_run_asks_for_packing(_flwr_launcher):
    """Phase 0 measured 72 client episodes never overlapping, so the GPU fraction is
    the lever with the largest ceiling. It is also the one that can OOM a full-profile
    run, where a single 6 308-image shard peaks at 15.9 GB of 16.3 -- so the default
    must stay 1.0 and packing must be something a run asks for."""
    assert "client-resources-num-gpus=1.0" in " ".join(stages._cmd_federate(Config()))
    packed = " ".join(stages._cmd_federate(Config(gpu_fraction=0.33)))
    assert "client-resources-num-gpus=0.33" in packed
    assert "client-resources-num-gpus=1.0" not in packed, "the old value must not survive"


def test_restoring_pyproject_keeps_an_uncommitted_edit(tmp_path, monkeypatch):
    """The restore used `git checkout -- pyproject.toml`, which puts back the
    *committed* file and so discards any uncommitted edit along with flwr's rewrite.
    Adding a run-config key and running the pipeline to test it deleted the key, and
    the next run failed on a value that had been on disk minutes earlier. The snapshot
    also makes the restore work where there is no git at all -- a source tarball."""
    from pipeline import runner

    project = tmp_path / "my-project"
    project.mkdir()
    pp = project / "pyproject.toml"
    monkeypatch.setattr(paths, "PROJECT", project)

    pp.write_text('[tool.flwr.app.config]\ncache = ""\n\n[tool.flwr.federations]\ndefault = "x"\n')
    run = runner.Run(Config())                      # snapshot taken here

    pp.write_text("# CONFIGURATION MIGRATION NOTICE\n# [tool.flwr.federations]\n")
    run._restore_pyproject()
    assert 'cache = ""' in pp.read_text(), "the uncommitted key must survive the run"
    assert "[tool.flwr.federations]" in pp.read_text(), "and flwr's rewrite must not"

    # A file flwr never touched is left exactly as the run found it, edits and all.
    pp.write_text("edited during the run\n")
    run._restore_pyproject()
    assert pp.read_text() == "edited during the run\n"


def test_the_cache_setting_reaches_the_client_and_is_declared_in_pyproject(_flwr_launcher):
    """flwr validates --run-config keys against [tool.flwr.app.config] and rejects the
    whole run for an undeclared one, so the key has to exist in pyproject even though
    the pipeline always overrides it. It is a run-level setting sent through the run
    config rather than a fit instruction: FedAvg shares one FitIns across all clients,
    which is how every vehicle once ended up training the same shard."""
    cmd = " ".join(stages._cmd_federate(Config(cache="ram")))
    assert 'cache="ram"' in cmd
    assert 'cache=""' in " ".join(stages._cmd_federate(Config()))

    declared = (paths.PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^cache = ", declared, re.M), "flwr rejects a run-config key it has never heard of"


def test_every_run_config_key_the_pipeline_sends_is_declared_in_pyproject(_flwr_launcher):
    """flwr validates --run-config against [tool.flwr.app.config] and rejects the WHOLE
    run for a key it has never heard of. So adding a key to the command without adding
    it to pyproject does not fall back to a default -- it kills every run, before the
    first round, for a reason that reads as a TOML error.

    Generic rather than one assertion per key, so the next lever added to the federate
    command is covered by this test on the day it is written.
    """
    cmd = stages._cmd_federate(Config())
    run_config = cmd[cmd.index("--run-config") + 1]
    sent = set(re.findall(r"(\w+)=", run_config))

    declared = (paths.PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    body = declared.split("[tool.flwr.app.config]", 1)[1].split("[tool.flwr", 1)[0]
    known = set(re.findall(r"^(\w+) *=", body, re.M))

    assert sent <= known, (
        f"the federate command sends run-config keys pyproject does not declare: "
        f"{sorted(sent - known)}. flwr will refuse the run.")


def test_fedbn_reaches_the_client_as_a_toml_boolean(_flwr_launcher):
    """Python's `True` is a bare word to a TOML parser, not a boolean, and flwr parses
    run-config values as TOML. `local_bn=True` would kill the run at parse."""
    on = " ".join(stages._cmd_federate(Config(local_bn=True)))
    assert "local_bn=true" in on and "local_bn=True" not in on
    assert "local_bn=false" in " ".join(stages._cmd_federate(Config()))


def test_the_round_one_freeze_reaches_the_federation(_flwr_launcher):
    """The lever is worthless if it stops at the pipeline. Off by default, because a
    freeze that arrives unasked changes every number this repo has already measured."""
    on = " ".join(stages._cmd_federate(Config(freeze_round1=10)))
    assert "freeze_round1=10" in on
    assert "freeze_round1=0" in " ".join(stages._cmd_federate(Config()))


def test_one_image_size_reaches_the_federation_the_baseline_and_the_holdout(_flwr_launcher):
    """A federation at 1024 scored against a ceiling at 640 is not a comparison. The
    baseline and evaluate stages already took this as a flag; the federation's own size
    was the piece that was unreachable, and the docstring of Config.imgsz said so."""
    cfg = Config(imgsz_override=1024)
    assert "imgsz=1024" in " ".join(stages._cmd_federate(cfg))
    assert "1024" in " ".join(stages._cmd_baseline(cfg))
    assert "1024" in " ".join(stages._cmd_evaluate(cfg))


def test_the_image_size_default_still_belongs_to_the_profile(_flwr_launcher):
    """Off means off: demo stays at 320 and full at 640, so adding the lever moves no
    number that has already been measured. `Config()` is the demo profile, which is
    what the 320 is -- and what this test asserted 640 for on its first draft."""
    assert Config().imgsz == 320
    assert Config(profile="full").imgsz == 640
    assert Config(profile="demo", imgsz_override=960).imgsz == 960
    assert Config(profile="full", imgsz_override=1024).imgsz == 1024


def test_a_gpu_fraction_ray_cannot_schedule_is_refused_rather_than_hung(capsys):
    """Ray accepts num_gpus > 1 per client and then places no client at all: the
    federation waits for workers that can never be scheduled, with nothing in any log
    saying why. Both entry points guard it, because both can set it."""
    from pipeline import runner

    with pytest.raises(SystemExit):
        runner.main(["--all", "--yes", "--gpu-fraction", "1.5"])
    assert "--gpu-fraction must be in (0, 1]" in capsys.readouterr().err

    import io

    from pipeline import server as srv

    handler = object.__new__(srv.Handler)
    handler.path = "/api/run"
    body = json.dumps({"gpu_fraction": 0}).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    seen = {}
    handler._json = lambda payload, code=200: seen.update(payload=payload, code=code)
    handler.do_POST()
    assert seen["code"] == 400 and "gpu_fraction must be in (0, 1]" in seen["payload"]["error"]


@pytest.fixture(autouse=True)
def _no_fleet_state_on_disk(monkeypatch):
    """The fleet check consults the holdout and the fleet manifest. A test must
    assert its own logic, not whatever the last real run left in the working tree --
    a demo run rewriting the manifest should not turn other tests red. The manifest's
    and the holdout's own effects each have a dedicated test that patches them back."""
    from pipeline import holdout as _h, vehicles as _v
    monkeypatch.setattr(_h, "names", lambda: set())
    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {})


def test_fleet_check_demands_a_shard_for_every_assignable_id():
    """The server picks batch ids from DEFAULT_BATCH_ID_RANGE (1..10) at random and
    cannot be told to stay within the vehicle count, so every id must resolve."""
    import json as _json
    from pipeline import vehicles as _v
    original = _v.load_fleet
    # partition is pinned to match the fake shards' 'x' condition rather than left to
    # the Config default, which moved to 'random' on 2026-09-02. This test is about
    # the shard COUNT; without the pin it fails on the partition check instead and
    # says nothing about what it is named for.
    cfg = Config(n_vehicles=6, partition="condition")
    try:
        _v.load_fleet = lambda: [{"vid": i, "condition": "x", "n_train": 300, "n_val": 60}
                                 for i in range(1, 7)]
        assert stages._check_fleet(cfg).satisfied is False
        _v.load_fleet = lambda: [{"vid": i, "condition": "x", "n_train": 300, "n_val": 60}
                                 for i in range(1, 11)]
        assert stages._check_fleet(cfg).satisfied is True
    finally:
        _v.load_fleet = original


# ------------------------------------------------------------------- server
def test_report_route_refuses_paths_outside_the_reports_dir():
    """`/reports/../../secret` must not escape. The check is `REPORTS in parents`."""
    from pipeline import server  # noqa: F401  (import guard: server must load cleanly)
    escaped = (paths.REPORTS / ".." / ".." / "CLAUDE.md").resolve()
    assert paths.REPORTS.resolve() not in escaped.parents

    inside = (paths.REPORTS / "20260101" / "report.html").resolve()
    assert paths.REPORTS.resolve() in inside.parents


def test_one_path_guard_covers_every_route_that_maps_a_url_to_a_file(tmp_path):
    """Reports and shard images share `safe_child`; both traversals must fail."""
    from pipeline import server

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "ok.jpg").write_bytes(b"x")
    (tmp_path / "secret.txt").write_text("credentials")

    assert server.safe_child(root, "sub/ok.jpg") == (root / "sub" / "ok.jpg").resolve()
    assert server.safe_child(root, "../secret.txt") is None
    assert server.safe_child(root, "..\\secret.txt") is None
    assert server.safe_child(root, "sub") is None            # a directory is not a file
    assert server.safe_child(root, "sub/missing.jpg") is None


def test_every_file_the_dashboard_imports_is_servable():
    """index.html references app.css and js/main.js; a 404 there renders a blank page."""
    from pipeline import server

    html = (server.STATIC / "index.html").read_text(encoding="utf-8")
    referenced = re.findall(r'(?:href|src)="/static/([^"]+)"', html)
    assert referenced, "index.html no longer references any static asset"
    for rel in referenced:
        target = server.safe_child(server.STATIC, rel)
        assert target is not None, f"{rel} is referenced but not servable"
        assert target.suffix in server.Handler.STATIC_TYPES, f"{rel} has no content type"

    # And every module those modules import, one level of transitive closure deep.
    for js in (server.STATIC / "js").glob("*.js"):
        for imported in re.findall(r'from "\./([^"]+)"', js.read_text(encoding="utf-8")):
            assert (server.STATIC / "js" / imported).is_file(), f"{js.name} imports missing {imported}"


def _run_js_check(tmp_path, name: str) -> None:
    """Run one of `pipeline/tests/js/*.mjs` against the real dashboard modules.

    The dashboard is served as plain files with no build step, so there is no JS test
    runner here and adding one would be a larger change than the code it guards. The
    modules are copied beside the check with a `type: module` marker, which is all
    node needs to import them.

    Skipped rather than failed without node — but note that GitHub's runners ship it,
    so CI does execute these.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip(f"node not installed; {name} is unchecked here")

    from pipeline import server
    work = tmp_path / "js"
    shutil.copytree(server.STATIC / "js", work)
    (work / "package.json").write_text('{"type":"module"}')
    shutil.copy(Path(__file__).parent / "js" / name, work / name)

    # console.assert writes to stderr and does NOT set the exit code, so an assertion
    # that fires would otherwise pass silently -- the exact class of bug this repo
    # keeps shipping. All three conditions are checked.
    out = subprocess.run([node, str(work / name)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Assertion failed" not in out.stderr, out.stderr
    assert "OK" in out.stdout, out.stdout


def test_label_overlay_boxes_land_where_the_label_file_says(tmp_path):
    """`cx cy w h` centred becomes `x y w h` cornered.

    A box drawn a few percent off looks plausible, and the overlay exists to be
    believed when it says a shard is mislabelled. So the conversion is executed, not
    read.
    """
    _run_js_check(tmp_path, "overlay_geometry.mjs")


def test_the_live_feed_says_what_it_is_showing_in_every_state(tmp_path):
    """Idle, training, started-but-nothing-written, and a vehicle nobody has heard of.

    The failure that matters is the quiet one: without the mtime in the image URL the
    browser serves round 1's mosaic for the whole run, and the panel looks alive while
    showing a fossil.
    """
    _run_js_check(tmp_path, "live_feed.mjs")


def test_train_artifact_route_serves_only_names_on_the_allowlist(tmp_path, monkeypatch):
    """The trainer's directory holds weights and a data yaml beside the pictures.

    Names are matched against `KINDS` rather than joined onto the run directory and
    guarded, so neither `../../CLAUDE.md` nor `weights/best.pt` can be fetched — and a
    future ultralytics release cannot publish a new file over HTTP by writing it.
    """
    from pipeline import train_artifacts

    run = tmp_path / "my-project" / "runs" / "detect" / "runs" / "fl" / "batch3"
    run.mkdir(parents=True)
    (run / "args.yaml").write_text("epochs: 1")
    (run / "labels.jpg").write_bytes(b"\xff\xd8jpeg")
    (run / "weights").mkdir()
    (run / "weights" / "best.pt").write_bytes(b"weights")
    monkeypatch.setattr(train_artifacts.paths, "PROJECT", tmp_path / "my-project")

    assert train_artifacts.run_dir(3) == run
    assert train_artifacts.artifact(3, "labels.jpg") == run / "labels.jpg"
    assert train_artifacts.artifact(3, "weights/best.pt") is None
    assert train_artifacts.artifact(3, "../../../CLAUDE.md") is None
    assert train_artifacts.artifact(3, "args.yaml") is None
    assert train_artifacts.artifact(4, "labels.jpg") is None      # no such vehicle
    # train_batch0.jpg is on the allowlist but was not written: absent, not an error.
    assert train_artifacts.artifact(3, "train_batch0.jpg") is None

    listed = train_artifacts.for_vehicle(3)
    assert [f["name"] for f in listed["files"]] == ["labels.jpg"]


def test_shard_label_boxes_are_normalised_and_a_bad_row_does_not_lose_the_good_ones(
        tmp_path, monkeypatch):
    """A label file with one broken row still has good rows.

    Raising here would blank the overlay for the whole frame, which is exactly the
    frame someone is looking at *because* they suspect its labels.
    """
    from pipeline import dataset_stats

    labels = tmp_path / "batch_7" / "labels" / "train"
    labels.mkdir(parents=True)
    (labels / "frame.txt").write_text(
        "2 0.5 0.5 0.25 0.5\n"        # good
        "9 0.1 0.2\n"                  # too few fields
        "car 0.1 0.2 0.3 0.4\n"        # class is not a number
        "0 0.9 0.9 0.05 0.05\n"        # good, and last
    )
    monkeypatch.setattr(dataset_stats.paths, "VEHICLE_BATCHES", tmp_path)

    got = dataset_stats.boxes(7, "frame.jpg")
    assert [b["cls"] for b in got] == [2, 0]
    assert got[0] == {"cls": 2, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.5}
    assert dataset_stats.boxes(7, "no-such-frame.jpg") == []


def test_shard_composition_counts_what_the_vehicle_actually_holds(tmp_path, monkeypatch):
    """The condition label is a claim; this counts the images behind it."""
    batches = tmp_path / "batch"
    shard = batches / "batch_2"
    (shard / "images" / "train").mkdir(parents=True)
    (shard / "images" / "val").mkdir(parents=True)
    (shard / "train.txt").write_text("a.jpg\nb.jpg\nc.jpg\n")
    (shard / "images" / "train" / "a.jpg").write_bytes(b"x")
    (shard / "images" / "val" / "v.jpg").write_bytes(b"x")

    monkeypatch.setattr(paths, "VEHICLE_BATCHES", batches)
    index = {"a.jpg": {"weather": "rainy", "scene": "city street", "timeofday": "night"},
             "b.jpg": {"weather": "rainy", "scene": "highway", "timeofday": "night"}}
    comp = vehicles.composition(2, index=index)

    assert comp["n_train"] == 3 and comp["n_val"] == 1
    assert comp["counts"]["weather"] == {"rainy": 2, "unknown": 1}
    assert list(comp["counts"]["weather"]) == ["rainy", "unknown"]   # sorted by frequency
    assert comp["samples"] == ["a.jpg"]         # only the image that exists on disk


def test_composition_never_triggers_an_attribute_index_build(monkeypatch):
    """It runs inside a request handler; streaming 1.45 GB of JSON there hangs the page."""
    monkeypatch.setattr(vehicles, "build_attribute_index",
                        lambda *a, **k: pytest.fail("composition must not build the index"))
    monkeypatch.setattr(vehicles, "ATTR_CACHE", Path("does-not-exist.json"))
    monkeypatch.setattr(vehicles, "_ATTR_MEMO", None)
    assert vehicles.cached_attributes() == {}


def test_live_state_is_derived_from_disk_not_from_the_event_bus(monkeypatch, tmp_path):
    """A run launched from the CLI must still light up the dashboard."""
    from pipeline import server, verify as _verify

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "server.1.log").write_text(
        "Aggregated parameters with checksum: 1.0\n"
        "Aggregated parameters with checksum: 2.0\n")
    (logs / "client.1.log").write_text(
        "Starting local training with batch_id=3, local_epochs=1\n"
        "Received weights with checksum: 10.0\n"
        "Sending back weights with checksum: 11.0\n")
    (logs / "metrics.csv").write_text(
        "round,stage,num_clients,loss,precision,recall,mAP50,mAP50-95,fitness\n"
        "1,evaluate,6,0.5,0.4,0.3,0.25,0.1,0.1\n")

    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    monkeypatch.setattr(_verify, "metrics_csv", lambda: logs / "metrics.csv")

    live = server.State().live()
    assert live["rounds_done"] == 2
    assert live["checksums"] == [1.0, 2.0]
    assert live["map50"] == [0.25]
    assert live["per_vehicle"]["3"]["received"] == 10.0
    assert live["per_vehicle"]["3"]["sent"] == 11.0


# ------------------------------------------------ per-vehicle learning
from pipeline import vehicle_metrics as vm  # noqa: E402


def _client_log(tmp_path, body):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "client.1.log").write_text(body)
    return logs


def test_per_vehicle_metrics_are_parsed_without_eval(tmp_path, monkeypatch):
    """The dict comes from a log file. literal_eval, never eval."""
    body = ("[Client] 3 Training done. metrics={'precision': 0.30, 'recall': 0.27, "
            "'mAP50': 0.23, 'mAP50-95': 0.11, 'fitness': 0.11, 'os': 'Windows'}\n")
    logs = _client_log(tmp_path, body)
    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    rounds = vm.per_vehicle_rounds()
    assert rounds["3"][0]["mAP50"] == 0.23
    assert rounds["3"][0]["mAP50_95"] == 0.11
    assert rounds["3"][0]["round"] == 1


def test_a_malicious_metrics_blob_cannot_execute(tmp_path, monkeypatch):
    logs = _client_log(tmp_path, "[Client] 1 Training done. metrics={'x': __import__('os')}\n")
    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    assert vm.per_vehicle_rounds() == {}      # rejected, not executed


def test_divergence_sums_to_about_zero(tmp_path, monkeypatch):
    rounds = {"1": [{"round": 1, "mAP50": 0.20}],
              "2": [{"round": 1, "mAP50": 0.30}],
              "3": [{"round": 1, "mAP50": 0.40}]}
    div = vm.divergence(rounds)
    assert abs(sum(v[0] for v in div.values())) < 1e-9
    assert div["1"][0] < 0 < div["3"][0]


def test_a_vehicle_that_never_trained_does_not_break_aggregation(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "log_dirs", lambda: [tmp_path])
    assert vm.per_vehicle_rounds() == {}
    assert vm.divergence({}) == {}
    assert vm.weight_movement() == {}


def test_missing_results_csv_degrades_rather_than_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT", tmp_path)
    assert vm.per_vehicle_epochs() == {}


def test_contribution_is_the_fedavg_weight():
    fleet = [{"vid": 1, "n_train": 300}, {"vid": 2, "n_train": 100}]
    c = vm.contribution(fleet)
    assert c["1"] == 0.75 and c["2"] == 0.25


def test_weight_movement_pairs_received_with_sent(tmp_path, monkeypatch):
    logs = _client_log(tmp_path,
        "Starting local training with batch_id=4, local_epochs=1\n"
        "Received weights with checksum: 100.0\n"
        "Sending back weights with checksum: 90.0\n")
    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    assert vm.weight_movement()["4"] == [10.0]


def test_fleet_json_records_val_counts(tmp_path):
    """It recorded n_train only, so every report printed 'val | ?'."""
    v = vehicles.Vehicle(1, "night", ["a.jpg"] * 5, ["b.jpg"] * 2)
    summary = v.to_summary()
    assert {k: summary[k] for k in ("vid", "condition", "n_train", "n_val")} == {
        "vid": 1, "condition": "night", "n_train": 5, "n_val": 2}
    assert len(summary["fingerprint"]) == 12


def test_a_fleet_fingerprint_proves_two_runs_saw_the_same_images():
    """Same config does not mean same data: a rebuilt fleet, a changed holdout or a
    different pool all produce different images under an identical config."""
    a = vehicles.Vehicle(1, "night", ["a.jpg", "b.jpg"], ["v.jpg"])
    same = vehicles.Vehicle(1, "night", ["a.jpg", "b.jpg"], ["v.jpg"])
    other_train = vehicles.Vehicle(1, "night", ["a.jpg", "c.jpg"], ["v.jpg"])
    other_val = vehicles.Vehicle(1, "night", ["a.jpg", "b.jpg"], ["w.jpg"])

    assert a.fingerprint() == same.fingerprint()
    assert a.fingerprint() != other_train.fingerprint()
    assert a.fingerprint() != other_val.fingerprint(), "the val split is part of the data too"


# ------------------------------------------------------- partition strategies
def test_random_partition_ignores_conditions():
    """The IID control case: no condition filter, every vehicle a uniform draw."""
    idx = _index()
    vs = vehicles.assign(3, 30, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=5, seed=2, partition="random")
    assert {v.condition for v in vs} == {"random mix"}
    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen)), "random slices must still be disjoint"


def test_mixed_partition_alternates():
    idx = _index()
    vs = vehicles.assign(4, 20, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=4, seed=5, partition="mixed")
    assert [v.condition for v in vs][1] == "random mix"
    assert [v.condition for v in vs][0] != "random mix"


def test_unknown_partition_is_rejected():
    idx = _index()
    with pytest.raises(ValueError):
        vehicles.assign(2, 10, index=idx, train_pool=set(idx), val_pool=set(idx),
                        partition="nonsense")


def _mixed_index(n=1200):
    """An index spanning several PROFILES, so a Dirichlet draw has groups to skew over."""
    kinds = [
        {"timeofday": "daytime", "scene": "city street", "weather": "clear"},   # daytime city
        {"timeofday": "night", "scene": "residential", "weather": "clear"},     # night
        {"timeofday": "daytime", "scene": "residential", "weather": "rainy"},   # rain / fog
        {"timeofday": "daytime", "scene": "highway", "weather": "clear"},       # highway
        {"timeofday": "dawn/dusk", "scene": "residential", "weather": "clear"}, # dawn / dusk
        {"timeofday": "daytime", "scene": "residential", "weather": "snowy"},   # snow
    ]
    return {f"img{i}.jpg": kinds[i % len(kinds)] for i in range(n)}


def _dominant_share(vehicle, index):
    counts = {}
    for name in vehicle.train:
        key = vehicles._group_of(index.get(name, {}))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / max(1, len(vehicle.train))


def test_dirichlet_alpha_is_the_skew_knob():
    """Small alpha concentrates a vehicle on one condition; large alpha flattens it.

    The direction is asserted, not a magic number: the draw is random, and pinning
    an exact share would make this a test of one seed rather than of the mechanism.
    """
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              partition="dirichlet")

    skewed = vehicles.assign(6, 60, seed=1, alpha=0.02, **kw)
    flat = vehicles.assign(6, 60, seed=1, alpha=200.0, **kw)

    skewed_share = sum(_dominant_share(v, idx) for v in skewed) / len(skewed)
    flat_share = sum(_dominant_share(v, idx) for v in flat) / len(flat)
    assert skewed_share > flat_share + 0.2, (skewed_share, flat_share)
    assert skewed_share > 0.8, "alpha=0.02 should put a vehicle almost entirely on one condition"


def test_dirichlet_slices_are_disjoint_and_deterministic():
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              partition="dirichlet", alpha=0.4, seed=11)
    vs = vehicles.assign(5, 50, **kw)

    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen)), "dirichlet slices must not overlap"
    assert all(len(v.train) == 50 for v in vs), "the per-client-mixture variant keeps sizes equal"
    assert [v.train for v in vehicles.assign(5, 50, **kw)] == [v.train for v in vs]
    assert all(v.condition.startswith("dirichlet") for v in vs)


def test_dirichlet_rejects_an_alpha_that_has_no_meaning():
    idx = _mixed_index(60)
    with pytest.raises(ValueError):
        vehicles.assign(2, 10, index=idx, train_pool=set(idx), val_pool=set(idx),
                        partition="dirichlet", alpha=0)


# ------------------------------------------------------------- quantity skew
def test_size_skew_gives_unequal_shards_that_still_sum_to_the_budget():
    """Skew redistributes the fleet's images; it must never change how many there are.

    A skewed run and an unskewed one have to be comparable, and they only are if both
    made the same number of image-visits. If skew changed the total, every comparison
    against a skewed arm would be confounded by budget -- which is the failure
    `compare.py` already refuses to let a config make.
    """
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              seed=3, partition="condition")

    flat = vehicles.assign(6, 100, size_skew=0.0, **kw)
    skewed = vehicles.assign(6, 100, size_skew=0.9, **kw)

    assert {v.n_train for v in flat} == {100}, "skew 0 must leave every shard equal"
    assert len({v.n_train for v in skewed}) > 1, "skew > 0 must make shards differ"
    assert sum(v.n_train for v in skewed) == 600, [v.n_train for v in skewed]


def test_size_skew_never_shards_below_the_batch_size():
    """A shard smaller than the batch takes no optimizer step and logs nothing wrong.

    That exact silent no-op is already in this repo's history, so a realism knob is
    not allowed to produce it however extreme the draw.
    """
    idx = _mixed_index(4000)
    floor = vehicles.size_floor(200)
    vs = vehicles.assign(6, 200, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=20, seed=7, size_skew=3.0)

    assert floor == 32, floor
    assert min(v.n_train for v in vs) >= floor, [v.n_train for v in vs]
    assert all(v.val for v in vs), "a vehicle with no val split cannot self-evaluate"


def test_size_skew_is_deterministic_and_keeps_slices_disjoint():
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              seed=11, size_skew=0.7, partition="dirichlet", alpha=0.4)
    vs = vehicles.assign(5, 80, **kw)

    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen)), "skewed slices must not overlap"
    assert [v.train for v in vehicles.assign(5, 80, **kw)] == [v.train for v in vs]


def test_skew_zero_draws_nothing_so_old_fleets_still_reproduce():
    """The default path must not touch the rng, or every fleet ever built changes.

    Determinism here is not a nicety: `fleet.meta.json` records a seed, and runs on
    disk are only comparable to future ones while that seed still means the same
    images.
    """
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              seed=4, partition="condition")
    assert ([v.train for v in vehicles.assign(4, 50, **kw)]
            == [v.train for v in vehicles.assign(4, 50, size_skew=0.0, **kw)])


def test_the_fleet_stage_expects_skewed_shards_to_be_small():
    """Without this the rebuild check fires every run -- and rebuilding rmtree's shards.

    The check asks "is any vehicle below per_vehicle images?", which is true by design
    for every skewed fleet. Left alone it would rmtree and relink the whole fleet on
    every invocation, including one made while a federation was in flight.
    """
    cfg = stages.Config(profile="demo", n_vehicles=2, size_skew=0.8)
    # A shard for every id the server can pick, sizes spread around per_vehicle=300.
    fleet = [{"vid": i, "n_train": 480 if i % 2 else 120, "condition": "night"}
             for i in paths.BATCH_IDS]
    meta = {"partition": cfg.partition, "seed": cfg.seed,
            "per_vehicle": cfg.per_vehicle, "holdout": 0, "size_skew": 0.8}

    with patch.object(vehicles, "load_fleet", return_value=fleet), \
         patch.object(vehicles, "load_fleet_meta", return_value=meta), \
         patch.object(stages.holdout, "names", return_value=set()):
        assert stages._check_fleet(cfg).satisfied, stages._check_fleet(cfg).detail

        thin = [*fleet[:-1], {**fleet[-1], "n_train": 4}]
        with patch.object(vehicles, "load_fleet", return_value=thin):
            assert not stages._check_fleet(cfg).satisfied, "below the floor is still stale"


def test_a_fleet_built_before_skew_existed_is_not_declared_stale():
    """`fleet.meta.json` files on disk have no size_skew key. Absent must read as 0."""
    cfg = stages.Config(profile="demo", n_vehicles=2)
    fleet = [{"vid": i, "n_train": cfg.per_vehicle, "condition": "x"} for i in paths.BATCH_IDS]
    meta = {"partition": cfg.partition, "seed": cfg.seed,
            "per_vehicle": cfg.per_vehicle, "holdout": 0}   # no size_skew

    with patch.object(vehicles, "load_fleet", return_value=fleet), \
         patch.object(vehicles, "load_fleet_meta", return_value=meta), \
         patch.object(stages.holdout, "names", return_value=set()):
        assert stages._check_fleet(cfg).satisfied, stages._check_fleet(cfg).detail


def test_the_registry_is_what_every_caller_reads():
    """Registering a partitioner is the only step; CLI choices follow from it."""
    assert vehicles.PARTITIONS == tuple(vehicles.PARTITIONERS)
    assert set(vehicles.PARTITIONS) >= {"condition", "random", "mixed", "dirichlet"}

    from pipeline import build_fleet, runner
    for parser, flag in ((build_fleet, None), (runner, None)):
        pass
    choices = runner.build_parser()._option_string_actions["--partition"].choices
    assert tuple(choices) == vehicles.PARTITIONS


def test_fleet_manifest_is_compared_rather_than_guessed(monkeypatch):
    """Label-sniffing could not tell condition from mixed, or one alpha from another."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "dirichlet a=0.5 - night 90%", "n_train": 300, "n_val": 60}
        for i in range(1, 11)])
    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {
        "partition": "dirichlet", "alpha": 0.5, "seed": 0, "per_vehicle": 300})

    same = Config(n_vehicles=6, partition="dirichlet", alpha=0.5)
    assert stages._check_fleet(same).satisfied is True

    other_alpha = stages._check_fleet(Config(n_vehicles=6, partition="dirichlet", alpha=0.1))
    assert other_alpha.satisfied is False and "alpha" in other_alpha.detail

    other_partition = stages._check_fleet(Config(n_vehicles=6, partition="condition"))
    assert other_partition.satisfied is False and "partition" in other_partition.detail


# ------------------------------------------------------ holdout and baseline
from pipeline import baseline as _baseline, holdout as _holdout  # noqa: E402


def test_holdout_selection_is_deterministic_across_processes():
    """Set iteration order varies per process; shuffling a set would give a
    different holdout on every machine and make two runs incomparable."""
    pool = {f"img{i}.jpg" for i in range(500)}
    first = _holdout.select(50, seed=4, val_pool=pool)
    second = _holdout.select(50, seed=4, val_pool=set(reversed(sorted(pool))))
    assert first == second
    assert _holdout.select(50, seed=5, val_pool=pool) != first
    assert len(set(first)) == 50


def test_holdout_refuses_to_invent_images_it_does_not_have():
    with pytest.raises(SystemExit):
        _holdout.select(500, val_pool={"a.jpg", "b.jpg"})


def test_no_vehicle_can_be_assigned_a_held_out_image():
    """The whole point: a client that self-evaluates on a holdout image makes the
    global metric partly self-referential."""
    idx = _mixed_index(600)
    pool = set(idx)
    held = set(_holdout.select(100, seed=2, val_pool=pool))

    vs = vehicles.assign(4, 40, index=idx, train_pool=pool, val_pool=pool,
                         val_per_vehicle=20, seed=1, exclude=held)
    for v in vs:
        assert not (set(v.train) & held), f"vehicle {v.vid} trains on held-out images"
        assert not (set(v.val) & held), f"vehicle {v.vid} self-evaluates on held-out images"


def test_a_fleet_that_predates_the_holdout_is_rebuilt(monkeypatch):
    """Otherwise the fleet stage says 'skip' and the contamination is never noticed."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "night", "n_train": 300, "n_val": 60} for i in range(1, 11)])
    monkeypatch.setattr(_holdout, "names", lambda: {f"h{i}.jpg" for i in range(1000)})

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {})
    predates = stages._check_fleet(Config(n_vehicles=6, partition="condition"))
    assert predates.satisfied is False and "holdout" in predates.detail

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {
        "partition": "condition", "seed": 0, "per_vehicle": 300, "holdout": 500})
    stale = stages._check_fleet(Config(n_vehicles=6, partition="condition"))
    assert stale.satisfied is False and "holdout" in stale.detail

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {
        "partition": "condition", "seed": 0, "per_vehicle": 300, "holdout": 1000})
    assert stages._check_fleet(Config(n_vehicles=6, partition="condition")).satisfied is True


class _FakeBox:
    """Ultralytics' Metric, in the shape that matters: the per-class arrays are
    indexed by POSITION in ap_class_index. It has NO nt_per_class -- that lives on
    the result, and reading it from here is what made every count come out '?'."""
    def __init__(self, present, ap50, ap):
        self.ap_class_index = present
        self.ap50, self.ap = ap50, ap
        self.p = [0.5] * len(present)
        self.r = [0.6] * len(present)


class _FakeResult:
    """DetMetrics: nt_per_class is indexed by CLASS ID and lives here, not on .box."""
    def __init__(self, box, names, nt=None):
        self.box, self.names = box, names
        if nt is not None:
            self.nt_per_class = nt


def test_instance_counts_come_from_the_result_not_from_the_box():
    """nt_per_class is a DetMetrics attribute. Reading it off result.box returns None
    for every class and the table renders '?' in the column that says how much the AP
    numbers mean -- silently, because a missing count is not an error."""
    box = _FakeBox(present=[0, 2], ap50=[0.3, 0.6], ap=[0.1, 0.35])
    result = _FakeResult(box, {0: "person", 2: "car"}, nt=[1277, 52, 10389])
    assert not hasattr(box, "nt_per_class"), "the fake must match the real shape"

    rows = _holdout.per_class(result, names={0: "person", 2: "car"})
    assert [r["instances"] for r in rows] == [1277, 10389]


def test_per_class_rows_are_named_by_class_id_not_by_array_position():
    """The trap: box.ap50[i] is the i-th class *present*, not class i. With class 0
    absent, zipping the arrays against names in order labels every row with its
    predecessor's name — a table that looks right and attributes `car`'s score to
    `person`."""
    names = {0: "person", 1: "rider", 2: "car", 3: "truck"}
    # Only classes 2 and 3 appear in the holdout.
    box = _FakeBox(present=[2, 3], ap50=[0.71, 0.42], ap=[0.40, 0.21])
    rows = _holdout.per_class(_FakeResult(box, names, nt=[0, 0, 5000, 300]), names=names)

    assert [r["name"] for r in rows] == ["car", "truck"]
    assert [r["AP50"] for r in rows] == [0.71, 0.42]
    assert [r["instances"] for r in rows] == [5000, 300]


def test_a_class_absent_from_the_holdout_is_omitted_rather_than_given_the_fleet_average():
    """Ultralytics' `maps` pre-fills absent classes with the overall mAP, so `train`
    — 29 instances fleet-wide, routinely absent from a 1000-image holdout — would be
    reported as scoring whatever the average was. Silent, and flattering."""
    names = {0: "person", 1: "car", 2: "train"}
    box = _FakeBox(present=[0, 1], ap50=[0.30, 0.70], ap=[0.15, 0.40])
    rows = _holdout.per_class(_FakeResult(box, names, nt=[100, 900, 0]), names=names)

    assert [r["class_id"] for r in rows] == [0, 1]
    assert "train" not in {r["name"] for r in rows}


def test_class_names_come_from_the_holdout_yaml_not_from_the_checkpoint():
    """`YOLO(weights).val()` names its classes from the MODEL. This project's
    checkpoints are built from yolov8s-13.yaml, which declares no `names:`, so
    Ultralytics defaults them to {0: "0", 1: "1", ...} -- and a whole per-class table
    rendered with every row labelled by its own index. The holdout's data.yaml has
    the real list."""
    box = _FakeBox(present=[0, 2], ap50=[0.3, 0.6], ap=[0.1, 0.35])
    placeholder = {i: str(i) for i in range(13)}      # what the checkpoint carries
    result = _FakeResult(box, placeholder, nt=[1277, 52, 10389])

    fell_back = _holdout.per_class(result, names=None)
    if not all(r["name"] == str(r["class_id"]) for r in fell_back):
        # A holdout exists on disk, so the yaml supplied real names.
        assert [r["name"] for r in fell_back] == ["person", "car"]

    # Explicit names always win, whatever the checkpoint claims.
    real = {0: "person", 2: "car"}
    assert [r["name"] for r in _holdout.per_class(result, names=real)] == ["person", "car"]


def test_holdout_fingerprint_separates_two_holdouts_that_record_identical_metadata():
    """size and seed describe the request, not the result: the same pair drawn from a
    val pool that has since grown gives different images and identical metadata. Two
    runs claiming 'the same holdout' need something checkable."""
    pool = {f"img{i}.jpg" for i in range(500)}
    a = _holdout.select(50, seed=4, val_pool=pool)
    b = _holdout.select(50, seed=4, val_pool=pool | {f"new{i}.jpg" for i in range(500)})

    assert _holdout.fingerprint(a) != _holdout.fingerprint(b), \
        "a grown pool produced a different holdout that fingerprints the same"
    assert _holdout.fingerprint(a) == _holdout.fingerprint(list(reversed(a))), \
        "fingerprint depends on listing order, so it cannot identify a holdout"


def test_the_report_pairs_a_class_delta_by_class_id_not_by_row_position():
    """A class present in the final round but absent from round 1 must get no delta,
    not its neighbour's. Rare classes drop out of a 1000-image holdout routinely, so
    this is the normal case rather than the edge one."""
    hold = [
        {"round": 1, "mAP50": 0.1, "per_class": [
            {"class_id": 2, "name": "car", "instances": 5000, "AP50": 0.60, "AP50-95": 0.3},
        ]},
        {"round": 2, "mAP50": 0.2, "per_class": [
            {"class_id": 0, "name": "person", "instances": 400, "AP50": 0.31, "AP50-95": 0.1},
            {"class_id": 2, "name": "car", "instances": 5000, "AP50": 0.66, "AP50-95": 0.4},
        ]},
    ]
    rows = [r for r in report._per_class_table(hold) if r.startswith("| car")
            or r.startswith("| person")]
    car = next(r for r in rows if r.startswith("| car"))
    person = next(r for r in rows if r.startswith("| person"))

    assert "+0.0600" in car, car
    assert "—" in person, "a class absent from round 1 was given a delta anyway"
    # car is 5000 of 5400 objects: the share column is the reason the mAP above is
    # mostly one class's number.
    assert "92.6%" in car, car
    assert "person" in "".join(rows)


def test_a_run_scored_before_per_class_existed_still_reports():
    """holdout_metrics.json from an earlier run has no per_class key at all."""
    assert report._per_class_table([{"round": 1, "mAP50": 0.1}]) == []
    assert report._per_class_table([{"round": 1, "mAP50": 0.1, "per_class": []}]) == []


# ------------------------------------------------------ live edge nodes
from pipeline import nodes as _nodes  # noqa: E402


@pytest.fixture(autouse=False)
def _clean_nodes():
    _nodes.reset()
    yield
    _nodes.reset()


def test_a_node_id_that_could_escape_a_path_is_refused_not_sanitised(_clean_nodes):
    """The id names a key and reaches a URL. Refused rather than cleaned: a silently
    renamed node is two rows in the listing and one confused operator."""
    for bad in ("../etc/passwd", "a/b", "", "a" * 33, "has space", ";rm -rf"):
        with pytest.raises(ValueError):
            _nodes.heartbeat(bad, {})
    assert _nodes.listing()["total"] == 0

    for good in ("cam-1", "cam.1", "A_9", "x"):
        _nodes.heartbeat(good, {})
    assert _nodes.listing()["total"] == 4


def test_an_oversized_frame_is_refused_rather_than_stored(_clean_nodes):
    with pytest.raises(ValueError):
        _nodes.heartbeat("cam-1", {}, frame=b"x" * (_nodes.MAX_FRAME_BYTES + 1))
    assert _nodes.frame("cam-1") is None


def test_only_the_newest_frame_per_node_is_kept(_clean_nodes):
    """Frames are held in memory, so an unbounded history would be a slow leak in a
    process that is meant to run for the length of a federation."""
    _nodes.heartbeat("cam-1", {}, frame=b"first")
    _nodes.heartbeat("cam-1", {}, frame=b"second")
    assert _nodes.frame("cam-1") == b"second"
    assert _nodes.listing()["total"] == 1


def test_a_node_that_stops_reporting_goes_offline_rather_than_disappearing(_clean_nodes):
    """Disappearing would read as 'never existed'. The dashboard has to be able to say
    a node was there and stopped, which is the failure an operator cares about."""
    now = 1000.0
    _nodes.heartbeat("cam-1", {"fps": 12}, now=now)
    live = _nodes.listing(now=now + 1)
    assert live["online"] == 1 and live["fleet_fps"] == 12.0

    gone = _nodes.listing(now=now + _nodes.OFFLINE_AFTER + 1)
    assert gone["total"] == 1, "the node vanished instead of going offline"
    assert gone["online"] == 0
    assert gone["fleet_fps"] == 0.0, "an offline node still counted toward fleet fps"


def test_the_registry_is_bounded(_clean_nodes):
    """Without a cap, a caller generating a fresh id per POST grows it forever."""
    for i in range(_nodes.MAX_NODES):
        _nodes.heartbeat(f"cam-{i}", {})
    with pytest.raises(ValueError):
        _nodes.heartbeat("one-too-many", {})
    # An existing node must still be able to report after the cap is reached.
    assert _nodes.heartbeat("cam-0", {"fps": 5})["fps"] == 5


def test_garbage_telemetry_cannot_break_the_dashboard(_clean_nodes):
    """Every field comes from another machine. A string where a float belongs must not
    reach the renderer, and open-ended `counts` must not grow without bound."""
    rec = _nodes.heartbeat("cam-1", {
        "fps": "not a number", "latency_ms": None, "detections": -5,
        "label": "L" * 500, "counts": {f"c{i}": i for i in range(200)},
    })
    assert rec["fps"] == 0.0 and rec["latency_ms"] == 0.0
    assert rec["detections"] == 0, "a negative count survived the clamp"
    assert len(rec["label"]) <= 64
    assert len(rec["counts"]) <= 32


def test_the_model_a_node_caches_on_is_content_addressed():
    """Re-running a federation rewrites global_round_1.pt with different weights under
    the same name. A node caching on the name would keep serving the previous run's
    model and never look stale."""
    info = _nodes.latest_model()
    if not info.get("available"):
        pytest.skip("no global checkpoint on disk")
    assert len(info["sha256"]) == 16
    assert info["name"].startswith("global_round_")
    assert "global_last" not in info["name"], \
        "global_last duplicates a round under a name that never changes"


def test_the_holdout_does_not_move_when_the_run_seed_does():
    """The holdout is the ruler; the run seed is what is being measured. They were the
    same field, so `--preset seeds` -- the one experiment whose whole job is to hold
    everything but the seed constant -- rebuilt the holdout for every arm and compared
    three scores taken on three different sets of images."""
    cmds = [" ".join(stages._cmd_holdout(Config(seed=s))) for s in (0, 1, 2)]
    assert len(set(cmds)) == 1, f"the holdout command varies with the run seed: {cmds}"
    assert "--seed 0" in cmds[0]

    # And it still moves when explicitly asked to.
    assert "--seed 7" in " ".join(stages._cmd_holdout(Config(holdout_seed=7)))


def test_the_experiment_driver_passes_through_the_levers_the_runner_has():
    """experiment.py builds runner command lines by hand, so a lever added to the
    runner is invisible here until someone adds it twice. Each of these was missing:
    partition was hardcoded 'condition' (which would have rebuilt an IID fleet as
    non-IID), gpu_fraction was never passed (every arm ran ~1.9x slower than the same
    run by hand), and `--all` retrained the centralised ceiling once per arm."""
    from pipeline import experiment

    cmd = " ".join(experiment.command({"seed": 3}, confirm=True))
    assert f"--partition {Config.partition}" in cmd, "partition is not the Config default"
    assert "--gpu-fraction" in cmd
    assert "--seed 3" in cmd

    swept = " ".join(experiment.command({"partition": "condition"}, confirm=False))
    assert "--partition condition" in swept, "an explicit arm setting was overridden"

    skipped = " ".join(experiment.command({"skip": "baseline"}, confirm=True))
    assert "--skip baseline" in skipped
    assert "--local-bn" in " ".join(experiment.command({"local_bn": True}, confirm=False))
    assert "--local-bn" not in " ".join(experiment.command({}, confirm=False))


# ------------------------------------------------- deployment engine
from pipeline import deploy as _deploy  # noqa: E402


def test_every_supernode_gets_its_own_clientappio_address():
    """Sharing one is the failure that looks like a hang: the second node binds
    nothing, registers nothing, and the federation waits forever for a client that
    never arrives."""
    addrs = []
    for i in range(5):
        cmd = _deploy.supernode_cmd(i, "127.0.0.1:9092")
        addrs.append(cmd[cmd.index("--clientappio-api-address") + 1])
    assert len(set(addrs)) == 5, f"SuperNodes share an address: {addrs}"
    assert all(a.startswith("127.0.0.1:") for a in addrs)
    # Every node dials the same fleet port; only their own API port differs.
    assert all("--superlink" in _deploy.supernode_cmd(i, "127.0.0.1:9092") for i in range(5))


def test_the_fleet_and_control_ports_are_not_the_same():
    """9092 is where SuperNodes dial in, 9093 is where `flwr run` submits. Pointing a
    SuperNode at the control port fails in a way that reads as a network problem."""
    assert _deploy.FLEET_PORT != _deploy.CONTROL_PORT
    link = _deploy.superlink_cmd("127.0.0.1")
    assert f"127.0.0.1:{_deploy.FLEET_PORT}" in link
    assert f"127.0.0.1:{_deploy.CONTROL_PORT}" in link
    # SuperNode API ports must not collide with either.
    used = {_deploy.FLEET_PORT, _deploy.CONTROL_PORT}
    assert not used & {_deploy.NODE_PORT_BASE + i for i in range(8)}


def test_the_federation_entry_is_appended_not_rewritten(tmp_path, monkeypatch):
    """~/.flwr/config.toml holds every federation on the machine, including the
    local-simulation one the rest of this pipeline runs on. Rewriting it would take
    the simulator down as a side effect of trying the deployment engine."""
    home = tmp_path
    monkeypatch.setattr(_deploy.Path, "home", staticmethod(lambda: home))
    cfg = home / ".flwr" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('[superlink.local-simulation]\naddress = ":local:"\n', encoding="utf-8")

    _deploy.ensure_federation("local-deployment", "127.0.0.1")
    text = cfg.read_text(encoding="utf-8")
    assert "[superlink.local-simulation]" in text, "the simulation federation was lost"
    assert "[superlink.local-deployment]" in text
    assert "insecure = true" in text

    # Idempotent: running the deployment twice must not duplicate the section.
    _deploy.ensure_federation("local-deployment", "127.0.0.1")
    assert cfg.read_text(encoding="utf-8").count("[superlink.local-deployment]") == 1


def test_the_run_config_the_deployment_submits_matches_the_simulation_one():
    """The two paths build their own command lines. A lever added to one and not the
    other means the deployment silently trains a different configuration -- and the
    keys must all be declared in pyproject or flwr refuses the whole run."""
    cfg = Config(rounds=3, local_epochs=2, n_vehicles=4, local_bn=True)
    deployed = " ".join(_deploy.run_cmd("local-deployment", cfg))
    simulated = " ".join(stages._cmd_federate(cfg))

    declared = (paths.PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    body = declared.split("[tool.flwr.app.config]", 1)[1].split("[tool.flwr", 1)[0]
    known = set(re.findall(r"^(\w+) *=", body, re.M))

    for cmdline, label in ((deployed, "deploy"), (simulated, "simulate")):
        sent = set(re.findall(r"(\w+)=", cmdline.split("--run-config", 1)[1]))
        assert sent <= known, f"{label} sends undeclared run-config keys: {sorted(sent - known)}"
    assert "local_bn=true" in deployed and "num_server_rounds=3" in deployed


def _rounds(*scores):
    return [{"round": i, "mAP50": s, "checkpoint": f"global_round_{i}.pt"}
            for i, s in enumerate(scores, 1)]


def test_a_long_run_reports_the_round_it_peaked_at_not_the_one_it_stopped_at():
    """Measured twice in this project: a warm-started model scored 0.2582 untrained
    against 0.2073 after two rounds, and a 1.667x-budget ceiling scored LOWER than a
    smaller one. Reporting the final checkpoint turns "we trained longer" into "we
    reported a worse model"."""
    info = _holdout.best_round(_rounds(0.10, 0.31, 0.28, 0.25))
    assert info["best_round"] == 2 and info["best"] == 0.31
    assert info["final_round"] == 4 and info["final"] == 0.25
    assert info["regression"] == 0.06, "the cost of the rounds after the peak"
    assert info["wasted_rounds"] == 2


def test_a_run_still_improving_reports_no_regression():
    info = _holdout.best_round(_rounds(0.10, 0.20, 0.30))
    assert info["best_round"] == 3 and info["wasted_rounds"] == 0
    assert info["regression"] == 0.0


def test_a_tie_goes_to_the_earlier_round():
    """Two rounds of equal score are not equally good: the earlier one cost less GPU,
    and preferring it is what makes `wasted_rounds` mean anything."""
    info = _holdout.best_round(_rounds(0.10, 0.30, 0.30))
    assert info["best_round"] == 2
    assert info["wasted_rounds"] == 1


def test_best_round_survives_a_curve_with_nothing_in_it():
    assert _holdout.best_round([]) == {}
    assert _holdout.best_round([{"round": 1}]) == {}, "a row with no score is not a best"


def test_promote_copies_rather_than_moves_so_the_curve_stays_reproducible(tmp_path, monkeypatch):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for i in (1, 2, 3):
        (ckpts / f"global_round_{i}.pt").write_bytes(bytes([i]))
    monkeypatch.setattr(_holdout.paths, "PROJECT", tmp_path)

    dst = _holdout.promote(_rounds(0.1, 0.9, 0.4))
    assert dst is not None and dst.name == "global_best.pt"
    assert dst.read_bytes() == bytes([2]), "promoted the wrong round"
    # The per-round checkpoints must survive: the curve is measured from them.
    assert all((ckpts / f"global_round_{i}.pt").exists() for i in (1, 2, 3))


def test_the_holdout_stage_runs_before_the_fleet_stage():
    """Order is load-bearing: a holdout carved afterwards is already in someone's
    val split."""
    names = [s.name for s in stages.STAGES]
    assert names.index("holdout") < names.index("fleet")
    assert names.index("evaluate") > names.index("federate")
    assert stages.BY_NAME["baseline"].gated, "it trains a whole model"


def test_evaluating_a_checkpoint_that_is_not_there_fails_loudly(tmp_path):
    """A zero would look like a bad model rather than a missing one."""
    with pytest.raises(SystemExit):
        _holdout.evaluate(tmp_path / "nope.pt")


def test_the_gap_is_only_reported_when_both_halves_exist(monkeypatch):
    monkeypatch.setattr(_baseline, "result", lambda: {})
    monkeypatch.setattr(_holdout, "curve", lambda: {"rounds": [{"mAP50": 0.4}]})
    assert _baseline.gap() == {}

    monkeypatch.setattr(_baseline, "result", lambda: {"mAP50": 0.5, "epochs": 24, "images": 8400})
    g = _baseline.gap()
    assert g["federated_mAP50"] == 0.4 and g["centralised_mAP50"] == 0.5
    assert abs(g["gap"] - 0.1) < 1e-9 and abs(g["retained"] - 0.8) < 1e-9


def test_the_baseline_budget_matches_the_federated_one():
    """rounds x local_epochs epochs over the pooled set is the same number of
    image-visits the fleet makes; anything else flatters one side."""
    cfg = Config(rounds=6, local_epochs=4)
    cmd = stages._cmd_baseline(cfg)
    assert "--rounds" in cmd and cmd[cmd.index("--rounds") + 1] == "6"
    assert cmd[cmd.index("--local-epochs") + 1] == "4"


def test_fleet_check_catches_a_partition_mismatch(monkeypatch):
    """A condition fleet must not be silently reused for a random run."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "night", "n_train": 300, "n_val": 60} for i in range(1, 11)])
    assert stages._check_fleet(Config(n_vehicles=6, partition="random")).satisfied is False
    assert stages._check_fleet(Config(n_vehicles=6, partition="condition")).satisfied is True


def test_the_report_leads_with_the_metric_no_client_could_flatter(monkeypatch):
    """A report that shows only self-evaluated numbers invites the comparison that
    the holdout exists to prevent.

    The per-client rows are stubbed for the same reason the holdout and baseline are:
    without them the section simply does not render, so this asserted the caveat was
    present only on a machine that had already completed a run.
    """
    from pipeline import baseline as _b, holdout as _h

    monkeypatch.setattr(logparse, "read_metrics_csv", lambda _p: [
        {"round": 1, "stage": "evaluate", "loss": 0.9, "precision": 0.5,
         "recall": 0.3, "mAP50": 0.35, "mAP50-95": 0.19}])
    monkeypatch.setattr(_h, "curve", lambda: {
        "holdout": {"size": 1000},
        "rounds": [{"round": 1, "mAP50": 0.35, "mAP50-95": 0.19, "precision": 0.5, "recall": 0.3},
                   {"round": 2, "mAP50": 0.43, "mAP50-95": 0.24, "precision": 0.6, "recall": 0.4}]})
    monkeypatch.setattr(_b, "gap", lambda: {
        "federated_mAP50": 0.43, "centralised_mAP50": 0.50, "gap": 0.07, "retained": 0.86})

    md = report.to_markdown(report.collect(config={"profile": "full"}))
    assert "## The honest global metric" in md
    assert "1000 images that no vehicle trained" in md
    assert "0.4300" in md and "0.5000" in md and "86.0%" in md
    # And the old number keeps its caveat rather than its old headline.
    assert "per client, on its own split" in md


def test_the_report_says_when_the_number_has_no_scale(monkeypatch):
    from pipeline import baseline as _b, holdout as _h

    monkeypatch.setattr(_h, "curve", lambda: {
        "holdout": {"size": 500},
        "rounds": [{"round": 1, "mAP50": 0.35, "mAP50-95": 0.19, "precision": 0.5, "recall": 0.3}]})
    monkeypatch.setattr(_b, "gap", lambda: {})
    md = report.to_markdown(report.collect(config={}))
    assert "still has no scale" in md


# ------------------------------------------------------------ shard validation
from pipeline import validate as _validate  # noqa: E402


def _shard(root, vid, train=("a.jpg",), val=("v.jpg",), label_text="0 0.5 0.5 0.2 0.2\n"):
    """A minimal but sound shard: listings, images and labels that agree."""
    shard = root / f"batch_{vid}"
    for split, names in (("train", train), ("val", val)):
        (shard / "images" / split).mkdir(parents=True, exist_ok=True)
        (shard / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            (shard / "images" / split / name).write_bytes(b"x")
            (shard / "labels" / split / f"{name.rsplit('.', 1)[0]}.txt").write_text(label_text)
        (shard / f"{split}.txt").write_text("".join(f"{n}\n" for n in names))
    return shard


def _checks(problems):
    return {p.check for p in problems}


def test_a_sound_fleet_reports_nothing(tmp_path):
    _shard(tmp_path, 1)
    _shard(tmp_path, 2, train=("b.jpg",), val=("w.jpg",))
    assert _validate.check_fleet(tmp_path, held=set()) == []


def test_an_image_without_a_label_is_caught(tmp_path):
    """It trains as a background image: the vehicle learns its condition is empty."""
    shard = _shard(tmp_path, 1)
    (shard / "labels" / "train" / "a.txt").unlink()
    assert "images with no label file" in _checks(_validate.check_fleet(tmp_path, held=set()))


def test_an_image_two_vehicles_both_hold_is_caught(tmp_path):
    """num_examples is FedAvg's weight, so a shared image votes twice."""
    _shard(tmp_path, 1, train=("a.jpg",))
    _shard(tmp_path, 2, train=("a.jpg",), val=("w.jpg",))
    assert ("images shared between two vehicles' train sets"
            in _checks(_validate.check_fleet(tmp_path, held=set())))


def test_an_image_in_both_train_and_val_is_caught(tmp_path):
    """Evaluation becomes a memory test and mAP stops meaning anything."""
    _shard(tmp_path, 1, train=("a.jpg",), val=("a.jpg",))
    assert ("images in both train and val of one shard"
            in _checks(_validate.check_fleet(tmp_path, held=set())))


def test_a_held_out_image_inside_a_shard_is_caught(tmp_path):
    """The holdout is the only honest metric; an image inside a shard undoes that."""
    _shard(tmp_path, 1, train=("a.jpg",))
    problems = _validate.check_fleet(tmp_path, held={"a.jpg"})
    assert "held-out images found inside a shard" in _checks(problems)


def test_a_listing_naming_a_file_that_is_not_there_is_caught(tmp_path):
    shard = _shard(tmp_path, 1)
    (shard / "train.txt").write_text("a.jpg\nghost.jpg\n")
    assert ("images listed but not materialised"
            in _checks(_validate.check_fleet(tmp_path, held=set())))


@pytest.mark.parametrize("text,why", [
    ("", "empty"),
    ("0 0.5 0.5\n", "too few fields"),
    ("x 0.5 0.5 0.2 0.2\n", "non-numeric"),
    ("99 0.5 0.5 0.2 0.2\n", "class id outside nc"),
    ("0 1.5 0.5 0.2 0.2\n", "unnormalised coordinates"),
])
def test_unusable_label_files_are_caught(tmp_path, text, why):
    _shard(tmp_path, 1, label_text=text)
    assert "unusable label files" in _checks(_validate.check_fleet(tmp_path, held=set())), why


def test_validation_repairs_nothing(tmp_path):
    """A validator that edits data hides the bug that produced the data."""
    shard = _shard(tmp_path, 1)
    (shard / "labels" / "train" / "a.txt").unlink()
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    _validate.check_fleet(tmp_path, held=set())
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


def test_validate_runs_before_the_gpu_stages():
    """Finding a broken shard after an hour of training is finding it too late."""
    names = [s.name for s in stages.STAGES]
    assert names.index("validate") > names.index("fleet")
    assert names.index("validate") < names.index("sanity") < names.index("federate")
    assert not stages.BY_NAME["validate"].gated, "seconds of scanning; never worth skipping"


# --------------------------------------------------------------- run comparison
from pipeline import compare as _compare  # noqa: E402


def _report(dir_, name, config, holdout_map=None, self_map=None, checksums=2):
    d = dir_ / name
    d.mkdir(parents=True)
    data = {"config": config, "checksums": list(range(checksums)), "learned": checksums > 1,
            "gpu": {"energy_wh": 10.0}, "stages": [{"seconds": 100}],
            "metrics": ([{"stage": "evaluate", "mAP50": self_map}] if self_map else [])}
    if holdout_map:
        data["holdout"] = {"rounds": [{"round": 1, "mAP50": holdout_map, "mAP50-95": 0.2}]}
    (d / "report.json").write_text(json.dumps(data))
    return d


def test_comparison_leads_with_the_holdout_number(tmp_path):
    _report(tmp_path, "20260101-000000", {"seed": 0, "rounds": 2}, holdout_map=0.40, self_map=0.46)
    _report(tmp_path, "20260102-000000", {"seed": 1, "rounds": 2}, holdout_map=0.42, self_map=0.44)
    runs = _compare.load(5, reports_dir=tmp_path)

    assert [r["holdout_mAP50"] for r in runs] == [0.40, 0.42]
    out = _compare.table(runs)
    assert "holdout mAP50" in out and out.index("holdout mAP50") < out.index("self mAP50")
    assert "+0.0200" in out


def test_comparison_says_when_more_than_one_setting_changed(tmp_path):
    """Two changed variables mean the difference cannot be attributed to either."""
    _report(tmp_path, "20260101-000000", {"seed": 0, "rounds": 2, "strategy": "fedavg"},
            holdout_map=0.40)
    _report(tmp_path, "20260102-000000", {"seed": 0, "rounds": 6, "strategy": "fedadam"},
            holdout_map=0.45)
    out = _compare.table(_compare.load(5, reports_dir=tmp_path))
    assert "WARNING" in out and "rounds" in out and "strategy" in out


def test_a_single_changed_setting_is_not_warned_about(tmp_path):
    _report(tmp_path, "20260101-000000", {"seed": 0, "rounds": 2}, holdout_map=0.40)
    _report(tmp_path, "20260102-000000", {"seed": 1, "rounds": 2}, holdout_map=0.41)
    out = _compare.table(_compare.load(5, reports_dir=tmp_path))
    assert "WARNING" not in out
    assert "seed" in out, "the varying setting should still be shown as a column"


def test_runs_without_a_holdout_number_are_called_out(tmp_path):
    """They predate the holdout, so they cannot be compared between fleets."""
    _report(tmp_path, "20260101-000000", {"seed": 0}, self_map=0.46)
    out = _compare.table(_compare.load(5, reports_dir=tmp_path))
    assert "no holdout number" in out


def test_a_corrupt_report_does_not_stop_the_comparison(tmp_path):
    _report(tmp_path, "20260101-000000", {"seed": 0}, holdout_map=0.40)
    bad = tmp_path / "20260102-000000"
    bad.mkdir()
    (bad / "report.json").write_text("{not json")
    assert len(_compare.load(5, reports_dir=tmp_path)) == 1


# ------------------------------------------------------------------ experiments
from pipeline import experiment as _exp  # noqa: E402


def test_every_arm_of_a_preset_changes_exactly_one_setting():
    """Two changed variables in one comparison explain neither."""
    base = {"profile": "demo", "vehicles": 6, "rounds": 2, "epochs": 1, "per_vehicle": 0}
    for preset, expect in (("seeds", "seed"), ("strategies", "strategy"),
                           ("partitions", "partition")):
        arms = _exp.arms_for(preset, base, [0, 1], ["fedavg", "fedadam"],
                             ["condition", "random"], [0.5])
        assert len(arms) == 2
        differing = {k for k in arms[0] if k != "label" and arms[0][k] != arms[1][k]}
        assert differing == {expect}, f"{preset} varies {differing}"


def test_the_alpha_preset_also_selects_the_partition_it_belongs_to():
    """alpha means nothing under condition partitioning; asking for it must switch."""
    arms = _exp.arms_for("alpha", {"profile": "demo"}, [], [], [], [0.05, 100.0])
    assert {a["partition"] for a in arms} == {"dirichlet"}
    assert [a["alpha"] for a in arms] == [0.05, 100.0]


def test_an_unknown_preset_is_refused():
    with pytest.raises(SystemExit):
        _exp.arms_for("vibes", {}, [], [], [], [])


def test_an_arm_becomes_a_real_runner_invocation():
    """Arms are driven through pipeline.runner, not a second code path that can
    drift from the one people actually use."""
    cmd = _exp.command({"profile": "full", "rounds": 6, "epochs": 4, "seed": 2,
                        "strategy": "fedadam", "partition": "dirichlet", "alpha": 0.3,
                        "per_vehicle": 1400}, confirm=True)
    assert cmd[1:4] == ["-m", "pipeline.runner", "--all"]
    for flag, value in (("--profile", "full"), ("--rounds", "6"), ("--epochs", "4"),
                        ("--seed", "2"), ("--strategy", "fedadam"),
                        ("--partition", "dirichlet"), ("--alpha", "0.3"),
                        ("--per-vehicle", "1400")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--yes" in cmd


def test_the_runbook_only_promises_commands_that_exist():
    """A runbook that names a module nobody wrote is worse than no runbook."""
    import re
    text = (REPO / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    modules = set(re.findall(r"python -m (pipeline\.[a-z_]+)", text))
    assert modules, "the runbook should name the entry points"
    for mod in modules:
        assert (REPO / "pipeline" / f"{mod.split('.')[1]}.py").is_file(), f"{mod} missing"
    for script in ("scripts/run_pipeline.ps1", "scripts/run_pipeline.sh"):
        assert (REPO / script).is_file(), f"{script} is referenced but absent"


def test_the_baseline_pools_only_the_shards_that_trained(monkeypatch, tmp_path):
    """A 6-vehicle run trains 6 of the 10 materialised shards. Pooling all ten hands
    the centralised model data the federation never saw, and the gap then measures
    the extra data."""
    from pipeline import baseline as _b, vehicle_metrics as _vm

    batches = tmp_path / "batch"
    for i in range(1, 11):
        (batches / f"batch_{i}").mkdir(parents=True)
        (batches / f"batch_{i}" / "train.txt").write_text(f"img{i}a.jpg\nimg{i}b.jpg\n")
    monkeypatch.setattr(paths, "VEHICLE_BATCHES", batches)
    monkeypatch.setattr(_vm, "per_vehicle_rounds", lambda: {"1": [], "2": [], "5": []})

    assert _b.trained_shards() == [1, 2, 5]
    assert len(_b.pooled_names()) == 6, "only the three shards that trained"
    assert len(_b.pooled_names(shards=list(range(1, 11)))) == 20


def test_parity_flags_a_ceiling_that_was_given_more_compute():
    """The run of 2026-08-06 pooled 14 000 images for 24 epochs against a federation
    that made 201 600 image-visits: 1.667x, and its retention figure is a bound."""
    from pipeline import baseline as _b

    over = _b.parity(images=14000, epochs=24, shards=6, per_vehicle=1400,
                     rounds=6, local_epochs=4)
    assert over["ratio"] == 1.667 and over["matched"] is False

    fair = _b.parity(images=8400, epochs=24, shards=6, per_vehicle=1400,
                     rounds=6, local_epochs=4)
    assert fair["ratio"] == 1.0 and fair["matched"] is True


def test_the_report_calls_an_over_provisioned_ceiling_what_it_is(monkeypatch):
    from pipeline import baseline as _b, holdout as _h

    monkeypatch.setattr(_h, "curve", lambda: {
        "holdout": {"size": 1000},
        "rounds": [{"round": 1, "mAP50": 0.43, "mAP50-95": 0.24,
                    "precision": 0.6, "recall": 0.4}]})
    monkeypatch.setattr(_b, "gap", lambda: {
        "federated_mAP50": 0.4334, "centralised_mAP50": 0.4771, "gap": 0.0437,
        "retained": 0.908, "matched": False, "budget_ratio": 1.667})
    md = report.to_markdown(report.collect(config={}))
    assert "lower bound" in md and "1.667" in md


def test_the_sanity_stage_does_not_shell_out_to_a_moving_target(monkeypatch, tmp_path):
    """`python -m ultralytics.cfg` ran until ultralytics 8.4 made cfg a package with
    no __main__, and the whole chain then halted at the first GPU stage. The sanity
    stage calls the same API a client calls, which cannot drift from it.

    The shard is faked rather than assumed present: `_cmd_sanity` refuses without one
    (the sibling test below asserts that), so this passed only on a machine that had
    already built a fleet — and failed on every clean clone, which is what CI is.
    """
    shards = tmp_path / "batch"
    (shards / "batch_1").mkdir(parents=True)
    (shards / "batch_1" / "data.yaml").write_text("nc: 13\n")
    monkeypatch.setattr(paths, "VEHICLE_BATCHES", shards)

    cmd = stages._cmd_sanity(Config(profile="demo"))
    assert cmd[1] == "-c", "invoke the API, not a console script that may not exist"
    body = cmd[2]
    assert "ultralytics.cfg" not in body
    assert "from ultralytics import YOLO" in body
    assert "imgsz=320" in body
    # And it trains on a shard that exists, not on the data.runtime.yaml a client
    # writes at runtime -- which made the stage pass only where it was not needed.
    assert "data.runtime.yaml" not in body
    # repr, because the snippet embeds the path as a Python literal -- on Windows a
    # raw path would turn every backslash into an escape.
    assert repr(str(paths.VEHICLE_BATCHES / "batch_1" / "data.yaml")) in body


def test_the_sanity_stage_refuses_to_run_without_a_shard(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "VEHICLE_BATCHES", tmp_path / "nothing")
    with pytest.raises(RuntimeError, match="fleet stage"):
        stages._cmd_sanity(Config())


def test_a_line_the_console_cannot_encode_does_not_kill_the_output_thread(monkeypatch):
    """On Windows a redirected stdout is cp1252, Ultralytics' progress bars are not,
    and an unguarded print raised inside the drain thread -- which then died, the pipe
    it was reading filled, and the stage failed for a reason nothing in the log
    explained. Reproduced with a real strict cp1252 stream."""
    import io
    from pipeline import runner as _r

    raw = io.BytesIO()
    strict = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(_r.sys, "stdout", strict)
    monkeypatch.setattr("sys.stdout", strict)

    _r._safe_print("progress ━━ bar")      # box drawing: not in cp1252
    strict.flush()

    monkeypatch.undo()
    written = raw.getvalue().decode("cp1252")
    assert "progress" in written and "bar" in written, written


def test_flwr_is_resolved_next_to_the_interpreter_not_from_path(monkeypatch, tmp_path):
    """A shell a person types into has the venv on PATH; one started by a script does
    not, and the stage died with '[WinError 2] The system cannot find the file
    specified' -- which names no file.

    The shim next to the interpreter is now the *evidence* that flwr is installed here
    rather than the thing that runs: Application Control blocks the generated .exe. What
    must still hold is that the environment is the interpreter's own, not PATH's.
    """
    fake = tmp_path / "flwr.exe"
    fake.write_bytes(b"")
    fake_py = tmp_path / "python.exe"
    monkeypatch.setattr(stages, "PY", str(fake_py))
    monkeypatch.setattr(stages.shutil, "which", lambda _name: "/another/env/bin/flwr")

    assert stages.flwr_launcher()[0] == str(fake_py),         "another environment's flwr on PATH must never win"
    assert stages._cmd_federate(Config())[0] == str(fake_py)


def test_a_missing_flwr_says_which_environment_to_install_it_into(monkeypatch, tmp_path):
    """The shim's existence is what proves flwr is installed in this environment, even
    though it is not what gets run. Without the check the failure would be a
    ModuleNotFoundError from three frames inside typer instead of a sentence."""
    monkeypatch.setattr(stages, "PY", str(tmp_path / "python.exe"))
    monkeypatch.setattr(stages.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="pip install flwr"):
        stages.flwr_launcher()


def test_every_subprocess_is_told_to_write_utf8():
    """flwr prints a flower emoji in its banner. With a redirected stdout on Windows
    the child gets cp1252 and dies with "'charmap' codec can't encode character
    '\U0001f338'" -- a federation failing for a reason unrelated to federation, and
    only when launched from a script rather than a terminal."""
    env = paths.subprocess_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] == "1"


def test_the_flwr_launcher_never_runs_the_blocked_shim():
    """A run halted at the federate stage with

        [WinError 4551] An Application Control policy has blocked this file

    on `Scripts\flwr.exe`. A console script is an unsigned executable generated on this
    machine, which is what those policies stop -- Smart App Control has already made conda
    unusable here for the same reason. The interpreter runs the same CLI with no .exe to
    block, so the launcher must be an interpreter invocation and must stay one."""
    import sys

    cmd = stages.flwr_launcher()

    assert cmd[0] == sys.executable, "the launcher must be this interpreter"
    assert cmd[1] == "-c" and "flwr.cli.app" in cmd[2]
    assert not any(str(part).endswith(".exe") and "flwr" in str(part).lower()
                   for part in cmd), "the shim is the thing that gets blocked"


def test_the_interpreters_scripts_directory_leads_the_path():
    """flwr resolves `flower-superlink` from PATH, not from its own location, so a
    non-interactive shell without the venv activated failed with
    'Unable to launch flower-superlink ... [WinError 2]'. Our children spawn children."""
    import os
    import sys

    env = paths.subprocess_env()
    scripts = str(Path(sys.executable).parent)
    entries = env["PATH"].split(os.pathsep)
    assert entries[0] == scripts
    assert sum(1 for e in entries if os.path.normcase(e) == os.path.normcase(scripts)) == 1


def test_the_scripts_directory_is_moved_to_the_front_not_merely_left_on_the_path(
        monkeypatch):
    """Present is not first, and only first is the guarantee that matters.

    The prepend used to be skipped whenever the directory was already anywhere on
    PATH. On a GitHub Windows runner `setup-python` puts it there behind PowerShell's
    own directory, so CI ran with the ordering this function claims to enforce
    silently absent — and "somewhere on PATH" is exactly the state in which another
    environment's `flower-superlink` resolves first.
    """
    import os
    import sys

    scripts = str(Path(sys.executable).parent)
    monkeypatch.setenv("PATH", os.pathsep.join(["/somewhere/else", scripts, "/later"]))

    entries = paths.subprocess_env()["PATH"].split(os.pathsep)
    assert entries[0] == scripts, "already-present is not the same as first"
    assert entries.count(scripts) == 1, "the old entry was left behind as a duplicate"
    assert "/somewhere/else" in entries and "/later" in entries, "the rest of PATH was dropped"


def test_checkpoints_from_a_previous_run_are_not_plotted_as_this_ones(tmp_path, monkeypatch):
    """A 2-round run leaves rounds 3..6 from a previous 6-round one in the directory.
    Scoring all of them drew a curve that jumped 0.014 -> 0.217 between round 2 and
    round 3: two different models plotted as one."""
    import os
    import time
    from pipeline import holdout as _h

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    old = time.time() - 3600
    for i in (3, 4, 5, 6):
        f = ckpts / f"global_round_{i}.pt"
        f.write_bytes(b"old")
        os.utime(f, (old, old))
    for i in (1, 2):
        (ckpts / f"global_round_{i}.pt").write_bytes(b"new")

    monkeypatch.setattr(paths, "PROJECT", tmp_path)
    assert [p.name for p in _h.checkpoints()] == ["global_round_1.pt", "global_round_2.pt"]


def test_a_demo_ceiling_does_not_overwrite_the_archive_of_a_full_one(tmp_path):
    """40 minutes of GPU time should not be replaceable by 30 seconds of it."""
    from pipeline import baseline as _b

    names = {f"baseline-{images}img-{epochs}ep.json"
             for images, epochs in ((8400, 24), (1200, 2))}
    assert len(names) == 2, "the archive name must distinguish the budgets"


def test_the_checksum_criterion_judges_one_run_not_a_pile_of_them(tmp_path):
    """Logs are per process and accumulate. Reading all of them gave eleven
    checksums for a three-round run, so the single most important signal in this
    project was being computed over a mixture."""
    import os
    import time

    old = time.time() - 3600
    (tmp_path / "server.111.log").write_text(
        "Aggregated parameters with checksum: 5.0\n"
        "Aggregated parameters with checksum: 5.0\n")     # a previous run that stalled
    os.utime(tmp_path / "server.111.log", (old, old))
    (tmp_path / "server.222.log").write_text(
        "Aggregated parameters with checksum: 1.0\n"
        "Aggregated parameters with checksum: 2.0\n")

    assert logparse.aggregate_checksums(tmp_path) == [1.0, 2.0]
    assert logparse.aggregate_checksums(tmp_path, all_runs=True) == [5.0, 5.0, 1.0, 2.0]
    # And the verdict follows this run, not the pile: the old run's stall must not
    # condemn a federation that is learning.
    assert logparse.federation_learned(tmp_path)[0] is True


def test_no_document_promises_a_command_with_a_mangled_path():
    r"""A backslash escape keeps eating part of a Windows path in a document.

    Three times now. Twice it was `.\scripts\run_pipeline.ps1` with the `\r` collapsed into a
    carriage return; on 2026-09-26 it was `Scripts\flwr.exe` with the `\f` collapsed into a
    formfeed -- in the environment-trap note about that very executable. Both leave a
    headline command nobody can run, and neither shows up in a rendered diff.

    So this no longer looks for one known string. No document may contain a control
    character at all, and the search is recursive: the formfeed landed in
    `docs/ENV_WINDOWS.md` and a sibling in `docs/findings/`, which two flat globs missed.
    """
    broken = b".\scripts\rrun_pipeline.ps1".replace(b"\rrun", b"\run")
    # Formfeed, vertical tab, backspace, bell, escape: what a collapsed \f, \v, \b, \a or
    # \e leaves behind. Tab and newline are excluded -- both are legitimate in Markdown.
    control = bytes([0x0c, 0x0b, 0x08, 0x07, 0x1b])

    for doc in [*REPO.glob("*.md"), *(REPO / "docs").rglob("*.md")]:
        body = doc.read_bytes()
        assert broken not in body, doc.name
        found = [f"0x{c:02x}" for c in control if bytes([c]) in body]
        assert not found, (
            f"{doc.relative_to(REPO)} contains {', '.join(found)} -- a backslash escape "
            f"was collapsed into a control character, so a path in it is unrunnable")


def test_a_stage_can_be_skipped_from_the_full_chain():
    """--all includes the gated baseline stage, so a script that also invoked
    pipeline.baseline afterwards trained the ceiling twice -- 40 minutes of GPU time
    for a duplicate."""
    full = [s.name for s in stages.resolve(None)]
    assert "baseline" in full

    without = [s.name for s in stages.resolve(None, skip="baseline")]
    assert "baseline" not in without
    assert without == [n for n in full if n != "baseline"], "nothing else moved"

    # Skipping by name, so a stage added later is still included by default rather
    # than silently missing from a hardcoded list.
    assert [s.name for s in stages.resolve("env,fleet", skip="fleet")] == ["env"]


def test_skipping_a_stage_that_does_not_exist_is_refused():
    with pytest.raises(SystemExit, match="unknown stage"):
        stages.resolve(None, skip="basline")


# ----------------------------------------------------------- data + plan views
from pipeline import dataset_stats as _ds, plan as _plan  # noqa: E402


def test_class_histogram_counts_instances_not_files(tmp_path):
    """Two objects in one file is two objects. The headline finding -- car is 55% of
    BDD100K -- is wrong by a factor if files are counted instead."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("2 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.1 0.1\n9 0.3 0.3 0.1 0.1\n")
    (labels / "b.txt").write_text("2 0.5 0.5 0.1 0.1\n")
    (labels / "notalabel.jpg").write_bytes(b"x")

    assert _ds.class_histogram(labels) == {"2": 3, "9": 1}
    assert _ds.class_histogram(tmp_path / "nope") == {}


def test_dataset_stats_are_cached_against_the_fleet_fingerprint(tmp_path, monkeypatch):
    """Reading 14 000 label files takes seconds; the 2-second poll must never do it.
    A rebuilt fleet has a new fingerprint and invalidates the cache by itself."""
    from pipeline import vehicles as _v

    monkeypatch.setattr(_ds, "CACHE", tmp_path / "stats.json")
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"fleet": [], "class_names": []}

    monkeypatch.setattr(_ds, "compute", counted)
    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {"fingerprint": "aaa"})

    _ds.cached()
    _ds.cached()
    assert calls["n"] == 1, "the second call must come from the cache"

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {"fingerprint": "bbb"})
    _ds.cached()
    assert calls["n"] == 2, "a different fleet must recount"


def test_the_plan_states_the_budget_both_sides_must_match():
    """The centralised epochs it quotes must be the ones that make the image-visits
    equal -- the arithmetic whose absence produced a 1.667x ceiling."""
    from pipeline import baseline as _b

    cfg = Config(profile="full", n_vehicles=6, rounds=6, local_epochs=4,
                 per_vehicle_override=1400)
    b = _plan.budget(cfg)
    assert b["image_visits"] == 6 * 1400 * 6 * 4 == 201600
    assert b["pooled_images"] == 8400

    par = _b.parity(b["pooled_images"], b["centralised_epochs_to_match"],
                    cfg.n_vehicles, cfg.per_vehicle, cfg.rounds, cfg.local_epochs)
    assert par["ratio"] == 1.0 and par["matched"] is True


def test_the_plan_warns_about_a_condition_it_cannot_fill(monkeypatch):
    """Asking for more images than the rarest condition holds turns a non-IID run
    into a nearly-IID one, silently."""
    monkeypatch.setattr(_plan.holdout, "names", lambda: {"x.jpg"})
    warns = _plan.warnings(Config(profile="full", partition="condition",
                                  per_vehicle_override=6308))
    assert any("rarest condition" in w for w in warns)

    quiet = _plan.warnings(Config(profile="demo", partition="condition", rounds=2))
    assert not any("rarest condition" in w for w in quiet)


def test_the_plan_warns_when_there_is_no_holdout(monkeypatch):
    monkeypatch.setattr(_plan.holdout, "names", lambda: set())
    warns = _plan.warnings(Config(rounds=2))
    assert any("comparable with nothing" in w for w in warns)


def test_the_plan_emits_commands_that_the_runner_accepts():
    """A plan quoting a command the CLI would reject is worse than none."""
    cfg = Config(profile="full", n_vehicles=6, rounds=6, local_epochs=4,
                 per_vehicle_override=1400, partition="dirichlet", alpha=0.3,
                 strategy="fedadam")
    from pipeline import runner

    parser = runner.build_parser()
    for entry in _plan.commands(cfg):
        parts = entry["cmd"].split()
        if parts[2:4] != ["pipeline.runner", "--all"]:
            continue
        parser.parse_args(parts[4:])        # raises SystemExit on anything unknown


# --------------------------------------------------------------------- docs tab
from pipeline import docs_index as _docs  # noqa: E402


def test_every_program_in_the_docs_index_exists_and_imports():
    """A docs page that names a module nobody wrote is worse than no docs page."""
    d = _docs.index()
    assert d["modules"], "the index should list the package's programs"
    for m in d["modules"]:
        path = REPO / m["module"]
        assert path.is_file(), m["module"]
        assert "could not import" not in m["doc"], f"{m['module']}: {m['doc'][:80]}"
        assert m["summary"], f"{m['module']} has no docstring first line"
        assert m["contributes"], f"{m['module']} does not say what it is for"


def test_the_docs_text_comes_from_the_modules_not_a_copy():
    """Copied prose rots. The summary must be the module's own first line."""
    from pipeline import holdout as _h

    entry = next(m for m in _docs.index()["modules"] if m["module"].endswith("holdout.py"))
    assert entry["summary"] == (_h.__doc__ or "").strip().splitlines()[0]


def test_the_documented_chain_is_the_chain_that_runs():
    chain = [s["name"] for s in _docs.index()["chain"]]
    assert chain == [s.name for s in stages.STAGES]
    gated = {s["name"] for s in _docs.index()["chain"] if s["gated"]}
    assert {"dataset", "sanity", "federate", "baseline"} <= gated


def test_the_documented_tabs_are_the_tabs_the_page_has():
    """A Docs tab describing a tab that does not exist is its own kind of lie."""
    html = (REPO / "pipeline" / "static" / "index.html").read_text(encoding="utf-8")
    for tab in _docs.index()["tabs"]:
        assert f'data-view="{tab["name"].lower()}"' in html, tab["name"]
    import re
    in_html = set(re.findall(r'data-view="([a-z]+)"', html))
    documented = {t["name"].lower() for t in _docs.index()["tabs"]}
    assert in_html == documented, (in_html ^ documented)


def test_every_element_the_run_form_reaches_for_exists():
    """`$("skew")` on an id the markup lacks throws, and the whole form stops working.

    control.js reads the form by id and index.html declares them; nothing links the two
    but this. The failure is silent in the worst way -- the page renders, and the first
    keystroke kills the estimate handler.
    """
    static = REPO / "pipeline" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "js" / "control.js").read_text(encoding="utf-8")

    ids = re.findall(r'id="([\w-]+)"', html)
    declared = set(ids)
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', js))
    wanted |= set(re.findall(r'"([\w-]+)"(?=[^\n]*\.forEach\(id)', js))

    missing = sorted(wanted - declared)
    assert not missing, f"control.js reads ids index.html does not declare: {missing}"
    # A duplicate id is worse than a missing one: getElementById returns the *first*
    # match and nothing errors. A "Size skew" input added as id="skew" would silently
    # have read the version-skew banner instead, and the form would have posted 0.
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"index.html declares these ids more than once: {dupes}"
    assert {"sizeSkew", "sizeSkewNote", "alpha", "partition"} <= wanted, "the form lost a control"


def test_no_view_module_reaches_for_an_element_the_markup_lacks():
    """The same failure as the test above, for every other view.

    `$("x")` on a missing id returns null and the next property access throws, which
    kills the whole render pass -- so one typo in a panel added to live.js takes the
    heartbeat, the GPU readouts and the criteria down with it, and the page just stops
    updating. Widened from control.js only, after a per-class panel added ids read by
    live.js that nothing was checking.
    """
    static = REPO / "pipeline" / "static"
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((static / "js").glob("*.js"))}
    declared = set(re.findall(r'id="([\w-]+)"', (static / "index.html").read_text(encoding="utf-8")))
    # Panels are rendered into the page by other modules, so an id can be perfectly
    # valid without appearing in index.html -- `consumedBody` is written by consumed.js
    # itself. Anything a module emits counts as declared; what this catches is an id
    # that NOTHING anywhere produces, which is always a typo or a deleted element.
    for src in sources.values():
        declared |= set(re.findall(r'id="([\w-]+)"', src))

    problems = {}
    for name, src in sources.items():
        # Only the literal `$("id")` form. Ids built at runtime (`$("v" + vid)`) are
        # the drawer's business and cannot be checked from here.
        wanted = set(re.findall(r'\$\("([\w-]+)"\)', src))
        missing = sorted(wanted - declared)
        if missing:
            problems[name] = missing
    assert not problems, f"view modules read ids nothing produces: {problems}"

    # Guard the guard: if the regex above ever stops matching, the test passes
    # vacuously and protects nothing.
    assert "perClass" in declared and "holdoutNote" in declared


def test_client_logs_are_scoped_to_the_run_that_is_being_measured(tmp_path):
    """`client.<pid>.log` accumulates across runs. Reading all of them made a
    six-vehicle run look like a nine-vehicle one, and the centralised ceiling was
    then trained on 12 600 images against a federation that saw 8 400 -- a ceiling
    with 1.5x the budget, reported as matched."""
    import os
    import time

    old = time.time() - 7200
    for name in ("client.111.log", "client.222.log"):
        f = tmp_path / name
        f.write_text("Starting local training with batch_id=7, local_epochs=1\n")
        os.utime(f, (old, old))

    server = tmp_path / "server.333.log"
    server.write_text("Aggregated parameters with checksum: 1.0\n")
    for name in ("client.444.log", "client.555.log"):
        (tmp_path / name).write_text("Starting local training with batch_id=3, local_epochs=1\n")

    current = {f.name for f in logparse.current_run_logs("client*.log", tmp_path)}
    assert current == {"client.444.log", "client.555.log"}
    assert len(list(logparse.iter_logs("client*.log", tmp_path))) == 4, "the old ones are still there"


def test_per_vehicle_metrics_follow_the_current_run(tmp_path, monkeypatch):
    """The dashboard's fleet panel and the baseline's shard list both come from here."""
    import os
    import time

    logs = tmp_path / "logs"
    logs.mkdir()
    old = time.time() - 7200
    stale = logs / "client.1.log"
    stale.write_text("[Client] 9 Training done. metrics={'mAP50': 0.9}\n")
    os.utime(stale, (old, old))

    (logs / "server.2.log").write_text("Aggregated parameters with checksum: 1.0\n")
    (logs / "client.3.log").write_text("[Client] 4 Training done. metrics={'mAP50': 0.2}\n")

    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    rounds = vm.per_vehicle_rounds()
    assert set(rounds) == {"4"}, "vehicle 9 belongs to a run that is over"


# ------------------------------------------------------------------- the ledger
from pipeline import ledger as _ledger  # noqa: E402


def _ledger_report(dir_, name, cfg, holdout_curve=(), seconds=100, wh=10.0, epochs=None):
    d = dir_ / name
    d.mkdir(parents=True)
    data = {
        "config": cfg, "checksums": [1.0, 2.0], "learned": True,
        "gpu": {"energy_wh": wh, "peak_mem_mib": 5000, "mean_util_pct": 40},
        "stages": [{"name": "federate", "status": "ok", "seconds": seconds},
                   {"name": "verify", "status": "ok", "seconds": 1}],
        "metrics": [{"stage": "evaluate", "mAP50": 0.5}],
        "holdout": {"rounds": [{"round": i + 1, "mAP50": v, "mAP50-95": v / 2}
                               for i, v in enumerate(holdout_curve)]},
        "learning": {"trained": ["1"], "conditions": {"1": "night"},
                     "rounds": {"1": [{"mAP50": 0.3}]},
                     "epochs": {"1": epochs or [{"box_loss": 1.5, "cls_loss": 1.0, "dfl_loss": 1.1}]}},
    }
    (d / "report.json").write_text(json.dumps(data))
    return d


def test_the_ledger_records_what_each_approach_cost_and_produced(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPORTS", tmp_path)
    _ledger_report(tmp_path, "20260101-000000",
                   {"strategy": "fedavg", "partition": "condition", "n_vehicles": 6,
                    "rounds": 6, "local_epochs": 4, "per_vehicle": 1400},
                   holdout_curve=(0.33, 0.41), seconds=3296, wh=82.2)

    r = _ledger.load()[0]
    assert r["data"]["image_visits"] == 6 * 1400 * 6 * 4
    assert r["data"]["effective_epochs"] == 24
    assert r["result"]["holdout_mAP50"] == 0.41
    assert r["time"]["seconds"] == 3297.0
    assert r["cost"]["wh_per_point"] == round(82.2 / 0.41, 1)
    assert "fedavg" in r["approach"] and "1400img" in r["approach"]
    assert r["learning"]["epochs"]["1"][0]["box"] == 1.5, "per-epoch rows must survive"


def test_a_federation_above_its_own_ceiling_is_flagged_not_celebrated(tmp_path, monkeypatch):
    """0.4334 against a ceiling of 0.35 does not mean federation beat centralised; it
    means the ceiling was stale."""
    monkeypatch.setattr(paths, "REPORTS", tmp_path)
    d = _ledger_report(tmp_path, "20260101-000000", {"n_vehicles": 2, "rounds": 1,
                                                     "local_epochs": 1, "per_vehicle": 10},
                       holdout_curve=(0.5,))
    data = json.loads((d / "report.json").read_text())
    data["baseline"] = {"centralised_mAP50": 0.4, "retained": 1.25, "matched": True}
    (d / "report.json").write_text(json.dumps(data))

    assert _ledger.load()[0]["result"]["ceiling_suspect"] is True


def test_repeats_of_one_approach_are_grouped_so_the_noise_floor_is_visible(tmp_path, monkeypatch):
    """A difference between two approaches smaller than the spread across repeats of
    one of them says nothing."""
    monkeypatch.setattr(paths, "REPORTS", tmp_path)
    cfg = {"strategy": "fedavg", "partition": "condition", "n_vehicles": 4,
           "rounds": 2, "local_epochs": 1, "per_vehicle": 300}
    _ledger_report(tmp_path, "20260101-000000", {**cfg, "seed": 0}, holdout_curve=(0.20,))
    _ledger_report(tmp_path, "20260102-000000", {**cfg, "seed": 1}, holdout_curve=(0.24,))
    _ledger_report(tmp_path, "20260103-000000", {**cfg, "strategy": "fedadam"}, holdout_curve=(0.22,))

    groups = {g["approach"]: g for g in _ledger.by_approach()}
    fedavg = next(g for k, g in groups.items() if k.startswith("fedavg"))
    assert fedavg["n"] == 2 and fedavg["spread"] == 0.04
    assert fedavg["mean"] == 0.22
    assert next(g for k, g in groups.items() if k.startswith("fedadam"))["spread"] is None


def test_an_empty_server_log_is_not_mistaken_for_the_current_run(tmp_path):
    """Importing my-project's server_app configures a CWD-relative logger, so
    `logs/server.<pid>.log` appears merely because something imported the module --
    running the test suite from the repo root creates one. Treating that as the
    current run made verify report "need >=2 rounds to tell, saw 0" straight after a
    six-round federation had succeeded."""
    import os
    import time

    real = tmp_path / "server.100.log"
    real.write_text("Aggregated parameters with checksum: 1.0\n"
                    "Aggregated parameters with checksum: 2.0\n")
    older = time.time() - 60
    os.utime(real, (older, older))

    # Newer, but it never aggregated anything: a process, not a run.
    (tmp_path / "server.200.log").write_text("[Server] Detected operating system: Windows\n")

    assert logparse.latest_run_log(tmp_path) == real
    assert logparse.aggregate_checksums(tmp_path) == [1.0, 2.0]
    assert logparse.federation_learned(tmp_path)[0] is True


def test_with_no_real_run_at_all_the_checksums_are_empty_not_wrong(tmp_path):
    (tmp_path / "server.1.log").write_text("[Server] starting\n")
    assert logparse.latest_run_log(tmp_path) is None
    assert logparse.aggregate_checksums(tmp_path) == []
    assert logparse.federation_learned(tmp_path)[0] is False


def test_the_mlflow_backend_is_a_database_because_the_filesystem_one_refuses_writes(monkeypatch):
    """mlflow 3.15 answers a file:// tracking URI with "The filesystem tracking backend
    (e.g., './mlruns') is in maintenance mode", so the sink logged nothing for an
    entire six-round run. The URI must also be absolute and POSIX-slashed: a Windows
    backslash escapes inside a SQLAlchemy URL, and a relative one puts a database
    wherever the subprocess was standing."""
    uri = paths.mlflow_uri()
    assert uri.startswith("sqlite:///"), uri
    assert "\\" not in uri, uri
    assert Path(uri.removeprefix("sqlite:///")).is_absolute(), uri

    # Ultralytics' callback reads the env var; the sink calls the function. If those
    # two disagree, a run's training curves and its federation facts land in different
    # stores and every chart is half a run.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert paths.subprocess_env()["MLFLOW_TRACKING_URI"] == uri


def test_a_federation_that_ran_is_actually_written_to_mlflow(monkeypatch):
    """pipeline.mlflow_sink has existed since the observability work and nothing ever
    called it, so MLflow held ultralytics' per-vehicle training curves and no
    federation at all -- no round-over-round checksum, which is the one signal that
    tells a federation that learns from one averaging its own input."""
    import contextlib

    from pipeline import mlflow_sink, runner

    calls = []
    monkeypatch.setattr(mlflow_sink, "run",
                        lambda name, params=None, nested=False: (
                            calls.append(("run", name)) or contextlib.nullcontext()))
    monkeypatch.setattr(mlflow_sink, "log_federation",
                        lambda log_dir, csv: calls.append(("federation", str(csv))))
    monkeypatch.setattr(mlflow_sink, "log_gpu", lambda summary: calls.append(("gpu", summary)))

    run = runner.Run(Config())
    run.results = [runner.StageResult("fleet", "ok"), runner.StageResult("federate", "ok")]
    run._log_to_mlflow({"energy_wh": 2.66})
    assert [c[0] for c in calls] == ["run", "federation", "gpu"]

    # A run that never federated has no federation facts to log, and an empty run in
    # the store is worse than no run: it looks like a federation that recorded nothing.
    calls.clear()
    run.results = [runner.StageResult("fleet", "ok")]
    run._log_to_mlflow({"energy_wh": 0.04})
    assert calls == []


def test_both_mlflow_writers_are_pointed_at_the_same_experiment(monkeypatch):
    """Ultralytics' callback falls back to trainer.args.project and creates its own
    experiment with no artifact location. MLflow then resolves artifacts against the
    CWD, which for every stage here is my-project -- one federation left 24 run
    directories under my-project/mlruns/, outside the store the sink reads."""
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    env = paths.subprocess_env()
    assert env["MLFLOW_EXPERIMENT_NAME"] == paths.MLFLOW_EXPERIMENT

    from pipeline import mlflow_sink
    assert mlflow_sink.EXPERIMENT == paths.MLFLOW_EXPERIMENT, "one name, defined once"


def test_a_refused_run_does_not_become_the_config_the_plan_previews():
    """/api/run validated the config it had already adopted, so a POST the server
    refused with a 400 still replaced the global one -- and /api/plan then previewed
    the stage table for a run that was never allowed to start. Harmless-looking
    because the next valid POST overwrote it, which is what kept it in the code."""
    import io

    from pipeline import server as srv

    before = srv.CONFIG
    handler = object.__new__(srv.Handler)          # do_POST needs no socket
    handler.path = "/api/run"
    body = json.dumps({"partition": "no-such-partition", "vehicles": 99}).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    seen = {}
    handler._json = lambda payload, code=200: seen.update(payload=payload, code=code)

    handler.do_POST()

    assert seen["code"] == 400 and "no-such-partition" in seen["payload"]["error"]
    assert srv.CONFIG is before, "a refused run must not replace the previewed config"


# ------------------------------------------------------------- the round profile
from pipeline import profile as _profile  # noqa: E402

# Two client episodes from the real six-round run, trimmed. The line that ends
# training is the same line that starts serialising the result.
PROFILE_LINES = "\n".join([
    "2026-08-06 00:57:35,355 - INFO - [Client] Creating FlowerClient instance from client_fn.",
    "2026-08-06 00:57:35,513 - INFO - [Client] YOLO model loaded successfully.",
    "2026-08-06 00:57:35,530 - INFO - [Client] Received weights with checksum: 705.23",
    "2026-08-06 00:57:35,539 - INFO - [Client] Successfully applied received weights to model",
    "2026-08-06 00:57:35,545 - INFO - [Client] Starting local training with batch_id=8, local_epochs=4",
    "2026-08-06 00:58:55,185 - INFO - [Client] 8 Training done. metrics={'mAP50': 0.35}",
    "2026-08-06 00:58:55,190 - INFO - [Client] Sending back weights with checksum: 236.48",
    "2026-08-06 00:58:55,338 - INFO - [Client] Creating FlowerClient instance from client_fn.",
    "2026-08-06 00:58:55,436 - INFO - [Client] YOLO model loaded successfully.",
    "2026-08-06 00:58:55,464 - INFO - [Client] Starting local training with batch_id=2, local_epochs=4",
    "2026-08-06 01:00:08,898 - INFO - [Client] 2 Training done. metrics={'mAP50': 0.33}",
    "2026-08-06 01:00:08,904 - INFO - [Client] Sending back weights with checksum: 205.59",
])


def test_the_phase_after_training_is_measured_rather_than_left_at_zero():
    """`Training done` closes training *and* opens serialisation. Stopping at the
    first marker a line matches reported weights_out as 0.0 s for every run --
    a phase that looks free because it was never measured, not because it is."""
    got = _profile.intervals(PROFILE_LINES, _profile.CLIENT_MARKERS)
    assert _profile.union_seconds(got["train"]) == pytest.approx(79.64 + 73.434, abs=0.01)
    assert _profile.union_seconds(got["weights_out"]) == pytest.approx(0.011, abs=0.001)
    assert _profile.union_seconds(got["construct"]) == pytest.approx(0.158 + 0.098, abs=0.001)


def test_serialised_clients_are_reported_as_serialised_not_averaged_away():
    """27 % mean utilisation is compatible with six clients each at 27 % and with one
    client at 27 % while five wait. Only the overlap count tells them apart, and the
    fix differs: `num-gpus < 1.0` for the second, the data path for the first."""
    eps = _profile.episodes(PROFILE_LINES)
    assert len(eps) == 2
    assert _profile.max_overlap(eps) == 1, "these two episodes are back to back, not concurrent"
    assert _profile.max_overlap([(0.0, 10.0), (5.0, 15.0), (5.5, 6.0)]) == 3


def test_two_runs_minutes_apart_are_profiled_apart(tmp_path):
    """The whole point of the profiler is a before/after comparison, and the arms of
    one are minutes apart. An mtime window with a minute of slack swallowed the
    previous arm: 48 episodes instead of 24, a wall clock spanning both runs, and a
    speed-up reading that was two runs added together. Membership comes from what the
    logs say, not from when the filesystem last touched them."""
    server = tmp_path / "server.10.log"
    server.write_text(
        "2026-08-06 00:57:26,281 - INFO - [Server] Aggregating 6 fit results\n"
        "2026-08-06 01:00:09,408 - INFO - [Server] Aggregated parameters with checksum: 159.9\n")
    (tmp_path / "client.11.log").write_text(PROFILE_LINES)

    # The next arm, four minutes later. Same shape, later clock -- and written last,
    # so every filesystem timestamp says it is the newest thing in the directory.
    (tmp_path / "client.12.log").write_text(PROFILE_LINES.replace("00:5", "01:1")
                                                         .replace("01:00:0", "01:20:0"))

    p = _profile.profile(server_log=server, log_dir=tmp_path)
    assert [Path(c).name for c in p["client_logs"]] == ["client.11.log"]
    assert p["episodes"] == 2, "the later run's episodes are not this run's"
    assert p["max_concurrent"] == 1
    assert p["wall_s"] < 300, "the wall clock must not span both runs"


def test_the_profiler_says_which_lever_the_measurement_points_at():
    """Phase 0 exists to choose between two fixes. A breakdown that leaves the
    choice to the reader has not made it."""
    serial_in_training = {"episodes": 6, "max_concurrent": 1, "train_share": 0.85}
    lines = " ".join(_profile.verdict(serial_in_training))
    assert "SERIALISED" in lines and "IN-TRAINING" in lines

    concurrent_overhead = {"episodes": 6, "max_concurrent": 3, "train_share": 0.40}
    lines = " ".join(_profile.verdict(concurrent_overhead))
    assert "CONCURRENT" in lines and "AROUND-TRAINING" in lines


def test_bdd_box_reconstruction_keeps_bikes_and_motors():
    """BDD calls them `bike` and `motor`; this project's classes are `bicycle` and
    `motorcycle`. Miss the aliases and the spot-check reports "one box too many" for
    every image containing either, which is a checker bug wearing a data bug's
    clothes -- it did exactly that on its first run. `lane` and `drivable area` carry
    poly2d and must stay out."""
    from pipeline import spotcheck

    rec = {"labels": [
        {"category": "bike", "box2d": {"x1": 0, "y1": 0, "x2": 1280, "y2": 720}},
        {"category": "motor", "box2d": {"x1": 640, "y1": 360, "x2": 1280, "y2": 720}},
        {"category": "lane", "poly2d": [[0, 0]]},
        {"category": "drivable area", "poly2d": [[0, 0]]},
    ]}
    boxes = spotcheck._boxes_from_bdd(rec, 1280, 720)

    assert [b[0] for b in boxes] == [spotcheck.CID["bicycle"], spotcheck.CID["motorcycle"]], \
        "the two aliases must map onto this project's class ids, and nothing else may"
    assert boxes[0][1:] == (0.5, 0.5, 1.0, 1.0), "a full-frame box is centre 0.5, size 1.0"
    assert boxes[1][1:] == (0.75, 0.75, 0.5, 0.5)
