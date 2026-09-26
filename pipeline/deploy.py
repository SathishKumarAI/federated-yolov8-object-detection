"""The Deployment Engine: real SuperLink + SuperNode processes, not the simulator.

Everything else in this project runs on Flower's **simulation** engine — one process,
Ray actors standing in for vehicles. That is the right tool for measuring, and it is
not what a fleet is. This starts the real thing: a SuperLink, N SuperNodes that dial
into it over gRPC, and `flwr run` submitting the app to the federation.

Five processes on one machine first, then five machines: the only difference is the
address the SuperNodes are given, which is the point of doing it locally first.

    python -m pipeline.deploy --nodes 2 --rounds 2 --epochs 1
    python -m pipeline.deploy --nodes 2 --dry-run          # print, start nothing
    python -m pipeline.deploy --nodes 5 --superlink-host 0.0.0.0   # let real machines in

On the other machines, one per shard:

    flower-supernode --insecure --superlink <this-host>:9092 \
                     --clientappio-api-address 127.0.0.1:9094

## Why this is not simply "the simulator with more processes"

**VRAM is the constraint, and it is a harder one here.** Under simulation Ray places
clients with `client-resources.num-gpus`, so `--gpu-fraction 0.33` caps the fleet at
three concurrent clients on one card. A SuperNode is an ordinary OS process: nothing
schedules it, nothing caps it, and each one holds its own CUDA context (~300–500 MiB
before a single weight is loaded) plus a full training footprint. Six of them on one
16 GB card will not fit — the measured demo-scale peak is already 12–15 GB for two to
three *scheduled* clients.

So `--nodes` defaults to **2** on one host. Past that, put the nodes on real machines
or run them on CPU. This is the number that bites first, so it is the default rather
than a footnote.

## The insecure default

`--insecure` everywhere, because this starts on loopback. It means **no TLS and no
authentication**: anything that can reach 9092 can join the federation as a vehicle and
receive the global model. Fine on a machine you own, not fine on a shared network. For
real hosts, generate certificates and drop `--insecure` — `flower-superlink` takes
`--ssl-certfile`, `--ssl-keyfile` and `--ssl-ca-certfile`.
"""
from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .stages import Config

#: Flower's defaults, restated because the ports are the thing people get wrong.
FLEET_PORT = 9092        # SuperNodes dial in here
CONTROL_PORT = 9093      # `flwr run` submits here
NODE_PORT_BASE = 9094    # each SuperNode's own ClientAppIO API; must be distinct


def executable(name: str) -> str:
    """Resolve next to THIS interpreter, never from PATH.

    A shell started by a script does not have the venv activated, so a bare
    `flower-superlink` finds whatever is first on the system PATH -- or nothing. Same
    reasoning as `stages.flwr_executable`.
    """
    here = Path(sys.executable).parent
    for candidate in (here / f"{name}.exe", here / name,
                      here / "Scripts" / f"{name}.exe", here / "bin" / name):
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"{name} not found beside {sys.executable} or on PATH. "
                     f"Install flwr into this interpreter's environment.")


def port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def wait_for_port(host: str, port: int, seconds: float, what: str) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if port_is_open(host, port):
            return True
        time.sleep(0.5)
    print(f"  ! {what} never opened {host}:{port} within {seconds:.0f}s")
    return False


def superlink_cmd(host: str) -> list[str]:
    return [executable("flower-superlink"), "--insecure",
            "--fleet-api-address", f"{host}:{FLEET_PORT}",
            "--exec-api-address", f"{host}:{CONTROL_PORT}"]


def supernode_cmd(index: int, superlink: str) -> list[str]:
    # Each SuperNode needs its OWN ClientAppIO address. Sharing one is the failure that
    # looks like a hang: the second node binds nothing, reports nothing, and the
    # federation waits forever for a client that never registers.
    return [executable("flower-supernode"), "--insecure",
            "--superlink", superlink,
            "--clientappio-api-address", f"127.0.0.1:{NODE_PORT_BASE + index}"]


def run_cmd(federation: str, cfg: Config) -> list[str]:
    from . import stages
    return [stages.flwr_executable(), "run", ".", federation, "--stream",
            "--run-config",
            f'num_server_rounds={cfg.rounds} local_epochs={cfg.local_epochs} '
            f'min_clients={cfg.n_vehicles} fraction_fit=1.0 '
            f'strategy="{cfg.strategy}" proximal_mu={cfg.proximal_mu} '
            f'cache="{cfg.cache}" local_bn={str(cfg.local_bn).lower()} '
            f'freeze_round1={cfg.freeze_round1} server_ema={cfg.server_ema} '
            f'fix_bn_from_round={cfg.fix_bn_from_round}']


def federation_entry(name: str, host: str) -> str:
    return (f"[superlink.{name}]\n"
            f'address = "{host}:{CONTROL_PORT}"\n'
            f"insecure = true\n")


def ensure_federation(name: str, host: str) -> Path:
    """Add the federation to ~/.flwr/config.toml if it is not already there.

    Appended, never rewritten: that file holds every federation on this machine,
    including the `local-simulation` one the rest of the pipeline runs on.
    """
    cfg = Path.home() / ".flwr" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    existing = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
    if f"[superlink.{name}]" in existing:
        return cfg
    cfg.write_text(existing.rstrip("\n") + "\n\n" + federation_entry(name, "127.0.0.1"),
                   encoding="utf-8")
    print(f"  added [superlink.{name}] to {cfg}")
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=2,
                    help="SuperNodes to start ON THIS MACHINE. Default 2 because each "
                         "is a real process with its own CUDA context and nothing "
                         "schedules them onto the card")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--strategy", default="fedavg")
    ap.add_argument("--local-bn", action="store_true")
    ap.add_argument("--freeze-round1", type=int, default=0)
    ap.add_argument("--server-ema", type=float, default=0.0)
    ap.add_argument("--fix-bn-from-round", type=int, default=0)
    ap.add_argument("--federation", default="local-deployment")
    ap.add_argument("--superlink-host", default="127.0.0.1",
                    help="0.0.0.0 to accept SuperNodes from other machines. No TLS and "
                         "no authentication: anything that reaches the fleet port can "
                         "join and receive the global model")
    ap.add_argument("--external-only", action="store_true",
                    help="start the SuperLink and stop, so the nodes can be real "
                         "machines. Prints the exact command each one runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config(rounds=args.rounds, local_epochs=args.epochs, n_vehicles=args.nodes,
                 strategy=args.strategy, local_bn=args.local_bn,
                 freeze_round1=args.freeze_round1, server_ema=args.server_ema,
                 fix_bn_from_round=args.fix_bn_from_round)
    superlink = f"{'127.0.0.1' if args.superlink_host == '0.0.0.0' else args.superlink_host}:{FLEET_PORT}"

    cmds = [("superlink", superlink_cmd(args.superlink_host))]
    if not args.external_only:
        cmds += [(f"supernode-{i}", supernode_cmd(i, superlink)) for i in range(args.nodes)]

    print(f"Deployment engine: 1 SuperLink + {0 if args.external_only else args.nodes} "
          f"local SuperNode(s)")
    print(f"  fleet   {args.superlink_host}:{FLEET_PORT}   (SuperNodes dial in)")
    print(f"  control {args.superlink_host}:{CONTROL_PORT}   (`flwr run` submits)")
    for name, cmd in cmds:
        print(f"  {name:<12} {' '.join(cmd)}")
    print(f"  submit       {' '.join(run_cmd(args.federation, cfg))}")

    if args.dry_run:
        print("\nDry run: nothing started.")
        return 0

    if args.nodes > 3 and not args.external_only:
        print(f"\n  ! {args.nodes} SuperNodes on one host. Each holds its own CUDA "
              f"context and nothing caps its VRAM the way Ray's client-resources does "
              f"under simulation. Expect out-of-memory rather than slowness.\n")

    ensure_federation(args.federation, args.superlink_host)
    env = paths.subprocess_env(data_root=paths.VEHICLE_ROOT)
    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, cmd in cmds:
            print(f"  starting {name} ...")
            procs.append((name, subprocess.Popen(cmd, cwd=paths.PROJECT, env=env)))
            if name == "superlink":
                if not wait_for_port(args.superlink_host, FLEET_PORT, 30, "SuperLink"):
                    return 1
                wait_for_port(args.superlink_host, CONTROL_PORT, 30, "SuperLink control")
            time.sleep(1.0)

        for name, p in procs:
            if p.poll() is not None:
                print(f"  ! {name} exited immediately with {p.returncode}")
                return 1

        if args.external_only:
            print(f"\nSuperLink up. On each machine:\n")
            print(f"  flower-supernode --insecure --superlink "
                  f"<this-host>:{FLEET_PORT} "
                  f"--clientappio-api-address 127.0.0.1:{NODE_PORT_BASE}\n")
            print("Then submit the app here:")
            print(f"  {' '.join(run_cmd(args.federation, cfg))}")
            print("\nCtrl-C to stop the SuperLink.")
            for _, p in procs:
                p.wait()
            return 0

        print(f"\n  submitting the app to '{args.federation}' ...\n")
        code = subprocess.run(run_cmd(args.federation, cfg),
                              cwd=paths.PROJECT, env=env).returncode
        print(f"\nflwr run exited {code}")
        return code
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        for name, p in reversed(procs):
            if p.poll() is None:
                print(f"  stopping {name}")
                p.terminate()
        for _, p in reversed(procs):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        # The pipeline's simulation runs kill a stale SuperLink before every
        # federation because it caches the CWD and environment of whichever run
        # started it. Leaving one alive here would poison the next simulation run.
        print("  deployment torn down")


if __name__ == "__main__":
    raise SystemExit(main())
