# Windows environment — why this repo does not use a conda env

**Short version:** Smart App Control blocks conda-forge's compiled stdlib extensions.
Build the environment on the **python.org** interpreter instead.

## The failure

`conda create -n fl_yolov8 python=3.11` succeeds, `torch` installs and runs on the
GPU, and then the first Ultralytics training run dies inside `torchvision`:

```
ImportError: DLL load failed while importing _bz2: An Application Control policy has blocked this file.
```

The chain is `ultralytics.engine.validator` → `import torchvision` →
`torchvision.datasets.utils` → `import bz2` → `_bz2.pyd`.

## The cause

Smart App Control is enforced on this host:

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy').VerifiedAndReputablePolicyState  # 1 = on
```

SAC allows a binary only if it is signed by a trusted publisher or has
established reputation. conda-forge's `_bz2.pyd` is unsigned. Tested and blocked on
**3.11, 3.12 and 3.13** conda envs — it is not a version problem. The `base` miniforge
env works only because it predates enforcement.

It is not Mark-of-the-Web either — the file carries no `Zone.Identifier` stream, so
`Unblock-File` changes nothing. SAC offers no per-file user allowlist, and turning it
off is one-way (re-enabling requires a Windows reset).

## The fix

python.org builds are PSF-signed, so SAC admits them:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv C:\Users\PRANAS\venvs\fl_yolov8
$py = "C:\Users\PRANAS\venvs\fl_yolov8\Scripts\python.exe"

# torch FIRST — Blackwell sm_120 needs cu128, and installing it before the project
# stops ultralytics pulling a default-CUDA torch from PyPI.
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 2.11.0+cu128 12.8 (12, 0)   <- (12, 0) is the sm_120 confirmation

cd my-project
& $py -m pip install -e ".[dev]"
```

**Python 3.12, not 3.13:** `flwr[simulation]` pulls `ray`, whose dependency marker is
`python>=3.11,<3.13` on Windows. On 3.13 the simulation extra resolves to nothing and
`flwr run` fails at import.

## Verified working set

| Package | Version |
|---|---|
| Python | 3.12.10 (python.org) |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| flwr | 1.33.0 (legacy `ServerApp(server_fn=...)` API still imports) |
| ray | 2.55.1 |
| ultralytics | 8.4.115 |
| GPU | RTX 5070 Ti 16 GB, driver 610.47, `sm_120` |

## Application Control blocks generated console scripts too

Measured 2026-09-26. A pipeline run halted at the federate stage:

```
[      running] federate  ...\Scripts\flwr.exe run . --stream --federation-config ...
[       failed] federate  [WinError 4551] An Application Control policy has blocked this file
HALTED at federate: [WinError 4551] An Application Control policy has blocked this file
```

Running it by hand gives `Permission denied`. Nothing about the environment changed on
purpose; the policy did. This is the same mechanism as the `_bz2.pyd` case above — an
unsigned binary produced on this machine — and a `pip install` console script is exactly
that: a generated `.exe` wrapper whose entire body is one import.

**The fix is to skip the wrapper, not to sign it.** `python -c "from flwr.cli.app import
app; app()"` runs the same CLI, in the same environment, with no executable to block, and
it is what `pip` generates the shim to do. `pipeline/stages.py::flwr_launcher` returns that
command; the shim is still looked for, but only as evidence that flwr is installed here, so
a missing install still fails with a sentence instead of a ModuleNotFoundError from inside
typer.

If another console script starts failing the same way — `flower-superlink`,
`flower-supernode`, `mlflow` — the same substitution applies. Find the entry point in the
package's metadata and call it with `python -c`.
