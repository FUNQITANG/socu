# CMake Presets (Windows + CUDA)

This repo includes a shared preset file: `CMakePresets.json`.

## Why presets

- Keep `cmake -S/-B/-D...` arguments out of your shell history.
- Make the required dependencies explicit via environment variables.
- Provide consistent build directories (`build-ninja/`) and cache variables.

## Important limitation (Windows + Ninja)

When using the `Ninja` generator on Windows **with MSVC as the host compiler**, NVCC must be able to find `cl.exe`/`link.exe` and the Windows SDK.

CMake presets can set environment variables, but they **cannot run** `Launch-VsDevShell.ps1` / `VsDevCmd.bat`.

So these presets **assume you are already inside a Visual Studio Developer Shell**.

### CUDA + MSVC compatibility note

CUDA's `nvcc` uses MSVC (`cl.exe`) as the *host compiler* on Windows. CUDA 13.x
officially supports Visual Studio 2019-2022; if your active developer shell is
from a newer VS (e.g. VS 2025 / MSVC 19.50), `nvcc` may fail during CMake's CUDA
compiler identification.

Passing `--allow-unsupported-compiler` can get you past the version check, but
it does not guarantee it will actually work; a common failure mode is:

- `nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)`

Fix: use a supported VS 2022 toolset (v143) for NVCC's host compiler.

You can do that in two ways:

1) Run CMake from a **VS 2022 x64 Developer Shell**.

2) Or explicitly point NVCC at VS2022's `cl.exe`:

- Set env var `SOCU_CUDA_HOST_COMPILER` to a full path to `cl.exe` (recommended)
- Or configure with `-DSOCU_CUDA_HOST_COMPILER=...`

This repo wires that into `CMAKE_CUDA_HOST_COMPILER` early enough (before
`project(... CUDA)`) so compiler identification uses the selected toolset.

If you want a single entry point that *always* loads the VS dev environment first,
use the helper scripts:

- `scripts/Configure-Windows.ps1` (loads VS dev shell, optionally activates venv, then runs `cmake --preset ...`)
- `scripts/Build-Windows.ps1` (configure + build)

The scripts support `SOCU_VS_INSTALL_DIR` to point at your VS root (e.g. `A:\\Visual Studio\\18\\Enterprise`).

To make the scripts also activate `./.venv`, set:

- `SOCU_ACTIVATE_VENV=1`

## Variables you can/should override

- `SOCU_CUDA_ROOT`
  - Default: `C:/CUDA`
  - Used for `CUDAToolkit_ROOT` and `CMAKE_CUDA_COMPILER`.

- `SOCU_NVCC_ALLOW_UNSUPPORTED_COMPILER`
  - Default: `ON` (Windows)
  - Enables `--allow-unsupported-compiler` early enough for CUDA compiler identification.

- `SOCU_CUDA_HOST_COMPILER`
  - Default: unset/empty
  - Full path to the `cl.exe` you want NVCC to use as host compiler (useful to
    pin to a VS 2022 / v143 toolset even if you run from another shell).

- `SOCU_BUILD_DEMO`
  - Default: `ON`
  - Turn OFF if you only want the Python package.

## Presets provided

- `win-ninja-release`
- `win-ninja-debug`
- `win-ninja-release-oneapi` (placeholder env var `SOCU_ONEAPI_ROOT`, intended for teams that also use Intel oneAPI)

## Per-machine overrides

For machine-specific paths, create a `CMakeUserPresets.json` (not committed).
A good pattern is to override only `SOCU_CUDA_ROOT` / `SOCU_ONEAPI_ROOT` there.
