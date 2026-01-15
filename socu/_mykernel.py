from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import Any, Optional


def _default_cuda_bin_dirs() -> list[Path]:
    # Best-effort. Users can also set PATH or call os.add_dll_directory themselves.
    candidates: list[Path] = []

    env_root = os.environ.get("SOCU_CUDA_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "bin")

    candidates.append(Path(r"C:\CUDA\bin"))
    return [p for p in candidates if p.exists()]


def _default_torch_cuda_dll_dirs() -> list[Path]:
    """Best-effort discovery of CUDA DLL locations from PyTorch installs.

    If you have a CUDA-enabled PyTorch wheel/conda env, it often ships cudart and
    other CUDA user-mode DLLs. Adding those directories can let this project run
    without bundling cudart in our own wheel.

    This does NOT replace the need for an NVIDIA driver.
    """

    candidates: list[Path] = []

    # Classic layout: <site-packages>/torch/lib/*.dll
    try:
        import torch  # type: ignore

        torch_dir = Path(torch.__file__).resolve().parent
        candidates.append(torch_dir / "lib")
    except Exception:
        pass

    # Newer pip layout: NVIDIA runtime packages under site-packages/nvidia/*/bin
    for root_str in sys.path:
        if not root_str:
            continue
        root = Path(root_str)
        candidates.extend(
            [
                root / "nvidia" / "cuda_runtime" / "bin",
                root / "nvidia" / "cublas" / "bin",
                root / "nvidia" / "cudnn" / "bin",
                root / "nvidia" / "cuda_nvrtc" / "bin",
                root / "nvidia" / "cuda_cupti" / "bin",
            ]
        )

    out: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if p.exists() and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _candidate_dll_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    candidates = []
    override = os.environ.get("SOCU_MYKERNEL_DLL")
    if override:
        candidates.append(Path(override))

    candidates.extend(
        [
            here / "_native" / "mykernel.dll",
            repo_root / "tests" / "_native" / "mykernel.dll",
            repo_root / "out" / "build" / "win-ninja-release" / "bin" / "mykernel.dll",
        ]
    )

    return [p for p in candidates if p.exists()]


class MyKernel:
    def __init__(self, dll_path: Optional[os.PathLike[str] | str] = None):
        # On Windows, dependencies must be discoverable at DLL-load time.
        if os.name == "nt":
            for p in _default_cuda_bin_dirs():
                os.add_dll_directory(str(p))
            for p in _default_torch_cuda_dll_dirs():
                os.add_dll_directory(str(p))

        if dll_path is None:
            candidates = _candidate_dll_paths()
            if not candidates:
                raise FileNotFoundError(
                    "Could not find mykernel.dll. Build+install it, or set SOCU_MYKERNEL_DLL."
                )
            dll_path = max(candidates, key=lambda p: p.stat().st_mtime)

        dll_path = Path(dll_path)
        os.add_dll_directory(str(dll_path.parent))

        self._dll_path = dll_path
        self._lib = ctypes.CDLL(str(dll_path))

        self._lib.create_stream.restype = ctypes.c_int64
        self._lib.destroy_stream.argtypes = [ctypes.c_int64]
        self._lib.destroy_stream.restype = None

        # Optional (newer builds):
        if hasattr(self._lib, "stream_synchronize"):
            self._lib.stream_synchronize.argtypes = [ctypes.c_int64]
            self._lib.stream_synchronize.restype = ctypes.c_int
        else:
            self._lib.stream_synchronize = None

        self._lib.syrk_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
        ]
        self._lib.syrk_update.restype = None

        # Optional timing helpers (newer builds):
        # - create_event/destroy_event
        # - event_record/event_synchronize
        # - event_elapsed_ms
        if hasattr(self._lib, "create_event"):
            self._lib.create_event.restype = ctypes.c_int64
            self._lib.destroy_event.argtypes = [ctypes.c_int64]
            self._lib.destroy_event.restype = None

            self._lib.event_record.argtypes = [ctypes.c_int64, ctypes.c_int64]
            self._lib.event_record.restype = ctypes.c_int

            self._lib.event_synchronize.argtypes = [ctypes.c_int64]
            self._lib.event_synchronize.restype = ctypes.c_int

            self._lib.event_elapsed_ms.argtypes = [ctypes.c_int64, ctypes.c_int64]
            self._lib.event_elapsed_ms.restype = ctypes.c_float
        else:
            self._lib.create_event = None
            self._lib.destroy_event = None
            self._lib.event_record = None
            self._lib.event_synchronize = None
            self._lib.event_elapsed_ms = None

    @property
    def dll_path(self) -> Path:
        return self._dll_path

    def create_stream(self) -> int:
        return int(self._lib.create_stream())

    def destroy_stream(self, stream: int) -> None:
        self._lib.destroy_stream(ctypes.c_int64(stream))

    def stream_synchronize(self, stream: int) -> int:
        if self._lib.stream_synchronize is None:
            raise AttributeError(
                "stream_synchronize not found in DLL. Rebuild/install a newer mykernel.dll."
            )
        return int(self._lib.stream_synchronize(ctypes.c_int64(stream)))

    def syrk_update(self, E_ptr: int, D_ptr: int, B: int, M: int, n: int, stream: int) -> None:
        self._lib.syrk_update(
            ctypes.c_void_p(E_ptr),
            ctypes.c_void_p(D_ptr),
            ctypes.c_int(B),
            ctypes.c_int(M),
            ctypes.c_int(n),
            ctypes.c_int64(stream),
        )

    def create_event(self) -> int:
        if self._lib.create_event is None:
            raise AttributeError("create_event not found in DLL. Rebuild/install a newer mykernel.dll.")
        return int(self._lib.create_event())

    def destroy_event(self, event: int) -> None:
        if self._lib.destroy_event is None:
            raise AttributeError("destroy_event not found in DLL. Rebuild/install a newer mykernel.dll.")
        self._lib.destroy_event(ctypes.c_int64(event))

    def event_record(self, event: int, stream: int) -> int:
        if self._lib.event_record is None:
            raise AttributeError("event_record not found in DLL. Rebuild/install a newer mykernel.dll.")
        return int(self._lib.event_record(ctypes.c_int64(event), ctypes.c_int64(stream)))

    def event_synchronize(self, event: int) -> int:
        if self._lib.event_synchronize is None:
            raise AttributeError(
                "event_synchronize not found in DLL. Rebuild/install a newer mykernel.dll."
            )
        return int(self._lib.event_synchronize(ctypes.c_int64(event)))

    def event_elapsed_ms(self, start_event: int, end_event: int) -> float:
        if self._lib.event_elapsed_ms is None:
            raise AttributeError("event_elapsed_ms not found in DLL. Rebuild/install a newer mykernel.dll.")
        return float(self._lib.event_elapsed_ms(ctypes.c_int64(start_event), ctypes.c_int64(end_event)))


def device_ptr(x: Any) -> int:
    # Warp arrays expose `.ptr` (int device pointer).
    if hasattr(x, "ptr"):
        return int(x.ptr)
    # CuPy exposes `.data.ptr`.
    if hasattr(x, "data") and hasattr(x.data, "ptr"):
        return int(x.data.ptr)
    # Torch tensors expose `.data_ptr()`.
    if hasattr(x, "data_ptr") and callable(x.data_ptr):
        return int(x.data_ptr())
    raise TypeError("Unsupported input: expected Warp/CuPy/Torch GPU array/tensor")
