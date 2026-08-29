from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from importlib.util import find_spec
from pathlib import Path


WINDOWS_NUMERIC_THREADING_CONTRACT = "windows-mkl-sequential-torch-openmp-v1"
NATIVE_NUMERIC_THREADING_CONTRACT = "platform-native-threading-v1"
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class NumericRuntimeConflictError(RuntimeError):
    """NumPy and Torch cannot safely share the current native runtime state."""


def configure_numeric_runtime() -> None:
    """Select the safe Windows MKL layer before NumPy or Torch is imported.

    Anaconda NumPy links MKL to ``Library/bin/libiomp5md.dll`` while PyPI Torch
    bundles a different ``torch/lib/libiomp5md.dll``.  MKL's sequential layer
    does not load Anaconda's Intel OpenMP runtime, leaving Torch as the sole
    OpenMP owner in this process.
    """

    if sys.platform != "win32":
        return
    current = os.environ.get("MKL_THREADING_LAYER")
    if current is None:
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    elif current.strip().casefold() == "sequential":
        # Normalize an equivalent explicit value before MKL consumes it.
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"


def numeric_threading_contract() -> str:
    """Return the provider-identity contract for native numeric threading."""

    return (
        WINDOWS_NUMERIC_THREADING_CONTRACT
        if sys.platform == "win32"
        else NATIVE_NUMERIC_THREADING_CONTRACT
    )


def ensure_torch_numpy_runtime_compatible() -> None:
    """Fail before importing Torch when the Windows process would native-abort."""

    if sys.platform != "win32":
        return
    configure_numeric_runtime()
    layer = os.environ.get("MKL_THREADING_LAYER", "")
    if layer != "SEQUENTIAL":
        raise NumericRuntimeConflictError(
            "Windows OpenCLIP/Qwen requires MKL_THREADING_LAYER=SEQUENTIAL so "
            "NumPy and Torch do not initialize duplicate Intel OpenMP runtimes"
        )
    duplicate_override = os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip()
    if duplicate_override and duplicate_override.casefold() not in _FALSE_VALUES:
        raise NumericRuntimeConflictError(
            "KMP_DUPLICATE_LIB_OK is an unsafe duplicate-runtime override and is "
            "not supported by Norma"
        )

    expected_torch_runtime = _bundled_torch_openmp_path()
    if expected_torch_runtime is None:
        return
    try:
        loaded_runtimes = _loaded_intel_openmp_paths()
    except OSError as error:
        raise NumericRuntimeConflictError(
            "Norma could not verify the loaded Windows OpenMP runtime before "
            "starting Torch"
        ) from error
    expected = _normalized_path(expected_torch_runtime)
    if any(_normalized_path(path) != expected for path in loaded_runtimes):
        raise NumericRuntimeConflictError(
            "a non-Torch Intel OpenMP runtime was initialized before Norma; "
            "restart with MKL_THREADING_LAYER=SEQUENTIAL before using OpenCLIP/Qwen"
        )


def _bundled_torch_openmp_path() -> Path | None:
    """Locate a wheel-bundled runtime without importing Torch itself."""

    spec = find_spec("torch")
    if spec is None or spec.submodule_search_locations is None:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "lib" / "libiomp5md.dll"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _loaded_intel_openmp_paths() -> tuple[Path, ...]:
    """Enumerate loaded modules that make a later Torch import unsafe.

    Importing Anaconda NumPy can load ``mkl_intel_thread*.dll`` before its
    linked ``libiomp5md.dll`` is initialized.  Seeing the threaded MKL backend
    is therefore already enough to fail closed: changing the environment to
    ``SEQUENTIAL`` after that DLL was loaded cannot switch MKL's native layer.
    """

    if sys.platform != "win32":
        return ()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = (
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

    process = kernel32.GetCurrentProcess()
    capacity = 256
    while True:
        modules = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(
            process,
            modules,
            ctypes.sizeof(modules),
            ctypes.byref(needed),
        ):
            raise OSError(ctypes.get_last_error(), "EnumProcessModules failed")
        if needed.value <= ctypes.sizeof(modules):
            count = needed.value // ctypes.sizeof(wintypes.HMODULE)
            break
        capacity = max(capacity * 2, needed.value // ctypes.sizeof(wintypes.HMODULE))

    paths: list[Path] = []
    for module in modules[:count]:
        buffer = ctypes.create_unicode_buffer(32768)
        length = psapi.GetModuleFileNameExW(process, module, buffer, len(buffer))
        if length == 0:
            raise OSError(ctypes.get_last_error(), "GetModuleFileNameExW failed")
        path = Path(buffer.value)
        module_name = path.name.casefold()
        if module_name == "libiomp5md.dll" or (
            module_name.startswith("mkl_intel_thread") and module_name.endswith(".dll")
        ):
            paths.append(path.resolve())
    return tuple(paths)


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


__all__ = [
    "NATIVE_NUMERIC_THREADING_CONTRACT",
    "NumericRuntimeConflictError",
    "WINDOWS_NUMERIC_THREADING_CONTRACT",
    "configure_numeric_runtime",
    "ensure_torch_numpy_runtime_compatible",
    "numeric_threading_contract",
]
