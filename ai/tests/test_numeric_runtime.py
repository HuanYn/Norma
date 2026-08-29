from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

# This assertion also proves that ai/tests/conftest.py ran before collection of
# this module.  That ordering protects test modules which import NumPy first.
if sys.platform == "win32":
    assert os.environ.get("MKL_THREADING_LAYER") == "SEQUENTIAL"

from ai import numeric_runtime


def test_windows_configuration_defaults_and_normalizes_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(numeric_runtime.sys, "platform", "win32")
    monkeypatch.delenv("MKL_THREADING_LAYER", raising=False)
    numeric_runtime.configure_numeric_runtime()
    assert os.environ["MKL_THREADING_LAYER"] == "SEQUENTIAL"

    monkeypatch.setenv("MKL_THREADING_LAYER", " sequential ")
    numeric_runtime.configure_numeric_runtime()
    assert os.environ["MKL_THREADING_LAYER"] == "SEQUENTIAL"


def test_windows_guard_rejects_nonsequential_and_unsafe_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(numeric_runtime.sys, "platform", "win32")
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")
    with pytest.raises(
        numeric_runtime.NumericRuntimeConflictError,
        match="MKL_THREADING_LAYER=SEQUENTIAL",
    ):
        numeric_runtime.ensure_torch_numpy_runtime_compatible()

    monkeypatch.setenv("MKL_THREADING_LAYER", "SEQUENTIAL")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    with pytest.raises(
        numeric_runtime.NumericRuntimeConflictError,
        match="unsafe duplicate-runtime override",
    ):
        numeric_runtime.ensure_torch_numpy_runtime_compatible()


def test_windows_guard_rejects_an_already_loaded_foreign_intel_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(numeric_runtime.sys, "platform", "win32")
    monkeypatch.setenv("MKL_THREADING_LAYER", "SEQUENTIAL")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setattr(
        numeric_runtime,
        "_bundled_torch_openmp_path",
        lambda: Path("C:/python/site-packages/torch/lib/libiomp5md.dll"),
    )
    monkeypatch.setattr(
        numeric_runtime,
        "_loaded_intel_openmp_paths",
        lambda: (Path("C:/conda/Library/bin/mkl_intel_thread.2.dll"),),
    )

    with pytest.raises(
        numeric_runtime.NumericRuntimeConflictError,
        match="non-Torch Intel OpenMP runtime",
    ):
        numeric_runtime.ensure_torch_numpy_runtime_compatible()


def test_openclip_and_qwen_entry_points_fail_closed_on_unsafe_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai.index.openclip_identity import resolve_openclip_backend
    from ai.rag.transformers_runtime import (
        LocalVLMUnavailableError,
        TransformersQwen3VLRuntime,
    )

    monkeypatch.setattr(numeric_runtime.sys, "platform", "win32")
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")

    with pytest.raises(numeric_runtime.NumericRuntimeConflictError):
        resolve_openclip_backend("cpu")
    with pytest.raises(LocalVLMUnavailableError):
        TransformersQwen3VLRuntime(tmp_path / "missing-model")


@pytest.mark.skipif(
    sys.platform != "win32" or find_spec("torch") is None,
    reason="the native OpenMP regression is specific to Windows Torch",
)
def test_contextual_numpy_then_default_openclip_and_numpy_again() -> None:
    import numpy as np

    from ai.index.openclip_identity import resolve_openclip_backend

    before = np.ones((32, 512), dtype=np.float64) @ np.ones(512, dtype=np.float64)
    assert resolve_openclip_backend("auto") in {"cpu", "cuda"}
    after = np.ones((32, 512), dtype=np.float64) @ np.ones(512, dtype=np.float64)

    assert before.tolist() == [512.0] * 32
    assert after.tolist() == [512.0] * 32
    loaded = tuple(
        str(path).replace("\\", "/").casefold()
        for path in numeric_runtime._loaded_intel_openmp_paths()
    )
    assert not any("/mkl_intel_thread" in path for path in loaded)
    openmp = tuple(path for path in loaded if path.endswith("/libiomp5md.dll"))
    assert len(openmp) == 1
    assert openmp[0].endswith("/torch/lib/libiomp5md.dll")


@pytest.mark.skipif(
    sys.platform != "win32" or find_spec("torch") is None,
    reason="the native OpenMP regression is specific to Windows Torch",
)
def test_default_openclip_cli_and_numpy_blas_share_one_safe_process() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = r"""
import json
import os
import tempfile

import ai
startup_layer = os.environ.get("MKL_THREADING_LAYER")
import numpy as np

before = float((np.ones((32, 512)) @ np.ones(512))[0])
from ai.cli import main
from ai.numeric_runtime import _loaded_intel_openmp_paths

with tempfile.TemporaryDirectory() as directory:
    first = main(["--data-dir", directory, "init"])
    second = main(["--data-dir", directory, "provider-status"])
after = float((np.ones((32, 512)) @ np.ones(512))[0])
solved = float(np.linalg.solve(np.eye(67) * 2.0, np.ones(67))[0])
loaded = [str(path).replace("\\", "/").casefold() for path in _loaded_intel_openmp_paths()]
print("NUMERIC_RUNTIME_RESULT=" + json.dumps({
    "first": first,
    "second": second,
    "startup_layer": startup_layer,
    "before": before,
    "after": after,
    "solved": solved,
    "loaded": loaded,
}))
"""
    environment = os.environ.copy()
    environment.pop("MKL_THREADING_LAYER", None)
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    marker = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("NUMERIC_RUNTIME_RESULT=")
    )
    result = json.loads(marker.removeprefix("NUMERIC_RUNTIME_RESULT="))
    assert result == {
        "first": 0,
        "second": 0,
        "startup_layer": "SEQUENTIAL",
        "before": 512.0,
        "after": 512.0,
        "solved": 0.5,
        "loaded": result["loaded"],
    }
    assert len(result["loaded"]) == 1
    assert result["loaded"][0].endswith("/torch/lib/libiomp5md.dll")
    assert "OMP: Error #15" not in completed.stderr


@pytest.mark.skipif(
    sys.platform != "win32" or find_spec("torch") is None,
    reason="the native OpenMP regression is specific to Windows Torch",
)
def test_numpy_imported_before_ai_fails_before_torch_native_abort(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = r"""
import os
import sys
import sysconfig

for import_path in {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}:
    if import_path:
        sys.path.insert(0, import_path)
import numpy as np

if os.environ.get("MKL_THREADING_LAYER") is not None:
    raise AssertionError("the late-import regression must start without a layer")
sys.path.insert(0, os.environ["NORMA_TEST_PROJECT_ROOT"])
import ai
from ai.index.embedding import (
    EmbeddingProviderUnavailableError,
    create_embedding_provider,
)

try:
    create_embedding_provider("openclip-multilingual", device="cpu")
except EmbeddingProviderUnavailableError as error:
    print("SAFE_CONFLICT=" + str(error))
else:
    raise AssertionError("late foreign OpenMP runtime was not rejected")
"""
    environment = os.environ.copy()
    environment.pop("MKL_THREADING_LAYER", None)
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    environment["NORMA_TEST_PROJECT_ROOT"] = str(project_root)
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SAFE_CONFLICT=a non-Torch Intel OpenMP runtime" in completed.stdout
    assert "OMP: Error #15" not in completed.stderr
