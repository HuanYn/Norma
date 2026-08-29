from __future__ import annotations

# Pytest loads this file before collecting test modules.  Some modules import
# NumPy at module scope, so install the same runtime contract as Norma's real
# entry points before those imports can lock MKL to its Intel OpenMP backend.
from ai.numeric_runtime import configure_numeric_runtime
import pytest


configure_numeric_runtime()


@pytest.fixture(autouse=True)
def _keep_unit_tests_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing explicit fake/baseline provider must never download a model."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
