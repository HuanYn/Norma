from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from scripts import start_deepseek as launcher


TEST_KEY = "fake-deepseek-key-for-launcher-tests"


def test_launcher_scopes_preset_and_key_to_child_environment(monkeypatch, capsys):
    monkeypatch.setenv("NORMA_VLM_API_KEY", "unrelated-existing-key")
    monkeypatch.setenv("NORMA_HOST", "0.0.0.0")
    monkeypatch.setenv("NORMA_PREFERENCE_MODE", "adaptive")
    before = dict(os.environ)
    monkeypatch.setattr(launcher.getpass, "getpass", lambda prompt: TEST_KEY)
    calls = []

    def run(args, *, cwd, env, check):
        calls.append((list(args), cwd, dict(env), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main() == 0
    assert len(calls) == 1
    args, cwd, environment, check = calls[0]
    assert args == [launcher.sys.executable, "-m", "ai", "web"]
    assert TEST_KEY not in " ".join(args)
    assert cwd == launcher.PROJECT_ROOT
    assert check is False
    assert environment["NORMA_VLM_API_KEY"] == TEST_KEY
    assert environment["NORMA_HOST"] == "127.0.0.1"
    assert environment["NORMA_VLM_BASE_URL"] == "https://api.deepseek.com"
    assert environment["NORMA_VLM_MODEL"] == "deepseek-flash"
    assert environment["NORMA_VLM_THINKING_MODE"] == "disabled"
    assert environment["NORMA_VLM_JSON_RESPONSE_FORMAT"] == "1"
    assert environment["NORMA_VLM_MAX_NEW_TOKENS"] == "1024"
    assert environment["NORMA_PREFERENCE_MODE"] == "record-only"
    assert dict(os.environ) == before
    captured = capsys.readouterr()
    assert TEST_KEY not in captured.out + captured.err
    assert "unrelated-existing-key" not in captured.out + captured.err


@pytest.mark.parametrize(
    "key", ["", " leading-space", "line\nbreak", "非ASCII", "x" * 4097]
)
def test_invalid_input_never_starts_child(monkeypatch, capsys, key):
    monkeypatch.setattr(launcher.getpass, "getpass", lambda prompt: key)
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *a, **k: pytest.fail("must not start")
    )
    assert launcher.main() == 2
    captured = capsys.readouterr()
    assert "nothing was started" in captured.out
    if key:
        assert key not in captured.out + captured.err


@pytest.mark.parametrize("failure", [EOFError, launcher.getpass.GetPassWarning])
def test_no_unsafe_input_fallback(monkeypatch, capsys, failure):
    def prompt(message):
        raise failure()

    monkeypatch.setattr(launcher.getpass, "getpass", prompt)
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *a, **k: pytest.fail("must not start")
    )
    assert launcher.main() == 2
    assert "Hidden input unavailable" in capsys.readouterr().out


def test_launch_errors_never_echo_child_environment(monkeypatch, capsys):
    monkeypatch.setattr(launcher.getpass, "getpass", lambda prompt: TEST_KEY)

    def fail(*args, **kwargs):
        raise OSError(TEST_KEY)

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    assert launcher.main() == 2
    captured = capsys.readouterr()
    assert TEST_KEY not in captured.out + captured.err


def test_child_exit_code_is_preserved(monkeypatch):
    monkeypatch.setattr(launcher.getpass, "getpass", lambda prompt: TEST_KEY)
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7)
    )
    assert launcher.main() == 7
