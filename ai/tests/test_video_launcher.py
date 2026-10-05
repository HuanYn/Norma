from __future__ import annotations

import os
import stat

import pytest

from scripts.video_worker import create_private_token, main, read_private_token


def test_create_token_exclusive_and_read(tmp_path, capsys):
    path = tmp_path / "worker-token"
    token = create_private_token(path)
    assert len(token) >= 32
    assert read_private_token(path) == token
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        create_private_token(path)
    assert capsys.readouterr().out == ""


def test_invalid_token_file_hides_content(tmp_path, capsys):
    path = tmp_path / "worker-token"
    path.write_text("short-secret")
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path / "jobs"), "--token-file", str(path)])
    assert "short-secret" not in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission test")
def test_group_readable_token_rejected(tmp_path):
    path = tmp_path / "token"
    create_private_token(path)
    path.chmod(0o640)
    with pytest.raises(ValueError, match="owner"):
        read_private_token(path)


def test_public_binding_requires_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("NORMA_VIDEO_WORKER_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "--host", "0.0.0.0"])


def test_ambiguous_token_sources_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NORMA_VIDEO_WORKER_TOKEN", "do-not-print-this-environment-value")
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "--token-file", str(tmp_path / "token")])
    assert "do-not-print-this-environment-value" not in capsys.readouterr().err
