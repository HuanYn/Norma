from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.video import client as video_client
from ai.video.client import VideoWorkerClient, VideoWorkerError


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, count):
        return b"\x00\x00\x00\x20ftyp" + b"bounded-test-mp4-header" * 3


def client():
    instance = VideoWorkerClient("http://127.0.0.1:8766")
    instance._opener = SimpleNamespace(open=lambda *args, **kwargs: Response())
    return instance


def test_failed_fsync_never_exposes_partial_artifact(tmp_path: Path, monkeypatch):
    target = tmp_path / "video.mp4"

    def fail(_):
        raise OSError("disk write failure")

    monkeypatch.setattr(video_client.os, "fsync", fail)
    with pytest.raises(VideoWorkerError, match="safely publish"):
        client().download("a" * 32, target)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_atomic_download_preserves_existing_file(tmp_path: Path):
    target = tmp_path / "video.mp4"
    target.write_bytes(b"existing")
    with pytest.raises(VideoWorkerError):
        client().download("a" * 32, target)
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_success_only_publishes_complete_file(tmp_path: Path):
    target = tmp_path / "video.mp4"
    client().download("a" * 32, target)
    assert target.read_bytes() == Response().read(256)
    assert list(tmp_path.iterdir()) == [target]
