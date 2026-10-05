import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("world_publish", Path(__file__).resolve().parents[2] / "scripts/build_world_demo.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.fixture
def output(tmp_path, monkeypatch):
    # Synthetic bytes ONLY test the checks; not passed off as actual model output.
    video = b"test-stub-not-a-real-video"
    (tmp_path / "offline-camera-smoke.mp4").write_bytes(video)
    result = {"status": "completed", "requested_frames": 165,
              "video_sha256": hashlib.sha256(video).hexdigest(),
              "generation_seconds": 123, "output_shape_cthw": [3, 165, 640, 384]}
    (tmp_path / "result.json").write_text(json.dumps(result))
    stream = {"width": 384, "height": 640, "nb_read_frames": "165",
              "r_frame_rate": "16/1", "duration": "10.3125"}
    monkeypatch.setattr(publisher.subprocess, "check_output", lambda *a, **k: json.dumps({"streams": [stream]}))
    return tmp_path, stream


def test_complete_matching_delivery(output):
    path, _ = output
    assert publisher.validate_video(path)["duration"] == 10.3125


@pytest.mark.parametrize("change", [{"nb_read_frames": "164"}, {"duration": "9.99"},
    {"r_frame_rate": "8/1"}, {"width": 400}, {"height": 624}])
def test_reject_short_slowed_truncated_or_mismatched_video(output, change):
    path, stream = output
    stream.update(change)
    with pytest.raises(ValueError):
        publisher.validate_video(path)


def test_reject_changed_file(output):
    path, _ = output
    (path / "offline-camera-smoke.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        publisher.validate_video(path)


def test_reject_incomplete_job(output):
    path, _ = output
    result = json.loads((path / "result.json").read_text())
    result["status"] = "failed"
    (path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="completed"):
        publisher.validate_video(path)
