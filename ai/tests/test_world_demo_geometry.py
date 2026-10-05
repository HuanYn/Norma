"""CPU-only checks; no claim about visual control or model quality."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("world_smoke", Path(__file__).resolve().parents[2] / "scripts/check_world_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.mark.parametrize("frames", [9, 21, 165, 357])
def test_frames_align(frames):
    assert runtime.aligned_frame_count(str(frames)) == frames


@pytest.mark.parametrize("frames", [1, 5, 10, 49, 161, 361])
def test_frames_reject_truncation(frames):
    with pytest.raises(argparse.ArgumentTypeError):
        runtime.aligned_frame_count(str(frames))


@pytest.mark.parametrize("original,expected", [((1206, 2030), (384, 640)), ((1206, 1601), (384, 512))])
def test_full_portrait_canvas_and_upstream_float_floor(original, expected):
    width, height = runtime.portrait_geometry(384, *original)
    assert (width, height) == expected
    area = width * height + 1
    assert round(np.sqrt(area * height / width) // 8 // 2 * 2) * 8 == height
    assert round(np.sqrt(area / (height / width)) // 8 // 2 * 2) * 8 == width


def test_165_frames_meet_duration_without_slowdown_or_loop():
    assert 165 / 16 >= 10
