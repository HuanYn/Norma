"""Offline camera mathematics tests; these do not imply model control quality."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ai.exploration import (
    ActionSequence,
    CameraAction,
    CameraIntrinsics,
    CameraPath,
    action_from_key,
    build_camera_path,
    control_request_fingerprint,
    validate_camera_poses,
    validate_lingbot_frame_count,
)


@pytest.fixture
def calibration() -> CameraIntrinsics:
    # An explicit synthetic calibration for mathematical tests, not estimated
    # from any photo and not claimed to be upstream's default calibration.
    return CameraIntrinsics(fx=600.0, fy=610.0, cx=416.0, cy=240.0)


def sequence(*specs: tuple[str, float, int], start: int = 0) -> ActionSequence:
    return ActionSequence(start_sequence=start, actions=tuple(
        CameraAction(sequence=start + index, action=name, magnitude=magnitude, frames=frames)
        for index, (name, magnitude, frames) in enumerate(specs)
    ))


@pytest.mark.parametrize(("key", "name", "magnitude"), [
    ("W", "forward", 1), ("s", "backward", 1), ("a", "left", 1), ("D", "right", 1),
    ("ArrowLeft", "look_left", 10), ("ArrowRight", "look_right", 10),
    ("ArrowUp", "look_up", 10), ("ArrowDown", "look_down", 10), (" ", "hold", 0),
])
def test_explicit_single_key_mapping(key, name, magnitude):
    action = action_from_key(key, sequence=5)
    assert (action.action, action.magnitude, action.sequence, action.frames) == (name, magnitude, 5, 8)


@pytest.mark.parametrize("key", ["wa", "forward", "Shift", "Space", " w", "", None, 1])
def test_no_unknown_or_simultaneous_controls(key):
    with pytest.raises(ValueError):
        action_from_key(key, sequence=0)


@pytest.mark.parametrize("override", [
    {"sequence": True}, {"sequence": -1}, {"sequence": "1"}, {"sequence": 1.0},
    {"frames": True}, {"frames": 0}, {"frames": 121}, {"frames": "8"}, {"frames": 8.5},
    {"magnitude": True}, {"magnitude": "1"}, {"magnitude": float("nan")},
    {"magnitude": float("inf")}, {"magnitude": -1}, {"magnitude": 0}, {"magnitude": 2.001},
    {"action": "jump"}, {"action": "roll"}, {"execute": "anything"},
])
def test_action_strict_bounds_and_extra_fields(override):
    data = {"sequence": 0, "action": "forward", "frames": 8, "magnitude": 1}
    with pytest.raises(ValidationError):
        CameraAction.model_validate(data | override)


def test_action_specific_amplitude_limits():
    assert action_from_key("ArrowRight", sequence=0, magnitude=45).magnitude == 45
    for name, magnitude in [("look_up", 45.1), ("look_down", 0), ("hold", 0.1)]:
        with pytest.raises(ValidationError):
            CameraAction(sequence=0, action=name, frames=8, magnitude=magnitude)


def test_sequence_number_and_shared_frame_count():
    actions = sequence(("forward", 1, 4), ("hold", 0, 4), start=17)
    assert actions.frame_count == 9
    assert actions.next_sequence == 19
    for indexes in [(17, 17), (17, 19), (18, 19), (18, 17)]:
        with pytest.raises(ValidationError, match="consecutive"):
            ActionSequence(start_sequence=17, actions=tuple(
                CameraAction(sequence=index, action="hold", magnitude=0, frames=4) for index in indexes
            ))
    with pytest.raises(ValidationError, match="overflow"):
        sequence(("hold", 0, 8), start=2**31 - 1)
    with pytest.raises(ValidationError):
        ActionSequence(start_sequence=True, actions=actions.actions)


@pytest.mark.parametrize("frame_counts", [[], [1], [3], [5], [120, 120, 120, 4], [4] * 33])
def test_sequence_bounded_and_never_padded(frame_counts):
    with pytest.raises(ValidationError):
        sequence(*[("hold", 0, count) for count in frame_counts])


@pytest.mark.parametrize(("name", "axis", "direction"), [
    ("forward", 2, 1), ("backward", 2, -1), ("left", 0, -1), ("right", 0, 1),
])
def test_translation_axes_magnitude_and_linear_intervals(calibration, name, axis, direction):
    path = build_camera_path(sequence((name, 2, 8)), calibration)
    assert path.poses.shape == (9, 4, 4)
    expected = np.zeros((9, 3))
    expected[:, axis] = direction * np.linspace(0, 2, 9)
    np.testing.assert_allclose(path.poses[:, :3, 3], expected, atol=1e-12)
    np.testing.assert_allclose(path.poses[:, :3, :3], np.repeat(np.eye(3)[None], 9, axis=0))
    np.testing.assert_array_equal(path.poses[0], np.eye(4))


@pytest.mark.parametrize(("name", "expected"), [
    ("look_right", [2**-0.5, 0, 2**-0.5]),
    ("look_left", [-2**-0.5, 0, 2**-0.5]),
    ("look_up", [0, -2**-0.5, 2**-0.5]),
    ("look_down", [0, 2**-0.5, 2**-0.5]),
])
def test_look_directions_in_opencv_not_opengl(calibration, name, expected):
    path = build_camera_path(sequence((name, 45, 8)), calibration)
    np.testing.assert_allclose(path.poses[-1, :3, :3] @ [0, 0, 1], expected, atol=1e-12)
    np.testing.assert_array_equal(path.poses[:, :3, 3], np.zeros((9, 3)))
    np.testing.assert_allclose(np.linalg.det(path.poses[:, :3, :3]), 1, atol=1e-12)


def test_turn_then_move_uses_current_camera_coordinates(calibration):
    path = build_camera_path(sequence(
        ("look_right", 45, 4), ("look_right", 45, 4), ("forward", 2, 4),
    ), calibration)
    np.testing.assert_allclose(path.poses[-1, :3, 3], [2, 0, 0], atol=1e-12)
    assert path.poses.shape == (13, 4, 4)  # No duplicate boundary frame.
    np.testing.assert_allclose(path.poses[10, :3, 3], [1, 0, 0], atol=1e-12)


def test_local_rotation_order_is_explicit(calibration):
    yaw_pitch = build_camera_path(sequence(("look_right", 45, 4), ("look_up", 45, 4)), calibration)
    pitch_yaw = build_camera_path(sequence(("look_up", 45, 4), ("look_right", 45, 4)), calibration)
    np.testing.assert_allclose(yaw_pitch.poses[-1, :3, 2], [0.5, -2**-0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(pitch_yaw.poses[-1, :3, 2], [2**-0.5, -0.5, 0.5], atol=1e-12)
    assert not np.allclose(yaw_pitch.poses[-1], pitch_yaw.poses[-1])


def test_opposites_return_to_initial_pose_and_hold_is_stationary(calibration):
    initial = np.eye(4)
    initial[:3, 3] = [3, 4, 5]
    actions = sequence(
        ("look_left", 30, 4), ("look_right", 30, 4),
        ("forward", 1.2, 4), ("backward", 1.2, 4), ("hold", 0, 4),
    )
    path = build_camera_path(actions, calibration, initial_pose=initial)
    np.testing.assert_allclose(path.poses[-1], initial, atol=1e-12)
    np.testing.assert_array_equal(path.poses[-5:], np.repeat(path.poses[-1][None], 5, axis=0))
    initial[0, 3] = 100
    assert path.poses[0, 0, 3] == 3  # Defensive copy.
    assert not path.poses.flags.writeable


def test_nonidentity_initial_rotation_is_respected(calibration):
    initial = np.array([[0, 0, 1, 10], [0, 1, 0, 20], [-1, 0, 0, 30], [0, 0, 0, 1]])
    path = build_camera_path(sequence(("forward", 1, 8)), calibration, initial_pose=initial)
    np.testing.assert_allclose(path.poses[-1, :3, 3], [11, 20, 30])
    np.testing.assert_array_equal(path.poses[-1, :3, :3], initial[:3, :3])


@pytest.mark.parametrize("mutation", ["scale", "reflection", "bottom", "nan", "translation"])
def test_invalid_pose_cannot_be_imported(mutation, calibration):
    pose = np.eye(4)
    if mutation == "scale":
        pose[0, 0] = 2
    elif mutation == "reflection":
        pose[0, 0] = -1
    elif mutation == "bottom":
        pose[3, 2] = 0.1
    elif mutation == "nan":
        pose[0, 3] = np.nan
    else:
        pose[0, 3] = 1_000_001
    with pytest.raises(ValueError):
        build_camera_path(sequence(("hold", 0, 8)), calibration, initial_pose=pose)


@pytest.mark.parametrize("poses", [np.eye(4), np.zeros((0, 4, 4)), np.zeros((362, 4, 4)),
    np.eye(4)[None].astype(str), np.eye(4)[None].astype(bool), np.zeros((2, 3, 4))])
def test_pose_shape_type_and_size_guards(poses):
    with pytest.raises(ValueError):
        validate_camera_poses(poses)


@pytest.mark.parametrize("override", [
    {"fx": 0}, {"fy": -1}, {"fx": float("inf")}, {"fx": True}, {"fy": "500"},
    {"cx": -1}, {"cx": 833}, {"cy": 481}, {"width": True}, {"height": 0}, {"extra": 1},
])
def test_intrinsics_validation(override):
    with pytest.raises(ValidationError):
        CameraIntrinsics.model_validate({"fx": 600, "fy": 600, "cx": 416, "cy": 240} | override)


def test_model_file_contract_is_absolute_float32_with_one_pixel_intrinsics_row(calibration):
    path = build_camera_path(sequence(("forward", 2, 8)), calibration)
    poses, intrinsics = path.lingbot_arrays()
    assert poses.shape == (9, 4, 4) and poses.dtype == np.float32
    assert intrinsics.shape == (1, 4) and intrinsics.dtype == np.float32
    np.testing.assert_array_equal(intrinsics, [[600, 610, 416, 240]])
    # No pre-normalization or relative conversion that upstream would apply twice.
    np.testing.assert_array_equal(poses[-1, :3, 3], [0, 0, 2])
    poses[0, 0, 0] = 0
    assert path.poses[0, 0, 0] == 1


def test_wrong_calibration_reference_is_not_silently_resized():
    calibration = CameraIntrinsics(width=640, height=384, fx=400, fy=400, cx=320, cy=192)
    path = build_camera_path(sequence(("hold", 0, 8)), calibration)
    with pytest.raises(ValueError, match="832x480"):
        path.lingbot_arrays()


@pytest.mark.parametrize(("frames", "latent"), [(9, 3), (21, 6), (33, 9), (81, 21), (357, 90)])
def test_upstream_chunk_alignment(frames, latent):
    assert validate_lingbot_frame_count(frames) == latent


@pytest.mark.parametrize("frames", [True, "9", 9.0, 0, 1, 5, 8, 49, 361, 365])
def test_would_truncate_or_invalid_frame_count_is_rejected(frames):
    with pytest.raises(ValueError):
        validate_lingbot_frame_count(frames)


def test_chunk_size_is_explicit_and_checked_at_export(calibration):
    path = build_camera_path(sequence(("hold", 0, 4)), calibration)
    with pytest.raises(ValueError, match="truncated"):
        path.lingbot_arrays()
    assert path.lingbot_arrays(latent_chunk_size=2)[0].shape[0] == 5
    for chunk in [True, 0, 33, "3", 3.5]:
        with pytest.raises(ValueError):
            validate_lingbot_frame_count(9, latent_chunk_size=chunk)


def test_fingerprint_stable_for_validated_json_and_signed_zero(calibration):
    actions = sequence(("hold", -0.0, 8))
    roundtrip = ActionSequence.model_validate_json(actions.model_dump_json())
    digest = control_request_fingerprint(actions, calibration)
    assert len(digest) == 64
    assert digest == control_request_fingerprint(roundtrip, calibration, initial_pose=np.eye(4))
    assert digest == control_request_fingerprint(sequence(("hold", 0, 8)), calibration)
    pose = np.eye(4)
    pose[0, 3] = -0.0
    assert digest == control_request_fingerprint(actions, calibration, initial_pose=pose)


def test_fingerprint_distinguishes_order_magnitude_frames_sequence_intrinsics_and_pose(calibration):
    batches = [
        sequence(("forward", 1, 4), ("look_up", 10, 4)),
        sequence(("look_up", 10, 4), ("forward", 1, 4)),
        sequence(("forward", 2, 4), ("look_up", 10, 4)),
        sequence(("forward", 1, 8), ("look_up", 10, 4)),
        sequence(("forward", 1, 4), ("look_up", 10, 4), start=5),
    ]
    hashes = {control_request_fingerprint(batch, calibration) for batch in batches}
    assert len(hashes) == len(batches)
    different_k = CameraIntrinsics(fx=610, fy=610, cx=416, cy=240)
    assert control_request_fingerprint(batches[0], different_k) not in hashes
    pose = np.eye(4)
    pose[2, 3] = 1
    assert control_request_fingerprint(batches[0], calibration, initial_pose=pose) not in hashes


def test_plain_dicts_are_not_accidentally_treated_as_validated_contracts(calibration):
    with pytest.raises(TypeError):
        build_camera_path({}, calibration)
    with pytest.raises(TypeError):
        control_request_fingerprint({}, calibration)
    with pytest.raises(TypeError):
        CameraPath(np.eye(4)[None], {})
