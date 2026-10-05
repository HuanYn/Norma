"""Bounded, CPU-only actions and an explicit OpenCV camera-file contract.

Contract reference (read, not imported or copied): Robbyant LingBot-World-v2,
commit 1895d300d8ac936401689b26389f51cbd36530eb, wan/image2video.py
and wan/utils/cam_utils.py, https://github.com/robbyant/lingbot-world-v2
(upstream is CC BY-NC-SA 4.0). This module independently constructs elementary
SE(3) transforms; it does not reproduce upstream generation/embedding code.

The reference consumes absolute camera-to-world poses (N,4,4) and one pixel
intrinsic row [fx,fy,cx,cy] for an 832x480 reference image. Camera axes are
+X right, +Y down, +Z forward, acting on column vectors. Our WASD/arrow mapping
and bounded movement units are Norma policy, NOT an upstream key/speed API.
Translation is in arbitrary units, not measured metres. The upstream loader
subsamples, converts to framewise relative poses, and normalizes translation;
absolute movement scale may therefore disappear during model conditioning.

No image geometry is recovered here. There is no model, network, rendering,
latent cache, session state, or claim of validated visual control/real-time use.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


CONTRACT_VERSION = "norma-opencv-controls-v1"
REFERENCE_WIDTH = 832
REFERENCE_HEIGHT = 480
MAX_FRAMES = 361
ActionName = Literal[
    "forward", "backward", "left", "right",
    "look_left", "look_right", "look_up", "look_down", "hold",
]
SequenceNumber = Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]
_MOVES = {"forward", "backward", "left", "right"}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class CameraAction(_Contract):
    """One local-axis action over `frames` NEW frame intervals.

    Magnitude is the total displacement (0 < value <= 2 arbitrary units),
    total look angle (0 < value <= 45 degrees), or exactly zero for hold.
    Opposite directions use different names, never negative magnitudes.
    Each action is interpolated from its starting pose without simultaneous
    movement/turning. Frames do not include the shared starting frame.
    """

    sequence: SequenceNumber
    action: ActionName
    frames: Annotated[int, Field(strict=True, ge=1, le=120)]
    magnitude: Annotated[float, Field(strict=True, ge=0, le=45, allow_inf_nan=False)]

    @model_validator(mode="after")
    def _magnitude_for_action(self) -> CameraAction:
        if self.action == "hold":
            if self.magnitude != 0:
                raise ValueError("hold magnitude must be zero")
        elif self.magnitude <= 0:
            raise ValueError("moving or looking requires positive magnitude")
        elif self.action in _MOVES and self.magnitude > 2:
            raise ValueError("translation magnitude must not exceed 2 arbitrary units")
        return self


class ActionSequence(_Contract):
    """Ordered batch; a future session must separately enforce expected sequence.

    No gaps, duplicate sequence numbers, silent padding, or frame truncation.
    The first pose plus new frame intervals must form 4k+1 input frames.
    This alone does NOT guarantee alignment to the chosen model chunk size;
    `CameraPath.lingbot_arrays` checks that additional requirement.
    """

    start_sequence: SequenceNumber = 0
    actions: Annotated[tuple[CameraAction, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def _ordered_and_bounded(self) -> ActionSequence:
        for offset, action in enumerate(self.actions):
            if action.sequence != self.start_sequence + offset:
                raise ValueError("actions must have consecutive sequence numbers from start_sequence")
        if self.next_sequence > 2**31 - 1:
            raise ValueError("next sequence number would overflow")
        intervals = sum(action.frames for action in self.actions)
        if not 4 <= intervals < MAX_FRAMES or intervals % 4:
            raise ValueError("total new frame intervals must be 4..360 and divisible by 4")
        return self

    @property
    def frame_count(self) -> int:
        return 1 + sum(action.frames for action in self.actions)

    @property
    def next_sequence(self) -> int:
        return self.start_sequence + len(self.actions)


class CameraIntrinsics(_Contract):
    """Pinhole pixel calibration for a stated image size, not inferred from it."""

    width: Annotated[int, Field(strict=True, ge=16, le=8192)] = REFERENCE_WIDTH
    height: Annotated[int, Field(strict=True, ge=16, le=8192)] = REFERENCE_HEIGHT
    fx: Annotated[float, Field(strict=True, gt=0, le=1_000_000, allow_inf_nan=False)]
    fy: Annotated[float, Field(strict=True, gt=0, le=1_000_000, allow_inf_nan=False)]
    cx: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    cy: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def _principal_point(self) -> CameraIntrinsics:
        if self.cx > self.width or self.cy > self.height:
            raise ValueError("principal point must be inside the reference image")
        return self


_KEYS: dict[str, ActionName] = {
    "w": "forward", "s": "backward", "a": "left", "d": "right",
    "arrowleft": "look_left", "arrowright": "look_right",
    "arrowup": "look_up", "arrowdown": "look_down", " ": "hold",
}


def action_from_key(
    key: str, *, sequence: int, frames: int = 8, magnitude: float | None = None,
) -> CameraAction:
    """Map one browser KeyboardEvent.key; no chord, free text or event execution."""
    if not isinstance(key, str) or key.lower() not in _KEYS:
        raise ValueError("supported keys are W/A/S/D, ArrowLeft/Right/Up/Down, and Space")
    name = _KEYS[key.lower()]
    if magnitude is None:
        magnitude = 0.0 if name == "hold" else 1.0 if name in _MOVES else 10.0
    return CameraAction(sequence=sequence, action=name, frames=frames, magnitude=magnitude)


def validate_camera_poses(poses: object) -> np.ndarray:
    """Return a defensive float64 copy of finite, proper SE(3) matrices.

    This checks matrix validity, NOT whether an arbitrary matrix was originally
    expressed as c2w rather than w2c; the latter is an explicit caller contract.
    """
    values = np.asarray(poses)
    if values.dtype.kind not in "fiu":
        raise ValueError("poses must be numeric, not strings, objects, or booleans")
    if values.ndim != 3 or values.shape[1:] != (4, 4) or not 1 <= len(values) <= MAX_FRAMES:
        raise ValueError("poses must have shape (N,4,4) with 1 <= N <= 361")
    values = values.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("poses must contain only finite numbers")
    if not np.allclose(values[:, 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("poses must have homogeneous last row [0,0,0,1]")
    rotation = values[:, :3, :3]
    if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("pose rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError("pose rotations must have determinant +1, not reflections")
    if np.abs(values[:, :3, 3]).max() > 1_000_000:
        raise ValueError("pose translation exceeds the bounded coordinate range")
    return values


def _initial_camera(initial_pose: object | None) -> np.ndarray:
    if initial_pose is None:
        return np.eye(4, dtype=np.float64)
    value = np.asarray(initial_pose)
    if value.shape != (4, 4):
        raise ValueError("initial_pose must be one (4,4) camera-to-world matrix")
    return validate_camera_poses(value[None])[0]


def validate_lingbot_frame_count(frame_count: int, *, latent_chunk_size: int = 3) -> int:
    """Reject inputs the pinned upstream would silently shorten; return latent count."""
    if type(frame_count) is not int or not 5 <= frame_count <= MAX_FRAMES or (frame_count - 1) % 4:
        raise ValueError("frame_count must be 4k+1, between 5 and 361")
    if type(latent_chunk_size) is not int or not 1 <= latent_chunk_size <= 32:
        raise ValueError("latent_chunk_size must be an integer in 1..32")
    latent_count = (frame_count - 1) // 4 + 1
    if latent_count % latent_chunk_size:
        raise ValueError("frame_count would be truncated by the upstream latent chunk alignment")
    return latent_count


@dataclass(frozen=True)
class CameraPath:
    """Mathematically valid camera conditioning, not generated or observed views."""

    poses: np.ndarray
    intrinsics: CameraIntrinsics

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be a validated CameraIntrinsics")
        poses = validate_camera_poses(self.poses)
        poses.flags.writeable = False
        object.__setattr__(self, "poses", poses)

    def lingbot_arrays(self, *, latent_chunk_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Return (poses.npy, intrinsics.npy) payloads; never write files or run a model.

        The caller must use this same frame count/chunk size when invoking the
        pinned upstream. No relative-pose or normalization preprocessing here:
        upstream does that itself. Calibration MUST target its fixed 832x480
        reference; arbitrary image resizing/cropping needs a separate adapter.
        """
        validate_lingbot_frame_count(len(self.poses), latent_chunk_size=latent_chunk_size)
        intrinsics = self.intrinsics
        if (intrinsics.width, intrinsics.height) != (REFERENCE_WIDTH, REFERENCE_HEIGHT):
            raise ValueError("pinned LingBot loader requires intrinsics calibrated for 832x480")
        row = [[intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]]
        return self.poses.astype(np.float32, copy=True), np.asarray(row, dtype=np.float32)


def _local_transform(action: CameraAction, fraction: float) -> np.ndarray:
    """Elementary right-handed axis rotation or translation, derived directly."""
    transform = np.eye(4, dtype=np.float64)
    amount = action.magnitude * fraction
    if action.action in _MOVES:
        coordinate = 2 if action.action in {"forward", "backward"} else 0
        direction = -1 if action.action in {"backward", "left"} else 1
        transform[coordinate, 3] = direction * amount
    elif action.action != "hold":
        # In OpenCV coordinates +yaw looks right; +pitch looks up.
        axis = 1 if action.action in {"look_left", "look_right"} else 0
        direction = -1 if action.action in {"look_left", "look_down"} else 1
        angle = math.radians(direction * amount)
        unit = np.eye(3)[axis]
        x, y, z = unit
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        # Rodrigues formula for an axis unit vector, no upstream implementation.
        transform[:3, :3] = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)
    return transform


def build_camera_path(
    actions: ActionSequence, intrinsics: CameraIntrinsics, *, initial_pose: object | None = None,
) -> CameraPath:
    """Piecewise single-axis local movement; each shared boundary occurs once."""
    if not isinstance(actions, ActionSequence) or not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("use validated ActionSequence and CameraIntrinsics inputs")
    start = _initial_camera(initial_pose)
    poses = [start]
    for action in actions.actions:
        poses.extend(start @ _local_transform(action, frame / action.frames) for frame in range(1, action.frames + 1))
        start = poses[-1]
    return CameraPath(poses=np.stack(poses), intrinsics=intrinsics)


def control_request_fingerprint(
    actions: ActionSequence, intrinsics: CameraIntrinsics, *, initial_pose: object | None = None,
) -> str:
    """Stable input digest for future idempotency checks, NOT a replay registry.

    A service still has to bind this to session ID, expected sequence, model
    revision, generation options and image hash, and persist/reject conflicts.
    It must not treat this digest alone as proof that an action was executed.
    """
    if not isinstance(actions, ActionSequence) or not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("use validated ActionSequence and CameraIntrinsics inputs")
    pose = _initial_camera(initial_pose)
    pose[pose == 0] = 0  # Canonicalize signed zero without rounding real inputs.
    payload = {
        "contract": CONTRACT_VERSION,
        "actions": actions.model_dump(mode="json"),
        "intrinsics": intrinsics.model_dump(mode="json"),
        "initial_pose": pose.tolist(),
    }
    # Float fields accept integer JSON inputs, but canonical output is uniform.
    for action in payload["actions"]["actions"]:
        if action["magnitude"] == 0:
            action["magnitude"] = 0.0
    for key in ("cx", "cy"):
        if payload["intrinsics"][key] == 0:
            payload["intrinsics"][key] = 0.0
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
