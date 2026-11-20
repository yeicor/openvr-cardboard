"""MediaPipe landmark geometry for hand and body tracking.

These helpers turn MediaPipe's normalised image landmarks into play-space
poses. They are plain numpy so they can be tested without a camera, model or
MediaPipe installation; :mod:`openvrcardboard.pose_tracking` owns the optional
runtime and the tracker classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .calibration import CameraModel, optical_point_to_play, optical_pose_to_play
from .mathutil import (
    clamp,
    from_matrix,
    from_two_vectors,
    normalize,
    yaw_only,
)

HAND_WRIST = 0
HAND_INDEX_MCP = 5
HAND_MIDDLE_MCP = 9
HAND_RING_MCP = 13
HAND_PINKY_MCP = 17
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_MIDDLE_TIP = 12
HAND_RING_TIP = 16
HAND_PINKY_TIP = 20
HAND_FINGERTIPS = (8, 12, 16, 20)

# MediaPipe Pose landmark indices.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_KNEE = 25
POSE_RIGHT_KNEE = 26
POSE_LEFT_ANKLE = 27
POSE_RIGHT_ANKLE = 28
POSE_LEFT_HEEL = 29
POSE_RIGHT_HEEL = 30
POSE_LEFT_FOOT = 31
POSE_RIGHT_FOOT = 32

# Driver tracker roles are named from the user's point of view ("waist"),
# while MediaPipe's landmark names are anatomical ("hips"). A role can map to
# more than one body joint so explicit tracker configurations keep working.
BODY_ROLE_JOINTS = {
    "waist": ("hips",),
    "pelvis": ("hips",),
    "spine_base": ("hips",),
    "chest": ("chest",),
    "spine_chest": ("chest",),
    "neck": ("chest",),
    "upper_chest": ("chest",),
    "left_foot": ("left_foot",),
    "left_ankle": ("left_foot",),
    "left_toe": ("left_foot",),
    "right_foot": ("right_foot",),
    "right_ankle": ("right_foot",),
    "right_toe": ("right_foot",),
    "left_shoulder": ("left_shoulder",),
    "left_clavicle": ("left_shoulder",),
    "right_shoulder": ("right_shoulder",),
    "right_clavicle": ("right_shoulder",),
    "left_hand": ("left_hand",),
    "left_wrist": ("left_hand",),
    "right_hand": ("right_hand",),
    "right_wrist": ("right_hand",),
    "left_knee": ("left_knee",),
    "right_knee": ("right_knee",),
    "head": ("head",),
}


def body_joint_names(roles: Optional[list[str]]) -> Optional[set[str]]:
    """Expand tracker role names into body landmark names.

    ``None`` means every joint; an empty list means no joints. Keeping those
    two cases distinct lets a user disable body trackers without disabling the
    body detector used for controller/HMD fallback.
    """
    if roles is None:
        return None
    names: set[str] = set()
    for role in roles:
        names.update(BODY_ROLE_JOINTS.get(role, (role,)))
    return names


class TrackingUnavailable(RuntimeError):
    """Raised when MediaPipe is not installed or cannot start."""


@dataclass(slots=True)
class HandObservation:
    side: str
    position: np.ndarray
    rotation: np.ndarray
    pinch: float
    grip: float
    # 21 x 2 pixel coordinates, kept for preview drawing.
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 2)))
    depth: float = 0.0
    confidence: float = 1.0
    valid: bool = True
    timestamp: float = 0.0

    def copy(self) -> "HandObservation":
        return HandObservation(
            side=self.side,
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            pinch=float(self.pinch),
            grip=float(self.grip),
            landmarks=self.landmarks.copy(),
            depth=float(self.depth),
            confidence=float(self.confidence),
            valid=bool(self.valid),
            timestamp=float(self.timestamp),
        )


@dataclass(slots=True)
class BodyObservation:
    # Play-space joint positions keyed by tracker role.
    joints: dict[str, np.ndarray] = field(default_factory=dict)
    # Yaw-only orientations keyed by tracker role.
    rotations: dict[str, np.ndarray] = field(default_factory=dict)
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((33, 2)))
    depth: float = 0.0
    confidence: float = 1.0
    valid: bool = True
    timestamp: float = 0.0

    def copy(self) -> "BodyObservation":
        return BodyObservation(
            joints={key: value.copy() for key, value in self.joints.items()},
            rotations={key: value.copy() for key, value in self.rotations.items()},
            landmarks=self.landmarks.copy(),
            depth=float(self.depth),
            confidence=float(self.confidence),
            valid=bool(self.valid),
            timestamp=float(self.timestamp),
        )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("landmarks must be an N x 2 array")
    points = points[:, :2].copy()
    if points.max(initial=0.0) <= 1.5 and points.min(initial=0.0) >= -0.5:
        # MediaPipe normalised landmarks.
        points[:, 0] *= max(1, width)
        points[:, 1] *= max(1, height)
    return points


def _basis_rotation(x_axis, y_axis, z_axis) -> np.ndarray:
    matrix = np.column_stack(
        (
            np.asarray(x_axis, dtype=np.float64).reshape(3),
            np.asarray(y_axis, dtype=np.float64).reshape(3),
            np.asarray(z_axis, dtype=np.float64).reshape(3),
        )
    )
    return from_matrix(matrix)


def _orthonormal(forward: np.ndarray, across: np.ndarray):
    forward = normalize_vector(forward)
    across = np.asarray(across, dtype=np.float64).reshape(3)
    across = across - forward * float(np.dot(across, forward))
    if np.linalg.norm(across) < 1e-8:
        across = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    across = normalize_vector(across)
    up = normalize_vector(np.cross(forward, across))
    return forward, normalize_vector(np.cross(up, forward)), up


def normalize_vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / length


def hand_inputs_from_landmarks(pixels: np.ndarray) -> tuple[float, float]:
    """Return ``(pinch, grip)`` in 0..1 from 21 hand pixels."""
    wrist = pixels[HAND_WRIST]
    palm_width = float(np.linalg.norm(pixels[HAND_INDEX_MCP] - pixels[HAND_PINKY_MCP]))
    palm_length = float(np.linalg.norm(pixels[HAND_MIDDLE_MCP] - wrist))
    if palm_width < 1e-3 or palm_length < 1e-3:
        return 0.0, 0.0

    pinch_distance = float(
        np.linalg.norm(pixels[HAND_THUMB_TIP] - pixels[HAND_INDEX_TIP])
    )
    pinch = clamp(1.0 - (pinch_distance / palm_width - 0.15) / 0.85, 0.0, 1.0)

    openness = 0.0
    for tip in HAND_FINGERTIPS:
        distance = float(np.linalg.norm(pixels[tip] - wrist)) / palm_length
        openness += clamp((distance - 0.75) / 1.05, 0.0, 1.0)
    grip = clamp(1.0 - openness / len(HAND_FINGERTIPS), 0.0, 1.0)
    return pinch, grip


def hand_pose_from_landmarks(
    landmarks,
    model: CameraModel,
    camera_position,
    camera_rotation,
    palm_width_m: float = 0.085,
    side: str = "right",
    max_depth_m: float = 6.0,
    min_depth_m: float = 0.12,
) -> Optional[HandObservation]:
    """Turn 21 normalised/pixel hand landmarks into a play-space pose."""
    height, width = model.height, model.width
    pixels = _pixels(np.asarray(landmarks, dtype=np.float64), width, height)
    if pixels.shape[0] < 21:
        return None
    palm_width_px = float(
        np.linalg.norm(pixels[HAND_INDEX_MCP] - pixels[HAND_PINKY_MCP])
    )
    if not math.isfinite(palm_width_px) or palm_width_px < 2.0:
        return None
    focal = max(1.0, 0.5 * (model.fx + model.fy))
    depth = focal * max(0.01, float(palm_width_m)) / palm_width_px
    depth = clamp(depth, min_depth_m, max_depth_m)

    def optical(index: int) -> np.ndarray:
        return model.unproject(float(pixels[index, 0]), float(pixels[index, 1]), depth)

    wrist = optical(HAND_WRIST)
    index_mcp = optical(HAND_INDEX_MCP)
    pinky_mcp = optical(HAND_PINKY_MCP)
    centre = np.mean(
        [wrist, index_mcp, optical(HAND_MIDDLE_MCP), optical(HAND_RING_MCP), pinky_mcp],
        axis=0,
    )
    position = optical_point_to_play(camera_position, camera_rotation, centre)

    forward_optical, across_optical, _ = _orthonormal(
        index_mcp - wrist, index_mcp - pinky_mcp
    )
    # Device frame: -Z along the fingers, +X to the anatomical right of the
    # hand. For a right hand the index is on the pinky side of +X and for a
    # left hand it is the other way around.
    sign = 1.0 if side == "left" else -1.0
    x_axis = across_optical * sign
    z_axis = -forward_optical
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    optical_rotation = _basis_rotation(x_axis, y_axis, z_axis)

    _, rotation = optical_pose_to_play(
        camera_position, camera_rotation, centre, optical_rotation
    )
    pinch, grip = hand_inputs_from_landmarks(pixels)
    return HandObservation(
        side=side,
        position=position,
        rotation=normalize(rotation),
        pinch=pinch,
        grip=grip,
        landmarks=pixels,
        depth=depth,
        confidence=1.0,
    )


def _visibility_ok(visibility, indices, minimum: float) -> bool:
    if visibility is None:
        return True
    try:
        values = [float(visibility[index]) for index in indices]
    except (TypeError, ValueError, IndexError):
        return True
    return all(math.isfinite(value) and value >= minimum for value in values)


def _body_depth(
    pixels: np.ndarray,
    model: CameraModel,
    shoulder_width_m: float,
    hip_width_m: float,
    visibility=None,
    min_visibility: float = 0.0,
) -> float:
    """Estimate body depth from visible body widths and torso height.

    Shoulder and hip widths shrink when the player turns, while the vertical
    shoulder-to-hip distance stays roughly constant. Taking a weighted median of
    all three estimates makes depth noticeably steadier than relying on a
    single pair, especially when one side of the body is partly occluded.
    """
    focal = max(1.0, 0.5 * (model.fx + model.fy))
    estimates: list[float] = []
    weights: list[float] = []
    pairs = (
        (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER, shoulder_width_m),
        (POSE_LEFT_HIP, POSE_RIGHT_HIP, hip_width_m),
    )
    for left, right, real_width in pairs:
        pixel_width = float(np.linalg.norm(pixels[left] - pixels[right]))
        if pixel_width <= 4.0:
            continue
        if visibility is not None:
            pair_visibility = min(
                float(visibility[left]), float(visibility[right])
            )
            if pair_visibility < min_visibility:
                continue
        else:
            pair_visibility = 1.0
        estimates.append(focal * real_width / pixel_width)
        weights.append(max(0.05, pair_visibility))

    # Torso height: shoulders to hips. This estimate is stable when the body
    # is rotated away from the camera, where width-based estimates collapse.
    shoulder_mid = 0.5 * (pixels[POSE_LEFT_SHOULDER] + pixels[POSE_RIGHT_SHOULDER])
    hip_mid = 0.5 * (pixels[POSE_LEFT_HIP] + pixels[POSE_RIGHT_HIP])
    torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_height > 4.0 and (
        visibility is None
        or _visibility_ok(
            visibility,
            (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER, POSE_LEFT_HIP, POSE_RIGHT_HIP),
            min_visibility,
        )
    ):
        torso_visibility = (
            1.0
            if visibility is None
            else min(
                float(visibility[POSE_LEFT_SHOULDER]),
                float(visibility[POSE_RIGHT_SHOULDER]),
                float(visibility[POSE_LEFT_HIP]),
                float(visibility[POSE_RIGHT_HIP]),
            )
        )
        estimates.append(focal * 0.50 / torso_height)
        weights.append(max(0.05, torso_visibility))

    if not estimates:
        return 2.5
    order = np.argsort(np.asarray(estimates, dtype=np.float64))
    ordered_estimates = np.asarray(estimates, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights)
    median_index = int(
        np.searchsorted(cumulative, 0.5 * float(cumulative[-1]))
    )
    median_index = min(median_index, len(ordered_estimates) - 1)
    return clamp(float(ordered_estimates[median_index]), 0.4, 8.0)


def _yaw_from_direction(direction, forward=(0.0, 0.0, -1.0)) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    direction[1] = 0.0
    if np.linalg.norm(direction) < 1e-8:
        return normalize(np.array([0.0, 0.0, 0.0, 1.0]))
    return yaw_only(from_two_vectors(forward, normalize_vector(direction)))


def body_pose_from_landmarks(
    landmarks,
    model: CameraModel,
    camera_position,
    camera_rotation,
    shoulder_width_m: float = 0.40,
    hip_width_m: float = 0.32,
    body_scale: float = 1.0,
    position_offset=(0.0, 0.0, 0.0),
    roles: Optional[list[str]] = None,
    visibility=None,
    min_visibility: float = 0.0,
    depth: Optional[float] = None,
) -> Optional[BodyObservation]:
    """Map MediaPipe Pose landmarks to play-space tracker poses.

    Landmarks below ``min_visibility`` are not turned into joints. That keeps a
    tracker from snapping to a hallucinated occluded limb, which MediaPipe
    sometimes reports with a confident-looking but impossible position.
    """
    pixels = _pixels(np.asarray(landmarks, dtype=np.float64), model.width, model.height)
    if pixels.shape[0] < 33:
        return None
    if depth is None:
        depth = _body_depth(
            pixels,
            model,
            shoulder_width_m,
            hip_width_m,
            visibility=visibility,
            min_visibility=min_visibility,
        )

    def optical(index: int) -> np.ndarray:
        return model.unproject(float(pixels[index, 0]), float(pixels[index, 1]), float(depth))

    def play(index: int) -> np.ndarray:
        return optical_point_to_play(camera_position, camera_rotation, optical(index))

    def visible(*indices: int) -> bool:
        return _visibility_ok(visibility, indices, min_visibility)

    hips_mid = 0.5 * (play(POSE_LEFT_HIP) + play(POSE_RIGHT_HIP))
    shoulder_mid = 0.5 * (play(POSE_LEFT_SHOULDER) + play(POSE_RIGHT_SHOULDER))
    hips_visible = visible(POSE_LEFT_HIP, POSE_RIGHT_HIP)
    shoulders_visible = visible(POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER)
    anchor = hips_mid if hips_visible or not shoulders_visible else shoulder_mid
    scale = clamp(float(body_scale), 0.2, 3.0)
    offset = np.asarray(position_offset, dtype=np.float64).reshape(3)

    def scaled(point: np.ndarray) -> np.ndarray:
        return anchor + (point - anchor) * scale + offset

    candidates = {
        "hips": (hips_mid, (POSE_LEFT_HIP, POSE_RIGHT_HIP)),
        "chest": (shoulder_mid, (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER)),
        "left_foot": (play(POSE_LEFT_ANKLE), (POSE_LEFT_ANKLE,)),
        "right_foot": (play(POSE_RIGHT_ANKLE), (POSE_RIGHT_ANKLE,)),
        "left_shoulder": (play(POSE_LEFT_SHOULDER), (POSE_LEFT_SHOULDER,)),
        "right_shoulder": (play(POSE_RIGHT_SHOULDER), (POSE_RIGHT_SHOULDER,)),
        "left_hand": (play(POSE_LEFT_WRIST), (POSE_LEFT_WRIST,)),
        "right_hand": (play(POSE_RIGHT_WRIST), (POSE_RIGHT_WRIST,)),
        "left_knee": (play(POSE_LEFT_KNEE), (POSE_LEFT_KNEE,)),
        "right_knee": (play(POSE_RIGHT_KNEE), (POSE_RIGHT_KNEE,)),
        "head": (play(0), (0,)),
    }
    accepted = body_joint_names(roles)
    joints: dict[str, np.ndarray] = {}
    for role, (point, required) in candidates.items():
        if accepted is not None and role not in accepted:
            continue
        if not visible(*required):
            continue
        joints[role] = scaled(point)

    # Yaw from the hip line, falling back to the shoulder line when the hips
    # are not visible or half-turned.
    left_hip = play(POSE_LEFT_HIP)
    right_hip = play(POSE_RIGHT_HIP)
    left_shoulder = play(POSE_LEFT_SHOULDER)
    right_shoulder = play(POSE_RIGHT_SHOULDER)
    if hips_visible:
        body_yaw = _yaw_from_direction(right_hip - left_hip)
        if np.linalg.norm(_horizontal(right_hip - left_hip)) < 0.05:
            body_yaw = _yaw_from_direction(right_shoulder - left_shoulder)
    elif shoulders_visible:
        body_yaw = _yaw_from_direction(right_shoulder - left_shoulder)
    else:
        body_yaw = normalize(np.array([0.0, 0.0, 0.0, 1.0]))

    rotations: dict[str, np.ndarray] = {}
    for role in joints:
        if role in ("left_foot", "right_foot"):
            heel = POSE_LEFT_HEEL if role == "left_foot" else POSE_RIGHT_HEEL
            toe = POSE_LEFT_FOOT if role == "left_foot" else POSE_RIGHT_FOOT
            if visible(heel, toe):
                rotations[role] = _yaw_from_direction(play(toe) - play(heel))
            else:
                rotations[role] = body_yaw
        else:
            rotations[role] = body_yaw

    return BodyObservation(
        joints=joints,
        rotations=rotations,
        landmarks=pixels,
        depth=float(depth),
        confidence=1.0,
    )


def _horizontal(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).reshape(3).copy()
    result[1] = 0.0
    return result


# ---------------------------------------------------------------------------
# MediaPipe Tasks wrappers
# ---------------------------------------------------------------------------
# Connection lists from the MediaPipe Tasks topology. Keeping them local means
# drawing works even before the optional MediaPipe import has happened.
HAND_CONNECTIONS = (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), (1, 2), (2, 3),
    (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
    (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
)
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
