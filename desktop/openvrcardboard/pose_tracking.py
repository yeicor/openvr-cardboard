"""Webcam hand and full-body tracking using MediaPipe.

MediaPipe Hands gives 21 image landmarks per hand and MediaPipe Pose gives 33
body landmarks. Both are monocular, so absolute depth is estimated from a known
physical width (palm width for hands, shoulder/hip width for the body). The
resulting points are unprojected with the calibrated camera model and converted
to OpenVR play space.

The MediaPipe dependency is optional and imported lazily. The geometry helpers
are plain numpy so they can be tested without a camera or model installed.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration import CameraModel, optical_pose_to_play, optical_point_to_play
from .filters import LandmarkSmoother, OneEuroFilter
from .mediapipe_models import (
    HAND_LANDMARKER,
    ModelUnavailable,
    pose_model_spec,
    resolve_model,
)
from .mathutil import (
    clamp,
    from_matrix,
    normalize,
    from_two_vectors,
    yaw_only,
)

LOGGER = logging.getLogger(__name__)

# Landmark indices from the MediaPipe hand topology.
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

_TASKS_API = None


def _load_mediapipe():
    """Import MediaPipe once and require the Tasks API.

    The legacy ``mp.solutions`` API was removed in recent builds. Every tracker
    now uses ``HandLandmarker``/``PoseLandmarker`` from ``mediapipe.tasks``.
    """
    global _TASKS_API
    if _TASKS_API is not None:
        return _TASKS_API
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TrackingUnavailable(
            "hand and body tracking need the 'mediapipe' package"
        ) from exc
    try:
        from mediapipe.tasks import python as tasks_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - old/incomplete wheels
        raise TrackingUnavailable(
            "this MediaPipe build has no Tasks API; install mediapipe>=1.2.3"
        ) from exc
    _TASKS_API = (mp, tasks_python, vision)
    return _TASKS_API


def _video_timestamp(now: float, last_ms: int) -> int:
    """Return a strictly increasing millisecond timestamp for VIDEO mode."""
    value = int(round(float(now) * 1000.0))
    if value <= last_ms:
        value = last_ms + 1
    return value


def _make_landmarker(create, requested: str, label: str):
    """Create a landmarker, trying the GPU delegate then CPU.

    ``auto`` prefers GPU but always falls back. An explicit ``gpu`` request
    also falls back so a missing OpenGL/Vulkan delegate never disables hand or
    body tracking; the actual delegate is reported by :meth:`status`.
    """
    order = ("gpu", "cpu") if str(requested).lower() in ("auto", "gpu") else ("cpu",)
    errors: list[str] = []
    for delegate in order:
        try:
            landmarker = create(delegate)
        except Exception as exc:  # MediaPipe raises generic runtime errors
            errors.append(f"{delegate}: {exc}")
            LOGGER.warning("MediaPipe %s %s delegate failed: %s", label, delegate, exc)
            continue
        if errors:
            LOGGER.warning("MediaPipe %s running on CPU after delegate fallback", label)
        return landmarker, delegate, errors
    raise TrackingUnavailable(
        f"MediaPipe {label} failed to start ({'; '.join(errors)})"
    )


def _normalised_points(landmarks) -> np.ndarray:
    return np.array([(point.x, point.y) for point in landmarks], dtype=np.float64)


def _close_landmarker(landmarker, label: str) -> None:
    """Close a MediaPipe landmarker without leaving a poisonous __del__.

    A landmarker whose GPU graph failed can raise (or leave the native graph
    half-closed) when its handle is destroyed. MediaPipe's ``__del__`` then
    retries the same native close during garbage collection, which can block
    interpreter shutdown forever. Clearing the handle after a failed close
    makes the finalizer a no-op while the process keeps running on CPU.
    """
    try:
        landmarker.close()
    except Exception as exc:  # pragma: no cover - depends on GPU driver failures
        LOGGER.warning("failed to close MediaPipe %s: %s", label, exc)
        try:
            landmarker._handle = None
        except Exception:
            pass


class HandTracker:
    """MediaPipe HandLandmarker wrapper with play-space outputs."""

    def __init__(
        self,
        model: CameraModel,
        camera_position,
        camera_rotation,
        palm_width_m: float = 0.085,
        max_fps: int = 30,
        swap_hands: bool = False,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        landmark_cutoff: float = 6.0,
        landmark_beta: float = 0.6,
        delegate: str = "auto",
        model_path: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.model = model
        self.camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        self.camera_rotation = normalize(camera_rotation)
        self.palm_width_m = max(0.03, float(palm_width_m))
        self.max_fps = max(1, int(max_fps))
        self.swap_hands = bool(swap_hands)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.delegate = str(delegate or "auto").lower()
        self.model_path = str(model_path) if model_path else ""
        self.model_dir = str(model_dir) if model_dir else ""
        self.landmark_cutoff = max(0.0, float(landmark_cutoff))
        self.landmark_beta = max(0.0, float(landmark_beta))
        self._landmarker = None
        self._delegate_used = ""
        self._last_timestamp_ms = 0
        self._last_time = 0.0
        self._last_result: dict[str, HandObservation] = {}
        self._landmark_filters: dict[str, LandmarkSmoother] = {}
        self._last_seen: dict[str, float] = {}
        self._inference_ms = 0.0
        self._inferences = 0
        self._error = ""

    def _ensure(self, force_cpu: bool = False) -> None:
        if self._landmarker is not None and not force_cpu:
            return
        if self._landmarker is not None:
            self._release_landmarker()
        if not self.model_path:
            try:
                self.model_path = str(
                    resolve_model(HAND_LANDMARKER, self.model_dir or None)
                )
            except ModelUnavailable as exc:
                self._error = str(exc)
                raise TrackingUnavailable(str(exc)) from exc
        mp, tasks_python, vision = _load_mediapipe()

        def create(kind: str):
            delegate = (
                tasks_python.BaseOptions.Delegate.GPU
                if kind == "gpu"
                else tasks_python.BaseOptions.Delegate.CPU
            )
            options = vision.HandLandmarkerOptions(
                base_options=tasks_python.BaseOptions(
                    model_asset_path=self.model_path, delegate=delegate
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return vision.HandLandmarker.create_from_options(options)

        try:
            landmarker, delegate, errors = _make_landmarker(
                create, "cpu" if force_cpu else self.delegate, "hand landmarker"
            )
        except TrackingUnavailable as exc:
            self._error = str(exc)
            raise
        self._landmarker = landmarker
        self._delegate_used = delegate
        self._last_timestamp_ms = 0
        self._error = "; ".join(errors)
        LOGGER.info(
            "MediaPipe hand landmarker started (%s delegate, %s)",
            delegate,
            Path(self.model_path).name,
        )

    def process(
        self, frame_bgr, timestamp: Optional[float] = None
    ) -> dict[str, HandObservation]:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.max_fps > 0 and self._last_time > 0.0:
            if now - self._last_time < 1.0 / self.max_fps:
                return self._last_result
        self._ensure()
        mp, _, _ = _load_mediapipe()
        image = np.asarray(frame_bgr)
        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        stamp_ms = _video_timestamp(now, self._last_timestamp_ms)
        started = time.perf_counter()
        try:
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp_ms
            )
        except Exception as exc:
            if self._delegate_used == "gpu":
                LOGGER.warning(
                    "MediaPipe hand GPU inference failed (%s); retrying on CPU", exc
                )
                try:
                    self._ensure(force_cpu=True)
                    stamp_ms = _video_timestamp(now, self._last_timestamp_ms)
                    result = self._landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                        stamp_ms,
                    )
                except Exception as retry_exc:
                    self._error = f"hand inference failed: {retry_exc}"
                    LOGGER.debug("MediaPipe hands failed: %s", retry_exc)
                    return self._last_result
            else:
                self._error = f"hand inference failed: {exc}"
                LOGGER.debug("MediaPipe hands failed: %s", exc)
                return self._last_result
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._inference_ms = (
            elapsed_ms
            if self._inference_ms <= 0.0
            else 0.85 * self._inference_ms + 0.15 * elapsed_ms
        )
        self._inferences += 1
        self._last_timestamp_ms = stamp_ms
        self._error = ""

        output: dict[str, HandObservation] = {}
        self._last_time = now
        seen: set[str] = set()
        for landmarks, handedness in zip(
            result.hand_landmarks or [], result.handedness or []
        ):
            label = "right"
            confidence = 1.0
            try:
                category = handedness[0]
                name = getattr(category, "category_name", "") or getattr(
                    category, "display_name", ""
                )
                if name:
                    label = str(name).lower()
                confidence = float(category.score)
            except (AttributeError, IndexError, TypeError, ValueError):
                pass
            if self.swap_hands:
                label = "left" if label == "right" else "right"
            points = _normalised_points(landmarks)
            pixels = _pixels(points, self.model.width, self.model.height)
            smoother = self._landmark_filters.get(label)
            last_seen = self._last_seen.get(label, 0.0)
            if smoother is None:
                smoother = LandmarkSmoother(
                    21,
                    2,
                    min_cutoff=self.landmark_cutoff,
                    beta=self.landmark_beta,
                )
                self._landmark_filters[label] = smoother
            elif last_seen > 0.0 and now - last_seen > 1.0:
                smoother.reset()
            pixels = smoother.filter(pixels, now)
            observation = hand_pose_from_landmarks(
                pixels,
                self.model,
                self.camera_position,
                self.camera_rotation,
                self.palm_width_m,
                label,
            )
            if observation is None:
                continue
            observation.confidence = confidence
            observation.timestamp = now
            output[label] = observation
            self._last_seen[label] = now
            seen.add(label)
        for label in list(self._landmark_filters):
            last_seen = self._last_seen.get(label, 0.0)
            if label not in seen and last_seen > 0.0 and now - last_seen > 1.0:
                self._landmark_filters.pop(label, None)
                self._last_seen.pop(label, None)
        self._last_result = output
        return output

    def status(self) -> dict:
        return {
            "ready": self._landmarker is not None,
            "model": (
                Path(self.model_path).name
                if self.model_path
                else HAND_LANDMARKER.filename
            ),
            "requested_delegate": self.delegate,
            "delegate": self._delegate_used or self.delegate,
            "model_path": self.model_path,
            "inference_ms": round(self._inference_ms, 2),
            "inferences": self._inferences,
            "error": self._error,
        }

    def draw(self, frame_bgr, observations: dict[str, HandObservation]):
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return frame_bgr
        for side, observation in observations.items():
            points = observation.landmarks.astype(np.int32)
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame_bgr,
                    tuple(points[start]),
                    tuple(points[end]),
                    (80, 200, 255) if side == "left" else (255, 170, 80),
                    2,
                )
            for point in points:
                cv2.circle(frame_bgr, tuple(point), 2, (255, 255, 255), -1)
            label = f"{side} pin {observation.pinch:.2f} grip {observation.grip:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                tuple(points[HAND_WRIST] + np.array([0, 18])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return frame_bgr

    def _release_landmarker(self) -> None:
        landmarker = self._landmarker
        self._landmarker = None
        self._delegate_used = ""
        if landmarker is not None:
            _close_landmarker(landmarker, "hand landmarker")

    def close(self) -> None:
        self._release_landmarker()
        self._last_timestamp_ms = 0
        self._last_result = {}
        self._landmark_filters.clear()
        self._last_seen.clear()


class BodyTracker:
    """MediaPipe PoseLandmarker wrapper producing sparse tracker poses."""

    def __init__(
        self,
        model: CameraModel,
        camera_position,
        camera_rotation,
        shoulder_width_m: float = 0.40,
        hip_width_m: float = 0.32,
        body_scale: float = 1.0,
        position_offset=(0.0, 0.0, 0.0),
        max_fps: int = 30,
        roles: Optional[list[str]] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pose_model: str = "full",
        min_visibility: float = 0.45,
        landmark_cutoff: float = 6.0,
        landmark_beta: float = 0.6,
        delegate: str = "auto",
        model_path: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.model = model
        self.camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        self.camera_rotation = normalize(camera_rotation)
        self.shoulder_width_m = float(shoulder_width_m)
        self.hip_width_m = float(hip_width_m)
        self.body_scale = float(body_scale)
        self.position_offset = tuple(float(v) for v in position_offset)
        self.max_fps = max(1, int(max_fps))
        # None keeps every joint; an empty list disables body-driven trackers
        # while still reporting the body observation for HMD/controller
        # fallback. Expand user-facing roles such as "waist" to MediaPipe's
        # "hips" landmark.
        self.roles = body_joint_names(roles)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.pose_model = str(pose_model or "full").lower()
        self._model_spec = pose_model_spec(self.pose_model)
        self.min_visibility = clamp(float(min_visibility), 0.0, 1.0)
        self.landmark_cutoff = max(0.0, float(landmark_cutoff))
        self.landmark_beta = max(0.0, float(landmark_beta))
        self.delegate = str(delegate or "auto").lower()
        self.model_path = str(model_path) if model_path else ""
        self.model_dir = str(model_dir) if model_dir else ""
        self._landmarker = None
        self._delegate_used = ""
        self._last_timestamp_ms = 0
        self._last_time = 0.0
        self._last_seen = 0.0
        self._last_result: Optional[BodyObservation] = None
        self._landmark_filter: Optional[LandmarkSmoother] = None
        self._inference_ms = 0.0
        self._inferences = 0
        self._error = ""
        self._depth_filter = OneEuroFilter(
            min_cutoff=max(1.0, self.landmark_cutoff * 0.4),
            beta=max(0.0, self.landmark_beta * 0.5),
        )

    def _ensure(self, force_cpu: bool = False) -> None:
        if self._landmarker is not None and not force_cpu:
            return
        if self._landmarker is not None:
            self._release_landmarker()
        if not self.model_path:
            try:
                self.model_path = str(
                    resolve_model(self._model_spec, self.model_dir or None)
                )
            except ModelUnavailable as exc:
                self._error = str(exc)
                raise TrackingUnavailable(str(exc)) from exc
        mp, tasks_python, vision = _load_mediapipe()

        def create(kind: str):
            delegate = (
                tasks_python.BaseOptions.Delegate.GPU
                if kind == "gpu"
                else tasks_python.BaseOptions.Delegate.CPU
            )
            options = vision.PoseLandmarkerOptions(
                base_options=tasks_python.BaseOptions(
                    model_asset_path=self.model_path, delegate=delegate
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            return vision.PoseLandmarker.create_from_options(options)

        try:
            landmarker, delegate, errors = _make_landmarker(
                create, "cpu" if force_cpu else self.delegate, "pose landmarker"
            )
        except TrackingUnavailable as exc:
            self._error = str(exc)
            raise
        self._landmarker = landmarker
        self._delegate_used = delegate
        self._last_timestamp_ms = 0
        self._error = "; ".join(errors)
        LOGGER.info(
            "MediaPipe pose landmarker started (%s delegate, %s)",
            delegate,
            Path(self.model_path).name,
        )

    def process(
        self, frame_bgr, timestamp: Optional[float] = None
    ) -> Optional[BodyObservation]:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.max_fps > 0 and self._last_time > 0.0:
            if now - self._last_time < 1.0 / self.max_fps:
                return self._last_result
        self._ensure()
        mp, _, _ = _load_mediapipe()
        image = np.asarray(frame_bgr)
        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        stamp_ms = _video_timestamp(now, self._last_timestamp_ms)
        started = time.perf_counter()
        try:
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp_ms
            )
        except Exception as exc:
            if self._delegate_used == "gpu":
                LOGGER.warning(
                    "MediaPipe pose GPU inference failed (%s); retrying on CPU", exc
                )
                try:
                    self._ensure(force_cpu=True)
                    stamp_ms = _video_timestamp(now, self._last_timestamp_ms)
                    result = self._landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                        stamp_ms,
                    )
                except Exception as retry_exc:
                    self._error = f"body inference failed: {retry_exc}"
                    LOGGER.debug("MediaPipe pose failed: %s", retry_exc)
                    return self._last_result
            else:
                self._error = f"body inference failed: {exc}"
                LOGGER.debug("MediaPipe pose failed: %s", exc)
                return self._last_result
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._inference_ms = (
            elapsed_ms
            if self._inference_ms <= 0.0
            else 0.85 * self._inference_ms + 0.15 * elapsed_ms
        )
        self._inferences += 1
        self._last_timestamp_ms = stamp_ms
        self._error = ""
        self._last_time = now
        if not result.pose_landmarks:
            self._last_result = None
            return None
        landmarks = result.pose_landmarks[0]
        points = _normalised_points(landmarks)
        visibility = np.array(
            [float(getattr(point, "visibility", 1.0)) for point in landmarks],
            dtype=np.float64,
        )
        pixels = _pixels(points, self.model.width, self.model.height)
        if self._landmark_filter is None:
            self._landmark_filter = LandmarkSmoother(
                33,
                2,
                min_cutoff=self.landmark_cutoff,
                beta=self.landmark_beta,
            )
            self._depth_filter.reset()
        elif self._last_seen > 0.0 and now - self._last_seen > 1.0:
            self._landmark_filter.reset()
            self._depth_filter.reset()
        pixels = self._landmark_filter.filter(pixels, now)
        raw_depth = _body_depth(
            pixels,
            self.model,
            self.shoulder_width_m,
            self.hip_width_m,
            visibility=visibility,
            min_visibility=self.min_visibility,
        )
        depth = self._depth_filter.filter(raw_depth, now)
        observation = body_pose_from_landmarks(
            pixels,
            self.model,
            self.camera_position,
            self.camera_rotation,
            self.shoulder_width_m,
            self.hip_width_m,
            self.body_scale,
            self.position_offset,
            self.roles,
            visibility=visibility,
            min_visibility=self.min_visibility,
            depth=depth,
        )
        self._last_seen = now
        if observation is not None:
            observation.confidence = float(np.mean(visibility)) if visibility.size else 0.0
            observation.timestamp = now
        self._last_result = observation
        return observation

    def status(self) -> dict:
        return {
            "ready": self._landmarker is not None,
            "model": (
                Path(self.model_path).name
                if self.model_path
                else self._model_spec.filename
            ),
            "variant": self.pose_model,
            "requested_delegate": self.delegate,
            "delegate": self._delegate_used or self.delegate,
            "model_path": self.model_path,
            "inference_ms": round(self._inference_ms, 2),
            "inferences": self._inferences,
            "error": self._error,
        }

    def draw(self, frame_bgr, observation: Optional[BodyObservation]):
        if observation is None:
            return frame_bgr
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return frame_bgr
        points = observation.landmarks.astype(np.int32)
        for start, end in POSE_CONNECTIONS:
            cv2.line(frame_bgr, tuple(points[start]), tuple(points[end]), (90, 230, 140), 2)
        for point in points:
            cv2.circle(frame_bgr, tuple(point), 2, (255, 255, 255), -1)
        cv2.putText(
            frame_bgr,
            f"body depth {observation.depth:.2f} m",
            (8, max(18, frame_bgr.shape[0] - 12) if hasattr(frame_bgr, "shape") else 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 255, 220),
            1,
            cv2.LINE_AA,
        )
        return frame_bgr

    def _release_landmarker(self) -> None:
        landmarker = self._landmarker
        self._landmarker = None
        self._delegate_used = ""
        if landmarker is not None:
            _close_landmarker(landmarker, "pose landmarker")

    def close(self) -> None:
        self._release_landmarker()
        self._last_timestamp_ms = 0
        self._last_result = None
        self._landmark_filter = None
        self._depth_filter.reset()
        self._last_seen = 0.0
