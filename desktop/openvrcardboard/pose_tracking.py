"""Webcam hand and full-body tracking using MediaPipe.

MediaPipe Hands gives 21 image landmarks per hand and MediaPipe Pose gives 33
body landmarks. The landmark geometry lives in
:mod:`openvrcardboard.pose_geometry`; this module owns the optional MediaPipe
dependency, the landmarker lifecycle and the tracker classes.

The MediaPipe dependency is optional and imported lazily.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration import CameraModel
from .filters import LandmarkSmoother, OneEuroFilter
from .mathutil import clamp, normalize
from .mediapipe_models import (
    HAND_LANDMARKER,
    ModelUnavailable,
    pose_model_spec,
    resolve_model,
)
from .pose_geometry import (
    HAND_CONNECTIONS,
    HAND_WRIST,
    POSE_CONNECTIONS,
    BodyObservation,
    HandObservation,
    TrackingUnavailable,
    _body_depth,
    _pixels,
    body_joint_names,
    body_pose_from_landmarks,
    hand_inputs_from_landmarks,
    hand_pose_from_landmarks,
)

LOGGER = logging.getLogger(__name__)

# Landmark geometry is re-exported so callers and tests can keep importing
# the tracking vocabulary from one module.
__all__ = [
    "BodyObservation",
    "HandObservation",
    "body_joint_names",
    "body_pose_from_landmarks",
    "hand_inputs_from_landmarks",
    "hand_pose_from_landmarks",
]

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
