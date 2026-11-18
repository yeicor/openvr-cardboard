"""Board-based ArUco tracking for the HMD, left hand and right hand.

This module replaces the old one-marker-per-device tracker.  Every frame is
decoded once, markers are grouped by their rigid board, and each board is
solved as a single PnP problem.  The resulting measurement is fed to an
independent :class:`~openvrcardboard.tracking_state.TrackedObject` filter,
which owns velocity, acceleration, uncertainty, prediction and the per-board
tracking state machine.

The detector is OpenCV's CPU ArUco implementation.  At 1920x1080 with fifteen
markers it measures roughly 2.5-5 ms on a current desktop CPU, so a custom GPU
detector only adds maintenance risk.  Detection defaults to full capture
resolution because the 40 mm peripheral markers are only about 33 pixels wide
at 2 m with a 60-degree camera.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .aruco import get_dictionary
from .boards import (
    BoardError,
    BoardPoseEstimate,
    BoardPoseEstimator,
    BoardSpec,
    board_axis_points,
    board_measurement_covariance,
    board_object_pose,
    board_optical_rotation,
    default_boards,
)
from .calibration import CameraModel
from .config import ARUCO_DICTIONARIES, CUSTOM_DICTIONARY_3X3
from .mathutil import (
    OPTICAL_TO_VIEW,
    conjugate,
    from_rotvec,
    multiply,
    normalize,
    rotate,
)
from .tracking_state import ObjectMeasurement, ObjectPose, TrackedObject, TrackingState

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MarkerDetection:
    """Filtered pose of one tracked object, keyed by its central marker ID.

    The name is kept for compatibility with the shared-state and fusion
    layers, which speak in terms of marker IDs.  A detection now represents a
    whole board, not one square marker.
    """

    marker_id: int
    position: np.ndarray
    rotation: np.ndarray
    optical_position: np.ndarray
    optical_rotation: np.ndarray
    reprojection_error: float
    distance: float
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    valid: bool = True
    board: str = ""
    marker_ids: tuple[int, ...] = ()
    inlier_ids: tuple[int, ...] = ()
    outlier_ids: tuple[int, ...] = ()
    marker_count: int = 0
    confidence: float = 0.0
    state: TrackingState = TrackingState.LOST
    position_sigma: float = 0.0
    orientation_sigma: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_rotation: np.ndarray = field(default_factory=lambda: np.zeros(4))
    capture_time: float = 0.0
    detection_time: float = 0.0
    pose_time: float = 0.0
    update_time: float = 0.0
    per_marker_error: dict[int, float] = field(default_factory=dict)
    board_corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))

    def copy(self) -> "MarkerDetection":
        return MarkerDetection(
            marker_id=int(self.marker_id),
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            optical_position=self.optical_position.copy(),
            optical_rotation=self.optical_rotation.copy(),
            reprojection_error=float(self.reprojection_error),
            distance=float(self.distance),
            corners=self.corners.copy(),
            valid=bool(self.valid),
            board=self.board,
            marker_ids=tuple(self.marker_ids),
            inlier_ids=tuple(self.inlier_ids),
            outlier_ids=tuple(self.outlier_ids),
            marker_count=int(self.marker_count),
            confidence=float(self.confidence),
            state=self.state,
            position_sigma=float(self.position_sigma),
            orientation_sigma=float(self.orientation_sigma),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            linear_acceleration=self.linear_acceleration.copy(),
            angular_acceleration=self.angular_acceleration.copy(),
            raw_position=self.raw_position.copy(),
            raw_rotation=self.raw_rotation.copy(),
            capture_time=float(self.capture_time),
            detection_time=float(self.detection_time),
            pose_time=float(self.pose_time),
            update_time=float(self.update_time),
            per_marker_error=dict(self.per_marker_error),
            board_corners=self.board_corners.copy(),
        )


class BoardTracker:
    """Detects five-marker boards and tracks three independent 6DOF objects."""

    def __init__(
        self,
        model: CameraModel,
        camera_position,
        camera_rotation,
        boards: Optional[dict[str, BoardSpec]] = None,
        dictionary: str = CUSTOM_DICTIONARY_3X3,
        detection_width: int = 0,
        refine_pose: bool = True,
        max_reprojection_error: float = 4.0,
        hold_time_s: float = 0.22,
        lost_timeout_s: float = 1.0,
        reacquire_frames: int = 3,
        process_noise: float = 3.0,
        angular_process_noise: float = 12.0,
        adaptive_noise: bool = True,
        static_noise_scale: float = 0.2,
        exposure_s: float = 0.005,
        min_markers_tracking: int = 3,
        min_confidence: float = 0.25,
        lost_position_sigma_m: float = 0.5,
        lost_orientation_sigma_deg: float = 30.0,
        reinitialize_gap_s: float = 0.30,
    ):
        self.model = model
        self.camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        self.camera_rotation = normalize(camera_rotation)
        self.boards = boards or default_boards()
        if not self.boards:
            raise BoardError("at least one board is required")
        self.dictionary_name = str(dictionary).upper()
        self.detection_width = max(0, int(detection_width))
        self.refine_pose = bool(refine_pose)
        self.max_reprojection_error = max(0.1, float(max_reprojection_error))
        self.exposure_s = max(0.0, float(exposure_s))
        self.min_markers_tracking = max(1, int(min_markers_tracking))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

        self._marker_to_board: dict[int, str] = {}
        for name, board in self.boards.items():
            for marker_id in board.marker_ids:
                if marker_id in self._marker_to_board:
                    raise BoardError(f"marker ID {marker_id} is used by more than one board")
                self._marker_to_board[int(marker_id)] = name

        self.estimator = BoardPoseEstimator(
            max_reprojection_error=self.max_reprojection_error,
            refine_pose=self.refine_pose,
        )
        self._cv2 = None
        self._detector = None
        self._detection_ms = 0.0
        self._timings: deque[float] = deque(maxlen=240)
        self._error = ""
        self._empty_frames = 0
        self._last_estimates: dict[str, BoardPoseEstimate] = {}
        self._last_previous_object: dict[str, np.ndarray] = {}
        self._last_previous_times: dict[str, float] = {}
        self._last_update_times: dict[str, float] = {}

        self.objects: dict[int, TrackedObject] = {}
        self.object_names: dict[str, TrackedObject] = {}
        for name, board in self.boards.items():
            tracker = TrackedObject(
                name=name,
                primary_marker_id=board.primary_id,
                process_noise=process_noise,
                angular_process_noise=angular_process_noise,
                adaptive_noise=adaptive_noise,
                static_noise_scale=static_noise_scale,
                max_prediction_s=lost_timeout_s,
                hold_time_s=hold_time_s,
                lost_timeout_s=lost_timeout_s,
                reacquire_frames=reacquire_frames,
                reacquire_blend_s=0.16,
                reinitialize_gap_s=reinitialize_gap_s,
                lost_position_sigma_m=lost_position_sigma_m,
                lost_orientation_sigma_deg=lost_orientation_sigma_deg,
                min_markers_tracking=min(
                    self.min_markers_tracking, len(board.markers)
                ),
                max_reprojection_px=self.max_reprojection_error,
                min_confidence=self.min_confidence,
            )
            self.objects[board.primary_id] = tracker
            self.object_names[name] = tracker

    # ------------------------------------------------------------------
    # OpenCV plumbing
    # ------------------------------------------------------------------
    def _load_cv2(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BoardError(
                "marker tracking needs the 'opencv-contrib-python' package"
            ) from exc
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
            raise BoardError("marker tracking needs OpenCV 4.7 or newer with aruco")
        self._cv2 = cv2
        return cv2

    def _make_detector(self):
        cv2 = self._load_cv2()
        name = self.dictionary_name
        try:
            dictionary = get_dictionary(cv2, name)
        except ValueError as exc:
            LOGGER.warning("%s; using %s", exc, CUSTOM_DICTIONARY_3X3)
            name = CUSTOM_DICTIONARY_3X3
            dictionary = get_dictionary(cv2, name)
        self.dictionary_name = name
        parameters = cv2.aruco.DetectorParameters()
        # The default threshold schedule is both faster and less prone to
        # false contours than the old wide schedule; the markers are high
        # contrast and the detector runs at full resolution.
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.minMarkerPerimeterRate = 0.012
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.01
        parameters.polygonalApproxAccuracyRate = 0.04
        parameters.minOtsuStdDev = 3.0
        parameters.perspectiveRemovePixelPerCell = 4
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return self._detector

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(
        self, frame_bgr, timestamp: Optional[float] = None
    ) -> dict[int, MarkerDetection]:
        """Detect boards in a BGR frame and return one entry per visible board."""
        cv2 = self._load_cv2()
        if frame_bgr is None:
            return {}
        started = time.monotonic()
        capture_time = started if timestamp is None else float(timestamp)
        image = np.asarray(frame_bgr)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        model = self.model.scaled(gray.shape[1], gray.shape[0])
        detection_started = time.perf_counter()
        try:
            corners, ids = self._detect_corners(cv2, gray)
            self._error = ""
        except Exception as exc:
            LOGGER.debug("board detection failed: %s", exc)
            self._error = f"ArUco detection failed: {exc}"
            return {}
        detection_end = time.monotonic()
        self._detection_ms = (time.perf_counter() - detection_started) * 1000.0
        self._timings.append(self._detection_ms)
        if ids is None or len(ids) == 0:
            self._empty_frames += 1
            return {}

        self._empty_frames = 0
        grouped: dict[str, dict[int, np.ndarray]] = {}
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            board_name = self._marker_to_board.get(marker_id)
            if board_name is None:
                continue
            grouped.setdefault(board_name, {})[marker_id] = np.asarray(
                marker_corners, dtype=np.float64
            ).reshape(4, 2)

        results: dict[int, MarkerDetection] = {}
        for name, board in self.boards.items():
            board_corners = grouped.get(name)
            if not board_corners:
                continue
            state = self.object_names[name]
            # The prior for the planar-ambiguity tie-break is the last
            # *accepted raw* orientation, advanced by the filter's angular
            # velocity over the gap.  Using the filtered pose itself would let
            # one over-confident velocity estimate steer the next solve onto
            # the mirrored IPPE branch.
            prior_rotation = None
            prior_time = self._last_previous_times.get(name)
            previous_object = self._last_previous_object.get(name)
            if previous_object is not None and prior_time is not None:
                gap = capture_time - prior_time
                if 0.0 <= gap <= 0.25:
                    prior_rotation = board_optical_rotation(
                        board,
                        previous_object,
                        self.camera_rotation,
                        OPTICAL_TO_VIEW,
                    )
                else:
                    prior_time = None
            estimate = self.estimator.estimate(
                board,
                board_corners,
                model,
                previous_rotation=prior_rotation,
                previous_time=prior_time,
                now=capture_time,
            )
            if estimate is None:
                continue
            pose_time = time.monotonic()
            self._last_estimates[name] = estimate

            raw_position, raw_rotation = board_object_pose(
                board,
                estimate.tvec,
                estimate.rotation,
                self.camera_position,
                self.camera_rotation,
                OPTICAL_TO_VIEW,
            )
            position_covariance, orientation_covariance = board_measurement_covariance(
                estimate,
                model,
                board,
                self.camera_rotation,
                OPTICAL_TO_VIEW,
            )
            speed = float(np.linalg.norm(state.filter.velocity))
            angular_speed = float(np.linalg.norm(state.filter.angular_velocity))
            confidence = self._measurement_confidence(
                estimate, board, model, speed, angular_speed
            )
            measurement = ObjectMeasurement(
                position=raw_position,
                rotation=raw_rotation,
                position_covariance=position_covariance,
                orientation_covariance=orientation_covariance,
                confidence=confidence,
                capture_time=capture_time,
                detection_time=detection_end,
                pose_time=pose_time,
                reprojection_error=estimate.reprojection_error,
                marker_count=estimate.marker_count,
                distance=estimate.distance,
                incidence_rad=estimate.incidence_rad,
                image_sigma_px=estimate.image_sigma_px,
                board_area_fraction=estimate.board_area_fraction,
                marker_ids=estimate.marker_ids,
                raw_position=raw_position,
                raw_rotation=raw_rotation,
            )
            update_time = time.monotonic()
            # The filter advances on capture time, never on processing time:
            # a delayed frame must not look like a long prediction step.  The
            # driver applies the only configured prediction horizon.
            state.observe(measurement, now=capture_time)
            accepted = state.last_detection_time == capture_time
            self._last_update_times[name] = capture_time
            filtered = state.pose_at(capture_time)
            if filtered is None:
                continue
            if accepted:
                # Only an accepted measurement may move the continuity prior;
                # otherwise a rejected outlier would steer the next solve.
                self._last_previous_object[name] = raw_rotation.copy()
                self._last_previous_times[name] = capture_time

            inlier_corners = np.concatenate(
                [estimate.corners[marker_id] for marker_id in estimate.marker_ids],
                axis=0,
            )
            all_corners = np.concatenate(
                [estimate.corners[marker_id] for marker_id in sorted(estimate.corners)],
                axis=0,
            )
            board_corners_px = self._project_board_outline(board, raw_position, raw_rotation, model)
            results[board.primary_id] = MarkerDetection(
                marker_id=board.primary_id,
                position=filtered.position.copy(),
                rotation=filtered.rotation.copy(),
                optical_position=estimate.tvec.copy(),
                optical_rotation=estimate.rotation.copy(),
                reprojection_error=float(estimate.reprojection_error),
                distance=float(estimate.distance),
                corners=all_corners,
                valid=bool(filtered.valid),
                board=name,
                marker_ids=tuple(estimate.corners.keys()),
                inlier_ids=tuple(estimate.marker_ids),
                outlier_ids=tuple(estimate.outlier_ids),
                marker_count=int(estimate.marker_count),
                confidence=float(confidence),
                state=filtered.state,
                position_sigma=float(filtered.position_sigma),
                orientation_sigma=float(filtered.orientation_sigma),
                velocity=filtered.velocity.copy(),
                angular_velocity=filtered.angular_velocity.copy(),
                linear_acceleration=filtered.linear_acceleration.copy(),
                angular_acceleration=filtered.angular_acceleration.copy(),
                raw_position=raw_position,
                raw_rotation=raw_rotation,
                capture_time=capture_time,
                detection_time=detection_end,
                pose_time=pose_time,
                update_time=update_time,
                per_marker_error=dict(estimate.per_marker_error),
                board_corners=board_corners_px,
            )
        return results

    def _detect_corners(self, cv2, gray):
        detector = self._detector if self._detector is not None else self._make_detector()
        height, width = gray.shape[:2]
        scale = 1.0
        detection_image = gray
        if self.detection_width and width > self.detection_width:
            scale = self.detection_width / float(width)
            detection_image = cv2.resize(
                gray,
                (self.detection_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        corners, ids, _ = detector.detectMarkers(detection_image)
        if ids is None or len(ids) == 0:
            # A distant peripheral marker can vanish in a downscaled image;
            # retry at full resolution every third empty frame.
            if scale < 1.0 and self._empty_frames % 3 == 2:
                corners, ids, _ = detector.detectMarkers(gray)
                scale = 1.0
            if ids is None or len(ids) == 0:
                return [], None
        mapped = [
            np.asarray(corner, dtype=np.float64).reshape(4, 2) / scale
            for corner in corners
        ]
        if scale < 1.0:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                50,
                0.01,
            )
            refined = []
            for corner in mapped:
                try:
                    refined.append(
                        cv2.cornerSubPix(
                            gray,
                            corner.astype(np.float32).reshape(-1, 1, 2),
                            (5, 5),
                            (-1, -1),
                            criteria,
                        )
                        .reshape(4, 2)
                        .astype(np.float64)
                    )
                except Exception:
                    refined.append(corner)
            mapped = refined
        return mapped, ids

    # ------------------------------------------------------------------
    # Quality and debug helpers
    # ------------------------------------------------------------------
    def _measurement_confidence(
        self,
        estimate: BoardPoseEstimate,
        board: BoardSpec,
        model: CameraModel,
        speed: float,
        angular_speed: float,
    ) -> float:
        max_error = max(0.5, self.max_reprojection_error)
        error_score = math.exp(-((estimate.reprojection_error / max_error) ** 2) * 2.0)
        count_score = min(1.0, 0.35 + 0.13 * estimate.marker_count)
        incidence_score = max(
            0.25, math.sqrt(max(0.0, math.cos(min(1.45, estimate.incidence_rad))))
        )
        # Motion blur is not directly measurable without exposure metadata, so
        # estimate the smear from the previous velocity and a configurable
        # exposure time.  It only reduces confidence; calibration is never
        # modified during tracking.
        f = max(1.0, 0.5 * (model.fx + model.fy))
        distance = max(0.1, estimate.distance)
        blur_px = self.exposure_s * (
            speed * f / distance
            + angular_speed * max(1.0, estimate.board_span_px * 0.5)
        )
        blur_score = math.exp(-blur_px / 8.0)
        spread_score = min(
            1.0,
            estimate.board_span_px
            / max(1.0, f * float(board.width_m) / distance),
        )
        quality = error_score * count_score * incidence_score * blur_score * spread_score
        return float(max(0.02, min(1.0, quality)))

    def _project_board_outline(
        self,
        board: BoardSpec,
        position: np.ndarray,
        rotation: np.ndarray,
        model: CameraModel,
    ) -> np.ndarray:
        """Project the board outline into the current image for debug drawing."""
        try:
            cv2 = self._load_cv2()
            points = board.outer_points
            q_view_board = multiply(rotation, board.object_quaternion)
            board_origin = position + rotate(rotation, board.object_offset)
            points_view = np.array(
                [rotate(q_view_board, point) + board_origin for point in points]
            )
            q_camera = multiply(self.camera_rotation, OPTICAL_TO_VIEW)
            points_optical = np.array(
                [
                    rotate(conjugate(q_camera), point - self.camera_position)
                    for point in points_view
                ]
            )
            projected, _ = cv2.projectPoints(
                points_optical.reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                model.matrix,
                model.distortion,
            )
            return np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        except Exception:
            return np.zeros((4, 2), dtype=np.float64)

    def object_states(self) -> dict[int, TrackedObject]:
        return dict(self.objects)

    def pose_at(self, marker_id: int, timestamp: float) -> Optional[ObjectPose]:
        tracker = self.objects.get(int(marker_id))
        if tracker is None:
            return None
        return tracker.pose_at(timestamp)

    def status(self) -> dict:
        timings = sorted(self._timings)
        count = len(timings)
        percentile = (
            (timings[min(count - 1, int(count * 0.95))] if count else 0.0)
        )
        return {
            "backend": "opencv-cpu",
            "dictionary": self.dictionary_name,
            "detection_ms": round(float(self._detection_ms), 3),
            "detection_p95_ms": round(float(percentile), 3),
            "error": self._error,
            "version": getattr(self._cv2, "__version__", "") if self._cv2 is not None else "",
            "width": self.detection_width,
            "refine_pose": self.refine_pose,
            "boards": [tracker.status() for tracker in self.object_names.values()],
        }

    def close(self) -> None:
        self._detector = None
        self._last_estimates.clear()

    # ------------------------------------------------------------------
    # Preview helpers
    # ------------------------------------------------------------------
    def draw(self, frame_bgr, detections: dict[int, MarkerDetection], now: Optional[float] = None):
        """Draw corners, raw/filtered/predicted axes, velocity and state."""
        cv2 = self._load_cv2()
        stamp = time.monotonic() if now is None else float(now)
        for marker_id, detection in detections.items():
            # Detected corners: green inliers, red outliers.
            points = detection.corners.astype(np.int32).reshape(-1, 1, 2)
            if len(points):
                cv2.polylines(frame_bgr, [points], True, (0, 220, 120), 1)
            outlier_ids = set(detection.outlier_ids)
            for marker, corners in self._last_estimates.get(detection.board).corners.items() if self._last_estimates.get(detection.board) else ():
                color = (0, 0, 230) if marker in outlier_ids else (0, 220, 120)
                cv2.polylines(
                    frame_bgr,
                    [np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)],
                    True,
                    color,
                    2,
                )
            board = self.boards.get(detection.board)
            if board is None:
                continue
            self._draw_axes(
                cv2,
                frame_bgr,
                board,
                detection.raw_position,
                detection.raw_rotation,
                (0, 220, 255),
                2,
            )
            predicted = self.pose_at(marker_id, stamp)
            filtered_position = detection.position
            filtered_rotation = detection.rotation
            if predicted is not None:
                filtered_position = predicted.position
                filtered_rotation = predicted.rotation
            self._draw_axes(
                cv2,
                frame_bgr,
                board,
                filtered_position,
                filtered_rotation,
                (255, 200, 0),
                2,
            )
            if predicted is not None:
                self._draw_axes(
                    cv2,
                    frame_bgr,
                    board,
                    predicted.position,
                    predicted.rotation,
                    (255, 0, 255),
                    1,
                )
                # Velocity arrow from the board origin.
                origin = self._project_play_point(filtered_position)
                tip = self._project_play_point(
                    filtered_position + predicted.velocity * 0.08
                )
                if origin is not None and tip is not None:
                    cv2.arrowedLine(
                        frame_bgr,
                        origin,
                        tip,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.25,
                    )
            centre = detection.corners.mean(axis=0)
            label = (
                f"{detection.board} {detection.state.value} "
                f"n={detection.marker_count} err={detection.reprojection_error:.1f}px "
                f"conf={detection.confidence:.2f}"
            )
            cv2.putText(
                frame_bgr,
                label,
                (int(centre[0]) + 6, int(centre[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 240, 180),
                1,
                cv2.LINE_AA,
            )
            detail = (
                f"d={detection.distance:.2f}m "
                f"sp={detection.position_sigma * 1000:.1f}mm "
                f"sq={math.degrees(detection.orientation_sigma):.2f}deg"
            )
            cv2.putText(
                frame_bgr,
                detail,
                (int(centre[0]) + 6, int(centre[1]) + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        return frame_bgr

    def _draw_axes(
        self,
        cv2,
        frame,
        board: BoardSpec,
        position: np.ndarray,
        rotation: np.ndarray,
        color,
        thickness: int,
    ) -> None:
        try:
            q_view_board = multiply(rotation, board.object_quaternion)
            origin_play = position + rotate(rotation, board.object_offset)
            axis_points = board_axis_points(board)
            origin = rotate(q_view_board, axis_points[0]) + origin_play
            xtip = rotate(q_view_board, axis_points[1]) + origin_play
            ytip = rotate(q_view_board, axis_points[2]) + origin_play
            ztip = rotate(q_view_board, axis_points[3]) + origin_play
            o = self._project_play_point(origin)
            x = self._project_play_point(xtip)
            y = self._project_play_point(ytip)
            z = self._project_play_point(ztip)
            if None in (o, x, y, z):
                return
            cv2.line(frame, o, x, color, thickness, cv2.LINE_AA)
            cv2.line(frame, o, y, color, thickness, cv2.LINE_AA)
            cv2.line(frame, o, z, color, thickness, cv2.LINE_AA)
        except Exception:
            return

    def _project_play_point(self, position: np.ndarray):
        try:
            cv2 = self._load_cv2()
            q_camera = multiply(self.camera_rotation, OPTICAL_TO_VIEW)
            optical = rotate(
                conjugate(q_camera),
                np.asarray(position, dtype=np.float64) - self.camera_position,
            )
            if optical[2] <= 1e-4:
                return None
            projected, _ = cv2.projectPoints(
                optical.reshape(1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                self.model.matrix,
                self.model.distortion,
            )
            point = np.asarray(projected, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(point)):
                return None
            return (int(round(point[0])), int(round(point[1])))
        except Exception:
            return None


__all__ = ["BoardError", "BoardTracker", "MarkerDetection"]
