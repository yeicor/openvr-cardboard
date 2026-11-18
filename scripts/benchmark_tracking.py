#!/usr/bin/env python3
"""Benchmark five-marker board tracking against the old single-marker pose.

The baseline solves only the large central marker with IPPE, exactly as the
retired one-marker tracker did.  The board tracker solves all visible markers
of the rigid board in one PnP problem, then filters and predicts the object
pose.  Both see the same rendered frames and the same detector parameters.

Examples:

    python scripts/benchmark_tracking.py
    python scripts/benchmark_tracking.py --width 1280 --height 720 --frames 120
    python scripts/benchmark_tracking.py --output /tmp/opencv-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = REPO_ROOT / "desktop"
if str(DESKTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(DESKTOP_ROOT))

from openvrcardboard.boards import (  # noqa: E402
    BoardSpec,
    board_object_pose,
    boards_from_config,
)
from openvrcardboard.config import AppConfig  # noqa: E402
from openvrcardboard.mathutil import (  # noqa: E402
    OPTICAL_TO_VIEW,
    angle_between,
    from_axis_angle,
    from_rotvec,
    normalize,
)
from openvrcardboard.fusion import PoseComposer  # noqa: E402
from openvrcardboard.markers import BoardTracker  # noqa: E402
from openvrcardboard.synthetic import SyntheticScene, _pose_rotation  # noqa: E402
from openvrcardboard.types import TrackingSnapshot  # noqa: E402

cv2 = None

# Benchmarks are run with the camera at the 2 m design range instead of the
# synthetic app's 2.6 m framing (which exists for MediaPipe body detection).
BENCHMARK_CAMERA_POSITION = (0.0, 1.20, -2.00)


@dataclass
class ErrorSeries:
    """Pose errors for one estimator over one scenario."""

    position_m: list[float] = field(default_factory=list)
    rotation_deg: list[float] = field(default_factory=list)
    detected: int = 0
    frames: int = 0
    timings_ms: list[float] = field(default_factory=list)

    def add(self, position_error: float, rotation_error: float) -> None:
        self.position_m.append(float(position_error))
        self.rotation_deg.append(float(rotation_error))

    def summary(self) -> dict:
        position = np.asarray(self.position_m, dtype=np.float64)
        rotation = np.asarray(self.rotation_deg, dtype=np.float64)
        return {
            "frames": int(self.frames),
            "detected": int(self.detected),
            "detection_rate": round(self.detected / max(1, self.frames), 4),
            "position_mm": _stats(position * 1000.0),
            "rotation_deg": _stats(rotation),
        }


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "p95": None, "max": None, "std": None}
    return {
        "mean": round(float(np.mean(values)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "max": round(float(np.max(values)), 3),
        "std": round(float(np.std(values)), 3),
    }


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _make_detector(dictionary_name: str):
    global cv2
    import cv2 as opencv

    from openvrcardboard.aruco import get_dictionary

    cv2 = opencv
    dictionary = get_dictionary(cv2, dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
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
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _detect(detector, frame) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(corner, dtype=np.float64).reshape(4, 2)
        for marker_id, corner in zip(np.asarray(ids).reshape(-1), corners)
    }


def _single_marker_pose(
    model,
    board: BoardSpec,
    marker_id: int,
    corners: np.ndarray,
    camera_position,
    camera_rotation,
):
    """Solve one marker exactly like the retired single-marker tracker."""
    marker = next(
        (item for item in board.markers if int(item.marker_id) == int(marker_id)),
        None,
    )
    if marker is None or corners is None:
        return None
    object_points = marker.object_points()
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    try:
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            model.matrix,
            model.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None
    if count <= 0 or not len(rvecs):
        return None
    best = int(np.argmin(np.asarray(errors, dtype=np.float64).reshape(-1)))
    rvec = np.asarray(rvecs[best], dtype=np.float64).reshape(3)
    tvec = np.asarray(tvecs[best], dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        model.matrix,
        model.distortion,
    )
    error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        np.asarray(projected, dtype=np.float64).reshape(4, 2)
                        - np.asarray(corners, dtype=np.float64).reshape(4, 2)
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )
    if error > 4.0:
        return None
    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                model.matrix,
                model.distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
    optical_rotation = normalize(from_rotvec(rvec))
    return board_object_pose(
        board,
        tvec,
        optical_rotation,
        camera_position,
        camera_rotation,
        OPTICAL_TO_VIEW,
    )


def _make_tracker(scene, boards: dict[str, BoardSpec], config: AppConfig):
    return BoardTracker(
        scene.model,
        scene.camera_position,
        scene.camera_rotation,
        boards=boards,
        dictionary=config.camera.dictionary,
        detection_width=0,
        refine_pose=config.camera.marker_refine_pose,
        max_reprojection_error=config.camera.max_reprojection_error,
        hold_time_s=config.tracking.marker_hold_s,
        lost_timeout_s=config.tracking.marker_lost_s,
        reacquire_frames=config.tracking.marker_reacquire_frames,
        process_noise=config.tracking.marker_process_noise,
        angular_process_noise=config.tracking.marker_angular_noise,
        exposure_s=config.tracking.marker_exposure_s,
        min_markers_tracking=config.tracking.marker_min_markers,
        min_confidence=config.tracking.marker_min_confidence,
        lost_position_sigma_m=config.tracking.marker_lost_position_sigma_m,
        lost_orientation_sigma_deg=config.tracking.marker_lost_orientation_sigma_deg,
        reinitialize_gap_s=config.tracking.marker_reinitialize_gap_s,
    )


def _noisy(frame: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return frame
    noise = rng.normal(0.0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class _HiResScene(SyntheticScene):
    """Synthetic scene rendered at high resolution before area downsampling.

    ``SyntheticScene`` normally rasterises the marker bitmap directly at the
    output resolution.  For a 40 mm marker that is only ~34 px wide at 2 m,
    the resampling bias is a noticeable fraction of a pixel, and because the
    80 mm centre and the 40 mm satellites are downsampled by different ratios
    the bias differs per marker.  A rigid board solve then absorbs that
    inconsistency as a small tilt.  Rendering at 4x and downsampling with
    ``INTER_AREA`` removes most of the artifact, so the benchmark measures the
    estimator instead of the rasteriser.
    """

    def _marker_bitmap(self, marker_id: int) -> np.ndarray:
        marker_id = int(marker_id)
        cached = self._marker_bitmaps.get(marker_id)
        if cached is not None:
            return cached
        cv2 = self._cv()
        generator = getattr(cv2.aruco, "generateImageMarker", None)
        if generator is None:  # OpenCV < 4.7
            image = cv2.aruco.drawMarker(self._dictionary, marker_id, 960)
        else:
            image = generator(self._dictionary, marker_id, 960)
        self._marker_bitmaps[marker_id] = image
        return image


def _make_scene(cls, width, height, boards, dictionary, poses=None):
    if poses is None:
        return cls(
            width,
            height,
            horizontal_fov_deg=60.0,
            dictionary=dictionary,
            boards=boards,
        )

    class FixedScene(cls):
        def object_poses(self, elapsed):
            return poses

    return FixedScene(
        width,
        height,
        horizontal_fov_deg=60.0,
        dictionary=dictionary,
        boards=boards,
    )


def _scene_pair(
    width,
    height,
    boards,
    dictionary,
    poses=None,
    supersample=4,
    camera_position=None,
):
    scene = _make_scene(SyntheticScene, width, height, boards, dictionary, poses)
    if supersample <= 1:
        if camera_position is not None:
            scene.camera_position = np.asarray(camera_position, dtype=np.float64)
        return scene, scene
    hi_scene = _make_scene(
        _HiResScene,
        width * supersample,
        height * supersample,
        boards,
        dictionary,
        poses,
    )
    if camera_position is not None:
        camera = np.asarray(camera_position, dtype=np.float64)
        scene.camera_position = camera
        hi_scene.camera_position = camera
    return scene, hi_scene


def _render_frame(scene, hi_scene, elapsed, supersample) -> np.ndarray:
    frame = hi_scene.render(elapsed)
    if hi_scene is scene or supersample <= 1:
        return frame
    return cv2.resize(
        frame,
        (scene.width, scene.height),
        interpolation=cv2.INTER_AREA,
    )


def _paint_marker(frame, scene, board, object_position, object_rotation, marker_ids):
    """Cover selected markers with the background colour."""
    for marker_id in marker_ids:
        marker = next(
            (item for item in board.markers if int(item.marker_id) == int(marker_id)),
            None,
        )
        if marker is None:
            continue
        corners = scene._project_board_points(
            board,
            object_position,
            object_rotation,
            marker.object_points(),
        )
        if corners is None:
            continue
        centre = np.mean(corners, axis=0)
        spread = corners - centre
        # Grow the patch so the marker's black border disappears too.
        patch = np.int32(centre + spread * 1.35)
        cv2.fillConvexPoly(frame, patch, (20, 20, 26), cv2.LINE_AA)


def _paint_board(frame, scene, board, object_position, object_rotation):
    """Cover the whole board, including its white background."""
    outline = scene._project_board_points(
        board,
        object_position,
        object_rotation,
        board.outer_points,
    )
    if outline is None:
        return
    centre = np.mean(outline, axis=0)
    patch = np.int32(centre + (outline - centre) * 1.08)
    cv2.fillConvexPoly(frame, patch, (20, 20, 26), cv2.LINE_AA)


def _pose_error(position, rotation, expected_position, expected_rotation):
    return (
        float(np.linalg.norm(np.asarray(position) - np.asarray(expected_position))),
        math.degrees(angle_between(rotation, expected_rotation)),
    )


def static_benchmark(
    width,
    height,
    frames,
    fps,
    sigma,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
    supersample: int = 4,
) -> dict:
    base_scene = SyntheticScene(
        width, height, horizontal_fov_deg=60.0,
        dictionary=config.camera.dictionary, boards=boards,
    )
    poses = base_scene.object_poses(0.0)
    scene, hi_scene = _scene_pair(
        width,
        height,
        boards,
        config.camera.dictionary,
        poses,
        supersample,
        BENCHMARK_CAMERA_POSITION,
    )
    tracker = _make_tracker(scene, boards, config)
    detector = _make_detector(config.camera.dictionary)
    rng = np.random.default_rng(seed)
    baseline = {name: ErrorSeries() for name in boards}
    raw = {name: ErrorSeries() for name in boards}
    filtered = {name: ErrorSeries() for name in boards}
    predicted = {name: ErrorSeries() for name in boards}
    baseline_positions = {name: [] for name in boards}
    filtered_positions = {name: [] for name in boards}
    baseline_rotations = {name: [] for name in boards}
    filtered_rotations = {name: [] for name in boards}
    base = time.monotonic() - frames / max(1.0, fps)
    horizon = float(config.tracking.pose_prediction_ms) / 1000.0

    for index in range(frames):
        frame = _noisy(
            _render_frame(scene, hi_scene, index / fps, supersample), rng, sigma
        )
        stamp = base + index / fps
        started = time.perf_counter()
        detections = tracker.detect(frame, timestamp=stamp)
        tracker_ms = (time.perf_counter() - started) * 1000.0
        corners = _detect(detector, frame)
        for name, board in boards.items():
            expected_position, expected_rotation = poses[name]
            primary = board.primary_id
            baseline_series = baseline[name]
            raw_series = raw[name]
            filtered_series = filtered[name]
            predicted_series = predicted[name]
            baseline_series.frames += 1
            raw_series.frames += 1
            filtered_series.frames += 1
            predicted_series.frames += 1
            baseline_series.timings_ms.append(tracker_ms)
            raw_series.timings_ms.append(tracker_ms)
            filtered_series.timings_ms.append(tracker_ms)
            predicted_series.timings_ms.append(tracker_ms)
            marker_corners = corners.get(primary)
            if marker_corners is not None:
                solved = _single_marker_pose(
                    scene.model,
                    board,
                    primary,
                    marker_corners,
                    scene.camera_position,
                    scene.camera_rotation,
                )
                if solved is not None:
                    baseline_series.detected += 1
                    baseline_series.add(
                        *_pose_error(
                            solved[0], solved[1], expected_position, expected_rotation
                        )
                    )
                    baseline_positions[name].append(np.asarray(solved[0]))
                    baseline_rotations[name].append(np.asarray(solved[1]))
            detection = detections.get(primary)
            if detection is not None:
                raw_series.detected += 1
                raw_series.add(
                    *_pose_error(
                        detection.raw_position,
                        detection.raw_rotation,
                        expected_position,
                        expected_rotation,
                    )
                )
            filtered_pose = tracker.pose_at(primary, stamp)
            if filtered_pose is not None:
                filtered_series.detected += 1
                filtered_series.add(
                    *_pose_error(
                        filtered_pose.position,
                        filtered_pose.rotation,
                        expected_position,
                        expected_rotation,
                    )
                )
                filtered_positions[name].append(np.asarray(filtered_pose.position))
                filtered_rotations[name].append(np.asarray(filtered_pose.rotation))
            future = tracker.pose_at(primary, stamp + horizon)
            if future is not None:
                predicted_series.detected += 1
                predicted_series.add(
                    *_pose_error(
                        future.position,
                        future.rotation,
                        expected_position,
                        expected_rotation,
                    )
                )

    result = {}
    for name in boards:
        result[name] = {
            "baseline_single_marker": baseline[name].summary(),
            "board_raw": raw[name].summary(),
            "board_filtered": filtered[name].summary(),
            "board_predicted": predicted[name].summary(),
            "baseline_drift_mm": _drift(baseline_positions[name]),
            "filtered_drift_mm": _drift(filtered_positions[name]),
            "baseline_jitter_mm": _jitter(baseline_positions[name]),
            "filtered_jitter_mm": _jitter(filtered_positions[name]),
            "baseline_rotation_jitter_deg": _rotation_jitter(
                baseline_rotations[name]
            ),
            "filtered_rotation_jitter_deg": _rotation_jitter(
                filtered_rotations[name]
            ),
        }
    result["sensor_noise_sigma"] = sigma
    result["prediction_ms"] = float(config.tracking.pose_prediction_ms)
    result["width"] = width
    result["height"] = height
    result["fps"] = fps
    return result


def _drift(positions: list[np.ndarray]) -> float | None:
    if len(positions) < 8:
        return None
    values = np.asarray(positions, dtype=np.float64)
    time_index = np.arange(len(values), dtype=np.float64)
    duration = max(1.0, float(len(values)))
    slope = np.array(
        [
            np.polyfit(time_index, values[:, axis], 1)[0] * duration
            for axis in range(3)
        ]
    )
    return round(float(np.linalg.norm(slope)) * 1000.0, 3)


def _jitter(positions: list[np.ndarray]) -> float | None:
    """RMS frame-to-frame position change in millimetres."""
    if len(positions) < 8:
        return None
    values = np.asarray(positions, dtype=np.float64)
    steps = np.diff(values, axis=0)
    return round(float(np.sqrt(np.mean(np.sum(steps * steps, axis=1)))) * 1000.0, 3)


def _rotation_jitter(rotations: list[np.ndarray]) -> float | None:
    """RMS frame-to-frame orientation change in degrees."""
    if len(rotations) < 8:
        return None
    steps = [
        math.degrees(angle_between(first, second))
        for first, second in zip(rotations, rotations[1:])
    ]
    return round(float(np.sqrt(np.mean(np.square(steps)))), 3)


def dynamic_benchmark(
    width,
    height,
    frames,
    fps,
    sigma,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
    supersample: int = 4,
) -> dict:
    scene, hi_scene = _scene_pair(
        width,
        height,
        boards,
        config.camera.dictionary,
        None,
        supersample,
        BENCHMARK_CAMERA_POSITION,
    )
    tracker = _make_tracker(scene, boards, config)
    detector = _make_detector(config.camera.dictionary)
    rng = np.random.default_rng(seed)
    baseline = {name: ErrorSeries() for name in boards}
    raw = {name: ErrorSeries() for name in boards}
    filtered = {name: ErrorSeries() for name in boards}
    predicted = {name: ErrorSeries() for name in boards}
    baseline_overshoot = {name: [] for name in boards}
    filtered_overshoot = {name: [] for name in boards}
    predicted_overshoot = {name: [] for name in boards}
    base = time.monotonic() - frames / max(1.0, fps, 1.0)
    horizon = float(config.tracking.pose_prediction_ms) / 1000.0

    for index in range(frames):
        elapsed = index / fps
        frame = _noisy(
            _render_frame(scene, hi_scene, elapsed, supersample), rng, sigma
        )
        stamp = base + index / fps
        detections = tracker.detect(frame, timestamp=stamp)
        corners = _detect(detector, frame)
        ground_truth = scene.board_poses(elapsed)
        future_truth = scene.board_poses(elapsed + horizon)
        for name, board in boards.items():
            expected_position, expected_rotation = ground_truth[name]
            future_position, future_rotation = future_truth[name]
            primary = board.primary_id
            baseline_series = baseline[name]
            raw_series = raw[name]
            filtered_series = filtered[name]
            predicted_series = predicted[name]
            baseline_series.frames += 1
            raw_series.frames += 1
            filtered_series.frames += 1
            predicted_series.frames += 1
            marker_corners = corners.get(primary)
            if marker_corners is not None:
                solved = _single_marker_pose(
                    scene.model, board, primary, marker_corners,
                    scene.camera_position, scene.camera_rotation,
                )
                if solved is not None:
                    baseline_series.detected += 1
                    baseline_series.add(
                        *_pose_error(
                            solved[0], solved[1], expected_position, expected_rotation
                        )
                    )
                    _track_overshoot(
                        baseline_overshoot[name], solved[0], expected_position,
                        expected_position, future_position,
                    )
            detection = detections.get(primary)
            if detection is not None:
                raw_series.detected += 1
                raw_series.add(
                    *_pose_error(
                        detection.raw_position, detection.raw_rotation,
                        expected_position, expected_rotation,
                    )
                )
            filtered_pose = tracker.pose_at(primary, stamp)
            if filtered_pose is not None:
                filtered_series.detected += 1
                filtered_series.add(
                    *_pose_error(
                        filtered_pose.position, filtered_pose.rotation,
                        expected_position, expected_rotation,
                    )
                )
                _track_overshoot(
                    filtered_overshoot[name], filtered_pose.position, expected_position,
                    expected_position, future_position,
                )
            future_pose = tracker.pose_at(primary, stamp + horizon)
            if future_pose is not None:
                predicted_series.detected += 1
                predicted_series.add(
                    *_pose_error(
                        future_pose.position, future_pose.rotation,
                        future_position, future_rotation,
                    )
                )
                _track_overshoot(
                    predicted_overshoot[name], future_pose.position, future_position,
                    expected_position, future_position,
                )

    result = {}
    for name in boards:
        result[name] = {
            "baseline_single_marker": baseline[name].summary(),
            "board_raw": raw[name].summary(),
            "board_filtered": filtered[name].summary(),
            "board_predicted": predicted[name].summary(),
            "baseline_overshoot_mm": _max_or_none(baseline_overshoot[name]),
            "filtered_overshoot_mm": _max_or_none(filtered_overshoot[name]),
            "predicted_overshoot_mm": _max_or_none(predicted_overshoot[name]),
        }
    result["sensor_noise_sigma"] = sigma
    result["prediction_ms"] = float(config.tracking.pose_prediction_ms)
    result["width"] = width
    result["height"] = height
    result["fps"] = fps
    return result


def _track_overshoot(target, estimate, current_truth, previous_truth, future_truth):
    direction = np.asarray(future_truth) - np.asarray(previous_truth)
    speed = float(np.linalg.norm(direction))
    if speed < 0.001:
        return
    ahead = float(np.dot(np.asarray(estimate) - np.asarray(current_truth), direction / speed))
    target.append(max(0.0, ahead))


def _max_or_none(values):
    return round(max(values) * 1000.0, 3) if values else None


def detection_benchmark(
    width,
    height,
    frames,
    fps,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
    supersample: int = 4,
) -> dict:
    """Detection rate by distance and viewing angle, plus occlusion groups."""
    board = boards["hmd"]
    detector = _make_detector(config.camera.dictionary)
    rng = np.random.default_rng(seed)
    results = {}

    for distance in (0.5, 1.0, 2.0):
        for angle in (0.0, 45.0, 60.0):
            object_rotation = from_axis_angle(
                (0.0, 1.0, 0.0), math.radians(angle)
            )
            poses = {
                "hmd": (
                    np.asarray(BENCHMARK_CAMERA_POSITION)
                    + np.array([0.0, 0.0, distance]),
                    object_rotation,
                )
            }
            scene, hi_scene = _scene_pair(
                width,
                height,
                {"hmd": board},
                config.camera.dictionary,
                poses,
                supersample,
                BENCHMARK_CAMERA_POSITION,
            )
            tracker = _make_tracker(scene, {"hmd": board}, config)
            baseline_hits = 0
            board_hits = 0
            marker_hits = 0
            marker_total = 0
            for index in range(frames):
                frame = _noisy(
                    _render_frame(scene, hi_scene, index / fps, supersample),
                    rng,
                    1.5,
                )
                stamp = time.monotonic() - frames / fps + index / fps
                corners = _detect(detector, frame)
                marker_total += sum(
                    1 for marker_id in corners if marker_id in board.marker_ids
                )
                if board.primary_id in corners:
                    marker_hits += 1
                    solved = _single_marker_pose(
                        scene.model, board, board.primary_id, corners[board.primary_id],
                        scene.camera_position, scene.camera_rotation,
                    )
                    if solved is not None:
                        baseline_hits += 1
                if board.primary_id in tracker.detect(frame, timestamp=stamp):
                    board_hits += 1
            results[f"{distance:.1f}m_{angle:.0f}deg"] = {
                "central_marker_rate": round(marker_hits / max(1, frames), 4),
                "mean_markers": round(marker_total / max(1, frames), 2),
                "baseline_solve_rate": round(baseline_hits / max(1, frames), 4),
                "board_solve_rate": round(board_hits / max(1, frames), 4),
            }

    # Occlusion groups at 1 m: all markers, left pair hidden and centre only.
    distance = 1.0
    poses = {
        "hmd": (
            np.asarray(BENCHMARK_CAMERA_POSITION)
            + np.array([0.0, 0.0, distance]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
    }
    scene, hi_scene = _scene_pair(
        width,
        height,
        {"hmd": board},
        config.camera.dictionary,
        poses,
        supersample,
        BENCHMARK_CAMERA_POSITION,
    )
    for label, hidden in (
        ("all_visible", ()),
        ("left_pair_hidden", tuple(int(item.marker_id) for item in board.markers[1:3])),
        (
            "centre_only",
            tuple(int(item.marker_id) for item in board.markers[1:]),
        ),
    ):
        tracker = _make_tracker(scene, {"hmd": board}, config)
        hits = 0
        errors = []
        for index in range(frames):
            frame = _noisy(
                _render_frame(scene, hi_scene, index / fps, supersample),
                rng,
                1.5,
            )
            _paint_marker(
                frame, scene, board, poses["hmd"][0], poses["hmd"][1], hidden
            )
            stamp = time.monotonic() - frames / fps + index / fps
            detection = tracker.detect(frame, timestamp=stamp).get(board.primary_id)
            if detection is not None:
                hits += 1
                errors.append(
                    np.linalg.norm(
                        detection.raw_position - poses["hmd"][0]
                    )
                    * 1000.0
                )
        results[f"occlusion_{label}"] = {
            "detection_rate": round(hits / max(1, frames), 4),
            "position_error_mm": _stats(np.asarray(errors)),
        }
    return results


def reacquisition_benchmark(
    width,
    height,
    frames,
    fps,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
    supersample: int = 4,
    occlusion_seconds: float = 1.0,
) -> dict:
    """Occlude a moving hand board, then measure the return to TRACKING."""
    scene, hi_scene = _scene_pair(
        width,
        height,
        boards,
        config.camera.dictionary,
        None,
        supersample,
        BENCHMARK_CAMERA_POSITION,
    )
    tracker = _make_tracker(scene, boards, config)
    detector = _make_detector(config.camera.dictionary)
    board = boards["left"]
    primary = board.primary_id
    rng = np.random.default_rng(seed)
    visible_before = max(6, frames // 3)
    occlusion_frames = max(3, int(round(max(0.05, float(occlusion_seconds)) * fps)))
    visible_after = frames - visible_before - occlusion_frames
    if visible_after < 4:
        raise SystemExit("reacquisition needs more frames for this occlusion")
    last_baseline = None
    previous_pose_after = None
    baseline_max_step = 0.0
    filtered_max_step = 0.0
    error_at_restore = None
    recovery_frames = None
    base = time.monotonic() - frames / max(1.0, fps)

    for index in range(frames):
        elapsed = index / fps
        occluded = visible_before <= index < visible_before + occlusion_frames
        frame = _noisy(
            _render_frame(scene, hi_scene, elapsed, supersample), rng, 1.5
        )
        ground_truth = scene.board_poses(elapsed)
        if occluded:
            position, rotation = ground_truth["left"]
            _paint_board(
                frame,
                scene,
                board,
                position,
                rotation,
            )
        stamp = base + index / fps
        detections = tracker.detect(frame, timestamp=stamp)
        pose = tracker.pose_at(primary, stamp)
        corners = _detect(detector, frame)
        solved = None
        if primary in corners:
            solved = _single_marker_pose(
                scene.model,
                board,
                primary,
                corners[primary],
                scene.camera_position,
                scene.camera_rotation,
            )
        if index < visible_before + occlusion_frames:
            if solved is not None:
                last_baseline = np.asarray(solved[0])
        else:
            if solved is not None and last_baseline is not None:
                if baseline_max_step == 0.0:
                    baseline_max_step = float(
                        np.linalg.norm(
                            np.asarray(solved[0]) - last_baseline
                        )
                        * 1000.0
                    )
            if primary in detections and pose is not None:
                if previous_pose_after is not None:
                    filtered_max_step = max(
                        filtered_max_step,
                        float(
                            np.linalg.norm(
                                np.asarray(pose.position) - previous_pose_after
                            )
                            * 1000.0
                        ),
                    )
                previous_pose_after = np.asarray(pose.position).copy()
                error = float(
                    np.linalg.norm(
                        np.asarray(pose.position) - ground_truth["left"][0]
                    )
                )
                if error_at_restore is None:
                    error_at_restore = error * 1000.0
                if (
                    recovery_frames is None
                    and pose.state.name == "TRACKING"
                    and error < 0.05
                ):
                    recovery_frames = index - (visible_before + occlusion_frames)

    return {
        "occlusion_seconds": round(occlusion_frames / fps, 3),
        "error_at_restore_mm": (
            None if error_at_restore is None else round(error_at_restore, 3)
        ),
        "recovery_frames_after_restore": recovery_frames,
        "recovery_seconds_after_restore": (
            None if recovery_frames is None else round(recovery_frames / fps, 3)
        ),
        "baseline_first_step_mm": (
            None if baseline_max_step == 0.0 else round(baseline_max_step, 3)
        ),
        "filtered_max_step_mm": round(filtered_max_step, 3),
        "note": "first_step is the output change between the last pose before "
        "the occlusion and the first pose after it; baseline snaps, the "
        "board tracker starts its blend from the predicted pose",
    }


def composed_static_benchmark(
    width,
    height,
    frames,
    fps,
    noise,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
    supersample: int = 4,
) -> dict:
    """Stationary HMD jitter after the full tracker -> composer path.

    The static benchmark measures the board filter directly.  This scenario
    also runs :class:`PoseComposer`, which is what actually reaches the driver,
    so a regression in source selection or output smoothing is visible.
    """
    board = boards["hmd"]
    position = np.array([0.0, 1.42, 0.0], dtype=np.float64)
    rotation = _pose_rotation(0.18, 0.05, -0.08)
    poses = {"hmd": (position, rotation)}
    scene, hi_scene = _scene_pair(
        width, height, {"hmd": board}, config.camera.dictionary,
        poses, supersample, BENCHMARK_CAMERA_POSITION,
    )
    tracker = _make_tracker(scene, {"hmd": board}, config)
    composer = PoseComposer(config.tracking, 0)
    rng = np.random.default_rng(seed)
    primary = board.primary_id
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    errors: list[float] = []
    base = time.monotonic() - frames / max(1.0, fps)
    warmup = min(10, frames // 3)

    for index in range(frames):
        frame = _noisy(
            _render_frame(scene, hi_scene, 0.0, supersample), rng, noise
        )
        stamp = base + index / fps
        tracker.detect(frame, timestamp=stamp)
        tracked = tracker.pose_at(primary, stamp)
        snapshot = TrackingSnapshot()
        if tracked is not None:
            snapshot.tracked_objects[primary] = tracked
        composed = composer.update(snapshot, stamp)
        hmd = composed.hmd
        if index < warmup or hmd is None or not hmd.valid:
            continue
        positions.append(np.asarray(hmd.position, dtype=np.float64).copy())
        rotations.append(np.asarray(hmd.rotation, dtype=np.float64).copy())
        errors.append(float(np.linalg.norm(hmd.position - position)) * 1000.0)

    values = np.asarray(positions, dtype=np.float64)
    mean = np.mean(values, axis=0) if values.size else np.zeros(3)
    radial = np.linalg.norm(values - mean, axis=1) if values.size else np.zeros(0)
    error_values = np.asarray(errors, dtype=np.float64)
    return {
        "samples": int(values.shape[0]),
        "position_std_mm": (
            round(
                float(np.sqrt(np.mean(np.sum((values - mean) ** 2, axis=1))))
                * 1000.0,
                3,
            )
            if values.size else None
        ),
        "position_p95_radial_mm": (
            round(_percentile(radial, 0.95) * 1000.0, 3) if radial.size else None
        ),
        "frame_jitter_mm": _jitter(positions),
        "rotation_jitter_deg": _rotation_jitter(rotations),
        "mean_error_mm": (
            round(float(np.mean(error_values)), 3) if error_values.size else None
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "note": "static board through BoardTracker and PoseComposer; "
        "position_std_mm is RMS radial deviation from the mean",
    }


def step_latency_benchmark(
    width,
    height,
    frames,
    fps,
    noise,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
    supersample: int = 4,
    step_seconds: float = 0.35,
    step_m: float = 0.12,
) -> dict:
    """Inject one position step and measure the composed HMD response.

    The board jumps by ``step_m`` along +X at ``step_seconds``.  The metric is
    the time to reach 63% and 90% of that step in the composed output, plus
    overshoot.  This separates filter lag from detection latency because the
    renderer still produces the same number of visible markers before and
    after the jump.
    """
    board = boards["hmd"]
    before = np.array([0.0, 1.42, 0.0], dtype=np.float64)
    after = before + np.array([step_m, 0.0, 0.0])
    rotation = _pose_rotation(0.10, 0.03, 0.0)
    step_time = max(0.05, float(step_seconds))

    def stepped_poses(elapsed):
        return {"hmd": (before if elapsed < step_time else after, rotation)}

    class _StepScene(SyntheticScene):
        def object_poses(self, elapsed):
            return stepped_poses(elapsed)

    class _StepHiResScene(_HiResScene):
        def object_poses(self, elapsed):
            return stepped_poses(elapsed)

    scene = _StepScene(
        width,
        height,
        horizontal_fov_deg=60.0,
        dictionary=config.camera.dictionary,
        boards={"hmd": board},
    )
    hi_scene = (
        scene
        if supersample <= 1
        else _StepHiResScene(
            width * supersample,
            height * supersample,
            horizontal_fov_deg=60.0,
            dictionary=config.camera.dictionary,
            boards={"hmd": board},
        )
    )
    camera = np.asarray(BENCHMARK_CAMERA_POSITION, dtype=np.float64)
    scene.camera_position = camera
    hi_scene.camera_position = camera
    tracker = _make_tracker(scene, {"hmd": board}, config)
    composer = PoseComposer(config.tracking, 0)
    rng = np.random.default_rng(seed)
    primary = board.primary_id
    base = time.monotonic() - frames / max(1.0, fps)
    samples = []
    for index in range(frames):
        elapsed = index / fps
        frame = _noisy(
            _render_frame(scene, hi_scene, elapsed, supersample), rng, noise
        )
        stamp = base + elapsed
        tracker.detect(frame, timestamp=stamp)
        tracked = tracker.pose_at(primary, stamp)
        snapshot = TrackingSnapshot()
        if tracked is not None:
            snapshot.tracked_objects[primary] = tracked
        composed = composer.update(snapshot, stamp)
        if composed.hmd is not None and composed.hmd.valid:
            samples.append((elapsed, composed.hmd.position.copy()))

    step_index = int(round(step_time * fps))
    pre = [pos for time_value, pos in samples if time_value < step_time]
    if not pre or len(samples) < step_index + 2:
        return {"error": "not enough samples around the step"}
    origin = np.mean(np.asarray(pre[-10:], dtype=np.float64), axis=0)
    direction = after - before
    distance = float(np.linalg.norm(direction))
    direction /= max(1e-9, distance)

    def progress(position):
        return float(np.dot(np.asarray(position) - origin, direction) / distance)

    times = []
    progress_values = []
    for time_value, pos in samples:
        if time_value < step_time:
            continue
        times.append(time_value - step_time)
        progress_values.append(progress(pos))
    progress_array = np.asarray(progress_values, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)

    def time_to(fraction: float):
        hits = np.nonzero(progress_array >= fraction)[0]
        return None if hits.size == 0 else float(times_array[hits[0]])

    t63 = time_to(0.63)
    t90 = time_to(0.90)
    return {
        "step_m": round(distance, 4),
        "step_at_s": round(step_time, 3),
        "time_to_63pct_ms": None if t63 is None else round(t63 * 1000.0, 1),
        "time_to_90pct_ms": None if t90 is None else round(t90 * 1000.0, 1),
        "max_progress": round(float(np.max(progress_array)), 4),
        "overshoot_pct": round(
            max(0.0, float(np.max(progress_array)) - 1.0) * 100.0, 2
        ),
        "final_progress": round(float(progress_array[-1]), 4),
        "width": width,
        "height": height,
        "fps": fps,
        "note": "composed HMD position after a rigid board step; the first "
        "frame at or above the fraction sets the latency",
    }


def runtime_benchmark(
    width,
    height,
    frames,
    fps,
    seed,
    config: AppConfig,
    boards: dict[str, BoardSpec],
) -> dict:
    scene = SyntheticScene(
        width, height, horizontal_fov_deg=60.0,
        dictionary=config.camera.dictionary, boards=boards,
    )
    scene.camera_position = np.asarray(BENCHMARK_CAMERA_POSITION, dtype=np.float64)
    tracker = _make_tracker(scene, boards, config)
    rng = np.random.default_rng(seed)
    total_ms = []
    loop_ms = []
    base = time.monotonic() - frames / max(1.0, fps)
    cpu_ms = 0.0
    wall_started = time.perf_counter()
    for index in range(frames):
        loop_started = time.perf_counter()
        frame = _noisy(scene.render(index / fps), rng, 1.0)
        stamp = base + index / fps
        detect_started = time.perf_counter()
        cpu_started = time.process_time()
        tracker.detect(frame, timestamp=stamp)
        cpu_ms += (time.process_time() - cpu_started) * 1000.0
        total_ms.append((time.perf_counter() - detect_started) * 1000.0)
        loop_ms.append((time.perf_counter() - loop_started) * 1000.0)
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    detections = list(tracker._timings)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    budget_ms = 1000.0 / 30.0
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "detect_mean_ms": round(float(np.mean(total_ms)), 3),
        "detect_p95_ms": round(_percentile(total_ms, 0.95), 3),
        "detect_p99_ms": round(_percentile(total_ms, 0.99), 3),
        "detect_max_ms": round(float(np.max(total_ms)), 3),
        "frame_mean_ms": round(float(np.mean(loop_ms)), 3),
        "frame_p95_ms": round(_percentile(loop_ms, 0.95), 3),
        "frame_p99_ms": round(_percentile(loop_ms, 0.99), 3),
        "mean_marker_detection_ms": round(float(np.mean(detections)), 3),
        "p95_marker_detection_ms": round(_percentile(detections, 0.95), 3),
        "detect_fps_from_mean": round(
            1000.0 / max(0.001, float(np.mean(total_ms))), 2
        ),
        "frame_fps_from_mean": round(
            1000.0 / max(0.001, float(np.mean(loop_ms))), 2
        ),
        "detect_frames_over_33ms": int(np.sum(np.asarray(total_ms) > budget_ms)),
        "frame_frames_over_33ms": int(np.sum(np.asarray(loop_ms) > budget_ms)),
        "cpu_time_ms": round(cpu_ms, 2),
        "mean_cpu_ms": round(cpu_ms / max(1, frames), 3),
        "wall_time_ms": round(wall_ms, 2),
        "peak_rss_mb": round(peak_mb, 1),
        "detection_width": 0,
        "note": "detect_* is BoardTracker.detect(); frame_* additionally "
        "includes synthetic frame rendering, so it is a pipeline upper bound",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=2.0,
                        help="Gaussian sensor noise sigma for static/dynamic tests")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tracking-benchmark.json",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=4,
        help="render scale for accuracy scenarios (1 disables supersampling)",
    )
    parser.add_argument(
        "--runtime-frames",
        type=int,
        default=0,
        help="frames for the runtime scenario (default: half of --frames)",
    )
    parser.add_argument("--skip-reacquire", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 12:
        raise SystemExit("--frames must be at least 12")
    if args.supersample < 1:
        raise SystemExit("--supersample must be at least 1")

    config = AppConfig()
    boards = {
        name: board
        for name, board in boards_from_config(config.tracking).items()
        if name in ("hmd", "left", "right")
    }
    started = time.monotonic()
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opencv": None,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "supersample": args.supersample,
    }
    print("static ...", file=sys.stderr)
    payload["static"] = static_benchmark(
        args.width, args.height, args.frames, args.fps,
        args.noise, args.seed, config, boards, args.supersample,
    )
    print("dynamic ...", file=sys.stderr)
    payload["dynamic"] = dynamic_benchmark(
        args.width, args.height, args.frames, args.fps,
        args.noise, args.seed + 1, config, boards, args.supersample,
    )
    print("detection ...", file=sys.stderr)
    payload["detection"] = detection_benchmark(
        args.width, args.height, max(20, args.frames // 3), args.fps,
        args.seed + 2, config, boards, args.supersample,
    )
    if not args.skip_reacquire:
        print("reacquisition ...", file=sys.stderr)
        payload["reacquisition"] = {
            f"{seconds:.1f}s": reacquisition_benchmark(
                args.width,
                args.height,
                args.frames,
                args.fps,
                args.seed + 3,
                config,
                boards,
                args.supersample,
                seconds,
            )
            for seconds in (0.2, 0.5, 1.0)
        }
    print("composed static ...", file=sys.stderr)
    payload["composed_static"] = composed_static_benchmark(
        args.width, args.height, args.frames, args.fps,
        args.noise, args.seed + 5, config, boards, args.supersample,
    )
    print("step latency ...", file=sys.stderr)
    payload["step_latency"] = step_latency_benchmark(
        args.width, args.height, args.frames, args.fps,
        args.noise, args.seed + 6, config, boards, args.supersample,
    )
    print("runtime ...", file=sys.stderr)
    runtime_frames = (
        args.runtime_frames if args.runtime_frames > 0
        else max(30, args.frames // 2)
    )
    payload["runtime"] = runtime_benchmark(
        args.width, args.height, runtime_frames, args.fps,
        args.seed + 4, config, boards,
    )
    import cv2 as opencv

    payload["opencv"] = opencv.__version__
    payload["elapsed_s"] = round(time.monotonic() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
