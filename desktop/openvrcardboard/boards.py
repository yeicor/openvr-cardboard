"""Rigid five-marker boards and board-level pose estimation.

A board is a flat, rigid plate carrying one large central ArUco marker and four
smaller peripheral markers.  All markers are measured in a single board frame
(``+X`` right, ``+Y`` up, ``+Z`` out of the printed face), so every visible
marker contributes corners to *one* PnP problem.  Solving and averaging five
independent marker poses is deliberately not supported: that approach throws
away the rigid geometry that makes a board accurate.

The default layout is 200 x 112 mm with an 80 mm central marker and 40 mm
peripheral markers at ``x = +/-70 mm`` and ``y = +/-26 mm``.  The four small
markers give a 180 mm horizontal baseline, the central marker keeps the board
usable when the periphery is occluded, and the 12 mm gap between the peripheral
pairs leaves a full 3x3-marker module of white quiet zone at the default
marker size.  Three boards carry the fifteen codes of the custom 3x3
dictionary; body trackers are MediaPipe-only and do not use markers.

OpenCV is imported lazily so configuration, UI and tests that do not need a
camera can still import this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .calibration import CameraModel
from .config import (
    DEFAULT_BOARD_CENTRAL_M,
    DEFAULT_BOARD_HEIGHT_M,
    DEFAULT_BOARD_PERIPHERAL_M,
    DEFAULT_BOARD_PERIPHERAL_X_M,
    DEFAULT_BOARD_PERIPHERAL_Y_M,
    DEFAULT_BOARD_WIDTH_M,
)
from .mathutil import angle_between, normalize, rotate

# ---------------------------------------------------------------------------
# Default physical layout
# ---------------------------------------------------------------------------
BOARD_WIDTH_M = DEFAULT_BOARD_WIDTH_M
BOARD_HEIGHT_M = DEFAULT_BOARD_HEIGHT_M
CENTRAL_MARKER_M = DEFAULT_BOARD_CENTRAL_M
PERIPHERAL_MARKER_M = DEFAULT_BOARD_PERIPHERAL_M
PERIPHERAL_X_M = DEFAULT_BOARD_PERIPHERAL_X_M
PERIPHERAL_Y_M = DEFAULT_BOARD_PERIPHERAL_Y_M


def _vec3(values: Iterable[float]) -> tuple[float, float, float]:
    items = tuple(float(value) for value in values)
    if len(items) != 3:
        raise ValueError(f"expected 3 values, got {len(items)}")
    return items[0], items[1], items[2]


def _quat4(values: Iterable[float]) -> tuple[float, float, float, float]:
    items = tuple(float(value) for value in values)
    if len(items) != 4:
        raise ValueError(f"expected 4 values, got {len(items)}")
    return items[0], items[1], items[2], items[3]


# The board is mounted on the front of a device.  Board +Z points out of the
# printed face (towards the camera); OpenVR device forward is -Z, so the board
# frame is a half turn around Y relative to the device frame.
DEFAULT_BOARD_ROTATION = (0.0, 1.0, 0.0, 0.0)

# Default IDs.  The first ID of each board is the central marker and is also
# the key used by the fusion layer for the device pose.
DEFAULT_BOARD_IDS: dict[str, tuple[int, int, int, int, int]] = {
    "hmd": (0, 1, 2, 3, 4),
    "left": (5, 6, 7, 8, 9),
    "right": (10, 11, 12, 13, 14),
}
BOARD_NAMES = ("hmd", "left", "right")


class BoardError(RuntimeError):
    """Raised when the board detector cannot be initialised."""


@dataclass(frozen=True, slots=True)
class MarkerSpec:
    """One square marker in board coordinates."""

    marker_id: int
    size_m: float
    center_x_m: float
    center_y_m: float
    role: str = "peripheral"

    @property
    def half_m(self) -> float:
        return float(self.size_m) * 0.5

    def object_points(self) -> np.ndarray:
        """Corner coordinates in the board frame (ArUco corner order)."""
        h = self.half_m
        x = float(self.center_x_m)
        y = float(self.center_y_m)
        # Top-left, top-right, bottom-right, bottom-left as seen from the
        # printed side.  +Y is up, so the top row has +h.
        return np.array(
            [
                [x - h, y + h, 0.0],
                [x + h, y + h, 0.0],
                [x + h, y - h, 0.0],
                [x - h, y - h, 0.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            "id": int(self.marker_id),
            "size_m": float(self.size_m),
            "center_m": [float(self.center_x_m), float(self.center_y_m)],
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class BoardSpec:
    """Geometry and object mounting of one rigid marker board."""

    name: str
    markers: tuple[MarkerSpec, ...]
    width_m: float = BOARD_WIDTH_M
    height_m: float = BOARD_HEIGHT_M
    # Pose of the board frame in the tracked object frame (metres, quaternion
    # x/y/z/w).  Calibrate these per device; the defaults put the board origin
    # at the object origin facing forward.
    object_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_rotation: tuple[float, float, float, float] = DEFAULT_BOARD_ROTATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip().lower())
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(
            self,
            "object_position",
            tuple(float(value) for value in self.object_position),
        )
        object.__setattr__(self, "object_rotation", normalize(self.object_rotation).tolist())

    @property
    def marker_ids(self) -> tuple[int, ...]:
        return tuple(marker.marker_id for marker in self.markers)

    @property
    def primary_id(self) -> int:
        return self.markers[0].marker_id

    @property
    def central(self) -> MarkerSpec:
        for marker in self.markers:
            if marker.role == "central":
                return marker
        return self.markers[0]

    def marker(self, marker_id: int) -> Optional[MarkerSpec]:
        for marker in self.markers:
            if marker.marker_id == int(marker_id):
                return marker
        return None

    @property
    def object_quaternion(self) -> np.ndarray:
        return normalize(self.object_rotation)

    @property
    def object_offset(self) -> np.ndarray:
        return np.asarray(self.object_position, dtype=np.float64).reshape(3)

    @property
    def outer_points(self) -> np.ndarray:
        """Four board corners in the board frame, used for debug drawing."""
        half_w = float(self.width_m) * 0.5
        half_h = float(self.height_m) * 0.5
        return np.array(
            [
                [-half_w, half_h, 0.0],
                [half_w, half_h, 0.0],
                [half_w, -half_h, 0.0],
                [-half_w, -half_h, 0.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width_m": float(self.width_m),
            "height_m": float(self.height_m),
            "object_position": [float(value) for value in self.object_position],
            "object_rotation": [float(value) for value in self.object_rotation],
            "markers": [marker.to_dict() for marker in self.markers],
        }


def default_board(name: str, ids: Optional[Iterable[int]] = None) -> BoardSpec:
    """Build one board with the default physical layout and custom IDs."""
    key = str(name).strip().lower()
    if key not in DEFAULT_BOARD_IDS:
        raise BoardError(f"unknown board {name!r}")
    values = tuple(int(value) for value in (ids if ids is not None else DEFAULT_BOARD_IDS[key]))
    if len(values) != 5:
        raise BoardError(f"board {key!r} needs exactly five marker IDs")
    if len(set(values)) != 5:
        raise BoardError(f"board {key!r} marker IDs must be unique")
    central = MarkerSpec(values[0], CENTRAL_MARKER_M, 0.0, 0.0, "central")
    peripherals = (
        MarkerSpec(values[1], PERIPHERAL_MARKER_M, -PERIPHERAL_X_M, PERIPHERAL_Y_M),
        MarkerSpec(values[2], PERIPHERAL_MARKER_M, -PERIPHERAL_X_M, -PERIPHERAL_Y_M),
        MarkerSpec(values[3], PERIPHERAL_MARKER_M, PERIPHERAL_X_M, PERIPHERAL_Y_M),
        MarkerSpec(values[4], PERIPHERAL_MARKER_M, PERIPHERAL_X_M, -PERIPHERAL_Y_M),
    )
    return BoardSpec(key, (central, *peripherals))


def default_boards(
    ids: Optional[dict[str, Iterable[int]]] = None,
    object_offsets: Optional[dict[str, Iterable[float]]] = None,
    object_rotations: Optional[dict[str, Iterable[float]]] = None,
) -> dict[str, BoardSpec]:
    """Return the three default boards, optionally with custom IDs/offsets."""
    boards: dict[str, BoardSpec] = {}
    for name in BOARDS_NAMES:
        board = default_board(name, None if ids is None else ids.get(name))
        position = (
            _vec3(object_offsets[name])
            if object_offsets and name in object_offsets
            else board.object_position
        )
        rotation = (
            _quat4(object_rotations[name])
            if object_rotations and name in object_rotations
            else board.object_rotation
        )
        boards[name] = BoardSpec(
            board.name,
            board.markers,
            board.width_m,
            board.height_m,
            position,
            rotation,
        )
    return boards


BOARDS_NAMES = BOARD_NAMES


def board_from_config(name: str, first_marker_id: int, config) -> BoardSpec:
    """Build a five-marker board from a ``BoardConfig``-like object."""
    first = int(first_marker_id)
    central = MarkerSpec(
        first, float(config.central_marker_m), 0.0, 0.0, "central"
    )
    peripheral_size = float(config.peripheral_marker_m)
    x = float(config.peripheral_x_m)
    y = float(config.peripheral_y_m)
    peripherals = (
        MarkerSpec(first + 1, peripheral_size, -x, y),
        MarkerSpec(first + 2, peripheral_size, -x, -y),
        MarkerSpec(first + 3, peripheral_size, x, y),
        MarkerSpec(first + 4, peripheral_size, x, -y),
    )
    return BoardSpec(
        name=str(name),
        markers=(central, *peripherals),
        width_m=float(config.width_m),
        height_m=float(config.height_m),
        object_position=_vec3(config.object_position),
        object_rotation=_quat4(config.object_rotation),
    )


def boards_from_config(tracking) -> dict[str, BoardSpec]:
    """Return the three device boards.

    ``tracking`` is duck-typed so this module stays independent of the config
    dataclasses: it needs ``boards.hmd/left/right`` and the three primary
    marker IDs.  Body trackers are MediaPipe-only, so no single-marker boards
    are built even when markers are enabled.
    """
    boards: dict[str, BoardSpec] = {}
    for name in BOARD_NAMES:
        board_config = getattr(tracking.boards, name)
        primary = int(getattr(tracking, f"{name}_marker_id"))
        boards[name] = board_from_config(name, primary, board_config)
    return boards


@dataclass(slots=True)
class BoardPoseEstimate:
    """Raw board pose in the optical camera frame plus quality evidence."""

    board: str
    marker_ids: tuple[int, ...]
    outlier_ids: tuple[int, ...]
    corners: dict[int, np.ndarray]
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    reprojection_error: float
    per_marker_error: dict[int, float]
    marker_count: int
    distance: float
    incidence_rad: float
    inlier_ratio: float
    board_span_px: float
    image_sigma_px: float = 1.0
    board_area_fraction: float = 0.0

    def copy(self) -> "BoardPoseEstimate":
        return BoardPoseEstimate(
            board=self.board,
            marker_ids=tuple(self.marker_ids),
            outlier_ids=tuple(self.outlier_ids),
            corners={key: value.copy() for key, value in self.corners.items()},
            rvec=self.rvec.copy(),
            tvec=self.tvec.copy(),
            rotation=self.rotation.copy(),
            reprojection_error=float(self.reprojection_error),
            per_marker_error=dict(self.per_marker_error),
            marker_count=int(self.marker_count),
            distance=float(self.distance),
            incidence_rad=float(self.incidence_rad),
            inlier_ratio=float(self.inlier_ratio),
            board_span_px=float(self.board_span_px),
            image_sigma_px=float(self.image_sigma_px),
            board_area_fraction=float(self.board_area_fraction),
        )


def _projected_points(
    cv2, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, matrix, distortion
) -> np.ndarray:
    projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def _corner_rms(projected: np.ndarray, observed: np.ndarray) -> float:
    difference = projected - observed
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def _quaternion_from_rvec(cv2, rvec: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return _quaternion_from_matrix(np.asarray(matrix, dtype=np.float64))


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    return normalize(np.array([x, y, z, w], dtype=np.float64))


class BoardPoseEstimator:
    """Solve one rigid pose from every visible marker of a board.

    The estimator uses RANSAC on the combined corner set when more than one
    marker is visible, refines the inliers with Levenberg-Marquardt, and then
    removes individual markers whose reprojection error is inconsistent with
    the known rigid geometry.  A single marker is solved with IPPE's square
    solver; the two planar solutions are disambiguated by continuity and
    reprojection error.
    """

    def __init__(
        self,
        max_reprojection_error: float = 4.0,
        refine_pose: bool = True,
        ransac_threshold_px: Optional[float] = None,
        ransac_iterations: int = 80,
    ):
        self.max_reprojection_error = max(0.1, float(max_reprojection_error))
        self.refine_pose = bool(refine_pose)
        self.ransac_threshold_px = max(
            0.5,
            float(ransac_threshold_px)
            if ransac_threshold_px is not None
            else min(6.0, self.max_reprojection_error),
        )
        self.ransac_iterations = max(10, int(ransac_iterations))
        self._cv2 = None

    def _load_cv2(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BoardError("board tracking needs the 'opencv-contrib-python' package") from exc
        if not hasattr(cv2, "aruco"):
            raise BoardError("board tracking needs OpenCV's aruco module")
        self._cv2 = cv2
        return cv2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(
        self,
        board: BoardSpec,
        corners: dict[int, np.ndarray],
        model: CameraModel,
        matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        previous_rotation: Optional[np.ndarray] = None,
        previous_time: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Optional[BoardPoseEstimate]:
        """Estimate ``board`` from ``{marker_id: corners(4,2)}``.

        ``previous_rotation`` is the last accepted board orientation in the
        optical camera frame.  It only breaks genuinely ambiguous planar
        solutions; reprojection error always remains the primary criterion.
        """
        cv2 = self._load_cv2()
        matrix = model.matrix if matrix is None else np.asarray(matrix, dtype=np.float64)
        distortion = (
            model.distortion
            if distortion is None
            else np.asarray(distortion, dtype=np.float64).reshape(-1)
        )

        known: dict[int, np.ndarray] = {}
        for marker_id, raw in corners.items():
            marker = board.marker(int(marker_id))
            if marker is None:
                continue
            points = np.asarray(raw, dtype=np.float64).reshape(4, 2)
            if not np.all(np.isfinite(points)):
                continue
            known[int(marker_id)] = points
        if not known:
            return None

        # Sort IDs for deterministic solving and debug output.
        known = dict(sorted(known.items()))
        marker_points: list[np.ndarray] = []
        for marker_id in known:
            marker = board.marker(marker_id)
            if marker is not None:
                marker_points.append(marker.object_points())
        object_points = np.concatenate(marker_points, axis=0)
        image_points = np.concatenate([known[marker_id] for marker_id in known], axis=0)
        inlier_ids = tuple(known.keys())
        solutions = self._solve(
            cv2, object_points, image_points, matrix, distortion, len(known)
        )
        if not solutions:
            return None

        solution = self._choose_solution(
            cv2,
            solutions,
            object_points,
            image_points,
            matrix,
            distortion,
            board,
            known,
            previous_rotation,
            previous_time,
            now,
        )
        if solution is None:
            return None

        # Remove markers whose individual reprojection is inconsistent with
        # the common rigid pose.  The loop is bounded by the marker count so a
        # bad detection cannot keep the solver oscillating.
        current_ids = list(inlier_ids)
        active_solution = solution
        for _ in range(len(current_ids) - 1):
            errors = self._marker_errors(
                cv2,
                active_solution,
                board,
                known,
                current_ids,
                matrix,
                distortion,
            )
            worst_id = max(errors, key=errors.get)
            worst = errors[worst_id]
            if worst <= self.max_reprojection_error:
                break
            # Keep the central marker and at least one marker when possible;
            # the caller will downgrade the tracking state if only one is left.
            if len(current_ids) <= 1:
                break
            current_ids = [value for value in current_ids if value != worst_id]
            subset_points: list[np.ndarray] = []
            for marker_id in current_ids:
                marker = board.marker(marker_id)
                if marker is not None:
                    subset_points.append(marker.object_points())
            object_subset = np.concatenate(subset_points, axis=0)
            image_subset = np.concatenate([known[marker_id] for marker_id in current_ids], axis=0)
            subset_solutions = self._solve(
                cv2, object_subset, image_subset, matrix, distortion, len(current_ids)
            )
            if not subset_solutions:
                break
            subset = self._choose_solution(
                cv2,
                subset_solutions,
                object_subset,
                image_subset,
                matrix,
                distortion,
                board,
                known,
                previous_rotation,
                previous_time,
                now,
            )
            if subset is None:
                break
            active_solution = subset

        return self._finalize(
            cv2,
            board,
            known,
            inlier_ids,
            current_ids,
            active_solution,
            matrix,
            distortion,
            model,
        )

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------
    def _solve(self, cv2, object_points, image_points, matrix, distortion, marker_count):
        """Return candidate ``(rvec, tvec)`` solutions for the point set."""
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        if marker_count == 1:
            if hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
                try:
                    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                        object_points,
                        image_points,
                        matrix,
                        distortion,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                    for index in range(int(count)):
                        candidates.append(
                            (
                                np.asarray(rvecs[index], dtype=np.float64).reshape(3),
                                np.asarray(tvecs[index], dtype=np.float64).reshape(3),
                            )
                        )
                except Exception:
                    candidates = []
            if not candidates:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    matrix,
                    distortion,
                    flags=getattr(cv2, "SOLVEPNP_ITERATIVE", 0),
                )
                if ok:
                    candidates.append((rvec.reshape(3), tvec.reshape(3)))
            return candidates

        # Multi-marker: RANSAC first.  Each marker contributes four corners;
        # with at least two markers eight correspondences are available.
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                matrix,
                distortion,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.ransac_threshold_px,
                confidence=0.995,
                flags=getattr(cv2, "SOLVEPNP_ITERATIVE", 0),
            )
            if ok and rvec is not None and tvec is not None:
                candidates.append((rvec.reshape(3), tvec.reshape(3)))
        except Exception:
            pass

        # IPPE's two planar solutions are valuable when the view is nearly
        # fronto-parallel and the RANSAC solution is noisy.
        if hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE"):
            try:
                count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                for index in range(int(count)):
                    candidates.append(
                        (
                            np.asarray(rvecs[index], dtype=np.float64).reshape(3),
                            np.asarray(tvecs[index], dtype=np.float64).reshape(3),
                        )
                    )
            except Exception:
                pass

        if not candidates:
            for flag in (
                getattr(cv2, "SOLVEPNP_IPPE", None),
                getattr(cv2, "SOLVEPNP_ITERATIVE", None),
            ):
                if flag is None:
                    continue
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        object_points, image_points, matrix, distortion, flags=flag
                    )
                except Exception:
                    continue
                if ok:
                    candidates.append((rvec.reshape(3), tvec.reshape(3)))
                    break
        return candidates

    def _choose_solution(
        self,
        cv2,
        solutions,
        object_points,
        image_points,
        matrix,
        distortion,
        board,
        corners,
        previous_rotation,
        previous_time=None,
        now=None,
    ):
        # Score each candidate on its best-supported markers instead of the
        # whole point set.  One misdetected peripheral marker must not make a
        # perfectly good RANSAC solution look invalid before the outlier loop
        # has had a chance to remove it.
        count = max(1, len(image_points) // 4)
        chunks = [slice(4 * index, 4 * index + 4) for index in range(count)]
        keep_count = count if count <= 2 else max(2, int(math.ceil(count * 0.6)))

        evaluated = []
        for rvec, tvec in solutions:
            if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                continue
            if float(tvec[2]) <= 1e-4:
                continue
            projected = _projected_points(cv2, object_points, rvec, tvec, matrix, distortion)
            errors = [
                _corner_rms(projected[chunk], image_points[chunk]) for chunk in chunks
            ]
            supported = [
                index
                for index, value in enumerate(errors)
                if math.isfinite(value) and value <= self.max_reprojection_error
            ]
            required = count if count == 1 else min(2, count)
            if len(supported) < required:
                continue
            supported = sorted(supported, key=lambda index: errors[index])[:keep_count]
            mask = np.concatenate(
                [np.arange(4 * index, 4 * index + 4) for index in supported]
            )
            subset_object = object_points[mask]
            subset_image = image_points[mask]
            error = _corner_rms(projected[mask], subset_image)
            refined = self._refine(
                cv2, subset_object, subset_image, matrix, distortion, rvec, tvec
            )
            if refined is not None:
                refined_projected = _projected_points(
                    cv2, subset_object, *refined, matrix, distortion
                )
                refined_error = _corner_rms(refined_projected, subset_image)
                if math.isfinite(refined_error) and refined_error <= error * 1.5:
                    rvec, tvec, error = refined[0], refined[1], refined_error
            evaluated.append((error, rvec, tvec))
        if not evaluated:
            return None

        best = min(evaluated, key=lambda item: item[0])
        if previous_rotation is not None:
            usable = True
            if previous_time is not None and now is not None:
                gap = float(now) - float(previous_time)
                usable = 0.0 <= gap <= 0.25
            if usable:
                scored = []
                for error, rvec, tvec in evaluated:
                    rotation = _quaternion_from_rvec(cv2, rvec)
                    angle = math.degrees(
                        float(angle_between(previous_rotation, rotation))
                    )
                    # The two planar solutions reproject almost identically, so
                    # sub-pixel noise alone can flip the board orientation by
                    # tens of degrees.  Continuity is therefore a strong
                    # tie-breaker; it still accepts a candidate up to two
                    # pixels worse, which covers fast motion and partial
                    # occlusion without following random flips.
                    score = error + 0.12 * min(angle, 90.0)
                    scored.append((score, angle, error, rvec, tvec))
                scored.sort(key=lambda item: item[0])
                score, angle, error, rvec, tvec = scored[0]
                if error <= best[0] + 2.0:
                    return rvec, tvec
        _, rvec, tvec = best
        return rvec, tvec

    def _refine(self, cv2, object_points, image_points, matrix, distortion, rvec, tvec):
        if not self.refine_pose or not hasattr(cv2, "solvePnPRefineLM"):
            return None
        try:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                matrix,
                distortion,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-5),
            )
        except Exception:
            return None
        return (
            np.asarray(refined_rvec, dtype=np.float64).reshape(3),
            np.asarray(refined_tvec, dtype=np.float64).reshape(3),
        )

    def _marker_errors(
        self, cv2, solution, board, corners, marker_ids, matrix, distortion
    ):
        rvec, tvec = solution
        errors: dict[int, float] = {}
        for marker_id in marker_ids:
            marker = board.marker(marker_id)
            projected = _projected_points(
                cv2, marker.object_points(), rvec, tvec, matrix, distortion
            )
            errors[int(marker_id)] = _corner_rms(projected, corners[int(marker_id)])
        return errors

    def _finalize(
        self,
        cv2,
        board,
        corners,
        original_ids,
        inlier_ids,
        solution,
        matrix,
        distortion,
        model,
    ):
        rvec, tvec = solution
        all_points = np.concatenate([corners[marker_id] for marker_id in inlier_ids], axis=0)
        object_points = np.concatenate(
            [board.marker(marker_id).object_points() for marker_id in inlier_ids], axis=0
        )
        projected = _projected_points(cv2, object_points, rvec, tvec, matrix, distortion)
        error = _corner_rms(projected, all_points)
        per_marker = self._marker_errors(
            cv2, solution, board, corners, inlier_ids, matrix, distortion
        )
        rotation = _quaternion_from_rvec(cv2, rvec)
        distance = float(np.linalg.norm(tvec))
        initial_ids = set(original_ids)
        outlier_ids = tuple(sorted(initial_ids - set(inlier_ids)))

        # Incidence angle between the board normal and the line of sight.
        normal = rotate(rotation, np.array([0.0, 0.0, 1.0]))
        line_of_sight = np.asarray(tvec, dtype=np.float64).reshape(3) / max(distance, 1e-6)
        incidence = math.acos(max(-1.0, min(1.0, abs(float(np.dot(normal, line_of_sight))))))
        span_px = self._span_px(all_points)
        area_fraction = self._area_fraction(all_points, model)
        image_sigma = max(0.1, float(error))
        return BoardPoseEstimate(
            board=board.name,
            marker_ids=tuple(int(value) for value in inlier_ids),
            outlier_ids=outlier_ids,
            corners={key: value.copy() for key, value in corners.items()},
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3).copy(),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3).copy(),
            rotation=normalize(rotation),
            reprojection_error=float(error),
            per_marker_error=per_marker,
            marker_count=len(inlier_ids),
            distance=distance,
            incidence_rad=float(incidence),
            inlier_ratio=float(len(inlier_ids)) / max(1, len(original_ids)),
            board_span_px=float(span_px),
            image_sigma_px=float(image_sigma),
            board_area_fraction=float(area_fraction),
        )

    @staticmethod
    def _span_px(points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        width = float(points[:, 0].max() - points[:, 0].min())
        height = float(points[:, 1].max() - points[:, 1].min())
        return math.hypot(width, height)

    @staticmethod
    def _area_fraction(points: np.ndarray, model: CameraModel) -> float:
        if points.shape[0] < 3:
            return 0.0
        try:
            import cv2

            hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
            area = float(cv2.contourArea(hull))
        except Exception:
            return 0.0
        return area / max(1.0, float(model.width * model.height))


def board_optical_rotation(
    board: BoardSpec,
    object_rotation: np.ndarray,
    camera_rotation: np.ndarray,
    optical_to_view: np.ndarray,
) -> np.ndarray:
    """Return the board orientation in the camera optical frame.

    This is the inverse of the rotation part of :func:`board_object_pose`.
    The pose estimator expects its continuity prior in the same frame as the
    PnP solution, so the estimated object pose is converted back here instead
    of comparing quaternions expressed in two different frames.
    """
    from .mathutil import conjugate, multiply

    q_camera = normalize(multiply(camera_rotation, optical_to_view))
    q_view_board = normalize(
        multiply(object_rotation, board.object_quaternion)
    )
    return normalize(multiply(conjugate(q_camera), q_view_board))


def board_object_pose(
    board: BoardSpec,
    optical_position: np.ndarray,
    optical_rotation: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    optical_to_view: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a board pose in the optical camera frame to the object frame.

    Returns ``(position, rotation)`` in play space.  The board mounting
    transform is applied here so callers never have to know whether the board
    origin coincides with the device origin.
    """
    from .mathutil import multiply

    q_camera = normalize(multiply(camera_rotation, optical_to_view))
    q_view_board = normalize(multiply(q_camera, optical_rotation))
    p_view_board = np.asarray(camera_position, dtype=np.float64).reshape(3) + rotate(
        q_camera, np.asarray(optical_position, dtype=np.float64).reshape(3)
    )
    q_board_object = board.object_quaternion
    q_object_board = np.array(
        [
            -q_board_object[0],
            -q_board_object[1],
            -q_board_object[2],
            q_board_object[3],
        ],
        dtype=np.float64,
    )
    q_view_object = normalize(multiply(q_view_board, q_object_board))
    p_view_object = p_view_board - rotate(q_view_object, board.object_offset)
    return p_view_object, q_view_object


def board_measurement_covariance(
    estimate: BoardPoseEstimate,
    model: CameraModel,
    board: BoardSpec,
    camera_rotation: np.ndarray,
    optical_to_view: np.ndarray,
    base_sigma_px: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position_covariance, orientation_covariance)`` in play space.

    The estimate is intentionally conservative: corner quantization, board
    manufacturing tolerance and calibration error add a floor that a pure
    pixel-noise model misses.  Depth uncertainty grows with ``z^2`` while
    lateral uncertainty grows with ``z``, which is the behaviour that keeps a
    Kalman filter from trusting a distant board too much.
    """
    from .mathutil import multiply, to_matrix

    z = max(0.05, float(estimate.distance))
    f = max(1.0, 0.5 * (float(model.fx) + float(model.fy)))
    sigma_px = max(float(base_sigma_px), float(estimate.image_sigma_px))
    count = max(1, int(estimate.marker_count))
    corner_count = max(4, 4 * count)
    incidence = max(0.25, math.cos(min(1.35, float(estimate.incidence_rad))))
    lateral = z * sigma_px / (f * math.sqrt(corner_count)) / incidence
    depth = (
        z
        * z
        * sigma_px
        / (f * max(0.05, float(board.width_m)) * math.sqrt(count))
        / incidence
    )
    # Calibration, corner-quantization and board-print tolerances, metres.
    # These floors are calibrated against real corner noise: a clean-looking
    # 0.5 px residual still contains correlated corner quantization, which is
    # what actually limits a small planar board at 1.5-2 m.  Treating the
    # residual as independent noise makes the filter overconfident.
    lateral = max(lateral, 0.003)
    depth = max(depth, 0.008)
    if count == 1:
        lateral *= 1.5
        depth *= 2.0
    covariance_optical = np.diag([lateral * lateral, lateral * lateral, depth * depth])

    q_camera = multiply(camera_rotation, optical_to_view)
    rotation = to_matrix(q_camera)
    position_covariance = rotation @ covariance_optical @ rotation.T

    angular = sigma_px / (f * math.sqrt(corner_count)) / incidence
    angular = max(angular, math.radians(1.7))
    if count == 1:
        angular *= 2.0
    orientation_covariance = np.eye(3) * (angular * angular)
    return position_covariance, orientation_covariance


def board_axis_points(board: BoardSpec, length_m: float = 0.03) -> np.ndarray:
    """Board-frame points for drawing X/Y/Z axes from the board origin."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [length_m, 0.0, 0.0],
            [0.0, length_m, 0.0],
            [0.0, 0.0, length_m],
        ],
        dtype=np.float64,
    )
