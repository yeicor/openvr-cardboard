"""Deterministic synthetic webcam scene used by tests, demos and the E2E run.

The scene draws a moving human figure carrying three rigid five-marker boards:
one on the headset and one on each hand.  Every board is rendered through the
same pinhole model the tracker uses, including its white quiet zone and all
five ArUco markers, so a real detector sees exactly the geometry the config
describes.  Board poses are available in play space for tests and benchmarks.

The renderer is test/demo tooling. It intentionally requires OpenCV (with the
``aruco`` module) and is only selected through ``camera.source = "synthetic"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .aruco import get_dictionary
from .boards import BoardSpec, default_boards
from .calibration import CameraModel
from .config import CUSTOM_DICTIONARY_3X3
from .mathutil import (
    OPTICAL_TO_VIEW,
    conjugate,
    from_axis_angle,
    multiply,
    normalize,
    rotate,
)

SYNTHETIC_CAMERA_POSITION = (0.0, 1.02, -2.60)
SYNTHETIC_CAMERA_ROTATION = (0.0, 1.0, 0.0, 0.0)
SYNTHETIC_HORIZONTAL_FOV_DEG = 65.0
SYNTHETIC_DICTIONARY = CUSTOM_DICTIONARY_3X3
SYNTHETIC_BOARD_NAMES = ("hmd", "left", "right")
# Every animated term is an integer harmonic of this period, so the motion
# (and therefore a cached clip) loops without a visible jump.
SYNTHETIC_LOOP_S = 6.0


def _phase(elapsed: float) -> float:
    return 2.0 * math.pi * ((float(elapsed) % SYNTHETIC_LOOP_S) / SYNTHETIC_LOOP_S)


def _sin(elapsed: float, harmonic: int, offset: float = 0.0) -> float:
    return math.sin(harmonic * _phase(elapsed) + offset)


class SyntheticError(RuntimeError):
    """Raised when the synthetic scene cannot be rendered."""


def _dictionary(cv2, name: str):
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise SyntheticError("OpenCV was built without the aruco module")
    try:
        return get_dictionary(cv2, name)
    except ValueError as exc:
        raise SyntheticError(str(exc)) from exc


def _pose_rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Object orientation facing the camera, with small live rotations.

    The synthetic camera sits at negative Z looking toward positive Z, so an
    object whose forward axis is ``-Z`` faces it. The board mounting rotation
    then turns the printed face toward the camera.
    """
    yaw_quat = normalize(from_axis_angle((0.0, 1.0, 0.0), yaw))
    pitch_quat = normalize(from_axis_angle((1.0, 0.0, 0.0), pitch))
    roll_quat = normalize(from_axis_angle((0.0, 0.0, 1.0), roll))
    return multiply(yaw_quat, multiply(pitch_quat, roll_quat))


@dataclass(slots=True)
class SyntheticScene:
    """Renders the animated test scene as BGR frames."""

    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = SYNTHETIC_HORIZONTAL_FOV_DEG
    dictionary: str = SYNTHETIC_DICTIONARY
    boards: dict[str, BoardSpec] = field(default_factory=default_boards)
    camera_position: np.ndarray = field(init=False)
    camera_rotation: np.ndarray = field(init=False)
    model: CameraModel = field(init=False)
    _cv2: object = field(init=False, default=None)
    _dictionary: object = field(init=False, default=None)
    _marker_bitmaps: dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.width = max(320, int(self.width))
        self.height = max(240, int(self.height))
        self.horizontal_fov_deg = max(10.0, min(150.0, float(self.horizontal_fov_deg)))
        self.boards = {
            str(name): board for name, board in dict(self.boards or {}).items()
        }
        if not self.boards:
            self.boards = default_boards()
        self.camera_position = np.asarray(
            SYNTHETIC_CAMERA_POSITION, dtype=np.float64
        )
        self.camera_rotation = normalize(
            np.asarray(SYNTHETIC_CAMERA_ROTATION, dtype=np.float64)
        )
        self.model = CameraModel.estimate(
            self.width, self.height, self.horizontal_fov_deg
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _play_to_optical(self, point) -> np.ndarray:
        value = np.asarray(point, dtype=np.float64) - self.camera_position
        value = rotate(conjugate(self.camera_rotation), value)
        return rotate(conjugate(OPTICAL_TO_VIEW), value)

    def _project(self, point):
        optical = self._play_to_optical(point)
        if optical[2] <= 1e-4:
            return None
        model = self.model
        screen = np.array(
            [
                model.fx * optical[0] / optical[2] + model.cx,
                model.fy * optical[1] / optical[2] + model.cy,
            ],
            dtype=np.float64,
        )
        return screen, float(optical[2])

    def _play_point(
        self,
        board: BoardSpec,
        object_position: np.ndarray,
        object_rotation: np.ndarray,
        board_point,
    ) -> np.ndarray:
        """Map a board-frame point through the object pose into play space."""
        local = rotate(board.object_quaternion, np.asarray(board_point, dtype=np.float64))
        local = local + board.object_offset
        return np.asarray(object_position, dtype=np.float64) + rotate(
            object_rotation, local
        )

    def _project_board_points(
        self,
        board: BoardSpec,
        object_position: np.ndarray,
        object_rotation: np.ndarray,
        points: np.ndarray,
    ) -> Optional[np.ndarray]:
        projected = []
        for point in np.asarray(points, dtype=np.float64):
            play = self._play_point(board, object_position, object_rotation, point)
            result = self._project(play)
            if result is None:
                return None
            projected.append(result[0])
        return np.asarray(projected, dtype=np.float32)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _cv(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SyntheticError(
                "the synthetic camera needs the 'opencv-contrib-python' package"
            ) from exc
        self._cv2 = cv2
        self._dictionary = _dictionary(cv2, self.dictionary)
        return cv2

    def _marker_bitmap(self, marker_id: int) -> np.ndarray:
        marker_id = int(marker_id)
        cached = self._marker_bitmaps.get(marker_id)
        if cached is not None:
            return cached
        cv2 = self._cv()
        generator = getattr(cv2.aruco, "generateImageMarker", None)
        if generator is None:  # OpenCV < 4.7
            image = cv2.aruco.drawMarker(self._dictionary, marker_id, 240)
        else:
            image = generator(self._dictionary, marker_id, 240)
        self._marker_bitmaps[marker_id] = image
        return image

    def _draw_board(
        self,
        frame,
        board: BoardSpec,
        object_position: np.ndarray,
        object_rotation: np.ndarray,
    ) -> None:
        cv2 = self._cv()
        outline = self._project_board_points(
            board, object_position, object_rotation, board.outer_points
        )
        if outline is None:
            return
        if (
            outline[:, 0].max() < 0
            or outline[:, 0].min() > self.width
            or outline[:, 1].max() < 0
            or outline[:, 1].min() > self.height
        ):
            return
        cv2.fillConvexPoly(
            frame, outline.astype(np.int32), (248, 248, 248), cv2.LINE_AA
        )
        cv2.polylines(
            frame,
            [outline.astype(np.int32)],
            True,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        for marker in board.markers:
            corners = self._project_board_points(
                board,
                object_position,
                object_rotation,
                marker.object_points(),
            )
            if corners is None:
                continue
            source = self._marker_bitmap(marker.marker_id)
            # A 240 px marker warped straight onto a 20 px quad point-samples
            # the bitmap and aliases whole modules at unlucky sub-pixel
            # offsets.  Area-average the source down to roughly twice the
            # destination size first so the synthetic frame is band-limited
            # like a real camera image.
            projected = max(
                float(np.ptp(corners[:, 0])), float(np.ptp(corners[:, 1]))
            )
            if projected > 0.0 and projected < source.shape[1] * 0.5:
                target = max(16, int(math.ceil(projected * 2.0)))
                source = cv2.resize(
                    source, (target, target), interpolation=cv2.INTER_AREA
                )
            source_height, source_width = source.shape
            x0 = max(0, int(np.floor(corners[:, 0].min())) - 2)
            y0 = max(0, int(np.floor(corners[:, 1].min())) - 2)
            x1 = min(self.width, int(np.ceil(corners[:, 0].max())) + 2)
            y1 = min(self.height, int(np.ceil(corners[:, 1].max())) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            local_corners = corners - np.array([x0, y0], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(
                np.array(
                    [
                        [-0.5, -0.5],
                        [source_width - 0.5, -0.5],
                        [source_width - 0.5, source_height - 0.5],
                        [-0.5, source_height - 0.5],
                    ],
                    dtype=np.float32,
                ),
                local_corners.astype(np.float32),
            )
            size = (x1 - x0, y1 - y0)
            warped = cv2.warpPerspective(
                cv2.cvtColor(source, cv2.COLOR_GRAY2BGR),
                transform,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            # Blend with a fractional alpha mask instead of a binary mask:
            # nearest-neighbour clipping removes the anti-aliased half pixel at
            # the marker border, which biases every detected corner inward and
            # shows up as a false board tilt in the synthetic benchmark.
            alpha = cv2.warpPerspective(
                np.full((source_height, source_width), 255, np.uint8),
                transform,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float32)[..., None] / 255.0
            region = frame[y0:y1, x0:x1].astype(np.float32)
            frame[y0:y1, x0:x1] = np.clip(
                region * (1.0 - alpha) + warped.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Scene motion
    # ------------------------------------------------------------------
    def body_joints(self, elapsed: float) -> dict[str, np.ndarray]:
        """Return an articulated human skeleton in play space.

        The figure walks, bends its elbows and knees, and turns its torso.
        Every term is a harmonic of :data:`SYNTHETIC_LOOP_S`, so tests can
        render exactly the same frame after one loop.
        """
        t = float(elapsed)
        sway = 0.10 * _sin(t, 1)
        depth = 0.06 * _sin(t, 1, math.pi / 2.0)
        bob = 0.030 * _sin(t, 3)
        root = np.array([sway, 0.0, depth])
        yaw = 0.22 * _sin(t, 2, 0.3)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        def body_point(x: float, y: float, z: float) -> np.ndarray:
            # Rotate the upper body around its vertical axis and then move it
            # with the root sway.
            return root + np.array(
                [cos_yaw * x + sin_yaw * z, y, -sin_yaw * x + cos_yaw * z]
            )

        head = body_point(0.0, 1.42 + bob, 0.0)
        neck = body_point(0.0, 1.27 + bob, 0.0)
        left_shoulder = body_point(-0.20, 1.25 + bob, 0.02)
        right_shoulder = body_point(0.20, 1.25 + bob, 0.02)
        left_elbow = body_point(
            -0.32,
            1.06 + bob + 0.05 * _sin(t, 2),
            -0.20 + 0.06 * _sin(t, 2, math.pi),
        )
        right_elbow = body_point(
            0.32,
            1.06 + bob - 0.05 * _sin(t, 2),
            -0.20 - 0.06 * _sin(t, 2, math.pi),
        )
        left_wrist = body_point(
            -0.28 + 0.05 * _sin(t, 3),
            1.17 + bob + 0.11 * _sin(t, 2, math.pi / 2.0),
            -0.60 + 0.08 * _sin(t, 2),
        )
        right_wrist = body_point(
            0.28 - 0.05 * _sin(t, 3, 0.7),
            1.17 + bob - 0.11 * _sin(t, 2, math.pi / 2.0),
            -0.60 - 0.08 * _sin(t, 2, 0.7),
        )
        left_hip = body_point(-0.11, 0.88, 0.0)
        right_hip = body_point(0.11, 0.88, 0.0)
        left_knee = body_point(
            -0.14 - 0.03 * _sin(t, 2),
            0.47 + 0.03 * _sin(t, 1, math.pi / 2.0),
            0.03 + 0.08 * _sin(t, 2, math.pi),
        )
        right_knee = body_point(
            0.14 + 0.03 * _sin(t, 2, math.pi),
            0.47 - 0.03 * _sin(t, 1, math.pi / 2.0),
            0.03 - 0.08 * _sin(t, 2),
        )
        left_ankle = body_point(
            -0.15 - 0.12 * _sin(t, 1),
            0.08 + 0.04 * abs(_sin(t, 1)),
            0.02 + 0.05 * _sin(t, 1, math.pi / 2.0),
        )
        right_ankle = body_point(
            0.15 + 0.12 * _sin(t, 1, math.pi),
            0.08 - 0.04 * abs(_sin(t, 1)),
            0.02 - 0.05 * _sin(t, 1, math.pi / 2.0),
        )
        return {
            "head": head,
            "neck": neck,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
        }

    def object_poses(self, elapsed: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return the HMD and hand object poses in play space for one frame."""
        t = float(elapsed)
        joints = self.body_joints(t)
        torso = _pose_rotation(0.22 * _sin(t, 2, 0.3))
        head = joints["head"]
        left_wrist = joints["left_wrist"]
        right_wrist = joints["right_wrist"]

        def offset(value) -> np.ndarray:
            return rotate(torso, np.asarray(value, dtype=np.float64))

        # Keep every board inside a 65-degree 720p view at its real distance:
        # hands sit about 1.0 m from the lens and the HMD about 1.5 m, so the
        # lateral excursion has to stay below roughly half a metre.
        return {
            "hmd": (
                head + offset((0.0, 0.20, -0.06)),
                _pose_rotation(
                    0.55 * _sin(t, 1),
                    0.20 * _sin(t, 2),
                    0.24 * _sin(t, 3, 0.7),
                ),
            ),
            "left": (
                left_wrist + offset((-0.18, 0.06, -0.06)),
                _pose_rotation(
                    0.55 * _sin(t, 2),
                    0.25 * _sin(t, 3, 0.4),
                    0.75 * _sin(t, 1, 0.2),
                ),
            ),
            "right": (
                right_wrist + offset((0.18, 0.06, -0.06)),
                _pose_rotation(
                    -0.55 * _sin(t, 2, 0.7),
                    0.25 * _sin(t, 3, 1.1),
                    -0.75 * _sin(t, 1, 0.6),
                ),
            ),
        }

    def board_poses(
        self, elapsed: float
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return ``name -> (position, rotation)`` for every configured board."""
        return {
            name: pose
            for name, pose in self.object_poses(elapsed).items()
            if name in self.boards
        }

    def marker_poses(
        self, elapsed: float
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return every marker centre and orientation in play space.

        The orientation is the board orientation: every marker on a rigid board
        shares it, and the marker face normal is board ``+Z``.
        """
        poses = self.object_poses(elapsed)
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for name, board in self.boards.items():
            object_position, object_rotation = poses.get(name, (None, None))
            if object_position is None:
                continue
            marker_rotation = normalize(
                multiply(object_rotation, board.object_quaternion)
            )
            for marker in board.markers:
                local = np.array(
                    [marker.center_x_m, marker.center_y_m, 0.0], dtype=np.float64
                )
                center = self._play_point(
                    board, object_position, object_rotation, local
                )
                result[int(marker.marker_id)] = (center, marker_rotation)
        return result

    # ------------------------------------------------------------------
    # Public rendering
    # ------------------------------------------------------------------
    def render(self, elapsed: float) -> np.ndarray:
        """Return one BGR frame for ``elapsed`` seconds since scene start."""
        cv2 = self._cv()
        t = float(elapsed)
        width, height = self.width, self.height
        frame = np.zeros((height, width, 3), np.uint8)
        for y in range(height):
            value = 26 + int(30 * y / max(1, height))
            frame[y, :] = (value + 6, value + 3, value)

        cv2.line(
            frame,
            (0, int(height * 0.83)),
            (width, int(height * 0.83)),
            (75, 78, 82),
            2,
        )
        for x in range(60, width, max(80, width // 8)):
            cv2.line(
                frame,
                (x, int(height * 0.15)),
                (width - 20, int(height * 0.15)),
                (50, 54, 58),
                1,
            )

        joints = self.body_joints(t)
        head = joints["head"]
        neck = joints["neck"]
        left_shoulder = joints["left_shoulder"]
        right_shoulder = joints["right_shoulder"]
        left_elbow = joints["left_elbow"]
        right_elbow = joints["right_elbow"]
        left_wrist = joints["left_wrist"]
        right_wrist = joints["right_wrist"]
        left_hip = joints["left_hip"]
        right_hip = joints["right_hip"]
        left_knee = joints["left_knee"]
        right_knee = joints["right_knee"]
        left_ankle = joints["left_ankle"]
        right_ankle = joints["right_ankle"]

        def project(point):
            projected = self._project(point)
            return None if projected is None else projected[0]

        def segment(start, end, color, thickness):
            first = project(start)
            second = project(end)
            if first is not None and second is not None:
                cv2.line(
                    frame,
                    tuple(np.int32(first)),
                    tuple(np.int32(second)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        segment(left_hip, left_knee, (52, 52, 62), 34)
        segment(left_knee, left_ankle, (52, 52, 62), 26)
        segment(right_hip, right_knee, (52, 52, 62), 34)
        segment(right_knee, right_ankle, (52, 52, 62), 26)
        for ankle in (left_ankle, right_ankle):
            point = project(ankle)
            if point is not None:
                cv2.ellipse(
                    frame,
                    tuple(np.int32(point)),
                    (34, 15),
                    0,
                    0,
                    360,
                    (30, 30, 36),
                    -1,
                    cv2.LINE_AA,
                )
        segment(left_shoulder, left_elbow, (150, 95, 45), 26)
        segment(left_elbow, left_wrist, (200, 165, 135), 18)
        segment(right_shoulder, right_elbow, (150, 95, 45), 26)
        segment(right_elbow, right_wrist, (200, 165, 135), 18)
        for wrist in (left_wrist, right_wrist):
            point = project(wrist)
            if point is not None:
                cv2.circle(
                    frame,
                    tuple(np.int32(point)),
                    15,
                    (200, 165, 135),
                    -1,
                    cv2.LINE_AA,
                )

        lt, rt = project(left_shoulder), project(right_shoulder)
        lh, rh = project(left_hip), project(right_hip)
        if all(value is not None for value in (lt, rt, lh, rh)):
            points = np.array([lt, rt, rh, lh], np.int32)
            cv2.fillConvexPoly(frame, points, (160, 100, 50), cv2.LINE_AA)
            cv2.polylines(frame, [points], True, (90, 55, 25), 2, cv2.LINE_AA)

        segment(neck, head, (200, 165, 135), 24)
        head_pixel = project(head)
        if head_pixel is not None:
            head_depth = self._project(head)[1]
            radius = max(18, int(0.105 * self.model.fx / head_depth))
            center = tuple(np.int32(head_pixel))
            cv2.circle(frame, center, radius, (205, 170, 140), -1, cv2.LINE_AA)
            cv2.circle(frame, center, radius, (130, 100, 80), 2, cv2.LINE_AA)
            # Hair and a mouth make the test figure read as a person to the
            # MediaPipe pose model instead of a geometric toy.
            cv2.ellipse(
                frame,
                (center[0], center[1] - int(radius * 0.62)),
                (int(radius * 1.02), int(radius * 0.55)),
                0,
                180,
                360,
                (45, 35, 30),
                -1,
                cv2.LINE_AA,
            )
            eye_radius = max(2, radius // 9)
            eye_y = center[1] - radius // 8
            cv2.circle(
                frame,
                (center[0] - radius // 3, eye_y),
                eye_radius,
                (30, 30, 30),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (center[0] + radius // 3, eye_y),
                eye_radius,
                (30, 30, 30),
                -1,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                frame,
                (center[0], center[1] + int(radius * 0.42)),
                (int(radius * 0.28), int(radius * 0.16)),
                0,
                0,
                180,
                (90, 60, 180),
                -1,
                cv2.LINE_AA,
            )

        for name, (position, rotation) in self.board_poses(t).items():
            self._draw_board(frame, self.boards[name], position, rotation)

        cv2.putText(
            frame,
            "SYNTHETIC WEBCAM",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"t = {t % SYNTHETIC_LOOP_S:4.1f} s",
            (width - 170, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        return frame


class SyntheticClip:
    """Lazily pre-generated, exactly looping synthetic video.

    Rendering all of the scene geometry costs about 10 ms at 720p, which is
    too slow for a 120 Hz test source. The clip renders each frame index once,
    stores it as JPEG, and decodes the cached copy on later loops. JPEG keeps
    the memory bounded (a 6-second 120 fps loop is roughly 60 MB instead of
    several gigabytes of raw BGR).

    Deriving the frame index from ``elapsed`` also makes playback independent
    of render speed: a slow first pass cannot shift the loop phase.
    """

    def __init__(
        self,
        scene: SyntheticScene,
        fps: int,
        quality: int = 92,
        max_frames: int = 600,
    ):
        self.scene = scene
        self.fps = max(1, int(fps))
        self.quality = max(50, min(98, int(quality)))
        self.max_frames = max(0, int(max_frames))
        self.loop_frames = max(1, int(round(self.fps * SYNTHETIC_LOOP_S)))
        self._cache: dict[int, bytes] = {}
        self._cv2 = None

    @property
    def cached_frames(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    def _cv(self):
        if self._cv2 is not None:
            return self._cv2
        self._cv2 = self.scene._cv()
        return self._cv2

    def render(self, elapsed: float) -> np.ndarray:
        cv2 = self._cv()
        index = int(round(float(elapsed) * self.fps)) % self.loop_frames
        cached = self._cache.get(index)
        if cached is not None:
            image = cv2.imdecode(
                np.frombuffer(cached, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is not None:
                return image
        frame = self.scene.render(index / float(self.fps))
        if self.max_frames <= 0 or len(self._cache) < self.max_frames:
            if index not in self._cache:
                ok, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
                )
                if ok:
                    self._cache[index] = buffer.tobytes()
        return frame
