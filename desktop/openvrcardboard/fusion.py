"""Sensor fusion and pose composition.

This module turns raw tracking observations into the device poses and controller
inputs that the OpenVR driver consumes. It is deliberately free of OpenCV, Qt
and socket code so the behaviour can be unit tested headlessly.

Coordinate conventions
----------------------
* Play space is OpenVR: +X right, +Y up, -Z forward (the HMD looks down -Z).
* Marker and body poses arriving in ``TrackingSnapshot`` are already in play
  space.
* Phone quaternions arrive in the Android world frame (ENU) with the standard
  Android device axes (+X right, +Y up the screen, +Z out of the screen).
  Because OpenVR play space is also +X right / +Y up / +Z towards the viewer,
  only the world frame needs converting: ``q_play = R_x(-90°) * q_android``.
  The device axes are left untouched.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .config import TrackingConfig
from .mathutil import (
    IDENTITY,
    angle_between,
    clamp,
    conjugate,
    from_axis_angle,
    from_rotvec,
    multiply,
    normalize,
    rotate,
    slerp,
    to_rotvec,
    yaw_only,
)
from .protocol import ControllerInput, DeviceId
from .types import PhoneSample, Pose, TrackingSnapshot

# Android world frame (X east, Y north, Z up) -> OpenVR play frame
# (X right/east, Y up, Z south/backward).
ANDROID_TO_OPENVR = from_axis_angle((1.0, 0.0, 0.0), -math.pi / 2.0)

# Android Surface.ROTATION_* values -> rotation around the device screen normal
# (+Z device axis, which is the axis pointing towards the viewer).
SCREEN_ROTATIONS = {
    0: 0.0,
    1: math.pi / 2.0,
    2: math.pi,
    3: -math.pi / 2.0,
}


from .filters import (  # noqa: F401 - re-exported for existing callers
    AdaptiveQuaternionFilter,
    LandmarkSmoother,
    OneEuroFilter,
    OneEuroVec3,
    _finite_vector,
    _smoothing_factor,
)

class PoseSmoother:
    """Smooths position and orientation and derives velocities."""

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.08,
        orientation_cutoff: float = 3.0,
        orientation_beta: float = 0.5,
    ):
        self.position_filter = OneEuroVec3(min_cutoff, beta)
        self.rotation_filter = AdaptiveQuaternionFilter(orientation_cutoff, orientation_beta)
        self.reset()

    def reset(self) -> None:
        self.position_filter.reset()
        self.rotation_filter.reset()
        self._initialized = False
        self._last_time = 0.0
        self._last_position = np.zeros(3, dtype=np.float64)
        self._last_rotation = IDENTITY.copy()
        self._last_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def predict(
        self,
        timestamp: float,
        max_horizon: float = 0.08,
        max_displacement: float = 0.20,
        max_angle: float = math.radians(20.0),
    ) -> Optional[Pose]:
        """Extrapolate the last filtered pose to ``timestamp``.

        Camera tracking updates at 24-60 Hz while the driver consumes poses at
        250 Hz. Repeating the last filtered pose makes motion visibly step;
        extrapolating with the filter's (already smoothed) velocity keeps the
        intermediate frames continuous. The horizon and displacement are
        clamped so a lost or noisy measurement can never fling a device across
        the play space.
        """
        if not self._initialized:
            return None
        stamp = float(timestamp)
        if not math.isfinite(stamp):
            return None
        horizon = stamp - self._last_time
        if horizon <= 1e-4:
            return None
        horizon = min(horizon, max(0.0, float(max_horizon)))
        if horizon <= 1e-4:
            return None

        velocity = self._last_velocity.copy()
        speed = float(np.linalg.norm(velocity))
        if speed < 0.02:
            velocity[:] = 0.0
        elif speed * horizon > max_displacement:
            velocity *= max_displacement / (speed * horizon)
        position = self._last_position + velocity * horizon

        omega = self._last_angular_velocity.copy()
        angular_speed = float(np.linalg.norm(omega))
        if angular_speed < 0.05:
            omega[:] = 0.0
        elif angular_speed * horizon > max_angle:
            omega *= max_angle / (angular_speed * horizon)
        rotation = (
            normalize(multiply(from_rotvec(omega * horizon), self._last_rotation))
            if angular_speed > 0.0
            else self._last_rotation.copy()
        )
        return Pose(
            position=position,
            rotation=rotation,
            velocity=velocity,
            angular_velocity=omega,
            timestamp=stamp,
            valid=True,
            source="predicted",
        )

    def update(self, pose: Pose, timestamp: Optional[float] = None) -> Pose:
        if not pose.valid:
            return pose.copy()

        stamp = float(timestamp if timestamp is not None else pose.timestamp)
        if not math.isfinite(stamp):
            stamp = time.monotonic()

        position = _finite_vector(pose.position, 3)
        rotation = normalize(pose.rotation)

        if not self._initialized:
            self._initialized = True
            self._last_time = stamp
            self._last_position = position
            self._last_rotation = rotation
            result = pose.copy()
            result.position = position
            result.rotation = rotation
            result.velocity = _finite_vector(pose.velocity, 3)
            result.angular_velocity = _finite_vector(pose.angular_velocity, 3)
            result.timestamp = stamp
            self._last_velocity = result.velocity.copy()
            self._last_angular_velocity = result.angular_velocity.copy()
            return result

        dt = stamp - self._last_time
        if dt <= 0.0:
            # Same measurement already filtered.
            cached = pose.copy()
            cached.position = self._last_position.copy()
            cached.rotation = self._last_rotation.copy()
            cached.velocity = self._last_velocity.copy()
            cached.angular_velocity = self._last_angular_velocity.copy()
            cached.timestamp = self._last_time
            return cached

        if dt > 1.0:
            # Tracking resumed after a long gap: restart from the measurement
            # instead of dragging the old pose in.
            self.reset()
            return self.update(pose, stamp)

        filtered_position = self.position_filter.filter(position, stamp)
        filtered_rotation = self.rotation_filter.filter(rotation, stamp)

        velocity = (filtered_position - self._last_position) / dt
        local_angular = to_rotvec(multiply(conjugate(self._last_rotation), filtered_rotation)) / dt
        # The wire format carries angular velocity in the tracking frame.
        angular_velocity = rotate(filtered_rotation, local_angular)

        result = pose.copy()
        result.position = filtered_position
        result.rotation = filtered_rotation
        result.velocity = velocity
        result.angular_velocity = angular_velocity
        result.timestamp = stamp

        self._last_time = stamp
        self._last_position = filtered_position
        self._last_rotation = filtered_rotation
        self._last_velocity = velocity
        self._last_angular_velocity = angular_velocity
        return result


def screen_rotation_quaternion(rotation: int) -> np.ndarray:
    """Return the rotation around the screen normal for an Android rotation."""
    return from_axis_angle((0.0, 0.0, 1.0), SCREEN_ROTATIONS.get(int(rotation) & 0x03, 0.0))


def android_quaternion_to_play(quaternion) -> np.ndarray:
    """Convert an Android rotation-vector quaternion to OpenVR play space."""
    return normalize(multiply(ANDROID_TO_OPENVR, normalize(quaternion)))


def apply_local_offset(pose: Pose, offset) -> Pose:
    """Translate ``pose`` along its local axes (marker/hand mounting offset)."""
    if offset is None:
        return pose
    vector = _finite_vector(offset, 3)
    if float(np.linalg.norm(vector)) < 1e-12:
        return pose
    result = pose.copy()
    result.position = result.position + rotate(result.rotation, vector)
    return result


def pose_is_fresh(pose: Optional[Pose], now: float, max_age: float) -> bool:
    if pose is None or not pose.valid:
        return False
    if pose.timestamp <= 0.0:
        return True
    return (now - pose.timestamp) <= max_age


def _invalid_copy(pose: Pose) -> Pose:
    result = pose.copy()
    result.valid = False
    result.velocity = np.zeros(3, dtype=np.float64)
    result.angular_velocity = np.zeros(3, dtype=np.float64)
    return result


@dataclass(slots=True)
class ComposedFrame:
    """Device poses and inputs ready to be sent to the driver."""

    hmd: Optional[Pose] = None
    left_controller: Optional[Pose] = None
    right_controller: Optional[Pose] = None
    trackers: dict[int, Pose] = field(default_factory=dict)
    inputs: dict[int, ControllerInput] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def devices(self) -> list[tuple[int, Pose]]:
        """Return ``(device_id, pose)`` pairs for every pose in the frame."""
        result: list[tuple[int, Pose]] = []
        if self.hmd is not None:
            result.append((int(DeviceId.HMD), self.hmd))
        if self.left_controller is not None:
            result.append((int(DeviceId.LEFT), self.left_controller))
        if self.right_controller is not None:
            result.append((int(DeviceId.RIGHT), self.right_controller))
        result.extend(sorted(self.trackers.items()))
        return result

    def copy(self) -> "ComposedFrame":
        return ComposedFrame(
            hmd=self.hmd.copy() if self.hmd is not None else None,
            left_controller=(
                self.left_controller.copy() if self.left_controller is not None else None
            ),
            right_controller=(
                self.right_controller.copy() if self.right_controller is not None else None
            ),
            trackers={key: pose.copy() for key, pose in self.trackers.items()},
            inputs=dict(self.inputs),
            sources=dict(self.sources),
            warnings=list(self.warnings),
        )

    def pose_for(self, device_id: int) -> Optional[Pose]:
        if device_id == DeviceId.HMD:
            return self.hmd
        if device_id == DeviceId.LEFT:
            return self.left_controller
        if device_id == DeviceId.RIGHT:
            return self.right_controller
        return self.trackers.get(int(device_id))


class PoseComposer:
    """Merges tracking snapshots into a single frame for the driver.

    Source selection is a preference order with a graceful fallback:

    * ``auto`` HMD: marker/body position + phone orientation when both exist,
      otherwise marker, body, phone, then the fixed manual pose.
    * ``auto`` controllers: marker -> MediaPipe hands -> body -> synthetic pose.
    * Explicit sources only use that source, but still fall back to the manual
      pose (HMD) or synthetic pose (controllers) if the device has never been
      tracked. Once a real pose has been seen and is then lost, the last pose is
      emitted with ``valid=False`` so the driver reports tracking loss instead
      of snapping to a synthetic location.
    """

    def __init__(
        self,
        tracking: TrackingConfig,
        tracker_count: int = 0,
        phone_estimator: Optional["PhonePoseEstimator"] = None,
    ):
        self.tracking = tracking
        self.tracker_count = max(0, min(int(tracker_count), 16))
        # The estimator owns the phone's yaw reference.  Passing it here lets
        # hybrid HMD mode fold the marker yaw into that reference instead of
        # overwriting the phone rotation every frame.
        self.phone_estimator = phone_estimator
        self._smoothers: dict[str, PoseSmoother] = {}
        self._smoothed_cache: dict[str, tuple[float, Pose]] = {}
        self._board_smoothers: dict[
            int, tuple[OneEuroVec3, AdaptiveQuaternionFilter]
        ] = {}
        self._ever_tracked_hmd = False
        self._ever_tracked_controller = {"left": False, "right": False}
        self._last_hmd: Optional[Pose] = None
        self._last_controller: dict[str, Optional[Pose]] = {"left": None, "right": None}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._smoothers.clear()
        self._smoothed_cache.clear()
        self._ever_tracked_hmd = False
        self._ever_tracked_controller = {"left": False, "right": False}
        self._last_hmd = None
        self._last_controller = {"left": None, "right": None}

    def update(self, snapshot: TrackingSnapshot, now: Optional[float] = None) -> ComposedFrame:
        if now is None:
            now = time.monotonic()
        frame = ComposedFrame()
        frame.hmd = self._compose_hmd(snapshot, now, frame)
        frame.left_controller = self._compose_controller("left", snapshot, now, frame)
        frame.right_controller = self._compose_controller("right", snapshot, now, frame)
        self._compose_trackers(snapshot, now, frame)
        frame.inputs = self._compose_inputs(snapshot)
        return frame

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------
    @property
    def max_age(self) -> float:
        return max(0.0, float(self.tracking.pose_max_age_s))

    def _fresh_marker(self, snapshot: TrackingSnapshot, marker_id: int, now: float) -> Optional[Pose]:
        pose = snapshot.markers.get(int(marker_id))
        return pose if pose_is_fresh(pose, now, self.max_age) else None

    def _smooth_board_output(self, marker_id: int, pose: Pose) -> Pose:
        """Optionally soften the final board pose for comfort tuning.

        The board's Kalman filter is the primary smoother.  Some users still
        prefer a still softer image and accept the extra latency, so this
        second stage is explicit, off by default and never replaces the
        source-level filter tuning.
        """
        cutoff = float(self.tracking.marker_output_smoothing_hz)
        if cutoff <= 0.0 or pose.timestamp <= 0.0:
            return pose
        smoother = self._board_smoothers.get(int(marker_id))
        if smoother is None:
            smoother = (
                OneEuroVec3(cutoff, 0.0),
                AdaptiveQuaternionFilter(cutoff, 0.0),
            )
            self._board_smoothers[int(marker_id)] = smoother
        position_filter, rotation_filter = smoother
        result = pose.copy()
        result.position = position_filter.filter(pose.position, pose.timestamp)
        result.rotation = rotation_filter.filter(pose.rotation, pose.timestamp)
        return result

    def _marker_pose(
        self, snapshot: TrackingSnapshot, marker_id: int, now: float
    ) -> tuple[Optional[Pose], bool]:
        """Return ``(pose, from_board_filter)`` for one primary marker ID.

        A live board tracker has already filtered and predicted the pose to
        ``now``; reusing it avoids stacking a second smoother over the Kalman
        filter. Plain snapshots without ``tracked_objects`` keep the legacy
        smoothing path.
        """
        tracked = snapshot.tracked_objects.get(int(marker_id))
        if tracked is not None:
            if not tracked.valid:
                return None, True
            return (
                self._smooth_board_output(
                    int(marker_id),
                    Pose(
                        position=np.array(tracked.position, copy=True),
                        rotation=np.array(tracked.rotation, copy=True),
                        velocity=np.array(tracked.velocity, copy=True),
                        angular_velocity=np.array(
                            tracked.angular_velocity, copy=True
                        ),
                        timestamp=float(tracked.timestamp),
                        valid=True,
                        source="markers",
                    ),
                ),
                True,
            )
        return self._fresh_marker(snapshot, marker_id, now), False

    def _fresh_hand(self, snapshot: TrackingSnapshot, side: str, now: float) -> Optional[Pose]:
        keys = (0, int(DeviceId.LEFT)) if side == "left" else (1, int(DeviceId.RIGHT))
        for key in keys:
            pose = snapshot.hands.get(key)
            if pose_is_fresh(pose, now, self.max_age):
                return pose
        return None

    def _fresh_body(self, snapshot: TrackingSnapshot, names: Iterable[str], now: float) -> Optional[Pose]:
        for name in names:
            pose = snapshot.body.get(name)
            if pose_is_fresh(pose, now, self.max_age):
                return pose
        return None

    def _smooth(self, key: str, pose: Pose, now: Optional[float] = None) -> Pose:
        if not self._smoothing_enabled:
            return pose.copy()
        smoother = self._smoothers.get(key)
        if smoother is None:
            smoother = PoseSmoother(
                min_cutoff=self.tracking.smoothing_min_cutoff,
                beta=self.tracking.smoothing_beta,
                orientation_cutoff=self.tracking.orientation_cutoff,
                orientation_beta=max(0.0, self.tracking.smoothing_beta * 5.0),
            )
            self._smoothers[key] = smoother
        filtered = None
        if pose.timestamp > 0.0:
            cached = self._smoothed_cache.get(key)
            if cached is not None and cached[0] == pose.timestamp:
                filtered = cached[1]
        if filtered is None:
            filtered = smoother.update(pose)
            if pose.timestamp > 0.0:
                self._smoothed_cache[key] = (pose.timestamp, filtered.copy())
        # Prediction is the driver's job.  Adding a horizon here as well
        # would stack two extrapolations and make fast motion overshoot.
        return filtered.copy()

    @property
    def _smoothing_enabled(self) -> bool:
        return (
            self.tracking.smoothing_min_cutoff > 0.0
            or self.tracking.smoothing_beta > 0.0
            or self.tracking.orientation_cutoff > 0.0
        )

    def _manual_hmd(self, now: float) -> Pose:
        return Pose(
            position=_finite_vector(self.tracking.manual_hmd_position, 3, [0.0, 1.65, 0.0]),
            rotation=normalize(self.tracking.manual_hmd_rotation),
            timestamp=now,
            source="manual",
        )

    def _synthetic_controller(self, side: str, now: float) -> Pose:
        position = (
            self.tracking.controller_offset_left
            if side == "left"
            else self.tracking.controller_offset_right
        )
        return Pose(
            position=_finite_vector(position, 3),
            rotation=IDENTITY.copy(),
            timestamp=now,
            source="synthetic",
        )

    def _marker_offset(self, key: str):
        return self.tracking.marker_offsets.get(key)

    def _hand_offset(self, key: str):
        return self.tracking.hand_offsets.get(key)

    def _pose_offset(self, key: str):
        return self.tracking.pose_offsets.get(key)

    # ------------------------------------------------------------------
    # HMD
    # ------------------------------------------------------------------
    def _compose_hmd(
        self, snapshot: TrackingSnapshot, now: float, frame: ComposedFrame
    ) -> Optional[Pose]:
        mode = (self.tracking.hmd_source or "auto").lower()

        if mode == "manual":
            pose = self._manual_hmd(now)
            self._last_hmd = pose.copy()
            frame.sources["hmd"] = "manual"
            return pose

        marker = None
        marker_filtered = False
        if self.tracking.markers_enabled:
            marker, marker_filtered = self._marker_pose(
                snapshot, self.tracking.hmd_marker_id, now
            )
        body = None
        if self.tracking.body_enabled and mode in ("auto", "body"):
            body = self._fresh_body(snapshot, ("head", "hmd", "neck", "nose"), now)
        phone = snapshot.phone_pose if pose_is_fresh(snapshot.phone_pose, now, self.max_age) else None

        pose: Optional[Pose] = None
        source = ""

        if mode == "phone":
            if phone is not None:
                pose = self._place_phone(phone, now)
                source = "phone"
        elif mode == "markers":
            if marker is not None:
                pose = marker if marker_filtered else self._smooth("hmd", marker, now)
                pose = apply_local_offset(pose, self._marker_offset("hmd"))
                source = "markers"
        elif mode == "body":
            if body is not None:
                pose = self._smooth("hmd", body, now)
                source = "body"
        else:  # auto
            if marker is not None:
                pose = marker if marker_filtered else self._smooth("hmd", marker, now)
                pose = apply_local_offset(pose, self._marker_offset("hmd"))
                source = "markers"
                if phone is not None:
                    # Hybrid: webcam gives the position and the slow absolute
                    # yaw, the phone's fast IMU gives the orientation between
                    # camera frames.  The marker residual is folded into the
                    # phone reference so marker yaw is never discarded.
                    pose.rotation = self._hybrid_rotation(pose.rotation, phone, now)
                    pose.angular_velocity = phone.angular_velocity.copy()
                    pose.orientation_only = False
                    source = "markers+phone"
            elif body is not None:
                pose = self._smooth("hmd", body, now)
                source = "body"
                if phone is not None:
                    pose.rotation = self._hybrid_rotation(pose.rotation, phone, now)
                    pose.angular_velocity = phone.angular_velocity.copy()
                    pose.orientation_only = False
                    source = "body+phone"
            elif phone is not None:
                pose = self._place_phone(phone, now)
                source = "phone"
                # The webcam position was lost but the phone IMU is still
                # alive. Hold the last known position instead of snapping
                # back to the neutral standing point.
                if self._last_hmd is not None and not self._last_hmd.orientation_only:
                    pose.position = self._last_hmd.position.copy()
                    pose.orientation_only = True
                    source = "phone+held"

        if pose is not None:
            pose = apply_local_offset(pose, self._pose_offset("hmd"))
            self._ever_tracked_hmd = True
            self._last_hmd = pose.copy()
            frame.sources["hmd"] = source
            return pose

        if self._ever_tracked_hmd and self._last_hmd is not None:
            frame.sources["hmd"] = f"{self._last_hmd.source or 'last'} (lost)"
            return _invalid_copy(self._last_hmd)

        pose = self._manual_hmd(now)
        self._last_hmd = pose.copy()
        frame.sources["hmd"] = "manual"
        if mode not in ("auto", "manual"):
            frame.warnings.append(f"no {mode} HMD pose; using manual pose")
        return pose

    def _hybrid_rotation(self, marker_rotation, phone: Pose, now: float) -> np.ndarray:
        """Blend a marker yaw into the phone reference and return the result."""
        if self.phone_estimator is not None:
            blended = self.phone_estimator.align_to_marker_orientation(
                marker_rotation, now
            )
            if blended is not None:
                return blended
        return phone.rotation.copy()

    def _place_phone(self, phone: Pose, now: float) -> Pose:
        pose = phone.copy()
        if pose.orientation_only and float(np.linalg.norm(pose.position)) < 1e-9:
            pose.position = _finite_vector(
                self.tracking.manual_hmd_position, 3, [0.0, 1.65, 0.0]
            )
        pose.timestamp = now
        return pose

    # ------------------------------------------------------------------
    # Controllers
    # ------------------------------------------------------------------
    def _compose_controller(
        self,
        side: str,
        snapshot: TrackingSnapshot,
        now: float,
        frame: ComposedFrame,
    ) -> Optional[Pose]:
        mode = (self.tracking.left_source if side == "left" else self.tracking.right_source)
        mode = (mode or "auto").lower()
        marker_id = self.tracking.left_marker_id if side == "left" else self.tracking.right_marker_id

        marker = None
        marker_filtered = False
        if self.tracking.markers_enabled and mode in ("auto", "markers"):
            marker, marker_filtered = self._marker_pose(snapshot, marker_id, now)
        hand = None
        if self.tracking.hands_enabled and mode in ("auto", "hands"):
            hand = self._fresh_hand(snapshot, side, now)
        body = None
        if self.tracking.body_enabled and mode in ("auto", "body"):
            body = self._fresh_body(snapshot, (f"{side}_hand", f"{side}_wrist", f"{side}_palm"), now)

        pose: Optional[Pose] = None
        source = ""
        if marker is not None:
            pose = (
                marker
                if marker_filtered
                else self._smooth(f"{side}:markers", marker, now)
            )
            pose = apply_local_offset(pose, self._marker_offset(side))
            source = "markers"
        elif hand is not None:
            pose = self._smooth(f"{side}:hands", hand, now)
            pose = apply_local_offset(pose, self._hand_offset(side))
            source = "hands"
        elif body is not None:
            pose = self._smooth(f"{side}:body", body, now)
            source = "body"
        elif mode == "synthetic" or (mode == "auto" and not self._ever_tracked_controller[side]):
            pose = self._synthetic_controller(side, now)
            source = "synthetic"

        if pose is not None:
            pose = apply_local_offset(pose, self._pose_offset(side))
            if source != "synthetic":
                self._ever_tracked_controller[side] = True
                self._last_controller[side] = pose.copy()
            frame.sources[side] = source
            return pose

        last = self._last_controller[side]
        if last is not None:
            frame.sources[side] = f"{last.source or 'last'} (lost)"
            return _invalid_copy(last)

        # Explicit source that has never produced a pose: keep the controller
        # usable at its configured neutral location.
        pose = self._synthetic_controller(side, now)
        frame.sources[side] = "synthetic"
        return pose

    # ------------------------------------------------------------------
    # Trackers
    # ------------------------------------------------------------------
    def _compose_trackers(
        self, snapshot: TrackingSnapshot, now: float, frame: ComposedFrame
    ) -> None:
        if self.tracker_count <= 0:
            return
        roles = self.tracking.tracker_roles
        for index in range(self.tracker_count):
            device_id = int(DeviceId.TRACKER_BASE) + index
            role = roles[index] if index < len(roles) else "waist"

            # Body trackers are MediaPipe- or externally-tracked only.  Marker
            # IDs are reserved for the HMD and hands, so a tracker can never
            # consume one of the fifteen printed codes.
            pose: Optional[Pose] = None
            source = ""
            body = None
            if self.tracking.body_enabled:
                body = self._fresh_body(snapshot, self._body_names_for_role(role), now)
            if body is not None:
                pose = self._smooth(f"tracker:{device_id}:body", body, now)
                source = f"body:{role}"
            else:
                external = snapshot.tracker_poses.get(device_id)
                if pose_is_fresh(external, now, self.max_age):
                    pose = external.copy()
                    source = "tracker"

            if pose is None:
                continue
            pose = apply_local_offset(pose, self._pose_offset(role))
            pose = apply_local_offset(pose, self._pose_offset(str(device_id)))
            frame.trackers[device_id] = pose
            frame.sources[f"tracker:{device_id}"] = source

    @staticmethod
    def _body_names_for_role(role: str) -> tuple[str, ...]:
        return {
            "waist": ("hips", "pelvis", "spine_base", "waist"),
            "chest": ("chest", "spine_chest", "neck", "upper_chest"),
            "left_foot": ("left_foot", "left_ankle", "left_toe"),
            "right_foot": ("right_foot", "right_ankle", "right_toe"),
            "left_shoulder": ("left_shoulder", "left_clavicle"),
            "right_shoulder": ("right_shoulder", "right_clavicle"),
            "left_hand": ("left_hand", "left_wrist"),
            "right_hand": ("right_hand", "right_wrist"),
        }.get(role, (role,))

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _compose_inputs(self, snapshot: TrackingSnapshot) -> dict[int, ControllerInput]:
        result: dict[int, ControllerInput] = {}

        def store(item: ControllerInput) -> None:
            if item is None:
                return
            device_id = int(item.device_id)
            if device_id not in (int(DeviceId.LEFT), int(DeviceId.RIGHT)):
                return
            result[device_id] = self._sanitize_input(item)

        # Hand inputs are the fallback; real gamepads win.
        for item in snapshot.hand_inputs.values():
            store(item)
        for item in snapshot.gamepads.values():
            store(item)
        return result

    @staticmethod
    def _sanitize_input(item: ControllerInput) -> ControllerInput:
        def finite(value: float, fallback: float) -> float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return fallback
            return value if math.isfinite(value) else fallback

        return ControllerInput(
            device_id=int(item.device_id),
            buttons=item.buttons,
            trigger=clamp(finite(item.trigger, 0.0), 0.0, 1.0),
            grip=clamp(finite(item.grip, 0.0), 0.0, 1.0),
            joystick_x=clamp(finite(item.joystick_x, 0.0), -1.0, 1.0),
            joystick_y=clamp(finite(item.joystick_y, 0.0), -1.0, 1.0),
            trackpad_x=clamp(finite(item.trackpad_x, 0.0), -1.0, 1.0),
            trackpad_y=clamp(finite(item.trackpad_y, 0.0), -1.0, 1.0),
        )


class PhonePoseEstimator:
    """Turns Android IMU samples into an orientation-only HMD pose.

    The estimator maps the Android world frame to play space, applies the
    screen rotation and removes the orientation captured at the last recenter.
    Position is fixed at ``neutral_position`` because a phone IMU cannot
    observe it.  Prediction belongs to the driver, so the estimator only
    extrapolates the newest sample to the consumer timestamp instead of adding
    a second configured horizon.

    A marker board can yaw-align the phone through
    :meth:`align_to_marker_orientation`.  The correction is folded into
    ``_reference`` as a low-passed ``q_marker * conj(q_phone)`` residual, so
    the fast gyro motion stays intact while the slow absolute yaw comes from
    the camera.
    """

    def __init__(
        self,
        neutral_position=(0.0, 1.65, 0.0),
        recenter_on_start: bool = True,
        auto_recenter: bool = True,
        orientation_cutoff: float = 0.0,
        orientation_beta: float = 0.0,
        max_prediction_ms: int = 50,
        timeout_s: float = 1.0,
        marker_blend_tau_s: float = 0.45,
    ):
        self.neutral_position = _finite_vector(neutral_position, 3, [0.0, 1.65, 0.0])
        self.recenter_on_start = bool(recenter_on_start)
        self.auto_recenter = bool(auto_recenter)
        self.orientation_cutoff = max(0.0, float(orientation_cutoff))
        self.orientation_beta = max(0.0, float(orientation_beta))
        self.max_prediction_s = max(0.0, float(max_prediction_ms) / 1000.0)
        self.timeout_s = max(0.0, float(timeout_s))
        self.marker_blend_tau_s = max(0.0, float(marker_blend_tau_s))
        self._filter = (
            AdaptiveQuaternionFilter(self.orientation_cutoff, self.orientation_beta)
            if self.orientation_cutoff > 0.0 or self.orientation_beta > 0.0
            else None
        )
        self.reset()

    def reset(self) -> None:
        self._reference: Optional[np.ndarray] = None
        # Reference before the marker correction is folded in.  Keeping the
        # two separate prevents the correction from being measured against
        # itself and slowly decaying back to the un-aligned yaw.
        self._reference_base: Optional[np.ndarray] = None
        self._marker_correction = IDENTITY.copy()
        self._marker_update_time = 0.0
        self._recenter_pending = self.recenter_on_start
        self._last_sample: Optional[PhoneSample] = None
        self._last_q_screen: Optional[np.ndarray] = None
        self._base_rotation: Optional[np.ndarray] = None
        self._base_timestamp = 0.0
        self._sample_timestamp = 0.0
        self._body_angular_velocity = np.zeros(3, dtype=np.float64)
        self._last_pose: Optional[Pose] = None
        self._rate_hz = 0.0
        self._last_sample_time = 0.0
        if self._filter is not None:
            self._filter.reset()

    @property
    def last_sample(self) -> Optional[PhoneSample]:
        return self._last_sample

    def set_neutral_position(self, position) -> None:
        self.neutral_position = _finite_vector(position, 3, [0.0, 1.65, 0.0])

    def recenter(self, now: Optional[float] = None) -> None:
        """Capture the current orientation as the new forward direction."""
        if self._last_q_screen is None:
            self._recenter_pending = True
            return
        self._reference = self._last_q_screen.copy()
        self._reference_base = self._reference.copy()
        self._marker_correction = IDENTITY.copy()
        self._marker_update_time = 0.0
        self._recenter_pending = False
        stamp = time.monotonic() if now is None else float(now)
        self._base_rotation = IDENTITY.copy()
        self._base_timestamp = stamp
        self._sample_timestamp = stamp
        self._last_pose = self._build_pose(IDENTITY.copy(), stamp, valid=True)

    def align_to(self, play_rotation, now: Optional[float] = None) -> None:
        """Make the current phone orientation read as ``play_rotation``.

        This is useful when a marker has already established the play-space
        forward direction and the phone should be yaw-aligned with it.
        """
        if self._last_q_screen is None:
            return
        target = normalize(play_rotation)
        self._reference = normalize(multiply(self._last_q_screen, conjugate(target)))
        self._reference_base = self._reference.copy()
        self._marker_correction = IDENTITY.copy()
        self._marker_update_time = 0.0
        self._recenter_pending = False
        stamp = time.monotonic() if now is None else float(now)
        self._base_rotation = target
        self._base_timestamp = stamp
        self._sample_timestamp = stamp

    def align_to_marker_orientation(
        self, marker_rotation, now: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Blend a marker's absolute yaw into the phone reference.

        ``q_marker * conj(q_phone)`` is the slowly varying residual between
        the camera and the phone.  It is low-passed so camera jitter and the
        phone's own gyro integration do not fight, then folded into
        ``_reference``.  Returns the corrected phone rotation, or ``None``
        when no phone sample is available yet.
        """
        if self._last_q_screen is None or self._reference is None:
            return None
        marker = normalize(marker_rotation)
        if not np.all(np.isfinite(marker)):
            return None
        stamp = time.monotonic() if now is None else float(now)
        if not math.isfinite(stamp):
            stamp = time.monotonic()

        # Phone orientation before the marker correction.  Measuring the
        # residual against the raw orientation keeps the low-pass from
        # converging to identity and erasing the alignment.  Only yaw is
        # folded in: the phone's gravity-stable pitch and roll are more
        # trustworthy than the planar board's tilt.
        base = self._reference_base if self._reference_base is not None else self._reference
        raw_phone = normalize(multiply(conjugate(base), self._last_q_screen))
        target = yaw_only(multiply(marker, conjugate(raw_phone)))

        if self.marker_blend_tau_s <= 1e-6:
            self._marker_correction = target
        else:
            dt = stamp - self._marker_update_time
            if self._marker_update_time <= 0.0 or dt <= 0.0 or dt > 1.0:
                alpha = 1.0
            else:
                alpha = 1.0 - math.exp(-dt / self.marker_blend_tau_s)
            self._marker_correction = yaw_only(
                slerp(self._marker_correction, target, max(0.0, min(1.0, alpha)))
            )
        self._marker_update_time = stamp
        self._reference = normalize(
            multiply(base, conjugate(self._marker_correction))
        )
        rotation = normalize(multiply(conjugate(self._reference), self._last_q_screen))
        sample_stamp = (
            self._sample_timestamp
            if self._sample_timestamp > 0.0
            else self._base_timestamp
        )
        self._base_rotation = rotation.copy()
        self._base_timestamp = sample_stamp
        horizon = min(max(0.0, stamp - sample_stamp), self.max_prediction_s)
        return self._predict(rotation, self._body_angular_velocity, horizon)

    def update(self, sample: PhoneSample, now: Optional[float] = None) -> Optional[Pose]:
        """Process a new IMU sample and return the current predicted pose."""
        if now is None:
            now = time.monotonic()
        now = float(now)
        if not math.isfinite(now):
            now = time.monotonic()

        quaternion = np.asarray(sample.quaternion, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(quaternion)) or float(np.linalg.norm(quaternion)) < 1e-8:
            return self._last_pose
        quaternion = normalize(quaternion)

        q_play = android_quaternion_to_play(quaternion)
        q_screen = multiply(q_play, screen_rotation_quaternion(sample.screen_rotation))
        if self._filter is not None:
            q_screen = self._filter.filter(q_screen, now)

        self._last_sample = sample
        self._last_q_screen = q_screen

        if self._reference is None:
            if self._recenter_pending or self.recenter_on_start:
                self._reference = q_screen.copy()
                self._reference_base = self._reference.copy()
                self._recenter_pending = False
        elif self._recenter_pending:
            self._reference = q_screen.copy()
            self._reference_base = self._reference.copy()
            self._marker_correction = IDENTITY.copy()
            self._marker_update_time = 0.0
            self._recenter_pending = False

        q_relative = (
            multiply(conjugate(self._reference), q_screen)
            if self._reference is not None
            else q_screen.copy()
        )

        # Gyroscope values are in the physical device frame. After applying the
        # screen rotation the sensor axes are the screen frame, hence the
        # inverse screen rotation on the angular velocity.
        gyroscope = _finite_vector(sample.gyroscope, 3)
        screen_rotation = screen_rotation_quaternion(sample.screen_rotation)
        body_omega = rotate(conjugate(screen_rotation), gyroscope)
        if float(np.linalg.norm(body_omega)) < 1e-5 and self._base_rotation is not None:
            dt = now - self._base_timestamp
            if 0.0 < dt < 0.5:
                body_omega = to_rotvec(
                    multiply(conjugate(self._base_rotation), q_relative)
                ) / dt

        rate_stamp = float(sample.timestamp)
        if math.isfinite(rate_stamp) and rate_stamp > 0.0:
            if self._last_sample_time > 0.0:
                delta = rate_stamp - self._last_sample_time
                if 1e-4 < delta < 1.0:
                    instant = 1.0 / delta
                    self._rate_hz = (
                        instant if self._rate_hz <= 0.0 else 0.9 * self._rate_hz + 0.1 * instant
                    )
            self._last_sample_time = rate_stamp

        self._base_rotation = q_relative.copy()
        self._base_timestamp = now
        self._sample_timestamp = now
        self._body_angular_velocity = body_omega

        pose = self._build_pose(q_relative, now, valid=True)
        self._last_pose = pose
        return pose

    @property
    def valid_age_s(self) -> float:
        """Maximum sample age that still counts as live tracking.

        Three missed samples at the observed rate is the useful definition;
        ``timeout_s`` remains a hard cutoff for a phone that vanished
        completely.
        """
        if self._rate_hz > 0.5:
            return min(self.timeout_s, 3.0 / self._rate_hz)
        return min(self.timeout_s, 0.12)

    def pose(self, now: Optional[float] = None) -> Optional[Pose]:
        """Return the latest pose, extrapolated to ``now``."""
        if self._last_pose is None or self._base_rotation is None:
            return None
        if now is None:
            now = time.monotonic()
        now = float(now)
        age = max(0.0, now - self._base_timestamp)
        horizon = min(age, self.max_prediction_s)
        rotation = self._predict(self._base_rotation, self._body_angular_velocity, horizon)
        stale = age > self.valid_age_s
        if stale:
            # A frozen orientation with a live gyro vector makes the driver
            # keep rotating a pose that no longer receives samples.
            self._body_angular_velocity = np.zeros(3, dtype=np.float64)
        valid = age <= self.timeout_s and not stale
        pose = self._build_pose(rotation, now, valid=valid)
        if stale:
            pose.angular_velocity = np.zeros(3, dtype=np.float64)
        self._last_pose = pose
        return pose

    def _predict(self, rotation: np.ndarray, body_omega: np.ndarray, seconds: float) -> np.ndarray:
        if seconds <= 0.0 or float(np.linalg.norm(body_omega)) < 1e-8:
            return rotation.copy()
        return normalize(multiply(rotation, from_rotvec(body_omega * seconds)))

    def _build_pose(self, rotation: np.ndarray, timestamp: float, valid: bool) -> Pose:
        world_omega = rotate(rotation, self._body_angular_velocity)
        return Pose(
            position=self.neutral_position.copy(),
            rotation=normalize(rotation),
            velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=world_omega,
            timestamp=float(timestamp),
            valid=bool(valid),
            orientation_only=True,
            source="phone",
        )
