"""Per-object 6DOF state estimation for board tracking.

Each tracked object (HMD, left hand, right hand) owns an independent
error-state Kalman filter.  The nominal state is position, linear velocity,
orientation quaternion and world-frame angular velocity; the 12-element error
state is ``[dp, dv, dtheta, domega]`` with orientation errors applied on the
left (``q_true = Exp(dtheta) * q``), so no Euler angles are ever filtered.

Measurements arrive as a rigid board pose from :mod:`openvrcardboard.boards`.
The filter predicts with the last reliable acceleration estimate, performs a
robust measurement update, and exposes a prediction at any output timestamp.
The per-object state machine implements ``TRACKING``, ``DEGRADED``,
``PREDICTING``, ``REACQUIRING`` and ``LOST`` without sharing mutable state
between objects.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .mathutil import (
    IDENTITY,
    clamp,
    conjugate,
    from_rotvec,
    multiply,
    normalize,
    slerp,
    to_rotvec,
)

_EPS = 1e-9
LOGGER = logging.getLogger(__name__)


class TrackingState(str, Enum):
    """Per-board tracking states exposed to the UI and diagnostics."""

    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    PREDICTING = "PREDICTING"
    REACQUIRING = "REACQUIRING"
    LOST = "LOST"


def _safe_sigma(covariance: np.ndarray) -> float:
    """Largest axis standard deviation of a 3x3 covariance."""
    try:
        values = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
        return float(math.sqrt(max(0.0, float(values[-1]))))
    except Exception:
        return float("inf")


@dataclass(slots=True)
class ObjectMeasurement:
    """One validated board pose with its uncertainty and timestamps."""

    position: np.ndarray
    rotation: np.ndarray
    position_covariance: np.ndarray
    orientation_covariance: np.ndarray
    confidence: float
    capture_time: float
    detection_time: float
    pose_time: float
    reprojection_error: float = 0.0
    marker_count: int = 5
    distance: float = 0.0
    incidence_rad: float = 0.0
    image_sigma_px: float = 1.0
    board_area_fraction: float = 0.0
    marker_ids: tuple[int, ...] = ()
    raw_position: Optional[np.ndarray] = None
    raw_rotation: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.rotation = normalize(self.rotation)
        self.position_covariance = np.asarray(
            self.position_covariance, dtype=np.float64
        ).reshape(3, 3)
        self.orientation_covariance = np.asarray(
            self.orientation_covariance, dtype=np.float64
        ).reshape(3, 3)
        self.confidence = float(max(0.01, min(1.0, self.confidence)))
        self.marker_ids = tuple(int(value) for value in self.marker_ids)
        if self.raw_position is not None:
            self.raw_position = np.asarray(self.raw_position, dtype=np.float64).reshape(3)
        if self.raw_rotation is not None:
            self.raw_rotation = normalize(self.raw_rotation)

    @property
    def position_sigma(self) -> float:
        return _safe_sigma(self.position_covariance)

    @property
    def orientation_sigma(self) -> float:
        return _safe_sigma(self.orientation_covariance)


@dataclass(slots=True)
class ObjectPose:
    """Predicted/filtered object pose at a requested output timestamp."""

    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    position_sigma: float
    orientation_sigma: float
    confidence: float
    state: TrackingState
    timestamp: float
    capture_time: float
    last_detection_time: float
    measurement_confidence: float = 0.0
    reprojection_error: float = 0.0
    marker_count: int = 0
    distance: float = 0.0
    valid: bool = True

    @property
    def orientation_sigma_deg(self) -> float:
        return math.degrees(self.orientation_sigma)

    def copy(self) -> "ObjectPose":
        return ObjectPose(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            linear_acceleration=self.linear_acceleration.copy(),
            angular_acceleration=self.angular_acceleration.copy(),
            position_sigma=float(self.position_sigma),
            orientation_sigma=float(self.orientation_sigma),
            confidence=float(self.confidence),
            state=self.state,
            timestamp=float(self.timestamp),
            capture_time=float(self.capture_time),
            last_detection_time=float(self.last_detection_time),
            measurement_confidence=float(self.measurement_confidence),
            reprojection_error=float(self.reprojection_error),
            marker_count=int(self.marker_count),
            distance=float(self.distance),
            valid=bool(self.valid),
        )


class _ErrorStateEkf:
    """Small 12-state multiplicative Kalman filter.

    The implementation is deliberately compact: all matrices are at most
    12x12, so a pure NumPy solve is faster and easier to audit than a large
    filtering dependency.
    """

    def __init__(
        self,
        linear_acceleration_noise: float = 1.0,
        angular_acceleration_noise: float = 2.0,
        acceleration_tau_s: float = 0.12,
        acceleration_clamp: float = 30.0,
        angular_acceleration_clamp: float = 200.0,
        max_prediction_s: float = 1.0,
        adaptive_noise: bool = True,
        static_noise_scale: float = 0.2,
    ):
        self.linear_acceleration_noise = max(1e-3, float(linear_acceleration_noise))
        self.angular_acceleration_noise = max(1e-3, float(angular_acceleration_noise))
        self.acceleration_tau_s = max(0.01, float(acceleration_tau_s))
        self.acceleration_clamp = max(0.1, float(acceleration_clamp))
        self.angular_acceleration_clamp = max(0.1, float(angular_acceleration_clamp))
        self.max_prediction_s = max(0.0, float(max_prediction_s))
        self.adaptive_noise = bool(adaptive_noise)
        self.static_noise_scale = clamp(float(static_noise_scale), 0.0, 1.0)
        # Speeds below the first threshold count as stationary; above the
        # second the filter uses its full configured process noise.
        self.stationary_speed_m_s = 0.02
        self.moving_speed_m_s = 0.10

        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.rotation = IDENTITY.copy()
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.linear_acceleration = np.zeros(3, dtype=np.float64)
        self.angular_acceleration = np.zeros(3, dtype=np.float64)

        self.covariance = np.eye(12, dtype=np.float64)
        self.initialized = False
        self.update_count = 0
        self.rejected_count = 0
        self.last_update_time = 0.0
        self.last_measurement_time = 0.0
        self.last_confidence = 0.0
        self.last_reprojection_error = 0.0
        self.last_marker_count = 0
        self.last_distance = 0.0
        # Accepted NIS values are compared with chi-square(6), whose mean is
        # 6.  A persistent mean far above or below that indicates measurement
        # floors that are too small or too large.
        self.nis_history: deque[float] = deque(maxlen=2000)
        self.nis_total = 0
        self.nis_gate_rejections = 0

    # ------------------------------------------------------------------
    # Initialisation and resets
    # ------------------------------------------------------------------
    def reset(
        self,
        position,
        rotation,
        timestamp: float,
        position_sigma: float = 0.02,
        orientation_sigma: float = 0.08,
    ) -> None:
        self.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.rotation = normalize(rotation)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.linear_acceleration = np.zeros(3, dtype=np.float64)
        self.angular_acceleration = np.zeros(3, dtype=np.float64)
        self.covariance = np.diag(
            [
                position_sigma**2,
                position_sigma**2,
                position_sigma**2,
                0.5**2,
                0.5**2,
                0.5**2,
                orientation_sigma**2,
                orientation_sigma**2,
                orientation_sigma**2,
                0.6**2,
                0.6**2,
                0.6**2,
            ]
        )
        self.initialized = True
        self.update_count = 0
        self.rejected_count = 0
        self.last_update_time = float(timestamp)
        self.last_measurement_time = float(timestamp)
        self.nis_history.clear()
        self.nis_total = 0
        self.nis_gate_rejections = 0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _linear_noise_scale(self, speed_m_s: float) -> float:
        """Process-noise scale for the current linear speed.

        At rest the estimate is dominated by measurement noise, so a smaller
        process noise gives a much steadier pose.  Once the board really moves
        the filter returns to its configured noise so it does not lag behind
        acceleration or direction changes.
        """
        if not self.adaptive_noise or self.static_noise_scale >= 1.0:
            return 1.0
        span = max(1e-6, self.moving_speed_m_s - self.stationary_speed_m_s)
        amount = clamp(
            (float(speed_m_s) - self.stationary_speed_m_s) / span, 0.0, 1.0
        )
        amount = amount * amount * (3.0 - 2.0 * amount)
        return self.static_noise_scale + (1.0 - self.static_noise_scale) * amount

    def _propagate(self, state: dict, dt: float, acceleration_gain: float) -> None:
        if dt <= 0.0:
            return
        acceleration = self.linear_acceleration * acceleration_gain
        state["position"] = (
            state["position"]
            + state["velocity"] * dt
            + 0.5 * acceleration * dt * dt
        )
        state["velocity"] = state["velocity"] + acceleration * dt
        state["rotation"] = normalize(
            multiply(from_rotvec(state["angular_velocity"] * dt), state["rotation"])
        )

        transition = np.eye(12, dtype=np.float64)
        transition[0:3, 3:6] = np.eye(3) * dt
        transition[6:9, 9:12] = np.eye(3) * dt

        sa = self.linear_acceleration_noise * self._linear_noise_scale(
            float(np.linalg.norm(state["velocity"]))
        )
        sal = self.angular_acceleration_noise
        process = np.zeros((12, 12), dtype=np.float64)
        process[0:3, 0:3] = np.eye(3) * (sa * dt * dt * 0.5) ** 2
        process[3:6, 3:6] = np.eye(3) * (sa * dt) ** 2
        process[6:9, 6:9] = np.eye(3) * (sal * dt * dt * 0.5) ** 2
        process[9:12, 9:12] = np.eye(3) * (sal * dt) ** 2
        state["covariance"] = (
            transition @ state["covariance"] @ transition.T + process
        )
        state["covariance"] = self._stabilize_covariance(state["covariance"])

    def _copy_state(self) -> dict:
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "rotation": self.rotation.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "covariance": self.covariance.copy(),
        }

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Symmetrise and floor eigenvalues so the matrix stays positive definite.

        Joseph-form updates are symmetric in exact arithmetic, but repeated
        floating-point updates can leave tiny negative eigenvalues that later
        make a solve fail or produce a huge gain.  A 1e-12 floor is far below
        any physical uncertainty, so it only removes numerical noise.
        """
        covariance = 0.5 * (covariance + covariance.T)
        try:
            values, vectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return covariance
        if values[0] < 1e-12:
            values = np.maximum(values, 1e-12)
            covariance = (vectors * values) @ vectors.T
            covariance = 0.5 * (covariance + covariance.T)
        return covariance

    def nis_summary(self) -> dict:
        """NIS statistics for calibration against chi-square(6).

        The expected mean is 6 for a correctly scaled filter.  A mean much
        larger than 6 means the measurement covariance is too optimistic
        (floors too small); much smaller means it is too pessimistic.
        """
        values = np.asarray(self.nis_history, dtype=np.float64)
        if values.size == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "expected": 6.0,
                "gate_rejections": int(self.nis_gate_rejections),
            }
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95.0)),
            "expected": 6.0,
            "gate_rejections": int(self.nis_gate_rejections),
        }

    def _acceleration_gain(self, horizon: float) -> float:
        # Acceleration is a short-term aid.  It contributes at most half of the
        # measured value and fades quickly, which prevents long-horizon
        # overshoot after a direction change.
        decay = math.exp(-max(0.0, horizon) / 0.10)
        return 0.5 * decay

    def predict_to(self, timestamp: float) -> tuple[dict, float]:
        """Return a predicted state copy and the horizon actually used."""
        if not self.initialized:
            return self._copy_state(), 0.0
        horizon = max(0.0, float(timestamp) - self.last_update_time)
        limited = min(horizon, self.max_prediction_s) if self.max_prediction_s > 0 else horizon
        state = self._copy_state()
        self._propagate(state, limited, self._acceleration_gain(limited))
        return state, limited

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------
    def update(
        self,
        position,
        rotation,
        position_covariance,
        orientation_covariance,
        confidence: float,
        timestamp: float,
    ) -> Optional[dict]:
        """Apply one pose measurement.

        Returns a diagnostics dictionary when the measurement was accepted and
        ``None`` when it was rejected by the innovation gate.  A rejected
        measurement leaves the state unchanged.
        """
        if not self.initialized:
            self.reset(position, rotation, timestamp)
            self.last_measurement_time = float(timestamp)
            self.last_confidence = float(confidence)
            return {"accepted": True, "nis": 0.0, "correction": 0.0, "angle": 0.0}

        dt = max(0.0, float(timestamp) - self.last_update_time)
        if dt > 0.0:
            state = self._nominal_state()
            self._propagate(state, dt, self._acceleration_gain(dt))
            self.position = state["position"]
            self.velocity = state["velocity"]
            self.rotation = state["rotation"]
            self.angular_velocity = state["angular_velocity"]
            self.covariance = state["covariance"]

        measured_position = np.asarray(position, dtype=np.float64).reshape(3)
        measured_rotation = normalize(rotation)
        residual_position = measured_position - self.position
        residual_rotation = to_rotvec(multiply(measured_rotation, conjugate(self.rotation)))

        position_covariance = np.asarray(position_covariance, dtype=np.float64).reshape(3, 3)
        orientation_covariance = np.asarray(
            orientation_covariance, dtype=np.float64
        ).reshape(3, 3)
        # A less reliable detection gets a larger measurement covariance.
        scale = 1.0 / max(0.05, min(1.0, float(confidence)))
        measurement_covariance = np.zeros((6, 6), dtype=np.float64)
        measurement_covariance[0:3, 0:3] = position_covariance * scale
        measurement_covariance[3:6, 3:6] = orientation_covariance * scale

        observation = np.zeros((6, 12), dtype=np.float64)
        observation[0:3, 0:3] = np.eye(3)
        observation[3:6, 6:9] = np.eye(3)

        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        residual = np.concatenate([residual_position, residual_rotation])
        try:
            solved = np.linalg.solve(innovation_covariance, residual)
            nis = float(residual @ solved)
        except np.linalg.LinAlgError:
            nis = float("inf")

        gate = 25.0  # chi-square 6 DOF, ~99.97%
        if not math.isfinite(nis):
            self.rejected_count += 1
            self.nis_gate_rejections += 1
            return None
        if nis > gate:
            # Down-weight instead of immediately discarding: one bad pixel
            # cluster should not freeze tracking, but two consecutive gross
            # outliers trigger the reacquisition path in TrackedObject.
            self.nis_gate_rejections += 1
            inflation = min(64.0, nis / gate)
            measurement_covariance *= inflation
            innovation_covariance = (
                observation @ self.covariance @ observation.T + measurement_covariance
            )
            try:
                solved = np.linalg.solve(innovation_covariance, residual)
                nis = float(residual @ solved)
            except np.linalg.LinAlgError:
                self.rejected_count += 1
                return None
            if nis > gate * 2.0:
                self.rejected_count += 1
                return None

        try:
            # Kalman gain K = P H^T S^-1.  Solve S X = H P and transpose
            # instead of forming S^-1, which is slower and less stable.
            gain = np.linalg.solve(
                innovation_covariance, observation @ self.covariance
            ).T
        except np.linalg.LinAlgError:
            self.rejected_count += 1
            return None
        correction = gain @ residual
        if not np.all(np.isfinite(correction)):
            self.rejected_count += 1
            return None

        previous_velocity = self.velocity.copy()
        previous_angular_velocity = self.angular_velocity.copy()
        self.position = self.position + correction[0:3]
        self.velocity = self.velocity + correction[3:6]
        self.rotation = normalize(
            multiply(from_rotvec(correction[6:9]), self.rotation)
        )
        self.angular_velocity = self.angular_velocity + correction[9:12]

        identity = np.eye(12, dtype=np.float64)
        residual_gain = identity - gain @ observation
        self.covariance = (
            residual_gain @ self.covariance @ residual_gain.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = self._stabilize_covariance(self.covariance)

        # Acceleration is derived from the velocity correction, then smoothed
        # with a time-aware first-order filter.  A genuine direction change
        # produces a large correction in one frame; the smoothing prevents
        # that single frame from driving the prediction.
        if dt > 1e-3:
            measured_acceleration = (self.velocity - previous_velocity) / dt
            measured_angular_acceleration = (
                self.angular_velocity - previous_angular_velocity
            ) / dt
            blend = 1.0 - math.exp(-dt / self.acceleration_tau_s)
            self.linear_acceleration += blend * (
                measured_acceleration - self.linear_acceleration
            )
            self.angular_acceleration += blend * (
                measured_angular_acceleration - self.angular_acceleration
            )
            speed = float(np.linalg.norm(self.linear_acceleration))
            if speed > self.acceleration_clamp:
                self.linear_acceleration *= self.acceleration_clamp / speed
            angular_speed = float(np.linalg.norm(self.angular_acceleration))
            if angular_speed > self.angular_acceleration_clamp:
                self.angular_acceleration *= (
                    self.angular_acceleration_clamp / angular_speed
                )

        self.last_update_time = float(timestamp)
        self.last_measurement_time = float(timestamp)
        self.last_confidence = float(confidence)
        self.update_count += 1
        self.rejected_count = 0
        self.nis_history.append(float(nis))
        if self.update_count % 250 == 0:
            summary = self.nis_summary()
            LOGGER.debug(
                "NIS mean %.2f median %.2f p95 %.2f (expected 6.0, n=%d, "
                "gate rejections %d)",
                summary["mean"],
                summary["median"],
                summary["p95"],
                summary["count"],
                summary["gate_rejections"],
            )
        return {
            "accepted": True,
            "nis": float(nis),
            "correction": float(np.linalg.norm(residual_position)),
            "angle": float(np.linalg.norm(residual_rotation)),
        }

    def _nominal_state(self) -> dict:
        return {
            "position": self.position,
            "velocity": self.velocity,
            "rotation": self.rotation,
            "angular_velocity": self.angular_velocity,
            "covariance": self.covariance,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def position_sigma(self) -> float:
        return _safe_sigma(self.covariance[0:3, 0:3])

    @property
    def orientation_sigma(self) -> float:
        return _safe_sigma(self.covariance[6:9, 6:9])


@dataclass(slots=True)
class _Blend:
    start_position: np.ndarray
    start_rotation: np.ndarray
    started_at: float
    duration: float

    def done(self, now: float) -> bool:
        return now >= self.started_at + self.duration


class TrackedObject:
    """Independent state machine and filter for one board-tracked object."""

    def __init__(
        self,
        name: str,
        primary_marker_id: int,
        *,
        process_noise: float = 1.0,
        angular_process_noise: float = 2.0,
        adaptive_noise: bool = True,
        static_noise_scale: float = 0.2,
        max_prediction_s: float = 1.0,
        hold_time_s: float = 0.22,
        lost_timeout_s: float = 1.0,
        reacquire_frames: int = 3,
        reacquire_blend_s: float = 0.16,
        reinitialize_gap_s: float = 0.30,
        lost_position_sigma_m: float = 0.5,
        lost_orientation_sigma_deg: float = 30.0,
        min_markers_tracking: int = 3,
        max_reprojection_px: float = 4.0,
        min_confidence: float = 0.25,
        acceleration_tau_s: float = 0.12,
    ):
        self.name = str(name)
        self.primary_marker_id = int(primary_marker_id)
        self.hold_time_s = max(0.02, float(hold_time_s))
        self.lost_timeout_s = max(self.hold_time_s, float(lost_timeout_s))
        self.reacquire_frames = max(1, int(reacquire_frames))
        self.reacquire_blend_s = max(0.0, float(reacquire_blend_s))
        self.reinitialize_gap_s = max(0.0, float(reinitialize_gap_s))
        self.lost_position_sigma_m = max(0.01, float(lost_position_sigma_m))
        self.lost_orientation_sigma = math.radians(
            max(0.1, float(lost_orientation_sigma_deg))
        )
        self.min_markers_tracking = max(1, int(min_markers_tracking))
        self.max_reprojection_px = max(0.1, float(max_reprojection_px))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.filter = _ErrorStateEkf(
            linear_acceleration_noise=process_noise,
            angular_acceleration_noise=angular_process_noise,
            acceleration_tau_s=acceleration_tau_s,
            max_prediction_s=max_prediction_s,
            adaptive_noise=adaptive_noise,
            static_noise_scale=static_noise_scale,
        )
        self.state = TrackingState.LOST
        self.last_detection_time = 0.0
        self.last_update_time = 0.0
        self.last_capture_time = 0.0
        self.last_measurement_confidence = 0.0
        self.last_reprojection_error = 0.0
        self.last_marker_count = 0
        self.last_distance = 0.0
        self.reacquire_streak = 0
        self.rejected_streak = 0
        self._blend: Optional[_Blend] = None
        self._last_output: Optional[ObjectPose] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    def observe(self, measurement: ObjectMeasurement, now: Optional[float] = None) -> TrackingState:
        """Feed one validated board measurement into the filter/state machine.

        ``now`` defaults to the measurement's capture timestamp.  Every
        timestamp in the tracker therefore shares one timebase; processing
        time is carried separately in ``measurement.detection_time``.
        """
        capture = float(measurement.capture_time)
        stamp = capture if now is None else float(now)
        with self._lock:
            had_state = self.filter.initialized
            gap = (
                capture - self.last_detection_time
                if self.last_detection_time > 0.0
                else math.inf
            )
            previous_state = self.state

            if not had_state:
                self.filter.reset(
                    measurement.position,
                    measurement.rotation,
                    capture,
                    position_sigma=max(0.005, measurement.position_sigma),
                    orientation_sigma=max(math.radians(0.1), measurement.orientation_sigma),
                )
                self.state = TrackingState.TRACKING
                self.last_detection_time = capture
                self.last_update_time = stamp
                self.last_capture_time = capture
                self._accept_metadata(measurement)
                self._remember_output()
                return self.state

            reinitialize = (
                previous_state == TrackingState.LOST
                or gap > self.reinitialize_gap_s
                or self.rejected_streak >= self.reacquire_frames
            )
            if reinitialize:
                # A long interruption invalidates velocity.  Restart from the
                # measurement and blend the visible output over a short window
                # so the virtual object does not snap.
                self.filter.reset(
                    measurement.position,
                    measurement.rotation,
                    capture,
                    position_sigma=max(0.005, measurement.position_sigma),
                    orientation_sigma=max(math.radians(0.1), measurement.orientation_sigma),
                )
                self.state = TrackingState.REACQUIRING
                self.reacquire_streak = 1
                self.rejected_streak = 0
                start = self._last_output
                if start is not None:
                    self._blend = _Blend(
                        start_position=start.position.copy(),
                        start_rotation=start.rotation.copy(),
                        started_at=stamp,
                        duration=self.reacquire_blend_s,
                    )
                self.last_detection_time = capture
                self.last_update_time = stamp
                self.last_capture_time = capture
                self._accept_metadata(measurement)
                self._remember_output()
                return self.state

            # During prediction/reacquisition the estimate is already
            # uncertain; inflate the measurement covariance so the correction
            # is gradual instead of a snap.
            position_covariance = measurement.position_covariance.copy()
            orientation_covariance = measurement.orientation_covariance.copy()
            confidence = measurement.confidence
            if previous_state in (TrackingState.PREDICTING, TrackingState.REACQUIRING):
                position_covariance *= 2.5
                orientation_covariance *= 2.5
                confidence *= 0.8

            diagnostics = self.filter.update(
                measurement.position,
                measurement.rotation,
                position_covariance,
                orientation_covariance,
                confidence,
                capture,
            )
            if diagnostics is None:
                self.rejected_streak += 1
                self.state = (
                    TrackingState.DEGRADED
                    if previous_state
                    in (TrackingState.TRACKING, TrackingState.DEGRADED)
                    else TrackingState.PREDICTING
                )
                if self.rejected_streak >= self.reacquire_frames:
                    self.state = TrackingState.REACQUIRING
                return self.state

            self.rejected_streak = 0
            self.last_detection_time = capture
            self.last_update_time = stamp
            self.last_capture_time = capture
            self._accept_metadata(measurement)

            good_measurement = (
                measurement.marker_count >= self.min_markers_tracking
                and measurement.reprojection_error <= self.max_reprojection_px
                and measurement.confidence >= self.min_confidence
            )
            if previous_state in (
                TrackingState.PREDICTING,
                TrackingState.REACQUIRING,
                TrackingState.LOST,
            ):
                self.reacquire_streak += 1
                if good_measurement and self.reacquire_streak >= self.reacquire_frames:
                    self.state = TrackingState.TRACKING
                    self.reacquire_streak = 0
                else:
                    self.state = TrackingState.REACQUIRING
            elif good_measurement:
                self.state = TrackingState.TRACKING
            else:
                self.state = TrackingState.DEGRADED
            self._remember_output()
            return self.state

    def _accept_metadata(self, measurement: ObjectMeasurement) -> None:
        self.last_measurement_confidence = float(measurement.confidence)
        self.last_reprojection_error = float(measurement.reprojection_error)
        self.last_marker_count = int(measurement.marker_count)
        self.last_distance = float(measurement.distance)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def pose_at(self, now: float) -> Optional[ObjectPose]:
        """Return the estimated pose at ``now`` or ``None`` before acquisition.

        The caller decides whether to ask for an earlier measurement time or
        for a future display time; this method never adds a hidden horizon.
        """
        stamp = float(now)
        with self._lock:
            if not self.filter.initialized:
                return None
            age = max(0.0, stamp - self.last_detection_time)
            if self.state not in (TrackingState.LOST,):
                if age > self.lost_timeout_s:
                    self.state = TrackingState.LOST
                elif age > self.hold_time_s and self.state != TrackingState.REACQUIRING:
                    self.state = TrackingState.PREDICTING

            state, horizon = self.filter.predict_to(stamp)
            position_covariance = state["covariance"][0:3, 0:3]
            orientation_covariance = state["covariance"][6:9, 6:9]
            position_sigma = _safe_sigma(position_covariance)
            orientation_sigma = _safe_sigma(orientation_covariance)
            if (
                self.state != TrackingState.LOST
                and (
                    position_sigma > self.lost_position_sigma_m
                    or orientation_sigma > self.lost_orientation_sigma
                )
            ):
                if age >= self.lost_timeout_s:
                    self.state = TrackingState.LOST

            confidence = (
                self.last_measurement_confidence
                * math.exp(-age / max(0.05, self.hold_time_s))
                * math.exp(-position_sigma / max(0.05, self.lost_position_sigma_m))
            )
            confidence = float(max(0.0, min(1.0, confidence)))
            output = ObjectPose(
                position=state["position"].copy(),
                rotation=state["rotation"].copy(),
                velocity=state["velocity"].copy(),
                angular_velocity=state["angular_velocity"].copy(),
                linear_acceleration=self.filter.linear_acceleration.copy(),
                angular_acceleration=self.filter.angular_acceleration.copy(),
                position_sigma=float(position_sigma),
                orientation_sigma=float(orientation_sigma),
                confidence=confidence,
                state=self.state,
                timestamp=stamp,
                capture_time=self.last_capture_time,
                last_detection_time=self.last_detection_time,
                measurement_confidence=self.last_measurement_confidence,
                reprojection_error=self.last_reprojection_error,
                marker_count=self.last_marker_count,
                distance=self.last_distance,
                valid=self.state != TrackingState.LOST,
            )
            if self._blend is not None:
                blend = self._blend
                if blend.done(stamp) or blend.duration <= 0.0:
                    self._blend = None
                else:
                    alpha = (stamp - blend.started_at) / blend.duration
                    alpha = max(0.0, min(1.0, alpha))
                    # Smoothstep avoids a velocity discontinuity at both ends.
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    output.position = (
                        blend.start_position * (1.0 - alpha) + output.position * alpha
                    )
                    output.rotation = slerp(
                        blend.start_rotation, output.rotation, alpha
                    )
            self._last_output = output.copy()
            return output

    def _remember_output(self) -> None:
        # Store the current filtered pose for the next reacquisition blend.
        output = ObjectPose(
            position=self.filter.position.copy(),
            rotation=self.filter.rotation.copy(),
            velocity=self.filter.velocity.copy(),
            angular_velocity=self.filter.angular_velocity.copy(),
            linear_acceleration=self.filter.linear_acceleration.copy(),
            angular_acceleration=self.filter.angular_acceleration.copy(),
            position_sigma=self.filter.position_sigma,
            orientation_sigma=self.filter.orientation_sigma,
            confidence=self.last_measurement_confidence,
            state=self.state,
            timestamp=self.last_update_time,
            capture_time=self.last_capture_time,
            last_detection_time=self.last_detection_time,
            measurement_confidence=self.last_measurement_confidence,
            reprojection_error=self.last_reprojection_error,
            marker_count=self.last_marker_count,
            distance=self.last_distance,
            valid=True,
        )
        self._last_output = output

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            pose = self._last_output
            return {
                "name": self.name,
                "primary_marker_id": self.primary_marker_id,
                "state": self.state.value,
                "confidence": round(float(self.last_measurement_confidence), 3),
                "reprojection_error": round(float(self.last_reprojection_error), 3),
                "marker_count": int(self.last_marker_count),
                "distance": round(float(self.last_distance), 3),
                "position_sigma": round(float(self.filter.position_sigma), 4),
                "orientation_sigma_deg": round(
                    math.degrees(self.filter.orientation_sigma), 3
                ),
                "velocity": [round(float(value), 4) for value in self.filter.velocity],
                "angular_velocity": [
                    round(float(value), 4) for value in self.filter.angular_velocity
                ],
                "linear_acceleration": [
                    round(float(value), 4)
                    for value in self.filter.linear_acceleration
                ],
                "last_detection_age": round(
                    max(0.0, time.monotonic() - self.last_detection_time), 3
                ),
                "pose": (
                    None
                    if pose is None
                    else {
                        "position": [round(float(value), 5) for value in pose.position],
                        "rotation": [round(float(value), 5) for value in pose.rotation],
                        "timestamp": round(float(pose.timestamp), 6),
                    }
                ),
            }


class SingleMarkerObjectTracker(TrackedObject):
    """Compatibility name for objects tracked without a board.

    The body/hand trackers do not publish full board states, so the fusion
    layer can still use the same estimator interface for them if desired.
    """

    pass


__all__ = [
    "ObjectMeasurement",
    "ObjectPose",
    "TrackingState",
    "TrackedObject",
    "SingleMarkerObjectTracker",
]
