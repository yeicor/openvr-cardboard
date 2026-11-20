import math

import numpy as np
import pytest

from openvrcardboard.config import TrackingConfig
from openvrcardboard.fusion import PoseComposer
from openvrcardboard.mathutil import (
    conjugate,
    from_axis_angle,
    multiply,
    normalize,
    to_rotvec,
)
from openvrcardboard.tracking_state import (
    ObjectMeasurement,
    TrackedObject,
    TrackingState,
    _ErrorStateEkf,
)
from openvrcardboard.types import TrackingSnapshot


def measurement(timestamp, position, rotation=(0.0, 0.0, 0.0, 1.0), **overrides):
    values = dict(
        position=np.asarray(position, dtype=np.float64),
        rotation=normalize(rotation),
        position_covariance=np.eye(3) * 0.002**2,
        orientation_covariance=np.eye(3) * math.radians(0.3) ** 2,
        confidence=0.9,
        capture_time=timestamp,
        detection_time=timestamp + 0.002,
        pose_time=timestamp + 0.003,
        reprojection_error=0.4,
        marker_count=5,
        distance=1.5,
        incidence_rad=0.1,
        image_sigma_px=0.4,
        board_area_fraction=0.01,
        marker_ids=(0, 1, 2, 3, 4),
    )
    values.update(overrides)
    return ObjectMeasurement(**values)


def tracker():
    return TrackedObject(
        "hmd",
        0,
        process_noise=3.0,
        angular_process_noise=12.0,
        hold_time_s=0.22,
        lost_timeout_s=1.0,
        reacquire_frames=3,
        reinitialize_gap_s=0.30,
    )


def feed(tracked, times, position, rotation=(0.0, 0.0, 0.0, 1.0)):
    states = []
    for index, stamp in enumerate(times):
        value = position(index) if callable(position) else position
        turn = rotation(index) if callable(rotation) else rotation
        states.append(tracked.observe(measurement(stamp, value, turn)))
    return states


def test_first_measurement_acquires_tracking():
    tracked = tracker()
    assert tracked.observe(measurement(10.0, (0.0, 1.5, 0.0))) == TrackingState.TRACKING
    pose = tracked.pose_at(10.0)
    assert pose is not None
    assert pose.valid
    assert pose.state == TrackingState.TRACKING
    assert np.linalg.norm(pose.position - [0.0, 1.5, 0.0]) < 1e-6


def test_constant_velocity_is_estimated_from_pose_measurements():
    tracked = tracker()
    times = np.arange(0.0, 1.0, 1.0 / 30.0)
    feed(tracked, times, lambda index: (0.01 * index, 1.5, 0.0))
    pose = tracked.pose_at(times[-1])
    assert pose is not None
    # 10 mm per 33.3 ms frame is 0.3 m/s; allow the filter a little lag.
    assert pose.velocity[0] == pytest.approx(0.3, abs=0.05)
    assert abs(pose.velocity[1]) < 0.03
    assert abs(pose.velocity[2]) < 0.03
    assert pose.position_sigma < 0.01


def test_prediction_moves_with_velocity_and_grows_uncertainty():
    tracked = tracker()
    times = np.arange(0.0, 1.0, 1.0 / 30.0)
    feed(tracked, times, lambda index: (0.01 * index, 1.5, 0.0))
    before = tracked.pose_at(times[-1])
    predicted = tracked.pose_at(times[-1] + 0.3)
    assert before is not None and predicted is not None
    assert predicted.state == TrackingState.PREDICTING
    assert predicted.position[0] > before.position[0] + 0.03
    assert predicted.position_sigma > before.position_sigma
    assert predicted.confidence < before.confidence
    # Prediction is capped, so a very long call cannot run away.
    far = tracked.pose_at(times[-1] + 5.0)
    assert far is not None
    assert far.state == TrackingState.LOST
    assert not far.valid


def test_rotation_is_filtered_as_a_quaternion():
    tracked = tracker()
    times = np.arange(0.0, 1.0, 1.0 / 30.0)
    yaw = from_axis_angle((0.0, 1.0, 0.0), 0.8)
    feed(
        tracked,
        times,
        lambda index: (0.0, 1.5, 0.0),
        lambda index: multiply(from_axis_angle((0.0, 1.0, 0.0), 0.002 * index), yaw),
    )
    pose = tracked.pose_at(times[-1])
    assert pose is not None
    assert np.linalg.norm(pose.rotation) == pytest.approx(1.0)
    angle = np.degrees(
        np.linalg.norm(to_rotvec(multiply(pose.rotation, conjugate(yaw))))
    )
    assert angle < 5.0


def test_gross_outlier_is_rejected_and_recovery_continues():
    tracked = tracker()
    times = np.arange(0.0, 1.0, 1.0 / 30.0)
    feed(tracked, times, lambda index: (0.0, 1.5, 0.0))
    assert tracked.observe(measurement(1.1, (3.0, 1.5, 0.0))) != TrackingState.TRACKING
    pose = tracked.pose_at(1.1)
    assert pose is not None
    assert np.linalg.norm(pose.position - (0.0, 1.5, 0.0)) < 0.02
    # The next plausible measurements recover normal tracking.
    states = feed(tracked, [1.13, 1.16, 1.19, 1.22], (0.0, 1.5, 0.0))
    assert states[-1] == TrackingState.TRACKING


def test_reacquisition_after_a_long_gap_blends_without_snapping():
    tracked = tracker()
    tracked.observe(measurement(0.0, (0.0, 1.5, 0.0)))
    for index in range(1, 8):
        tracked.observe(measurement(index / 30.0, (0.0, 1.5, 0.0)))
    previous = tracked.pose_at(7.0 / 30.0)
    assert previous is not None
    # A 0.5 s gap exceeds the reinitialize threshold.
    state = tracked.observe(measurement(0.74, (0.4, 1.5, 0.0)))
    assert state == TrackingState.REACQUIRING
    blended = tracked.pose_at(0.75)
    assert blended is not None
    assert 0.0 < blended.position[0] < 0.4
    assert blended.state == TrackingState.REACQUIRING


def test_degraded_state_when_markers_are_missing():
    tracked = tracker()
    tracked.observe(measurement(0.0, (0.0, 1.5, 0.0)))
    for index in range(1, 5):
        tracked.observe(measurement(index / 30.0, (0.0, 1.5, 0.0)))
    state = tracked.observe(
        measurement(5 / 30.0, (0.0, 1.5, 0.0), marker_count=2, marker_ids=(0, 1))
    )
    assert state == TrackingState.DEGRADED
    assert tracked.pose_at(5 / 30.0).state == TrackingState.DEGRADED


def test_adaptive_noise_drops_at_rest_and_recovers_with_motion():
    adaptive = _ErrorStateEkf(
        linear_acceleration_noise=1.0,
        adaptive_noise=True,
        static_noise_scale=0.2,
    )
    assert adaptive._linear_noise_scale(0.0) == pytest.approx(0.2)
    assert adaptive._linear_noise_scale(0.02) == pytest.approx(0.2)
    assert adaptive._linear_noise_scale(0.06) == pytest.approx(0.6)
    assert adaptive._linear_noise_scale(0.10) == pytest.approx(1.0)
    assert adaptive._linear_noise_scale(2.0) == pytest.approx(1.0)

    fixed = _ErrorStateEkf(
        linear_acceleration_noise=1.0,
        adaptive_noise=False,
        static_noise_scale=0.2,
    )
    assert fixed._linear_noise_scale(0.0) == pytest.approx(1.0)


def test_ten_thousand_updates_keep_covariance_positive_definite():
    """A long stationary run must not accumulate negative eigenvalues."""
    tracked = tracker()
    timestamp = 0.0
    for index in range(10_000):
        timestamp = index / 100.0
        tracked.observe(measurement(timestamp, (0.0, 1.5, 0.0)))
    covariance = tracked.filter.covariance
    assert np.all(np.isfinite(covariance))
    assert covariance == pytest.approx(covariance.T, abs=1e-12)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues[0] > 0.0
    pose = tracked.pose_at(timestamp)
    assert pose is not None
    assert pose.valid
    assert np.linalg.norm(pose.position - (0.0, 1.5, 0.0)) < 0.01


def test_composed_stationary_hmd_is_steadier_than_raw_measurements():
    """The tracker -> composer path must reject measurement jitter at rest."""
    tracked = tracker()
    config = TrackingConfig(markers_enabled=True)
    config.marker_output_smoothing_hz = 0.0
    composer = PoseComposer(config)
    rng = np.random.default_rng(3)
    centre = np.array([0.1, 1.6, -0.4])
    raw = []
    composed = []

    for index in range(360):
        timestamp = index / 90.0
        position = centre + rng.normal(0.0, 0.003, 3)
        tracked.observe(measurement(timestamp, position))
        pose = tracked.pose_at(timestamp)
        assert pose is not None
        snapshot = TrackingSnapshot()
        snapshot.tracked_objects[0] = pose
        frame = composer.update(snapshot, now=timestamp)
        assert frame.hmd is not None and frame.hmd.valid
        if index > 10:
            raw.append(position)
            composed.append(np.asarray(frame.hmd.position, dtype=np.float64))

    def jitter(values):
        steps = np.diff(np.asarray(values, dtype=np.float64), axis=0)
        return float(np.sqrt(np.mean(np.sum(steps * steps, axis=1))))

    raw_jitter = jitter(raw)
    composed_jitter = jitter(composed)
    assert composed_jitter < 0.5 * raw_jitter
    assert np.mean(composed, axis=0) == pytest.approx(centre, abs=0.005)
