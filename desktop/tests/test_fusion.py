import math
import time

import numpy as np
import pytest

from openvrcardboard.config import TrackingConfig
from openvrcardboard.fusion import (
    AdaptiveQuaternionFilter,
    ComposedFrame,
    OneEuroFilter,
    OneEuroVec3,
    PhonePoseEstimator,
    PoseComposer,
    PoseSmoother,
    android_quaternion_to_play,
    apply_local_offset,
    pose_is_fresh,
    screen_rotation_quaternion,
)
from openvrcardboard.mathutil import (
    IDENTITY,
    angle_between,
    from_axis_angle,
    from_euler_zyx_deg,
    multiply,
    normalize,
    rotate,
    to_euler_zyx_deg,
)
from openvrcardboard.protocol import Buttons, ControllerInput, DeviceId
from openvrcardboard.tracking_state import ObjectPose, TrackingState
from openvrcardboard.types import PhoneSample, Pose, TrackingSnapshot


def make_pose(position=(0.0, 0.0, 0.0), rotation=IDENTITY, timestamp=0.0, **kwargs) -> Pose:
    return Pose(
        position=np.array(position, dtype=np.float64),
        rotation=normalize(rotation),
        timestamp=float(timestamp),
        **kwargs,
    )


def make_tracked_object(position, timestamp):
    return ObjectPose(
        position=np.array(position, dtype=np.float64),
        rotation=IDENTITY.copy(),
        velocity=np.zeros(3, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
        linear_acceleration=np.zeros(3, dtype=np.float64),
        angular_acceleration=np.zeros(3, dtype=np.float64),
        position_sigma=0.003,
        orientation_sigma=math.radians(1.0),
        confidence=0.9,
        state=TrackingState.TRACKING,
        timestamp=float(timestamp),
        capture_time=float(timestamp),
        last_detection_time=float(timestamp),
        measurement_confidence=0.9,
        valid=True,
    )


def world_yaw_phone_quaternion(yaw_deg: float) -> np.ndarray:
    """Android quaternion for an upright phone rotated by ``yaw_deg``."""
    upright = from_axis_angle((1.0, 0.0, 0.0), math.pi / 2.0)
    yaw = from_axis_angle((0.0, 0.0, 1.0), math.radians(yaw_deg))
    return multiply(yaw, upright)


def phone_sample(
    quaternion=IDENTITY,
    gyroscope=(0.0, 0.0, 0.0),
    screen_rotation: int = 0,
    accuracy: int = 3,
) -> PhoneSample:
    return PhoneSample(
        sequence=1,
        timestamp=1.0,
        quaternion=np.asarray(quaternion, dtype=np.float64),
        gyroscope=np.asarray(gyroscope, dtype=np.float64),
        acceleration=np.zeros(3),
        accuracy=accuracy,
        received_at=1.0,
        screen_rotation=screen_rotation,
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def test_one_euro_filter_converges_and_ignores_non_finite():
    filter_ = OneEuroFilter(1.0, 0.0)
    for index in range(200):
        value = filter_.filter(2.0, index * 0.01)
    assert value == pytest.approx(2.0, abs=1e-9)
    assert filter_.filter(float("nan"), 3.0) == pytest.approx(2.0)


def test_one_euro_filter_tracks_steps_faster_with_high_cutoff():
    slow = OneEuroFilter(1.0, 0.0)
    fast = OneEuroFilter(20.0, 0.0)
    for index in range(20):
        stamp = index * 0.01
        slow.filter(0.0, stamp)
        fast.filter(0.0, stamp)
    slow_value = slow.filter(1.0, 0.2)
    fast_value = fast.filter(1.0, 0.2)
    assert fast_value > slow_value


def test_one_euro_vec3_filters_each_axis():
    filter_ = OneEuroVec3(10.0, 0.0)
    for index in range(5):
        result = filter_.filter((1.0, 2.0, 3.0), index * 0.01)
    assert result == pytest.approx([1.0, 2.0, 3.0])


def test_adaptive_quaternion_filter_preserves_orientation():
    filter_ = AdaptiveQuaternionFilter(5.0, 0.1)
    target = from_euler_zyx_deg(30.0, 20.0, 10.0)
    result = target
    for index in range(10):
        result = filter_.filter(target, index * 0.01)
    assert angle_between(result, target) < math.radians(0.5)


def test_adaptive_quaternion_filter_holds_a_single_frame_glitch():
    filter_ = AdaptiveQuaternionFilter(5.0, 0.1)
    target = from_euler_zyx_deg(30.0, 20.0, 10.0)
    flipped = multiply(from_axis_angle((0.0, 1.0, 0.0), math.pi), target)
    result = target
    for index in range(10):
        result = filter_.filter(target, index * 0.01)
    assert angle_between(result, target) < math.radians(0.5)

    # A one-frame 180 degree jump must be held, never chased with a raised
    # alpha; the next normal sample is accepted immediately afterwards.
    held = filter_.filter(flipped, 0.10)
    assert angle_between(held, result) < 1e-6
    recovered = filter_.filter(target, 0.11)
    assert angle_between(recovered, target) < math.radians(0.5)


def test_adaptive_quaternion_filter_accepts_a_consistent_teleport():
    filter_ = AdaptiveQuaternionFilter(5.0, 0.1)
    target = from_euler_zyx_deg(30.0, 20.0, 10.0)
    flipped = multiply(from_axis_angle((0.0, 1.0, 0.0), math.pi), target)
    filter_.filter(target, 0.0)
    filter_.filter(target, 0.01)
    first = filter_.filter(flipped, 0.02)
    second = filter_.filter(flipped, 0.03)
    assert angle_between(first, target) < 1e-6
    assert angle_between(second, target) < 1e-6
    accepted = filter_.filter(flipped, 0.04)
    assert angle_between(accepted, flipped) < 1e-6



def test_pose_smoother_derives_world_angular_velocity():
    smoother = PoseSmoother(10.0, 0.0, 20.0, 0.0)
    rotation = IDENTITY
    result = make_pose()
    for index in range(51):
        timestamp = index * 0.01
        rotation = from_axis_angle((0.0, 1.0, 0.0), math.radians(90.0) * index / 50.0)
        result = smoother.update(make_pose(rotation=rotation, timestamp=timestamp), timestamp)
    assert result.angular_velocity[1] == pytest.approx(math.pi, abs=0.1)
    assert abs(result.angular_velocity[0]) < 0.1
    assert abs(result.angular_velocity[2]) < 0.1


def test_pose_smoother_restarts_after_long_gap():
    smoother = PoseSmoother()
    smoother.update(make_pose(position=(0, 0, 0), timestamp=0.0), 0.0)
    result = smoother.update(make_pose(position=(5, 0, 0), timestamp=2.0), 2.0)
    assert result.position == pytest.approx([5.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Phone pose estimation
# ---------------------------------------------------------------------------
def test_android_upright_phone_maps_to_identity_play_rotation():
    quaternion = world_yaw_phone_quaternion(0.0)
    result = android_quaternion_to_play(quaternion)
    assert angle_between(result, IDENTITY) < 1e-6


def test_phone_estimator_recenters_initial_heading():
    estimator = PhonePoseEstimator(recenter_on_start=True)
    pose = estimator.update(phone_sample(world_yaw_phone_quaternion(35.0)), now=10.0)
    assert pose is not None
    assert pose.orientation_only
    assert angle_between(pose.rotation, IDENTITY) < 1e-6
    assert pose.position == pytest.approx([0.0, 1.65, 0.0])


def test_phone_estimator_keeps_relative_yaw_after_recenter():
    estimator = PhonePoseEstimator(recenter_on_start=True)
    estimator.update(phone_sample(world_yaw_phone_quaternion(0.0)), now=10.0)
    pose = estimator.update(phone_sample(world_yaw_phone_quaternion(45.0)), now=10.01)
    yaw, pitch, roll = to_euler_zyx_deg(pose.rotation)
    assert yaw == pytest.approx(45.0, abs=0.5)
    assert pitch == pytest.approx(0.0, abs=0.5)
    assert roll == pytest.approx(0.0, abs=0.5)


def test_phone_estimator_applies_screen_rotation():
    estimator = PhonePoseEstimator(recenter_on_start=True)
    estimator.update(phone_sample(screen_rotation=0), now=10.0)
    pose = estimator.update(phone_sample(screen_rotation=1), now=10.01)
    _, _, roll = to_euler_zyx_deg(pose.rotation)
    assert abs(roll) == pytest.approx(90.0, abs=1.0)


def test_phone_estimator_recenter_removes_relative_rotation():
    estimator = PhonePoseEstimator(recenter_on_start=True)
    estimator.update(phone_sample(world_yaw_phone_quaternion(0.0)), now=10.0)
    estimator.update(phone_sample(world_yaw_phone_quaternion(60.0)), now=10.01)
    estimator.recenter(now=10.02)
    pose = estimator.update(phone_sample(world_yaw_phone_quaternion(60.0)), now=10.03)
    assert angle_between(pose.rotation, IDENTITY) < 1e-6


def test_phone_estimator_predicts_with_gyroscope():
    estimator = PhonePoseEstimator(recenter_on_start=True, max_prediction_ms=20)
    estimator.update(phone_sample(world_yaw_phone_quaternion(0.0)), now=10.0)
    estimator.update(
        phone_sample(world_yaw_phone_quaternion(0.0), gyroscope=(0.0, 0.0, 2.0)),
        now=10.01,
    )
    pose = estimator.pose(now=10.03)
    _, _, roll = to_euler_zyx_deg(pose.rotation)
    assert abs(roll) == pytest.approx(math.degrees(0.04), abs=0.5)


def test_phone_estimator_marks_stale_pose_invalid():
    estimator = PhonePoseEstimator(recenter_on_start=True, timeout_s=0.2)
    estimator.update(phone_sample(), now=10.0)
    pose = estimator.pose(now=10.5)
    assert pose is not None
    assert not pose.valid


def test_phone_estimator_align_to_places_current_heading():
    estimator = PhonePoseEstimator(recenter_on_start=True)
    estimator.update(phone_sample(world_yaw_phone_quaternion(10.0)), now=10.0)
    target = from_euler_zyx_deg(90.0, 0.0, 0.0)
    estimator.align_to(target, now=10.01)
    pose = estimator.update(phone_sample(world_yaw_phone_quaternion(10.0)), now=10.02)
    assert angle_between(pose.rotation, target) < 1e-6


def test_phone_estimator_marker_alignment_blends_yaw_only():
    estimator = PhonePoseEstimator(recenter_on_start=True, marker_blend_tau_s=0.0)
    estimator.update(phone_sample(world_yaw_phone_quaternion(0.0)), now=10.0)
    marker = from_euler_zyx_deg(30.0, 12.0, -7.0)
    aligned = estimator.align_to_marker_orientation(marker, now=10.01)
    yaw, pitch, roll = to_euler_zyx_deg(aligned)
    assert yaw == pytest.approx(30.0, abs=1.0)
    # The phone's gravity-aligned pitch and roll survive the yaw correction.
    assert pitch == pytest.approx(0.0, abs=0.5)
    assert roll == pytest.approx(0.0, abs=0.5)


def test_phone_estimator_marker_alignment_predicts_between_samples():
    estimator = PhonePoseEstimator(
        recenter_on_start=True, max_prediction_ms=50, marker_blend_tau_s=0.0
    )
    estimator.update(
        phone_sample(world_yaw_phone_quaternion(0.0), gyroscope=(0.0, 0.0, 2.0)),
        now=10.0,
    )
    aligned = estimator.align_to_marker_orientation(IDENTITY, now=10.02)
    _, _, roll = to_euler_zyx_deg(aligned)
    assert abs(roll) == pytest.approx(math.degrees(0.04), abs=0.5)


def test_phone_estimator_marker_alignment_stays_locked_across_ticks():
    estimator = PhonePoseEstimator(recenter_on_start=True, marker_blend_tau_s=0.0)
    estimator.update(phone_sample(world_yaw_phone_quaternion(0.0)), now=10.0)
    marker = from_euler_zyx_deg(40.0, 0.0, 0.0)
    aligned = None
    for tick in range(10):
        aligned = estimator.align_to_marker_orientation(
            marker, now=10.01 + 0.01 * tick
        )
    assert aligned is not None
    yaw, _, _ = to_euler_zyx_deg(aligned)
    assert yaw == pytest.approx(40.0, abs=0.1)



# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_composer_manual_and_synthetic_fallbacks():
    config = TrackingConfig(markers_enabled=False, hands_enabled=False, body_enabled=False)
    config.manual_hmd_position = [0.1, 1.7, 0.2]
    composer = PoseComposer(config)
    frame = composer.update(TrackingSnapshot(), now=5.0)
    assert frame.hmd is not None
    assert frame.hmd.position == pytest.approx([0.1, 1.7, 0.2])
    assert frame.sources["hmd"] == "manual"
    assert frame.left_controller is not None
    assert frame.right_controller is not None
    assert frame.sources["left"] == "synthetic"
    assert frame.sources["right"] == "synthetic"


def test_composer_hybrid_hmd_uses_marker_position_and_phone_rotation():
    config = TrackingConfig(markers_enabled=True)
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.markers[0] = make_pose(position=(0.2, 1.6, -0.4), rotation=IDENTITY, timestamp=10.0)
    phone_rotation = from_euler_zyx_deg(25.0, 5.0, 0.0)
    snapshot.phone_pose = make_pose(
        position=(0.0, 1.65, 0.0),
        rotation=phone_rotation,
        timestamp=10.0,
        orientation_only=True,
        source="phone",
    )
    frame = composer.update(snapshot, now=10.01)
    assert frame.hmd.source == ""
    assert frame.hmd.position == pytest.approx([0.2, 1.6, -0.4])
    assert angle_between(frame.hmd.rotation, phone_rotation) < 1e-6
    assert frame.sources["hmd"] == "markers+phone"
    assert not frame.hmd.orientation_only


def test_composer_controller_source_priority_and_offsets():
    config = TrackingConfig(markers_enabled=True, hands_enabled=True, body_enabled=True)
    config.marker_offsets = {"left": [0.0, 0.05, 0.0]}
    config.hand_offsets = {"left": [0.0, 0.0, -0.02]}
    config.pose_offsets = {"right": [0.0, 0.0, 0.1]}
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.markers[5] = make_pose(position=(1.0, 1.0, 1.0), timestamp=10.0)
    snapshot.markers[10] = make_pose(position=(2.0, 2.0, 2.0), timestamp=10.0)
    snapshot.hands[0] = make_pose(position=(9.0, 9.0, 9.0), timestamp=10.0)
    snapshot.hands[1] = make_pose(position=(8.0, 8.0, 8.0), timestamp=10.0)
    snapshot.body["right_hand"] = make_pose(position=(7.0, 7.0, 7.0), timestamp=10.0)
    frame = composer.update(snapshot, now=10.01)
    assert frame.sources["left"] == "markers"
    assert frame.left_controller.position == pytest.approx([1.0, 1.05, 1.0])
    assert frame.sources["right"] == "markers"
    assert frame.right_controller.position == pytest.approx([2.0, 2.0, 2.1])


def test_composer_hands_fall_back_before_body():
    config = TrackingConfig(markers_enabled=False, hands_enabled=True, body_enabled=True)
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.hands[0] = make_pose(position=(4.0, 5.0, 6.0), timestamp=10.0)
    snapshot.body["left_hand"] = make_pose(position=(1.0, 1.0, 1.0), timestamp=10.0)
    frame = composer.update(snapshot, now=10.01)
    assert frame.sources["left"] == "hands"
    assert frame.left_controller.position == pytest.approx([4.0, 5.0, 6.0])


def test_composer_marks_last_pose_invalid_when_marker_disappears():
    config = TrackingConfig(markers_enabled=True)
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.markers[5] = make_pose(position=(1.0, 1.5, -0.5), timestamp=10.0)
    first = composer.update(snapshot, now=10.01)
    assert first.left_controller.valid

    lost = composer.update(TrackingSnapshot(), now=10.02)
    assert lost.left_controller is not None
    assert not lost.left_controller.valid
    assert lost.left_controller.position == pytest.approx([1.0, 1.5, -0.5])
    assert "lost" in lost.sources["left"]


def test_composer_rejects_stale_sources():
    config = TrackingConfig(markers_enabled=True)
    config.pose_max_age_s = 0.1
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.markers[5] = make_pose(position=(1.0, 1.0, 1.0), timestamp=10.0)
    composer.update(snapshot, now=10.01)
    frame = composer.update(snapshot, now=10.5)
    assert not frame.left_controller.valid


def test_composer_uses_board_filter_output_without_extra_smoothing():
    config = TrackingConfig(markers_enabled=True)
    config.marker_output_smoothing_hz = 0.0
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.tracked_objects[0] = make_tracked_object((0.1, 1.6, -0.4), 10.0)
    frame = composer.update(snapshot, now=10.01)
    assert frame.sources["hmd"] == "markers"
    assert frame.hmd is not None
    assert frame.hmd.position == pytest.approx([0.1, 1.6, -0.4])


def test_composer_optional_board_output_smoothing_softens_a_step():
    config = TrackingConfig(markers_enabled=True)
    config.marker_output_smoothing_hz = 4.0
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.tracked_objects[5] = make_tracked_object((0.0, 0.0, 0.0), 10.0)
    composer.update(snapshot, now=10.01)

    moved = TrackingSnapshot()
    moved.tracked_objects[5] = make_tracked_object((1.0, 0.0, 0.0), 10.01)
    frame = composer.update(moved, now=10.02)
    assert frame.left_controller is not None
    assert 0.0 < frame.left_controller.position[0] < 1.0


def test_composer_gamepad_inputs_override_hand_inputs_and_are_clamped():
    config = TrackingConfig()
    composer = PoseComposer(config)
    snapshot = TrackingSnapshot()
    snapshot.hand_inputs[1] = ControllerInput(
        device_id=DeviceId.LEFT, trigger=0.3, joystick_x=0.2
    )
    snapshot.gamepads[1] = ControllerInput(
        device_id=DeviceId.LEFT,
        buttons=Buttons.TRIGGER_CLICK,
        trigger=4.0,
        joystick_x=-3.0,
    )
    snapshot.gamepads[2] = ControllerInput(
        device_id=DeviceId.RIGHT, grip=-2.0, joystick_y=0.5
    )
    frame = composer.update(snapshot, now=10.0)
    left = frame.inputs[int(DeviceId.LEFT)]
    right = frame.inputs[int(DeviceId.RIGHT)]
    assert left.trigger == 1.0
    assert left.joystick_x == -1.0
    assert left.buttons == Buttons.TRIGGER_CLICK
    assert right.grip == 0.0
    assert right.joystick_y == 0.5


def test_composer_maps_trackers_from_body_joints():
    config = TrackingConfig(body_enabled=True)
    config.tracker_roles = ["left_foot", "right_foot"]
    composer = PoseComposer(config, tracker_count=2)
    snapshot = TrackingSnapshot()
    snapshot.body["left_foot"] = make_pose(position=(0.1, 0.1, 0.0), timestamp=10.0)
    snapshot.body["right_foot"] = make_pose(position=(-0.1, 0.1, 0.0), timestamp=10.0)
    frame = composer.update(snapshot, now=10.01)
    assert frame.trackers[100].position == pytest.approx([0.1, 0.1, 0.0])
    assert frame.trackers[101].position == pytest.approx([-0.1, 0.1, 0.0])
    assert frame.sources["tracker:100"] == "body:left_foot"


def test_composed_frame_devices_and_pose_for():
    frame = ComposedFrame(
        hmd=make_pose(),
        left_controller=make_pose(),
        right_controller=make_pose(),
        trackers={100: make_pose(), 101: make_pose()},
    )
    ids = [device_id for device_id, _ in frame.devices()]
    assert ids == [0, 1, 2, 100, 101]
    assert frame.pose_for(100) is frame.trackers[100]
    assert frame.pose_for(42) is None


def test_apply_local_offset_rotates_translation():
    pose = make_pose(rotation=from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0))
    offset = apply_local_offset(pose, (1.0, 0.0, 0.0))
    assert offset.position == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_pose_is_fresh_handles_invalid_and_zero_timestamp():
    now = time.monotonic()
    stale = make_pose(timestamp=now - 5.0)
    assert not pose_is_fresh(stale, now, 0.5)
    assert pose_is_fresh(make_pose(timestamp=0.0), now, 0.5)
    invalid = make_pose(timestamp=now)
    invalid.valid = False
    assert not pose_is_fresh(invalid, now, 0.5)
