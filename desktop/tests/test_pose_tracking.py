import math

import numpy as np
import pytest

from openvrcardboard.calibration import CameraModel, camera_pose
from openvrcardboard.config import CameraConfig
from openvrcardboard.filters import LandmarkSmoother
from openvrcardboard.mathutil import rotate
from openvrcardboard.pose_tracking import (
    body_pose_from_landmarks,
    hand_inputs_from_landmarks,
    hand_pose_from_landmarks,
)


def make_hand(pixels_width=60.0, open_hand=True):
    """Synthetic right-hand landmarks in pixels, palm facing the camera."""
    points = np.zeros((21, 2), dtype=np.float64)
    wrist = np.array([640.0, 520.0])
    index_mcp = np.array([640.0 + pixels_width / 2, 420.0])
    pinky_mcp = np.array([640.0 - pixels_width / 2, 420.0])
    middle_mcp = np.array([640.0, 410.0])
    ring_mcp = np.array([620.0, 420.0])
    points[0] = wrist
    points[5] = index_mcp
    points[9] = middle_mcp
    points[13] = ring_mcp
    points[17] = pinky_mcp
    tip_distance = 170.0 if open_hand else 20.0
    for base, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        points[tip] = points[base] + np.array([0.0, -tip_distance])
    points[4] = np.array([740.0, 470.0]) if open_hand else np.array([690.0, 460.0])
    points[1:4] = [points[4] + np.array([-12.0, 12.0]), points[4] + np.array([-6.0, 18.0]), points[4]]
    return points


def make_body():
    points = np.full((33, 2), 0.5, dtype=np.float64)
    # Player's left appears on the right of a camera facing them.
    points[11] = (740.0, 300.0)  # left shoulder
    points[12] = (540.0, 300.0)  # right shoulder
    points[23] = (710.0, 450.0)  # left hip
    points[24] = (570.0, 450.0)  # right hip
    points[25] = (710.0, 540.0)  # left knee
    points[26] = (570.0, 540.0)
    points[27] = (700.0, 650.0)  # left ankle
    points[28] = (580.0, 650.0)
    points[29] = (700.0, 675.0)  # left heel
    points[30] = (580.0, 675.0)
    points[31] = (730.0, 660.0)  # left foot index
    points[32] = (550.0, 660.0)
    points[15] = (760.0, 430.0)  # left wrist
    points[16] = (520.0, 430.0)
    points[0] = (640.0, 160.0)  # nose
    return points


@pytest.fixture
def camera():
    model = CameraModel.estimate(1280, 720)
    position, rotation = camera_pose(CameraConfig())
    return model, position, rotation


def test_open_hand_inputs(camera):
    model, position, rotation = camera
    pixels = make_hand(open_hand=True)
    pinch, grip = hand_inputs_from_landmarks(pixels)
    assert pinch < 0.3
    assert grip < 0.3

    closed = make_hand(open_hand=False)
    closed[4] = closed[8] + np.array([3.0, 3.0])
    pinch, grip = hand_inputs_from_landmarks(closed)
    assert pinch > 0.8
    assert grip > 0.5


def test_hand_pose_is_in_front_of_camera(camera):
    model, position, rotation = camera
    observation = hand_pose_from_landmarks(
        make_hand(), model, position, rotation, palm_width_m=0.085, side="right"
    )
    assert observation is not None
    # Palm width 60 px at f=1108 px -> ~1.57 m depth.
    assert 1.4 < observation.depth < 1.75
    # Centre pixel is below the principal point, so the hand is below the
    # camera and between the camera and the player.
    assert observation.position[1] < 1.5
    assert observation.position[1] > 0.9
    assert -0.5 < observation.position[2] < 0.5
    assert np.linalg.norm(observation.rotation) == pytest.approx(1.0)


def test_hand_points_fingers_up_in_play_space(camera):
    model, position, rotation = camera
    observation = hand_pose_from_landmarks(
        make_hand(), model, position, rotation, palm_width_m=0.085, side="right"
    )
    fingers = rotate(observation.rotation, np.array([0.0, 0.0, -1.0]))
    assert fingers[1] > 0.9
    right = rotate(observation.rotation, np.array([1.0, 0.0, 0.0]))
    assert right[0] > 0.9


def test_left_hand_mirrors_local_x(camera):
    model, position, rotation = camera
    observation = hand_pose_from_landmarks(
        make_hand(), model, position, rotation, palm_width_m=0.085, side="left"
    )
    right = rotate(observation.rotation, np.array([1.0, 0.0, 0.0]))
    assert right[0] < -0.9


def test_hand_rejects_degenerate_landmarks(camera):
    model, position, rotation = camera
    assert (
        hand_pose_from_landmarks(
            np.zeros((21, 2)), model, position, rotation, side="right"
        )
        is None
    )


def test_body_joints_are_placed_anatomically(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(
        make_body(), model, position, rotation, roles=["hips", "chest", "left_foot", "right_foot"]
    )
    assert observation is not None
    assert set(observation.joints) == {"hips", "chest", "left_foot", "right_foot"}
    assert observation.joints["left_foot"][0] < observation.joints["right_foot"][0]
    assert observation.joints["chest"][1] > observation.joints["hips"][1]
    assert observation.joints["hips"][1] > observation.joints["left_foot"][1]
    assert observation.depth > 1.5
    for rotation in observation.rotations.values():
        assert np.linalg.norm(rotation) == pytest.approx(1.0)


def test_body_roles_filter(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(
        make_body(), model, position, rotation, roles=[]
    )
    assert observation is not None
    assert observation.joints == {}


def test_body_role_aliases(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(
        make_body(), model, position, rotation, roles=["waist"]
    )
    assert observation is not None
    assert set(observation.joints) == {"hips"}


def test_body_scale_and_offset(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(
        make_body(),
        model,
        position,
        rotation,
        body_scale=1.0,
        position_offset=(0.0, 0.25, 0.0),
    )
    assert observation is not None
    # Every accepted joint is lifted by the configured offset.
    assert "hips" in observation.joints
    assert observation.joints["hips"][1] > 0.8


def test_body_yaw_follows_shoulder_line(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(make_body(), model, position, rotation)
    assert observation is not None
    forward = rotate(observation.rotations["hips"], np.array([0.0, 0.0, -1.0]))
    assert abs(forward[1]) < 1e-6 or abs(forward[1]) < 1e-9
    assert np.linalg.norm(forward) == pytest.approx(1.0)


def test_landmark_smoother_reduces_subpixel_jitter():
    smoother = LandmarkSmoother(1, 2, min_cutoff=4.0, beta=0.0)
    values = []
    for index in range(40):
        x = 100.0 + (0.5 if index % 2 else -0.5)
        filtered = smoother.filter([[x, 50.0]], index / 60.0)
        values.append(float(filtered[0, 0]))
    assert np.std(values[10:]) < 0.2


def test_body_visibility_drops_occluded_joints(camera):
    model, position, rotation = camera
    visibility = np.ones(33, dtype=np.float64)
    visibility[[27, 28, 29, 30, 31, 32]] = 0.1
    observation = body_pose_from_landmarks(
        make_body(),
        model,
        position,
        rotation,
        visibility=visibility,
        min_visibility=0.45,
    )
    assert observation is not None
    assert "hips" in observation.joints
    assert "left_foot" not in observation.joints
    assert "right_foot" not in observation.joints


def test_body_visibility_can_drop_every_joint(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(
        make_body(),
        model,
        position,
        rotation,
        visibility=np.zeros(33),
        min_visibility=0.45,
    )
    assert observation is not None
    assert observation.joints == {}


def test_body_depth_override_is_used(camera):
    model, position, rotation = camera
    observation = body_pose_from_landmarks(
        make_body(), model, position, rotation, depth=3.25
    )
    assert observation is not None
    assert observation.depth == pytest.approx(3.25)


def test_observation_copy_keeps_timestamp(camera):
    model, position, rotation = camera
    observation = hand_pose_from_landmarks(
        make_hand(), model, position, rotation, side="right"
    )
    assert observation is not None
    observation.timestamp = 12.5
    assert observation.copy().timestamp == pytest.approx(12.5)
