import numpy as np
import pytest

from openvrcardboard.boards import (
    BOARD_HEIGHT_M,
    BOARD_WIDTH_M,
    CENTRAL_MARKER_M,
    PERIPHERAL_MARKER_M,
    PERIPHERAL_X_M,
    PERIPHERAL_Y_M,
    BoardPoseEstimator,
    board_from_config,
    board_measurement_covariance,
    board_object_pose,
    boards_from_config,
    default_boards,
)
from openvrcardboard.config import AppConfig
from openvrcardboard.mathutil import (
    OPTICAL_TO_VIEW,
    conjugate,
    multiply,
    to_rotvec,
)
from openvrcardboard.synthetic import SyntheticScene

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV aruco module is required for board tests", allow_module_level=True)


def make_scene():
    boards = default_boards()
    return SyntheticScene(1280, 720, boards=boards), boards


def exact_corners(scene, board, object_position, object_rotation):
    return {
        marker.marker_id: scene._project_board_points(
            board, object_position, object_rotation, marker.object_points()
        )
        for marker in board.markers
    }


def test_default_boards_use_the_configured_geometry():
    boards = boards_from_config(AppConfig().tracking)
    board = boards["hmd"]
    assert board.marker_ids == (0, 1, 2, 3, 4)
    assert board.width_m == pytest.approx(BOARD_WIDTH_M)
    assert board.height_m == pytest.approx(BOARD_HEIGHT_M)
    assert board.central.size_m == pytest.approx(CENTRAL_MARKER_M)
    assert board.marker(1).size_m == pytest.approx(PERIPHERAL_MARKER_M)
    assert (board.marker(1).center_x_m, board.marker(1).center_y_m) == pytest.approx(
        (-PERIPHERAL_X_M, PERIPHERAL_Y_M)
    )
    assert boards["left"].marker_ids == (5, 6, 7, 8, 9)
    assert boards["right"].marker_ids == (10, 11, 12, 13, 14)


def test_board_pnp_recovers_an_exact_synthetic_pose():
    scene, boards = make_scene()
    board = boards["hmd"]
    position, rotation = scene.object_poses(0.0)["hmd"]
    corners = exact_corners(scene, board, position, rotation)
    estimate = BoardPoseEstimator(max_reprojection_error=2.0).estimate(
        board, corners, scene.model
    )
    assert estimate is not None
    assert estimate.marker_count == 5
    assert estimate.outlier_ids == ()
    assert estimate.reprojection_error < 1e-3
    solved_position, solved_rotation = board_object_pose(
        board,
        estimate.tvec,
        estimate.rotation,
        scene.camera_position,
        scene.camera_rotation,
        OPTICAL_TO_VIEW,
    )
    assert np.linalg.norm(solved_position - position) < 1e-5
    error = np.degrees(
        np.linalg.norm(to_rotvec(multiply(solved_rotation, conjugate(rotation))))
    )
    assert error < 1e-3


def test_board_pnp_rejects_one_corrupted_marker():
    scene, boards = make_scene()
    board = boards["hmd"]
    position, rotation = scene.object_poses(0.0)["hmd"]
    corners = exact_corners(scene, board, position, rotation)
    corrupted = dict(corners)
    corrupted[3] = corrupted[3] + np.array([35.0, -25.0], dtype=np.float64)
    estimate = BoardPoseEstimator(max_reprojection_error=3.0).estimate(
        board, corrupted, scene.model
    )
    assert estimate is not None
    assert 3 in estimate.outlier_ids
    assert estimate.marker_count == 4
    assert estimate.reprojection_error < 1.0
    solved_position, _ = board_object_pose(
        board,
        estimate.tvec,
        estimate.rotation,
        scene.camera_position,
        scene.camera_rotation,
        OPTICAL_TO_VIEW,
    )
    assert np.linalg.norm(solved_position - position) < 0.01


def test_board_pnp_solves_from_the_central_marker_alone():
    scene, boards = make_scene()
    board = boards["left"]
    position, rotation = scene.object_poses(0.5)["left"]
    corners = exact_corners(scene, board, position, rotation)
    estimate = BoardPoseEstimator(max_reprojection_error=2.0).estimate(
        board, {board.primary_id: corners[board.primary_id]}, scene.model
    )
    assert estimate is not None
    assert estimate.marker_count == 1
    solved_position, _ = board_object_pose(
        board,
        estimate.tvec,
        estimate.rotation,
        scene.camera_position,
        scene.camera_rotation,
        OPTICAL_TO_VIEW,
    )
    # A single square has a planar ambiguity, but the chosen solution must be
    # geometrically consistent with the observed square.
    assert np.linalg.norm(solved_position - position) < 0.03


def test_board_covariance_is_positive_definite_and_range_aware():
    scene, boards = make_scene()
    board = boards["hmd"]
    position, rotation = scene.object_poses(0.0)["hmd"]
    corners = exact_corners(scene, board, position, rotation)
    estimator = BoardPoseEstimator(max_reprojection_error=2.0)
    near = estimator.estimate(board, corners, scene.model)
    assert near is not None
    far = near.copy()
    far.tvec = far.tvec * 2.0
    far.distance = float(np.linalg.norm(far.tvec))
    near_position, near_orientation = board_measurement_covariance(
        near, scene.model, board, scene.camera_rotation, OPTICAL_TO_VIEW
    )
    far_position, far_orientation = board_measurement_covariance(
        far, scene.model, board, scene.camera_rotation, OPTICAL_TO_VIEW
    )
    for covariance in (near_position, near_orientation, far_position, far_orientation):
        assert np.allclose(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    # Depth uncertainty grows faster than lateral uncertainty, so the largest
    # eigenvalue must grow by more than the 2x range change.
    assert np.linalg.eigvalsh(far_position)[-1] > 2.5 * np.linalg.eigvalsh(near_position)[-1]


def test_board_from_config_uses_every_custom_field():
    config = AppConfig().tracking
    board = board_from_config("custom", 20, config.boards.left)
    assert board.primary_id == 20
    assert board.marker_ids == (20, 21, 22, 23, 24)
    assert board.object_position == pytest.approx(tuple(config.boards.left.object_position))
    assert board.object_rotation == pytest.approx(tuple(config.boards.left.object_rotation))
