import math
import time

import numpy as np
import pytest

from openvrcardboard.calibration import CameraModel, camera_model_for_config
from openvrcardboard.config import AppConfig, CameraConfig
from openvrcardboard.boards import boards_from_config
from openvrcardboard.markers import BoardTracker
from openvrcardboard.mathutil import angle_between

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV aruco module is required for synthetic scene tests", allow_module_level=True)

from openvrcardboard.synthetic import (  # noqa: E402
    SYNTHETIC_BOARD_NAMES,
    SYNTHETIC_HORIZONTAL_FOV_DEG,
    SyntheticScene,
)


def make_scene():
    boards = boards_from_config(AppConfig().tracking)
    return SyntheticScene(1280, 720, boards=boards), boards


def make_tracker(scene, config=None):
    config = config or AppConfig()
    return BoardTracker(
        scene.model,
        scene.camera_position,
        scene.camera_rotation,
        boards=scene.boards,
        dictionary=config.camera.dictionary,
        max_reprojection_error=8.0,
    )


@pytest.fixture(scope="module")
def scene():
    return make_scene()[0]


def test_synthetic_scene_renders_moving_board_frame(scene):
    first = scene.render(0.0)
    later = scene.render(1.6)
    assert first.shape == (720, 1280, 3)
    assert first.dtype == np.uint8
    difference = np.abs(first.astype(np.int16) - later.astype(np.int16)).mean()
    assert difference > 1.0
    assert int((first.max(axis=2) > 200).sum()) > 1000


def test_synthetic_scene_exposes_every_marker_pose(scene):
    poses = scene.marker_poses(0.0)
    assert set(poses) == set(range(15))
    # Every marker on a rigid board shares the board orientation.
    for name in SYNTHETIC_BOARD_NAMES:
        board = scene.boards[name]
        for marker in board.markers:
            position, rotation = poses[marker.marker_id]
            assert np.all(np.isfinite(position))
            assert np.linalg.norm(rotation) == pytest.approx(1.0)


def test_synthetic_boards_decode_into_six_dof_poses(scene):
    tracker = make_tracker(scene)
    base = time.monotonic()
    samples = []
    for index in range(0, 121):
        elapsed = index / 30.0
        detections = tracker.detect(scene.render(elapsed), timestamp=base + elapsed)
        if index in (0, 60, 120):
            samples.append((elapsed, detections))
    for elapsed, detections in samples:
        assert set(detections) == {0, 5, 10}
        assert all(item.valid for item in detections.values())
        assert all(item.marker_count >= 3 for item in detections.values())
        assert all(item.reprojection_error < 2.0 for item in detections.values())
        ground_truth = scene.board_poses(elapsed)
        for marker_id, name in ((0, "hmd"), (5, "left"), (10, "right")):
            expected = ground_truth[name][0]
            # The synthetic scene deliberately renders 40 mm peripheral
            # markers at 2.6 m on a 720p frame, which is harsher than the
            # 1080p / 2 m deployment target.  PnP position error at that scale
            # is dominated by the renderer's sub-pixel edge bias, so allow a
            # wider bound than the production benchmark.
            assert np.linalg.norm(detections[marker_id].position - expected) < 0.10

    # The HMD board stays near head height while the controllers stay below it
    # and on opposite sides, which is what gives the OpenXR app 6DoF motion.
    for _, detections in samples:
        assert detections[0].position[1] == pytest.approx(1.55, abs=0.12)
        assert detections[5].position[0] < -0.2
        assert detections[10].position[0] > 0.2
        assert detections[5].position[1] < detections[0].position[1] - 0.2
        assert detections[10].position[1] < detections[0].position[1] - 0.2

    first = samples[0][1]
    last = samples[-1][1]
    assert np.linalg.norm(last[0].position - first[0].position) > 0.03
    assert np.linalg.norm(last[5].position - first[5].position) > 0.03


def test_synthetic_board_rotation_stays_continuous_under_noise(scene):
    tracker = make_tracker(scene)
    rng = np.random.default_rng(7)
    base = time.monotonic()
    previous = {}
    worst = 0.0
    for index in range(90):
        elapsed = index / 30.0
        frame = scene.render(elapsed).astype(np.int16)
        frame = np.clip(frame + rng.normal(0.0, 2.0, frame.shape), 0, 255).astype(np.uint8)
        detections = tracker.detect(frame, timestamp=base + elapsed)
        for marker_id, detection in detections.items():
            last = previous.get(marker_id)
            if last is not None:
                worst = max(
                    worst,
                    math.degrees(angle_between(last, detection.rotation)),
                )
            previous[marker_id] = detection.rotation
    # Without IPPE candidate continuity the planar ambiguity produces frequent
    # 60-180 degree flips even on this clean synthetic footage.
    assert worst < 30.0


def test_synthetic_camera_model_ignores_calibration_file(tmp_path):
    config = CameraConfig(
        source="synthetic",
        width=1280,
        height=720,
        horizontal_fov_deg=SYNTHETIC_HORIZONTAL_FOV_DEG,
        calibration_file=str(tmp_path / "missing.json"),
    )
    model, warnings = camera_model_for_config(config, tmp_path)
    expected = CameraModel.estimate(1280, 720, SYNTHETIC_HORIZONTAL_FOV_DEG)
    assert warnings == []
    assert model.fx == pytest.approx(expected.fx)
    assert model.fy == pytest.approx(expected.fy)
    assert model.cx == pytest.approx(expected.cx)
    assert model.cy == pytest.approx(expected.cy)
