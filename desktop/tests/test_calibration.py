import json
import math
from pathlib import Path

import numpy as np
import pytest

from openvrcardboard.calibration import (
    CameraModel,
    camera_model_for_config,
    camera_pose,
    direction_from_pixel,
    local_point_from_pixel,
    optical_point_to_play,
)
from openvrcardboard.config import CameraConfig
from openvrcardboard.mathutil import IDENTITY, OPTICAL_TO_VIEW, normalize, rotate

BASE = Path(__file__).resolve().parent


def test_estimate_centres_principal_point_and_fov():
    model = CameraModel.estimate(1280, 720, 90.0)
    assert model.cx == pytest.approx(640.0)
    assert model.cy == pytest.approx(360.0)
    # A 90° horizontal FOV means fx equals half the width.
    assert model.fx == pytest.approx(640.0)
    assert model.fy == pytest.approx(640.0)


def test_project_and_unproject_roundtrip():
    model = CameraModel.estimate(640, 480)
    optical = np.array([0.2, -0.15, 2.5])
    u, v = model.project(optical)
    recovered = model.unproject(u, v, 2.5)
    assert recovered == pytest.approx(optical)


def test_scale_changes_intrinsics_proportionally():
    model = CameraModel.estimate(1280, 720)
    scaled = model.scaled(640, 360)
    assert scaled.width == 640
    assert scaled.height == 360
    assert scaled.fx == pytest.approx(model.fx / 2)
    assert scaled.fy == pytest.approx(model.fy / 2)
    assert scaled.cx == pytest.approx(model.cx / 2)
    assert scaled.cy == pytest.approx(model.cy / 2)


def test_save_load_roundtrip_and_scaling(tmp_path):
    model = CameraModel(1280, 720, 900.0, 910.0, 640.0, 360.0, [0.1, -0.2, 0.0, 0.0, 0.05])
    path = model.save(tmp_path / "camera_calibration.json")
    values = json.loads(path.read_text())
    assert values["camera_matrix"][0][0] == pytest.approx(900.0)

    loaded = CameraModel.load(path, 1280, 720)
    assert loaded.fx == pytest.approx(900.0)
    assert loaded.distortion[:2] == pytest.approx([0.1, -0.2])

    smaller = CameraModel.load(path, 640, 360)
    assert smaller.fx == pytest.approx(450.0)
    assert smaller.cx == pytest.approx(320.0)


def test_load_rejects_bad_matrix(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"width": 100, "height": 100, "camera_matrix": [[1, 2]]}))
    with pytest.raises(ValueError):
        CameraModel.load(path, 100, 100)


def test_model_for_config_falls_back_with_warning(tmp_path):
    config = CameraConfig(
        width=800, height=600, calibration_file="missing.json", horizontal_fov_deg=90.0
    )
    model, warnings = camera_model_for_config(config, tmp_path)
    assert model.width == 800
    assert model.height == 600
    assert model.fx == pytest.approx(400.0)
    assert warnings


def test_model_for_config_loads_file(tmp_path):
    CameraModel(800, 600, 500.0, 500.0, 400.0, 300.0).save(
        tmp_path / "camera_calibration.json"
    )
    config = CameraConfig(width=800, height=600)
    model, warnings = camera_model_for_config(config, tmp_path)
    assert not warnings
    assert model.fx == pytest.approx(500.0)


def test_camera_pose_default_looks_towards_origin():
    position, rotation = camera_pose(CameraConfig())
    assert position == pytest.approx([0.0, 1.5, -1.5])
    # Camera at -Z looking towards +Z means 180° around Y.
    forward = rotate(rotation, np.array([0.0, 0.0, -1.0]))
    assert forward == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_optical_to_play_maps_centre_ray_forward():
    position, rotation = camera_pose(CameraConfig())
    optical = np.array([0.0, 0.0, 2.0])
    world = optical_point_to_play(position, rotation, optical)
    # 2 m in front of the camera at z=-1.5 is z=+0.5 at camera height.
    assert world == pytest.approx([0.0, 1.5, 0.5], abs=1e-9)


def test_optical_down_is_world_down():
    position, rotation = camera_pose(CameraConfig())
    optical = np.array([0.0, 0.5, 2.0])  # +Y is down in optical coordinates
    world = optical_point_to_play(position, rotation, optical)
    assert world == pytest.approx([0.0, 1.0, 0.5], abs=1e-9)


def test_local_point_from_pixel_uses_intrinsics():
    position, rotation = camera_pose(CameraConfig())
    model = CameraModel.estimate(1280, 720)
    centre = local_point_from_pixel(model, position, rotation, 640, 360, 2.0)
    assert centre == pytest.approx([0.0, 1.5, 0.5], abs=1e-9)
    offset = local_point_from_pixel(model, position, rotation, 740, 360, 2.0)
    # Pixels to the right of the image are to the camera's right, which is
    # world -X for the default camera looking back at the player.
    assert offset[0] < centre[0]


def test_direction_from_pixel_points_into_play_space():
    _, rotation = camera_pose(CameraConfig())
    model = CameraModel.estimate(1280, 720)
    centre = direction_from_pixel(model, rotation, 640, 360)
    assert centre == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    right = direction_from_pixel(model, rotation, 1200, 360)
    # The default camera looks back at the player from -Z, so its right side
    # is world -X.
    assert right[0] < 0.0
    up = direction_from_pixel(model, rotation, 640, 100)
    assert up[1] > 0.0
    assert np.linalg.norm(right) == pytest.approx(1.0)


def test_optical_to_view_rotates_180_about_x():
    rotated = rotate(OPTICAL_TO_VIEW, np.array([0.0, 1.0, 0.0]))
    assert rotated == pytest.approx([0.0, -1.0, 0.0])
    rotated = rotate(OPTICAL_TO_VIEW, np.array([0.0, 0.0, 1.0]))
    assert rotated == pytest.approx([0.0, 0.0, -1.0])
