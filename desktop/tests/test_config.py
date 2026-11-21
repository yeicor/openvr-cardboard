import json

import pytest

from openvrcardboard.config import (
    CONFIG_VERSION,
    AppConfig,
    load_config,
    save_config,
    validate_config,
)


def test_default_boards_use_disjoint_five_id_ranges():
    config = AppConfig()
    validate_config(config)
    assert CONFIG_VERSION == 7
    assert (config.tracking.hmd_marker_id, config.tracking.left_marker_id) == (0, 5)
    assert config.tracking.right_marker_id == 10
    assert config.camera.dictionary == "DICT_3X3_15"
    hmd = config.tracking.boards.hmd
    assert hmd.width_m == pytest.approx(0.200)
    assert hmd.height_m == pytest.approx(0.112)
    assert hmd.central_marker_m == pytest.approx(0.080)
    assert hmd.peripheral_marker_m == pytest.approx(0.040)
    assert (hmd.peripheral_x_m, hmd.peripheral_y_m) == pytest.approx((0.070, 0.026))


def test_synthetic_camera_forces_known_geometry():
    config = AppConfig()
    config.camera.source = "synthetic"
    warnings = validate_config(config)
    assert config.camera.horizontal_fov_deg == pytest.approx(65.0)
    assert config.camera.dictionary == "DICT_3X3_15"
    assert config.camera.position == [0.0, 1.02, -2.60]
    assert config.camera.rotation == [0.0, 1.0, 0.0, 0.0]
    assert config.camera.max_reprojection_error == pytest.approx(8.0)
    assert config.camera.marker_detection_width == 0
    assert any("synthetic camera forces" in warning for warning in warnings)


def test_phone_camera_defaults_and_port_collision():
    config = AppConfig()
    assert config.phone.camera_enabled is True
    assert config.phone.camera_port == 8766
    assert config.phone.camera_bitrate_mbps == pytest.approx(40.0)
    assert config.phone.camera_timeout_s == pytest.approx(0.5)
    assert validate_config(config) == []

    config.phone.camera_port = config.phone.video_port
    warnings = validate_config(config)
    assert config.phone.camera_enabled is False
    assert any("phone.camera_port must differ" in warning for warning in warnings)


def test_phone_camera_settings_are_clamped():
    config = AppConfig()
    config.phone.camera_port = 0
    config.phone.camera_bitrate_mbps = 500.0
    config.phone.camera_timeout_s = 0.0
    validate_config(config)
    assert config.phone.camera_port == 1
    assert config.phone.camera_bitrate_mbps == pytest.approx(200.0)
    assert config.phone.camera_timeout_s == pytest.approx(0.1)


def test_comfort_filter_settings_are_clamped():
    config = AppConfig()
    config.tracking.marker_static_noise_scale = 4.0
    config.tracking.marker_output_smoothing_hz = -3.0
    config.tracking.marker_process_noise = -1.0
    validate_config(config)
    assert config.tracking.marker_static_noise_scale == pytest.approx(1.0)
    assert config.tracking.marker_output_smoothing_hz == pytest.approx(0.0)
    assert config.tracking.marker_process_noise == pytest.approx(0.05)
    assert config.tracking.marker_adaptive_noise is True


def test_unknown_camera_source_falls_back_to_webcam():
    config = AppConfig()
    config.camera.source = "webcam"
    assert validate_config(config) == []
    config.camera.source = "hologram"
    warnings = validate_config(config)
    assert config.camera.source == "webcam"
    assert any("camera source" in warning for warning in warnings)


def test_openxr_capture_enables_bridge_and_forces_sbs():
    config = AppConfig()
    config.camera.source = "synthetic"
    config.openxr.enabled = False
    config.phone.capture_source = "openxr"
    config.phone.layout = "mono"
    warnings = validate_config(config)
    assert config.openxr.enabled is True
    assert config.phone.layout == "sbs"
    assert any("OpenXR capture requires" in warning for warning in warnings)
    assert config.openxr.remote_port == 4242
    assert config.openxr.capture_port == 39102
    assert config.openxr.manage_service is True


def test_openxr_ports_must_differ():
    config = AppConfig()
    config.openxr.enabled = True
    config.openxr.remote_port = 40000
    config.openxr.capture_port = 40000
    warnings = validate_config(config)
    assert (config.openxr.remote_port, config.openxr.capture_port) == (4242, 39102)
    assert any("ports are the same" in warning for warning in warnings)


def test_synthetic_gamepads_allow_shared_index():
    config = AppConfig()
    config.gamepad.synthetic = True
    config.gamepad.left_index = 0
    config.gamepad.right_index = 0
    warnings = validate_config(config)
    assert not any("same index" in warning for warning in warnings)


def test_overlapping_board_ids_are_repaired():
    config = AppConfig()
    config.tracking.left_marker_id = 1
    warnings = validate_config(config)
    assert (config.tracking.hmd_marker_id, config.tracking.left_marker_id) == (0, 5)
    assert config.tracking.right_marker_id == 10
    assert any("disjoint five-ID ranges" in warning for warning in warnings)


def test_board_ids_respect_dictionary_capacity():
    config = AppConfig()
    config.camera.dictionary = "DICT_4X4_50"
    config.tracking.right_marker_id = 48
    warnings = validate_config(config)
    assert config.tracking.right_marker_id == 10
    assert any("disjoint five-ID ranges" in warning for warning in warnings)


def test_impossible_board_geometry_falls_back_to_defaults():
    config = AppConfig()
    config.tracking.boards.hmd.peripheral_y_m = 0.06
    warnings = validate_config(config)
    assert config.tracking.boards.hmd.peripheral_y_m == pytest.approx(0.026)
    assert any("geometry is impossible" in warning for warning in warnings)


def test_board_mounting_rotation_is_normalised():
    config = AppConfig()
    config.tracking.boards.hmd.object_rotation = [2.0, 0.0, 0.0, 0.0]
    validate_config(config)
    assert config.tracking.boards.hmd.object_rotation == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )


def test_config_round_trips_board_geometry(tmp_path):
    config = AppConfig()
    config.tracking.boards.left.peripheral_x_m = 0.072
    config.tracking.boards.right.object_position = [0.0, -0.02, 0.01]
    path = save_config(config, tmp_path / "config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config_version"] == CONFIG_VERSION
    loaded, warnings = load_config(path)
    assert loaded.tracking.boards.left.peripheral_x_m == pytest.approx(0.072)
    assert loaded.tracking.boards.right.object_position == pytest.approx(
        [0.0, -0.02, 0.01]
    )
    assert not any("configuration version" in warning for warning in warnings)


def test_unknown_dictionary_falls_back_to_default():
    config = AppConfig()
    config.camera.dictionary = "DICT_QUANTUM_9"
    warnings = validate_config(config)
    assert config.camera.dictionary == "DICT_3X3_15"
    assert any("unknown ArUco dictionary" in warning for warning in warnings)


def test_old_config_version_is_discarded(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "config_version": CONFIG_VERSION - 1,
                "camera": {"marker_size_m": 0.5},
                "tracking": {"left_marker_id": 1},
            }
        ),
        encoding="utf-8",
    )
    config, warnings = load_config(path)
    assert config.camera.dictionary == "DICT_3X3_15"
    assert config.tracking.left_marker_id == 5
    assert any("ignoring configuration version" in warning for warning in warnings)
