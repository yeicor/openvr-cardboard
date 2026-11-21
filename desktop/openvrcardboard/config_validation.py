"""Configuration clamping and validation.

Kept separate from :mod:`openvrcardboard.config` so the dataclasses stay easy
to scan; ``config.validate_config`` and the numeric helpers are re-exported for
callers that historically imported them from ``config``.
"""

from __future__ import annotations

import math
from typing import Any

from .config import (
    ARUCO_DICTIONARIES,
    CUSTOM_DICTIONARY_3X3,
    DICTIONARY_CAPACITY,
    AppConfig,
    BoardConfig,
)


def _float_list(value: Any, length: int, fallback: list[float]) -> list[float]:
    try:
        result = [float(v) for v in list(value)]
    except (TypeError, ValueError):
        return list(fallback)
    if len(result) != length or not all(math.isfinite(v) for v in result):
        return list(fallback)
    return result


def _vector_dict(value: Any, length: int) -> dict[str, list[float]]:
    """Normalise a ``name -> vector`` mapping, dropping invalid entries."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[float]] = {}
    for key, vector in value.items():
        try:
            numbers = [float(v) for v in list(vector)]
        except (TypeError, ValueError):
            continue
        if len(numbers) != length or not all(math.isfinite(v) for v in numbers):
            continue
        result[str(key)] = numbers
    return result


def _clamp(value: int | float, low: int | float, high: int | float, cast=float):
    try:
        value = cast(value)
    except (TypeError, ValueError):
        value = cast(low)
    if isinstance(value, float) and not math.isfinite(value):
        value = cast(low)
    return max(low, min(high, value))


def _board_geometry_valid(board: BoardConfig) -> bool:
    """Check that every marker fits, and no two markers overlap."""
    half_w = board.width_m * 0.5
    half_h = board.height_m * 0.5
    central = board.central_marker_m * 0.5
    peripheral = board.peripheral_marker_m * 0.5
    px = abs(board.peripheral_x_m)
    py = abs(board.peripheral_y_m)
    if not (half_w > central and half_h > central):
        return False
    if peripheral <= 0.0 or px <= 0.0 or py <= 0.0:
        return False
    # Every peripheral black square must stay inside the board outline.
    if px + peripheral > half_w or py + peripheral > half_h:
        return False
    # Central versus peripheral: rectangles are separate when either axis is.
    if px < central + peripheral and py < central + peripheral:
        return False
    # The two peripherals on the same side must not touch.
    if py * 2.0 < peripheral * 2.0:
        return False
    return True


def _reset_board_geometry(board: BoardConfig, warnings: list[str], name: str) -> None:
    defaults = BoardConfig()
    warnings.append(
        f"board {name!r} geometry is impossible or overlapping; using the "
        f"default {defaults.width_m * 1000:.0f}x{defaults.height_m * 1000:.0f} mm layout"
    )
    board.width_m = defaults.width_m
    board.height_m = defaults.height_m
    board.central_marker_m = defaults.central_marker_m
    board.peripheral_marker_m = defaults.peripheral_marker_m
    board.peripheral_x_m = defaults.peripheral_x_m
    board.peripheral_y_m = defaults.peripheral_y_m


def _validate_board(board: BoardConfig, name: str, warnings: list[str]) -> None:
    board.width_m = _clamp(board.width_m, 0.05, 3.0, float)
    board.height_m = _clamp(board.height_m, 0.03, 2.0, float)
    board.central_marker_m = _clamp(board.central_marker_m, 0.005, min(board.width_m, board.height_m), float)
    board.peripheral_marker_m = _clamp(board.peripheral_marker_m, 0.004, board.central_marker_m, float)
    board.peripheral_x_m = _clamp(board.peripheral_x_m, 0.005, board.width_m * 0.5, float)
    board.peripheral_y_m = _clamp(board.peripheral_y_m, 0.004, board.height_m * 0.5, float)
    board.object_position = _float_list(board.object_position, 3, [0.0, 0.0, 0.0])
    board.object_rotation = _float_list(board.object_rotation, 4, [0.0, 1.0, 0.0, 0.0])
    norm = math.sqrt(sum(value * value for value in board.object_rotation))
    if norm < 1e-6:
        board.object_rotation = [0.0, 1.0, 0.0, 0.0]
    else:
        board.object_rotation = [value / norm for value in board.object_rotation]
    if not _board_geometry_valid(board):
        _reset_board_geometry(board, warnings, name)


def validate_config(config: AppConfig) -> list[str]:
    """Clamp values and normalise lists. Returns human-readable warnings."""
    warnings: list[str] = []
    cfg = config

    cfg.driver.udp_port = int(_clamp(cfg.driver.udp_port, 1, 65535, int))
    cfg.driver.state_rate_hz = int(_clamp(cfg.driver.state_rate_hz, 30, 500, int))
    cfg.driver.pose_timeout_ms = int(_clamp(cfg.driver.pose_timeout_ms, 100, 10000, int))
    if cfg.driver.controller_profile not in ("vive_controller", "knuckles", "oculus_touch"):
        warnings.append(f"unknown controller profile {cfg.driver.controller_profile!r}; using vive_controller")
        cfg.driver.controller_profile = "vive_controller"
    cfg.driver.tracker_count = int(_clamp(cfg.driver.tracker_count, 0, 16, int))

    cfg.camera.index = int(_clamp(cfg.camera.index, 0, 32, int))
    if cfg.camera.source not in ("webcam", "synthetic"):
        warnings.append(f"unknown camera source {cfg.camera.source!r}; using webcam")
        cfg.camera.source = "webcam"
    cfg.camera.width = int(_clamp(cfg.camera.width, 320, 7680, int))
    cfg.camera.height = int(_clamp(cfg.camera.height, 240, 4320, int))
    cfg.camera.fps = int(_clamp(cfg.camera.fps, 5, 240, int))
    cfg.camera.fourcc = str(cfg.camera.fourcc or "").strip()[:4]
    cfg.camera.allow_mode_fallback = bool(cfg.camera.allow_mode_fallback)
    cfg.camera.verify_mode = bool(cfg.camera.verify_mode)
    cfg.camera.autofocus = bool(cfg.camera.autofocus)
    cfg.camera.focus = _clamp(cfg.camera.focus, -1.0, 1023.0, float)
    cfg.camera.auto_exposure = bool(cfg.camera.auto_exposure)
    cfg.camera.exposure = _clamp(cfg.camera.exposure, -1.0, 100000.0, float)
    cfg.camera.gain = _clamp(cfg.camera.gain, -1.0, 1024.0, float)
    cfg.camera.auto_white_balance = bool(cfg.camera.auto_white_balance)
    cfg.camera.white_balance = _clamp(cfg.camera.white_balance, -1.0, 10000.0, float)
    cfg.camera.auto_marker_control = bool(cfg.camera.auto_marker_control)
    cfg.camera.camera_aggressiveness = _clamp(
        cfg.camera.camera_aggressiveness, 0.0, 1.0, float
    )
    cfg.camera.camera_target_brightness = _clamp(
        cfg.camera.camera_target_brightness, 0.1, 0.9, float
    )
    cfg.camera.camera_max_gain = _clamp(cfg.camera.camera_max_gain, 0.0, 1024.0, float)
    cfg.camera.camera_timestamp_offset_ms = _clamp(
        cfg.camera.camera_timestamp_offset_ms, -1000.0, 1000.0, float
    )
    cfg.camera.camera_timestamp_source = str(
        cfg.camera.camera_timestamp_source or "auto"
    ).strip().lower()
    if cfg.camera.camera_timestamp_source not in ("auto", "pos_msec", "read"):
        warnings.append(
            f"unknown camera timestamp source {cfg.camera.camera_timestamp_source!r}; "
            "using auto"
        )
        cfg.camera.camera_timestamp_source = "auto"
    cfg.camera.position = _float_list(cfg.camera.position, 3, [0.0, 1.5, -1.5])
    cfg.camera.rotation = _float_list(cfg.camera.rotation, 4, [0.0, 1.0, 0.0, 0.0])
    dictionary = str(cfg.camera.dictionary or CUSTOM_DICTIONARY_3X3).strip().upper()
    if dictionary not in ARUCO_DICTIONARIES:
        warnings.append(
            f"unknown ArUco dictionary {cfg.camera.dictionary!r}; using "
            f"{CUSTOM_DICTIONARY_3X3}"
        )
        dictionary = CUSTOM_DICTIONARY_3X3
    cfg.camera.dictionary = dictionary
    cfg.camera.marker_detection_width = int(
        _clamp(cfg.camera.marker_detection_width, 0, 3840, int)
    )
    cfg.camera.marker_refine_pose = bool(cfg.camera.marker_refine_pose)
    cfg.camera.max_reprojection_error = _clamp(cfg.camera.max_reprojection_error, 0.1, 100.0, float)
    cfg.camera.mirror_preview = bool(cfg.camera.mirror_preview)
    cfg.camera.horizontal_fov_deg = _clamp(cfg.camera.horizontal_fov_deg, 10.0, 150.0, float)
    cfg.camera.phone_calibration_file = (
        str(cfg.camera.phone_calibration_file or "").strip()
        or "phone_calibrations.json"
    )
    cfg.camera.calibration_board_cols = int(
        _clamp(cfg.camera.calibration_board_cols, 3, 20, int)
    )
    cfg.camera.calibration_board_rows = int(
        _clamp(cfg.camera.calibration_board_rows, 3, 20, int)
    )
    cfg.camera.calibration_square_mm = _clamp(
        cfg.camera.calibration_square_mm, 5.0, 200.0, float
    )
    cfg.camera.calibration_min_views = int(
        _clamp(cfg.camera.calibration_min_views, 4, 60, int)
    )
    if cfg.camera.source == "synthetic":
        # The synthetic scene is rendered with known, fixed geometry. Forcing
        # these values guarantees that marker poses match play space instead
        # of depending on a real camera calibration.
        forced = {
            "horizontal_fov_deg": (65.0, cfg.camera.horizontal_fov_deg),
            "dictionary": (CUSTOM_DICTIONARY_3X3, cfg.camera.dictionary),
            "position": ([0.0, 1.02, -2.60], list(cfg.camera.position)),
            "rotation": ([0.0, 1.0, 0.0, 0.0], list(cfg.camera.rotation)),
            "max_reprojection_error": (8.0, cfg.camera.max_reprojection_error),
            "marker_detection_width": (0, cfg.camera.marker_detection_width),
        }
        for name, (value, current) in forced.items():
            if current != value:
                warnings.append(
                    f"synthetic camera forces {name}={value!r} (was {current!r})"
                )
                setattr(cfg.camera, name, value)

    cfg.tracking.pose_max_age_s = _clamp(cfg.tracking.pose_max_age_s, 0.05, 5.0, float)
    cfg.tracking.mediapipe_max_fps = int(_clamp(cfg.tracking.mediapipe_max_fps, 1, 120, int))
    cfg.tracking.inference_width = int(_clamp(cfg.tracking.inference_width, 0, 3840, int))
    cfg.tracking.inference_height = int(_clamp(cfg.tracking.inference_height, 0, 2160, int))
    cfg.tracking.body_model = str(cfg.tracking.body_model or "full").strip().lower()
    if cfg.tracking.body_model not in ("lite", "full", "heavy"):
        warnings.append(f"unknown body model {cfg.tracking.body_model!r}; using full")
        cfg.tracking.body_model = "full"
    cfg.tracking.mediapipe_delegate = str(
        cfg.tracking.mediapipe_delegate or "auto"
    ).strip().lower()
    if cfg.tracking.mediapipe_delegate not in ("auto", "gpu", "cpu"):
        warnings.append(
            f"unknown MediaPipe delegate {cfg.tracking.mediapipe_delegate!r}; using auto"
        )
        cfg.tracking.mediapipe_delegate = "auto"
    cfg.tracking.model_dir = str(cfg.tracking.model_dir or "").strip()
    cfg.tracking.marker_offsets = _vector_dict(cfg.tracking.marker_offsets, 3)
    cfg.tracking.marker_rotation_offsets = _vector_dict(
        cfg.tracking.marker_rotation_offsets, 4
    )
    cfg.tracking.hand_offsets = _vector_dict(cfg.tracking.hand_offsets, 3)
    cfg.tracking.pose_offsets = _vector_dict(cfg.tracking.pose_offsets, 3)
    cfg.tracking.smoothing_min_cutoff = _clamp(cfg.tracking.smoothing_min_cutoff, 0.0, 100.0, float)
    cfg.tracking.smoothing_beta = _clamp(cfg.tracking.smoothing_beta, 0.0, 10.0, float)
    cfg.tracking.orientation_cutoff = _clamp(cfg.tracking.orientation_cutoff, 0.0, 100.0, float)
    cfg.tracking.pose_prediction_ms = int(_clamp(cfg.tracking.pose_prediction_ms, 0, 200, int))
    cfg.tracking.marker_hold_s = _clamp(cfg.tracking.marker_hold_s, 0.02, 2.0, float)
    cfg.tracking.marker_lost_s = _clamp(
        cfg.tracking.marker_lost_s, cfg.tracking.marker_hold_s, 10.0, float
    )
    cfg.tracking.marker_reacquire_frames = int(
        _clamp(cfg.tracking.marker_reacquire_frames, 1, 60, int)
    )
    cfg.tracking.marker_process_noise = _clamp(
        cfg.tracking.marker_process_noise, 0.05, 100.0, float
    )
    cfg.tracking.marker_angular_noise = _clamp(
        cfg.tracking.marker_angular_noise, 0.05, 400.0, float
    )
    cfg.tracking.marker_adaptive_noise = bool(cfg.tracking.marker_adaptive_noise)
    cfg.tracking.marker_static_noise_scale = _clamp(
        cfg.tracking.marker_static_noise_scale, 0.0, 1.0, float
    )
    cfg.tracking.marker_output_smoothing_hz = _clamp(
        cfg.tracking.marker_output_smoothing_hz, 0.0, 60.0, float
    )
    cfg.tracking.marker_exposure_s = _clamp(
        cfg.tracking.marker_exposure_s, 0.0, 0.05, float
    )
    cfg.tracking.marker_min_markers = int(
        _clamp(cfg.tracking.marker_min_markers, 1, 5, int)
    )
    cfg.tracking.marker_min_confidence = _clamp(
        cfg.tracking.marker_min_confidence, 0.0, 1.0, float
    )
    cfg.tracking.marker_lost_position_sigma_m = _clamp(
        cfg.tracking.marker_lost_position_sigma_m, 0.01, 10.0, float
    )
    cfg.tracking.marker_lost_orientation_sigma_deg = _clamp(
        cfg.tracking.marker_lost_orientation_sigma_deg, 0.1, 180.0, float
    )
    cfg.tracking.marker_reinitialize_gap_s = _clamp(
        cfg.tracking.marker_reinitialize_gap_s, 0.0, 5.0, float
    )
    cfg.tracking.body_min_visibility = _clamp(cfg.tracking.body_min_visibility, 0.0, 1.0, float)
    cfg.tracking.landmark_cutoff = _clamp(cfg.tracking.landmark_cutoff, 0.0, 100.0, float)
    cfg.tracking.landmark_beta = _clamp(cfg.tracking.landmark_beta, 0.0, 10.0, float)
    cfg.tracking.hand_trigger_pinch = _clamp(cfg.tracking.hand_trigger_pinch, 0.05, 0.95, float)
    cfg.tracking.manual_hmd_position = _float_list(
        cfg.tracking.manual_hmd_position, 3, [0.0, 1.65, 0.0]
    )
    cfg.tracking.manual_hmd_rotation = _float_list(
        cfg.tracking.manual_hmd_rotation, 4, [0.0, 0.0, 0.0, 1.0]
    )
    cfg.tracking.controller_offset_left = _float_list(
        cfg.tracking.controller_offset_left, 3, [-0.25, 1.25, -0.35]
    )
    cfg.tracking.controller_offset_right = _float_list(
        cfg.tracking.controller_offset_right, 3, [0.25, 1.25, -0.35]
    )
    cfg.tracking.body_scale = _clamp(cfg.tracking.body_scale, 0.2, 3.0, float)
    cfg.tracking.body_position_offset = _float_list(
        cfg.tracking.body_position_offset, 3, [0.0, 0.0, 0.0]
    )
    allowed_roles = {
        "waist",
        "chest",
        "left_foot",
        "right_foot",
        "left_shoulder",
        "right_shoulder",
        "left_hand",
        "right_hand",
    }
    cfg.tracking.tracker_roles = [
        role if role in allowed_roles else "waist" for role in cfg.tracking.tracker_roles[:16]
    ]
    while len(cfg.tracking.tracker_roles) < 16:
        cfg.tracking.tracker_roles.append("waist")
    for board_name in ("hmd", "left", "right"):
        _validate_board(
            getattr(cfg.tracking.boards, board_name), board_name, warnings
        )

    capacity = DICTIONARY_CAPACITY.get(cfg.camera.dictionary, 0)
    default_bases = {"hmd": 0, "left": 5, "right": 10}
    bases: dict[str, int] = {}
    for name in ("hmd", "left", "right"):
        bases[name] = int(
            _clamp(
                getattr(cfg.tracking, f"{name}_marker_id"),
                0,
                max(0, capacity - 1),
                int,
            )
        )
    spans = {name: set(range(base, base + 5)) for name, base in bases.items()}
    invalid_boards = capacity < 15 or any(
        max(span) >= capacity for span in spans.values()
    )
    overlapping = len(spans["hmd"] | spans["left"] | spans["right"]) != 15
    if invalid_boards or overlapping:
        warnings.append(
            "the HMD, left and right boards need three disjoint five-ID ranges "
            f"inside {cfg.camera.dictionary} (IDs 0-{max(0, capacity - 1)}); "
            "using 0-14"
        )
        if capacity < 15:
            cfg.camera.dictionary = CUSTOM_DICTIONARY_3X3
            capacity = 15
        bases = dict(default_bases)
    for name, base in bases.items():
        setattr(cfg.tracking, f"{name}_marker_id", base)

    cfg.phone.imu_port = int(_clamp(cfg.phone.imu_port, 1, 65535, int))
    cfg.phone.video_port = int(_clamp(cfg.phone.video_port, 1, 65535, int))
    cfg.phone.camera_enabled = bool(cfg.phone.camera_enabled)
    cfg.phone.camera_port = int(_clamp(cfg.phone.camera_port, 1, 65535, int))
    cfg.phone.camera_bitrate_mbps = _clamp(
        cfg.phone.camera_bitrate_mbps, 1.0, 200.0, float
    )
    cfg.phone.camera_timeout_s = _clamp(cfg.phone.camera_timeout_s, 0.1, 10.0, float)
    if cfg.phone.camera_enabled and cfg.phone.camera_port in (
        cfg.phone.imu_port,
        cfg.phone.video_port,
    ):
        warnings.append(
            "phone.camera_port must differ from phone.imu_port and "
            "phone.video_port; phone camera disabled"
        )
        cfg.phone.camera_enabled = False
    if cfg.phone.layout not in ("mono", "sbs", "top_bottom"):
        warnings.append(f"unknown phone layout {cfg.phone.layout!r}; using sbs")
        cfg.phone.layout = "sbs"
    if cfg.phone.capture_source == "openxr" and cfg.phone.layout != "sbs":
        warnings.append("OpenXR eye capture is side-by-side; forcing phone layout sbs")
        cfg.phone.layout = "sbs"
    if cfg.phone.capture_source not in ("test", "screen", "camera", "game", "openxr"):
        warnings.append(f"unknown capture source {cfg.phone.capture_source!r}; using test")
        cfg.phone.capture_source = "test"
    if cfg.phone.capture_source == "openxr" and not cfg.openxr.enabled:
        warnings.append("OpenXR capture requires openxr.enabled; enabling it")
        cfg.openxr.enabled = True
    cfg.phone.width = int(_clamp(cfg.phone.width, 160, 3840, int))
    cfg.phone.height = int(_clamp(cfg.phone.height, 120, 2160, int))
    cfg.phone.fps = int(_clamp(cfg.phone.fps, 1, 120, int))
    if cfg.phone.codec not in ("auto", "jpeg", "hevc", "av1", "h264"):
        warnings.append(f"unknown phone codec {cfg.phone.codec!r}; using auto")
        cfg.phone.codec = "auto"
    if cfg.phone.encoder in ("cpu", "libx264", "software"):
        cfg.phone.encoder = "x264"
    if cfg.phone.encoder not in ("auto", "nvenc", "x264"):
        warnings.append(f"unknown phone encoder {cfg.phone.encoder!r}; using auto")
        cfg.phone.encoder = "auto"
    cfg.phone.bitrate_mbps = _clamp(cfg.phone.bitrate_mbps, 1.0, 200.0, float)
    cfg.phone.keyframe_interval_s = _clamp(cfg.phone.keyframe_interval_s, 0.1, 10.0, float)
    cfg.phone.jpeg_quality = int(_clamp(cfg.phone.jpeg_quality, 20, 95, int))
    lens = cfg.phone.lens
    lens.lens_separation_mm = _clamp(lens.lens_separation_mm, 20.0, 120.0, float)
    lens.eye_screen_distance_mm = _clamp(lens.eye_screen_distance_mm, 10.0, 200.0, float)
    lens.screen_width_mm = _clamp(lens.screen_width_mm, 0.0, 400.0, float)
    lens.screen_height_mm = _clamp(lens.screen_height_mm, 0.0, 400.0, float)
    lens.fov_deg = _clamp(lens.fov_deg, 30.0, 150.0, float)
    lens.distortion_k1 = _clamp(lens.distortion_k1, -1.0, 1.0, float)
    lens.distortion_k2 = _clamp(lens.distortion_k2, -1.0, 1.0, float)
    lens.chromatic_aberration = _clamp(lens.chromatic_aberration, 0.0, 0.1, float)
    lens.image_scale = _clamp(lens.image_scale, 0.5, 1.5, float)
    lens.eye_offset_x = _clamp(lens.eye_offset_x, -0.25, 0.25, float)
    lens.eye_offset_y = _clamp(lens.eye_offset_y, -0.25, 0.25, float)
    lens.overscan = _clamp(lens.overscan, 0.0, 0.25, float)
    cfg.phone.capture_region = [int(v) for v in cfg.phone.capture_region[:4]]
    if cfg.phone.capture_region and len(cfg.phone.capture_region) != 4:
        cfg.phone.capture_region = []

    cfg.gamepad.synthetic = bool(cfg.gamepad.synthetic)
    cfg.gamepad.poll_hz = int(_clamp(cfg.gamepad.poll_hz, 50, 1000, int))
    cfg.gamepad.hotplug_interval_s = _clamp(
        cfg.gamepad.hotplug_interval_s, 0.25, 10.0, float
    )
    cfg.gamepad.deadzone = _clamp(cfg.gamepad.deadzone, 0.0, 0.6, float)
    cfg.gamepad.trigger_threshold = _clamp(cfg.gamepad.trigger_threshold, 0.05, 0.95, float)
    cfg.gamepad.left_index = int(_clamp(cfg.gamepad.left_index, 0, 15, int))
    cfg.gamepad.right_index = int(_clamp(cfg.gamepad.right_index, 0, 15, int))
    if cfg.gamepad.left_index == cfg.gamepad.right_index and not cfg.gamepad.synthetic:
        warnings.append("left and right gamepads use the same index")
    if cfg.gamepad.controller_profile == "auto":
        cfg.gamepad.controller_profile = cfg.driver.controller_profile

    cfg.openxr.enabled = bool(cfg.openxr.enabled)
    cfg.openxr.manage_service = bool(cfg.openxr.manage_service)
    cfg.openxr.remote_port = int(_clamp(cfg.openxr.remote_port, 1, 65535, int))
    cfg.openxr.capture_port = int(_clamp(cfg.openxr.capture_port, 1, 65535, int))
    cfg.openxr.send_rate_hz = int(_clamp(cfg.openxr.send_rate_hz, 20, 250, int))
    cfg.openxr.view_width = int(_clamp(cfg.openxr.view_width, 320, 7680, int))
    cfg.openxr.view_height = int(_clamp(cfg.openxr.view_height, 240, 4320, int))
    cfg.openxr.ipd_m = _clamp(cfg.openxr.ipd_m, 0.03, 0.1, float)
    cfg.openxr.remote_host = str(cfg.openxr.remote_host or "127.0.0.1")
    cfg.openxr.capture_host = str(cfg.openxr.capture_host or "127.0.0.1")
    if cfg.openxr.capture_port == cfg.openxr.remote_port:
        warnings.append("OpenXR capture and remote ports are the same; using 39102/4242")
        cfg.openxr.capture_port = 39102
        cfg.openxr.remote_port = 4242

    cfg.ui.preview_width = int(_clamp(cfg.ui.preview_width, 240, 1920, int))
    cfg.ui.preview_fps = int(_clamp(cfg.ui.preview_fps, 1, 60, int))
    return warnings
