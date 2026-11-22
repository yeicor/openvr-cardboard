"""Persistent desktop configuration."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .camera_modes import DEFAULT_CAMERA_MODE

APP_NAME = "openvrcardboard"
CONFIG_VERSION = 7

# The project ships a custom 3x3 dictionary. Its larger modules survive motion
# blur and distance better than any predefined 4x4/5x5 dictionary, and fifteen
# codes cover the HMD and both hand boards exactly.
CUSTOM_DICTIONARY_3X3 = "DICT_3X3_15"

# Dictionaries accepted by the ArUco detector. Keep this list in sync with the
# UI and with OpenCV's cv2.aruco module.
ARUCO_DICTIONARIES = (
    CUSTOM_DICTIONARY_3X3,
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16H5",
    "DICT_APRILTAG_25H9",
    "DICT_APRILTAG_36H10",
    "DICT_APRILTAG_36H11",
)

DICTIONARY_CAPACITY = {
    CUSTOM_DICTIONARY_3X3: 15,
    "DICT_4X4_50": 50,
    "DICT_4X4_100": 100,
    "DICT_4X4_250": 250,
    "DICT_4X4_1000": 1000,
    "DICT_5X5_50": 50,
    "DICT_5X5_100": 100,
    "DICT_5X5_250": 250,
    "DICT_5X5_1000": 1000,
    "DICT_6X6_50": 50,
    "DICT_6X6_100": 100,
    "DICT_6X6_250": 250,
    "DICT_6X6_1000": 1000,
    "DICT_ARUCO_ORIGINAL": 1024,
    "DICT_APRILTAG_16H5": 30,
    "DICT_APRILTAG_25H9": 35,
    "DICT_APRILTAG_36H10": 2320,
    "DICT_APRILTAG_36H11": 587,
}


def default_config_dir() -> Path:
    override = os.environ.get("OPENVRCARDBOARD_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "OpenVR Cardboard"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenVR Cardboard"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "openvrcardboard"


def default_config_path() -> Path:
    return default_config_dir() / "config.json"


@dataclass
class DriverConfig:
    udp_host: str = "127.0.0.1"
    udp_port: int = 39100
    state_rate_hz: int = 250
    pose_timeout_ms: int = 1000
    enable_virtual_hmd: bool = True
    controller_profile: str = "vive_controller"
    tracker_count: int = 6


# 3x3 markers need one full module (size / 5) of white quiet zone between
# neighbours.  The 112 mm board height leaves 12 mm between the peripheral
# pairs and 10 mm to the cut line at the default 40 mm marker size.
DEFAULT_BOARD_WIDTH_M = 0.200
DEFAULT_BOARD_HEIGHT_M = 0.112
DEFAULT_BOARD_CENTRAL_M = 0.080
DEFAULT_BOARD_PERIPHERAL_M = 0.040
DEFAULT_BOARD_PERIPHERAL_X_M = 0.070
DEFAULT_BOARD_PERIPHERAL_Y_M = 0.026


@dataclass
class BoardConfig:
    """Physical geometry of one rigid five-marker board.

    All distances are metres measured from the board origin at the centre of
    the large marker: ``+X`` right, ``+Y`` up and ``+Z`` out of the printed
    face. The four peripheral markers sit at ``(+/-x, +/-y)``.
    """

    width_m: float = DEFAULT_BOARD_WIDTH_M
    height_m: float = DEFAULT_BOARD_HEIGHT_M
    central_marker_m: float = DEFAULT_BOARD_CENTRAL_M
    peripheral_marker_m: float = DEFAULT_BOARD_PERIPHERAL_M
    peripheral_x_m: float = DEFAULT_BOARD_PERIPHERAL_X_M
    peripheral_y_m: float = DEFAULT_BOARD_PERIPHERAL_Y_M
    # Pose of the board frame in the tracked object frame. Calibrate these per
    # device; the default puts the board origin at the object origin, facing
    # forward (board +Z is the printed face, device forward is -Z).
    object_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    object_rotation: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 0.0])


@dataclass
class BoardsConfig:
    hmd: BoardConfig = field(default_factory=BoardConfig)
    left: BoardConfig = field(default_factory=BoardConfig)
    right: BoardConfig = field(default_factory=BoardConfig)


@dataclass
class CameraConfig:
    source: str = "webcam"  # webcam, synthetic
    index: int = 0
    # Start at the top of the shared fallback ladder. Cameras that cannot
    # deliver 1080p120 are retried at 720p120, 1080p90, ... before the
    # negotiated mode is accepted.
    width: int = DEFAULT_CAMERA_MODE.width
    height: int = DEFAULT_CAMERA_MODE.height
    fps: int = DEFAULT_CAMERA_MODE.fps
    backend: str = "auto"  # auto, v4l2, dshow, msmf, any
    # MJPG is what makes 1080p60 possible on most USB 2/3 webcams. It must be
    # selected *before* the resolution and frame rate; the driver negotiates
    # its mode list from the pixel format.
    fourcc: str = "MJPG"
    allow_mode_fallback: bool = True
    verify_mode: bool = True
    autofocus: bool = True
    focus: float = -1.0
    auto_exposure: bool = True
    exposure: float = -1.0
    gain: float = -1.0
    # Marker-aware automatic camera control. When enabled it replaces the
    # camera's own exposure loop: the capture thread shortens exposure for the
    # least motion blur that still detects every marker the scene offers, and
    # raises gain just enough to keep the image usable. Aggressiveness scales
    # both the step size and how often the loop acts; 0 disables adaptation
    # and leaves the manual/automatic controls untouched.
    auto_marker_control: bool = True
    camera_aggressiveness: float = 0.5
    camera_target_brightness: float = 0.52
    camera_max_gain: float = 64.0
    # Constant correction subtracted from the camera timestamp. Positive
    # values move poses earlier (towards the start of exposure). POS_MSEC is
    # used when the backend provides a usable clock; otherwise the timestamp
    # is taken from the read call and corrected by this value.
    camera_timestamp_offset_ms: float = 0.0
    camera_timestamp_source: str = "auto"  # auto, pos_msec, read
    auto_white_balance: bool = True
    white_balance: float = -1.0
    calibration_file: str = "camera_calibration.json"
    # Phone cameras are calibrated per reported camera key so switching phones
    # never overwrites another device's intrinsics.
    phone_calibration_file: str = "phone_calibrations.json"
    horizontal_fov_deg: float = 60.0
    # Camera pose in play space. Identity rotation looks down -Z with +Y up,
    # matching OpenVR HMD axes. The default camera sits in front of the player
    # and looks back toward the origin.
    position: list[float] = field(default_factory=lambda: [0.0, 1.5, -1.5])
    rotation: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 0.0])
    # The shipped 3x3 dictionary has exactly fifteen codes: five for each of
    # the HMD, left hand and right hand. Body tracking is MediaPipe-only.
    dictionary: str = CUSTOM_DICTIONARY_3X3
    # ArUco detection runs at full capture resolution by default: the 40 mm
    # peripheral markers are only ~33 px wide at 2 m with a 60-degree lens, so
    # downscaling loses them. 0 disables the downscale.
    marker_detection_width: int = 0
    marker_refine_pose: bool = True
    max_reprojection_error: float = 4.0
    mirror_preview: bool = False
    # Camera calibration board. The defaults match the 9x6-inner-corner
    # chessboard described by the calibration wizard and docs/SETUP.md.
    calibration_board_cols: int = 9
    calibration_board_rows: int = 6
    calibration_square_mm: float = 25.0
    calibration_min_views: int = 12


@dataclass
class TrackingConfig:
    markers_enabled: bool = True
    hands_enabled: bool = False
    body_enabled: bool = True
    hmd_source: str = "auto"  # auto, phone, markers, body, manual
    left_source: str = "auto"  # auto, markers, hands, body, synthetic
    right_source: str = "auto"
    hmd_marker_id: int = 0
    left_marker_id: int = 5
    right_marker_id: int = 10
    # Each device board uses five consecutive IDs starting at the value above;
    # the first ID is the large central marker and the key used by fusion.
    # The three five-ID ranges cover the entire 15-code dictionary. Body
    # trackers are driven by MediaPipe and never claim marker IDs.
    boards: BoardsConfig = field(default_factory=BoardsConfig)
    tracker_roles: list[str] = field(
        default_factory=lambda: [
            "waist",
            "chest",
            "left_foot",
            "right_foot",
            "left_shoulder",
            "right_shoulder",
        ]
    )
    marker_offsets: dict[str, list[float]] = field(default_factory=dict)
    marker_rotation_offsets: dict[str, list[float]] = field(default_factory=dict)
    hand_offsets: dict[str, list[float]] = field(default_factory=dict)
    hand_swap: bool = False
    mediapipe_max_fps: int = 30
    # MediaPipe's models see a 256x256 crop internally, so running inference on
    # a 1080p frame only costs RGB conversion and cache misses. 960 px keeps
    # small hands detectable while making 60 FPS inference practical.
    inference_width: int = 960
    inference_height: int = 0  # 0 = preserve aspect ratio
    # MediaPipe Tasks model selection. The hand landmarker has a single model;
    # the pose landmarker trades latency for accuracy with lite/full/heavy.
    body_model: str = "full"  # lite, full, heavy
    mediapipe_delegate: str = "auto"  # auto, gpu, cpu
    model_dir: str = ""  # empty = the per-user cache or bundled models
    pose_max_age_s: float = 0.25
    # One-Euro pose filtering for MediaPipe hands/body. Board-tracked devices
    # are filtered once by their own error-state Kalman filter, so the composer
    # does not run One Euro on top of a board pose.
    smoothing_min_cutoff: float = 2.0
    smoothing_beta: float = 0.25
    orientation_cutoff: float = 4.0
    # The driver is the single prediction owner. Poses are sent at their
    # measured timestamps and the driver applies this horizon through
    # ``poseTimeOffset``; no source adds its own hidden prediction horizon.
    pose_prediction_ms: int = 35
    # Board tracker: hold/lost timeouts, state-machine reacquisition, filter
    # noise and measurement quality thresholds.
    marker_hold_s: float = 0.22
    marker_lost_s: float = 1.0
    marker_reacquire_frames: int = 3
    marker_process_noise: float = 1.0
    marker_angular_noise: float = 2.0
    # The filter normally uses the full process noise while the board moves
    # and scales it down when the object is nearly still.  This removes most
    # static jitter without adding lag to real motion.
    marker_adaptive_noise: bool = True
    marker_static_noise_scale: float = 0.2
    # Optional extra One-Euro smoothing of the final board pose.  It is off
    # by default because the Kalman filter already smooths the signal; users
    # who prefer a softer image can enable it and accept the added latency.
    marker_output_smoothing_hz: float = 0.0
    marker_exposure_s: float = 0.005
    marker_min_markers: int = 3
    marker_min_confidence: float = 0.25
    marker_lost_position_sigma_m: float = 0.5
    marker_lost_orientation_sigma_deg: float = 30.0
    marker_reinitialize_gap_s: float = 0.30
    body_min_visibility: float = 0.45
    landmark_cutoff: float = 6.0
    landmark_beta: float = 0.6
    hand_inputs: bool = True
    hand_trigger_pinch: float = 0.35
    manual_hmd_position: list[float] = field(default_factory=lambda: [0.0, 1.65, 0.0])
    manual_hmd_rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    controller_offset_left: list[float] = field(default_factory=lambda: [-0.25, 1.25, -0.35])
    controller_offset_right: list[float] = field(default_factory=lambda: [0.25, 1.25, -0.35])
    phone_auto_recenter: bool = True
    phone_recenter_on_start: bool = True
    body_scale: float = 1.0
    body_position_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pose_offsets: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class LensConfig:
    """Cardboard lens/display geometry sent to the phone.

    These values let the phone align the two eye images with the physical
    lenses. They do not change tracking; they only affect the displayed image.
    """

    lens_separation_mm: float = 63.0
    eye_screen_distance_mm: float = 45.0
    screen_width_mm: float = 0.0  # 0 = use the phone's reported panel size
    screen_height_mm: float = 0.0
    fov_deg: float = 90.0
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    chromatic_aberration: float = 0.0
    image_scale: float = 1.0
    eye_offset_x: float = 0.0  # fraction of one eye width
    eye_offset_y: float = 0.0  # fraction of the eye height
    overscan: float = 0.0


@dataclass
class PhoneConfig:
    imu_port: int = 39101
    video_enabled: bool = True
    video_port: int = 8765
    # Phone-as-webcam: the phone uploads H.264 access units to this WebSocket
    # port and the desktop uses them as the tracking camera while connected.
    camera_enabled: bool = True
    camera_port: int = 8766
    camera_bitrate_mbps: float = 40.0
    # Preferred phone-camera mode sent in camera_welcome. The phone applies the
    # same fallback ladder, starting from the closest entry when this target
    # cannot be opened.
    camera_width: int = DEFAULT_CAMERA_MODE.width
    camera_height: int = DEFAULT_CAMERA_MODE.height
    camera_fps: int = DEFAULT_CAMERA_MODE.fps
    # Fall back to the configured camera when no phone frame arrived for this
    # long (covers Wi-Fi stalls as well as an explicit disconnect).
    camera_timeout_s: float = 0.5
    token: str = ""
    layout: str = "sbs"  # mono, sbs, top_bottom
    capture_source: str = "test"  # test, screen, camera, game, openxr
    capture_monitor: int = 1
    capture_region: list[int] = field(default_factory=list)  # x, y, w, h; empty = monitor
    # Per-eye dimensions for stereo layouts; full frame dimensions for mono.
    # 1200x1080 per eye is 2400x1080 in the panel once side-by-side, which is
    # the native landscape resolution of a modern 1080p phone.
    width: int = 1200
    height: int = 1080
    fps: int = 90
    # auto uses NVENC HEVC/AV1 when an NVIDIA GPU and ffmpeg are available and
    # falls back to the JPEG path everywhere else (including CI). Explicit
    # h264 can use the CPU libx264 backend when phone.encoder allows it.
    codec: str = "auto"  # auto, jpeg, hevc, av1, h264
    bitrate_mbps: float = 40.0
    keyframe_interval_s: float = 1.0
    encoder: str = "auto"  # auto, nvenc, x264
    jpeg_quality: int = 65
    test_pattern: str = "sbs"  # sbs, grid, bars
    camera_index: int = 1
    stream_on_launch: bool = False
    lens: LensConfig = field(default_factory=LensConfig)


@dataclass
class GamepadConfig:
    enabled: bool = True
    backend: str = "auto"  # auto, pygame, evdev
    synthetic: bool = False
    poll_hz: int = 250
    hotplug_interval_s: float = 1.0
    deadzone: float = 0.12
    trigger_threshold: float = 0.5
    left_index: int = 0
    right_index: int = 1
    rumble: bool = True
    invert_left_y: bool = True
    invert_right_y: bool = True
    controller_profile: str = "auto"


@dataclass
class OpenXRConfig:
    """Loopback links to Monado and the OpenXR capture shim."""

    enabled: bool = True
    remote_host: str = "127.0.0.1"
    remote_port: int = 4242
    capture_host: str = "127.0.0.1"
    capture_port: int = 39102
    send_rate_hz: int = 100
    view_width: int = 2400
    view_height: int = 1080
    ipd_m: float = 0.063
    # Start the bundled monado-service while the desktop app runs so other
    # OpenXR applications can use the registered runtime. Disable this when an
    # external service (for example the recorded E2E harness) owns the runtime.
    manage_service: bool = True


@dataclass
class UIConfig:
    preview_width: int = 560
    preview_fps: int = 30
    dark_theme: bool = True
    start_tracking_on_launch: bool = True
    minimize_to_tray: bool = False


@dataclass
class AppConfig:
    config_version: int = CONFIG_VERSION
    driver: DriverConfig = field(default_factory=DriverConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    gamepad: GamepadConfig = field(default_factory=GamepadConfig)
    openxr: OpenXRConfig = field(default_factory=OpenXRConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _merge_dataclass(target: Any, values: dict[str, Any]) -> None:
    for item in fields(target):
        if item.name not in values:
            continue
        current = getattr(target, item.name)
        incoming = values[item.name]
        if is_dataclass(current) and isinstance(incoming, dict):
            _merge_dataclass(current, incoming)
        elif isinstance(current, dict) and isinstance(incoming, dict):
            merged = dict(current)
            merged.update(incoming)
            setattr(target, item.name, merged)
        else:
            setattr(target, item.name, incoming)


# Validation lives in a separate module to keep this file focused on the
# dataclasses and persistence. Re-exported for existing callers.
from .config_validation import (  # noqa: E402,F401
    _clamp,
    _float_list,
    _vector_dict,
    validate_config,
)


def load_config(path: str | Path | None = None) -> tuple[AppConfig, list[str]]:
    config = AppConfig()
    warnings: list[str] = []
    target = Path(path).expanduser() if path else default_config_path()
    if target.exists():
        try:
            values = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                warnings.append(f"config root is not an object: {target}")
            else:
                try:
                    version = int(values.get("config_version", 0))
                except (TypeError, ValueError):
                    version = 0
                if version != CONFIG_VERSION:
                    warnings.append(
                        f"ignoring configuration version {version} at {target}; "
                        f"starting fresh with version {CONFIG_VERSION}"
                    )
                else:
                    _merge_dataclass(config, values)
        except (OSError, ValueError) as exc:
            warnings.append(f"could not read config {target}: {exc}")
    warnings.extend(validate_config(config))
    return config, warnings


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    validate_config(config)
    target = Path(path).expanduser() if path else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(config)
    payload["config_version"] = CONFIG_VERSION
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def config_dir_for(path: str | Path | None) -> Path:
    return (Path(path).expanduser().parent if path else default_config_path().parent)


def resolve_relative(path_value: str, base_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()
