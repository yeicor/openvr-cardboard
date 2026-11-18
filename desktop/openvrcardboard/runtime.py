"""Application runtime: wires capture, tracking, fusion, driver and phone."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .calibration import camera_model_for_config, camera_pose
from .config import (
    AppConfig,
    config_dir_for,
    load_config,
    save_config,
    validate_config,
)
from .driver_link import DriverLink
from .fusion import PhonePoseEstimator, PoseComposer
from .game import GameInput
from .gamepad import GamepadManager
from .openxr import OpenXRManager
from .phone_link import PhoneImuReceiver
from .pipeline import FusionWorker, TrackingWorker
from .protocol import DeviceId
from .state import SharedState
from .types import PhoneSample, RuntimeStatus
from .video import VideoError, VideoStreamServer

LOGGER = logging.getLogger(__name__)


class Runtime:
    """Owns every background component and exposes lifecycle/status."""

    def __init__(self, config: AppConfig, config_path: Optional[Path] = None):
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.base_dir = config_dir_for(self.config_path)
        warnings: list[str] = []

        self.model, model_warnings = camera_model_for_config(
            config.camera, self.base_dir
        )
        warnings.extend(model_warnings)
        self.camera_position, self.camera_rotation = camera_pose(config.camera)

        self.state = SharedState()
        for warning in warnings:
            self.state.add_note(warning)
        self._warnings = warnings

        self.driver = DriverLink(
            host=config.driver.udp_host,
            port=config.driver.udp_port,
            rate_hz=config.driver.state_rate_hz,
            prediction_ms=config.tracking.pose_prediction_ms,
            on_haptic=self._on_haptic,
        )
        self.phone = PhoneImuReceiver(
            port=config.phone.imu_port, on_sample=self._on_phone_sample
        )
        self.estimator = PhonePoseEstimator(
            neutral_position=config.tracking.manual_hmd_position,
            recenter_on_start=config.tracking.phone_recenter_on_start,
            auto_recenter=config.tracking.phone_auto_recenter,
        )
        self.composer = PoseComposer(
            config.tracking,
            config.driver.tracker_count,
            phone_estimator=self.estimator,
        )
        self.gamepads = GamepadManager(self.state, config.gamepad)
        self.openxr = OpenXRManager(config.openxr, on_haptic=self._on_openxr_haptic)
        self.tracking = TrackingWorker(
            self.state,
            config,
            self.model,
            self.camera_position,
            self.camera_rotation,
        )
        self.fusion = FusionWorker(
            self.state,
            config,
            self.composer,
            self.estimator,
            self.driver,
            rate_hz=config.driver.state_rate_hz,
            on_frame=self.openxr.update_frame,
        )
        self.video: Optional[VideoStreamServer] = None

        self._lock = threading.RLock()
        self._running = False
        self._video_error = ""
        self._started_at = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def start(self, start_tracking: Optional[bool] = None) -> list[str]:
        """Start all components. Returns notes for anything that failed."""
        problems: list[str] = []
        with self._lock:
            if self._running:
                return problems
            self._running = True
            self._started_at = time.monotonic()

            try:
                self.driver.start()
            except OSError as exc:
                problems.append(f"driver link: {exc}")
                self.state.set_status(last_error=f"driver link: {exc}")
            try:
                self.phone.start()
            except OSError as exc:
                problems.append(f"phone IMU: {exc}")
                self.state.set_status(last_error=f"phone IMU: {exc}")

            if self.config.openxr.enabled:
                try:
                    self.openxr.start()
                except OSError as exc:
                    problems.append(f"OpenXR bridge: {exc}")
                    self.state.set_status(last_error=f"OpenXR bridge: {exc}")

            if self.config.gamepad.enabled:
                self.gamepads.start()
            else:
                self.state.add_note("gamepad emulation is disabled")

            should_track = (
                self.config.ui.start_tracking_on_launch
                if start_tracking is None
                else bool(start_tracking)
            )
            if should_track:
                try:
                    self.tracking.start()
                except Exception as exc:
                    problems.append(f"tracking: {exc}")
                    self.state.set_status(last_error=f"tracking: {exc}")

            self.fusion.start()

            if self.config.phone.stream_on_launch and self.config.phone.video_enabled:
                try:
                    self.start_video()
                except VideoError as exc:
                    problems.append(f"video: {exc}")
                    self._video_error = str(exc)
        for problem in problems:
            self.state.add_note(problem)
            LOGGER.warning("%s", problem)
        LOGGER.info("runtime started")
        return problems

    def stop(self) -> None:
        with self._lock:
            if not any(
                (
                    self._running,
                    self.tracking.running,
                    self.fusion.running,
                    self.gamepads.running,
                    self.openxr.running,
                    self.phone.local_port,
                    self.driver.local_port,
                    self.video is not None,
                )
            ):
                return
            self._running = False
            self.stop_video()
            self.fusion.stop()
            self.tracking.stop()
            self.gamepads.stop()
            self.openxr.stop()
            self.phone.stop()
            self.driver.stop()
        LOGGER.info("runtime stopped")

    def __enter__(self) -> "Runtime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Tracking controls
    # ------------------------------------------------------------------
    def start_tracking(self) -> list[str]:
        problems: list[str] = []
        try:
            self.tracking.start()
        except Exception as exc:
            problems.append(f"tracking: {exc}")
        self.state.set_hand_input_options(
            self.config.tracking.hand_inputs,
            self.config.tracking.hand_trigger_pinch,
        )
        return problems

    def stop_tracking(self) -> None:
        self.tracking.stop()
        self.state.clear_markers()
        self.state.clear_hands()
        self.state.set_body(None)
        self.state.set_preview(None)

    def recenter(self) -> None:
        """Re-zero the phone orientation and ask the phone to recenter too."""
        self.state.request_recenter()
        if self.video is not None:
            try:
                self.video.recenter()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    @property
    def video_error(self) -> str:
        return self._video_error

    def start_video(self) -> None:
        if self.config.phone.video_enabled is False:
            raise VideoError("phone video is disabled in the configuration")
        with self._lock:
            if self.video is not None and self.video.running:
                return
            server = VideoStreamServer(
                self.config.phone,
                view_provider=self._game_input,
                capture_provider=self.openxr.capture,
            )
            server.start()
            self.video = server
            self._video_error = ""
            self.state.add_note(
                f"phone video streaming on port {server.local_port}"
            )

    def stop_video(self) -> None:
        with self._lock:
            if self.video is None:
                return
            try:
                self.video.stop()
            except Exception as exc:
                LOGGER.debug("video stop failed: %s", exc)
            self.video = None

    def toggle_video(self) -> bool:
        if self.video is not None and self.video.running:
            self.stop_video()
            return False
        self.start_video()
        return True

    def _game_input(self) -> GameInput:
        orientation, steer_x, steer_y, trigger = self.state.game_inputs()
        return GameInput(
            orientation=orientation,
            steer_x=steer_x,
            steer_y=steer_y,
            trigger=trigger,
        )

    def video_stats(self):
        video = self.video
        return video.stats() if video is not None else None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def save(self) -> Path:
        target = save_config(self.config, self.config_path)
        self.state.add_note(f"configuration saved to {target}")
        return target

    def apply_config(self, config: AppConfig) -> list[str]:
        """Replace config and restart components so changes take effect."""
        warnings = validate_config(config)
        old_config = self.config
        was_running = self._running
        was_tracking = self.tracking.running
        was_video = self.video is not None and self.video.running
        # The phone IMU receiver binds its UDP port in the constructor, so the
        # previous runtime must be stopped before the replacement is built.
        self.stop()
        try:
            new_runtime = Runtime(config, self.config_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("could not rebuild runtime")
            warnings.append(f"could not apply configuration: {exc}")
            try:
                restored = Runtime(old_config, self.config_path)
            except Exception:
                LOGGER.exception("could not restore previous runtime")
                self.state.add_note(warnings[-1])
                return warnings
            self.__dict__.update(restored.__dict__)
            for warning in warnings:
                self.state.add_note(warning)
            if was_running:
                self.start(start_tracking=was_tracking)
                self._restore_video(was_video, warnings)
            return warnings
        # Move the freshly built components into this instance so callers
        # holding a Runtime reference keep working.
        self.__dict__.update(new_runtime.__dict__)
        for warning in warnings:
            self.state.add_note(warning)
        if was_running:
            # Do not silently start a camera the user had stopped; the driver,
            # phone link and video follow their own configuration.
            self.start(start_tracking=was_tracking)
            self._restore_video(was_video, warnings)
        return warnings

    def _restore_video(self, was_video: bool, warnings: list[str]) -> None:
        """Restart manually-started video after a config reload."""
        if not was_video or self.video is not None:
            return
        if not self.config.phone.video_enabled:
            return
        try:
            self.start_video()
        except VideoError as exc:
            warnings.append(f"video: {exc}")
            self.state.add_note(f"video: {exc}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> RuntimeStatus:
        status = self.state.status()
        phone_stats = self.phone.stats()
        status.phone_rate_hz = phone_stats.rate_hz
        status.phone_address = phone_stats.address
        status.phone_last_seen = phone_stats.last_seen
        status.phone_accurate = phone_stats.accurate
        link_stats = self.driver.stats()
        status.driver_connected = link_stats.connected
        status.driver_packets = link_stats.send_packets
        if link_stats.last_status is not None:
            last = link_stats.last_status
            status.driver_status = {
                "device_count": int(last.device_count),
                "uptime_ms": int(last.uptime_ms),
                "state_packets": int(last.state_packets),
                "parse_errors": int(last.parse_errors),
            }
        video_stats = self.video_stats()
        if video_stats is not None:
            status.video_clients = video_stats.clients
            status.video_fps = video_stats.fps
            status.video_frames = video_stats.frames
            status.video_last_error = video_stats.last_error
        status.gamepads = [
            f"{side}: {info['name']}"
            for side, info in self.gamepads.status().get("pads", {}).items()
        ]
        status.openxr = self.openxr.stats()
        if self._video_error:
            status.video_last_error = self._video_error
        return status

    def tracking_status(self) -> dict:
        camera = self.tracking.camera.stats()
        return {
            "running": self.tracking.running,
            "camera": {
                "fps": camera.fps,
                "frame_ms": camera.frame_ms,
                "width": camera.width,
                "height": camera.height,
                "frames": camera.frames,
                "error": camera.last_error,
                "opened": camera.opened,
                "fourcc": camera.fourcc,
                "mode_ok": camera.mode_ok,
                "requested_width": camera.requested_width,
                "requested_height": camera.requested_height,
                "requested_fps": camera.requested_fps,
                "inference_ms": self.state.status().inference_ms,
                "timestamp_source": camera.timestamp_source,
                "exposure": camera.exposure,
                "gain": camera.gain,
                "brightness": camera.brightness,
                "detected_markers": camera.detected_markers,
                "visible_markers": camera.visible_markers,
                "auto_control": camera.auto_control,
                "auto_actions": camera.auto_actions,
            },
            "markers": self.markers_available(),
            "hands": self.hands_available(),
            "body": self.body_available(),
            "marker_status": (
                self.tracking.markers.status()
                if self.tracking.markers is not None
                else {}
            ),
            "hand_status": (
                self.tracking.hands.status()
                if self.tracking.hands is not None
                else {}
            ),
            "body_status": (
                self.tracking.body.status()
                if self.tracking.body is not None
                else {}
            ),
            "gamepads": self.gamepads.status(),
            "phone": {
                "rate_hz": self.phone.stats().rate_hz,
                "address": self.phone.stats().address,
                "samples": self.phone.stats().samples,
            },
            "driver": {
                "connected": self.driver.stats().connected,
                "packets": self.driver.stats().send_packets,
                "prediction_ms": self.driver.stats().prediction_ms,
            },
            "openxr": self.openxr.stats(),
        }

    def markers_available(self) -> bool:
        return self.tracking.markers is not None

    def hands_available(self) -> bool:
        return self.tracking.hands is not None

    def body_available(self) -> bool:
        return self.tracking.body is not None

    def diagnostics(self) -> dict:
        """JSON-safe snapshot used by ``--status-json`` and the E2E harness."""
        now = time.monotonic()
        status = self.status()
        snapshot = self.state.snapshot(now)
        frame = self.fusion.last_frame

        def pose_dict(pose) -> Optional[dict]:
            if pose is None:
                return None
            return {
                "position": [round(float(value), 5) for value in pose.position],
                "rotation": [round(float(value), 5) for value in pose.rotation],
                "valid": bool(pose.valid),
                "orientation_only": bool(pose.orientation_only),
                "source": pose.source,
                "age_s": round(max(0.0, now - pose.timestamp), 3),
            }

        markers = {}
        for marker_id, pose in snapshot.markers.items():
            markers[str(marker_id)] = pose_dict(pose)
        body = {
            name: pose_dict(pose) for name, pose in snapshot.body.items()
        }
        inputs = {}
        for device_id, item in snapshot.gamepads.items():
            inputs[str(device_id)] = dataclasses.asdict(item)
        composed = None
        if frame is not None:
            composed = {
                "hmd": pose_dict(frame.hmd),
                "left": pose_dict(frame.left_controller),
                "right": pose_dict(frame.right_controller),
                "trackers": {
                    str(device_id): pose_dict(pose)
                    for device_id, pose in frame.trackers.items()
                },
                "inputs": {
                    str(device_id): dataclasses.asdict(item)
                    for device_id, item in frame.inputs.items()
                },
                "sources": dict(frame.sources),
            }
        capture_events = self.openxr.capture.events()
        return {
            "time": time.time(),
            "monotonic": now,
            "status": dataclasses.asdict(status),
            "tracking": self.tracking_status(),
            "openxr": self.openxr.stats(),
            "capture_events": capture_events[-40:],
            "markers": markers,
            "marker_details": self.state.marker_details(),
            "body": body,
            "gamepad_inputs": inputs,
            "composed": composed,
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_haptic(self, event) -> None:
        try:
            self.gamepads.rumble(
                event.device_id, event.amplitude, event.frequency, event.duration
            )
        except Exception:
            LOGGER.debug("could not route haptic event", exc_info=True)

    def _on_openxr_haptic(self, hand: str, amplitude: float, duration: float) -> None:
        device_id = DeviceId.LEFT if hand == "left" else DeviceId.RIGHT
        try:
            self.gamepads.rumble(device_id, amplitude, 0.0, duration)
        except Exception:
            LOGGER.debug("could not route OpenXR haptic", exc_info=True)

    def _on_phone_sample(self, sample: PhoneSample) -> None:
        self.state.set_phone(sample)


def create_runtime(path: Optional[str] = None) -> Runtime:
    config, warnings = load_config(path)
    runtime = Runtime(config, Path(path) if path else None)
    for warning in warnings:
        runtime.state.add_note(warning)
    return runtime
