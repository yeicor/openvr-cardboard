"""Runtime data structures shared between tracking, networking and the UI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .protocol import ControllerInput
from .tracking_state import ObjectPose


@dataclass(slots=True)
class Pose:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    timestamp: float = 0.0
    valid: bool = True
    orientation_only: bool = False
    source: str = ""

    def copy(self) -> "Pose":
        return Pose(
            position=np.array(self.position, dtype=np.float64, copy=True),
            rotation=np.array(self.rotation, dtype=np.float64, copy=True),
            velocity=np.array(self.velocity, dtype=np.float64, copy=True),
            angular_velocity=np.array(self.angular_velocity, dtype=np.float64, copy=True),
            timestamp=float(self.timestamp),
            valid=bool(self.valid),
            orientation_only=bool(self.orientation_only),
            source=self.source,
        )


@dataclass(slots=True)
class PhoneSample:
    sequence: int
    timestamp: float
    quaternion: np.ndarray
    gyroscope: np.ndarray
    acceleration: np.ndarray
    accuracy: int
    received_at: float
    source_address: str = ""
    screen_rotation: int = 0

    def copy(self) -> "PhoneSample":
        return PhoneSample(
            sequence=self.sequence,
            timestamp=float(self.timestamp),
            quaternion=np.array(self.quaternion, copy=True),
            gyroscope=np.array(self.gyroscope, copy=True),
            acceleration=np.array(self.acceleration, copy=True),
            accuracy=int(self.accuracy),
            received_at=float(self.received_at),
            source_address=self.source_address,
            screen_rotation=int(self.screen_rotation),
        )


@dataclass(slots=True)
class TrackingSnapshot:
    markers: dict[int, Pose] = field(default_factory=dict)
    # Full 6DOF filter state per board primary marker ID, already predicted to
    # the snapshot timestamp. Fusion uses this instead of smoothing the
    # 30 Hz marker pose a second time.
    tracked_objects: dict[int, ObjectPose] = field(default_factory=dict)
    hands: dict[int, Pose] = field(default_factory=dict)
    hand_inputs: dict[int, ControllerInput] = field(default_factory=dict)
    body: dict[str, Pose] = field(default_factory=dict)
    tracker_poses: dict[int, Pose] = field(default_factory=dict)
    phone: Optional[PhoneSample] = None
    phone_pose: Optional[Pose] = None
    gamepads: dict[int, ControllerInput] = field(default_factory=dict)
    gamepad_names: dict[int, str] = field(default_factory=dict)

    def copy(self) -> "TrackingSnapshot":
        return TrackingSnapshot(
            markers={key: value.copy() for key, value in self.markers.items()},
            tracked_objects={
                key: value.copy() for key, value in self.tracked_objects.items()
            },
            hands={key: value.copy() for key, value in self.hands.items()},
            hand_inputs={key: value for key, value in self.hand_inputs.items()},
            body={key: value.copy() for key, value in self.body.items()},
            tracker_poses={key: value.copy() for key, value in self.tracker_poses.items()},
            phone=self.phone.copy() if self.phone else None,
            phone_pose=self.phone_pose.copy() if self.phone_pose else None,
            gamepads=dict(self.gamepads),
            gamepad_names=dict(self.gamepad_names),
        )


@dataclass(slots=True)
class RuntimeStatus:
    """Mutable status bag copied under the shared-state lock."""

    started_at: float = field(default_factory=time.monotonic)
    camera_fps: float = 0.0
    camera_frame_ms: float = 0.0
    camera_resolution: tuple[int, int] = (0, 0)
    camera_error: str = ""
    inference_ms: float = 0.0
    marker_count: int = 0
    hands_visible: bool = False
    body_visible: bool = False
    phone_rate_hz: float = 0.0
    phone_address: str = ""
    phone_last_seen: float = 0.0
    phone_accurate: bool = True
    gamepads: list[str] = field(default_factory=list)
    openxr: dict = field(default_factory=dict)
    driver_connected: bool = False
    driver_packets: int = 0
    driver_status: dict = field(default_factory=dict)
    video_clients: int = 0
    video_fps: float = 0.0
    video_frames: int = 0
    video_last_error: str = ""
    last_error: str = ""
    notes: list[str] = field(default_factory=list)

    def copy(self) -> "RuntimeStatus":
        result = RuntimeStatus(
            started_at=self.started_at,
            camera_fps=self.camera_fps,
            camera_frame_ms=self.camera_frame_ms,
            camera_resolution=self.camera_resolution,
            camera_error=self.camera_error,
            inference_ms=self.inference_ms,
            marker_count=self.marker_count,
            hands_visible=self.hands_visible,
            body_visible=self.body_visible,
            phone_rate_hz=self.phone_rate_hz,
            phone_address=self.phone_address,
            phone_last_seen=self.phone_last_seen,
            phone_accurate=self.phone_accurate,
            gamepads=list(self.gamepads),
            openxr=dict(self.openxr),
            driver_connected=self.driver_connected,
            driver_packets=self.driver_packets,
            driver_status=dict(self.driver_status),
            video_clients=self.video_clients,
            video_fps=self.video_fps,
            video_frames=self.video_frames,
            video_last_error=self.video_last_error,
            last_error=self.last_error,
            notes=list(self.notes),
        )
        return result
