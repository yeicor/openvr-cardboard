"""Thread-safe shared state between producers and the fusion worker.

Producers (camera/marker tracker, MediaPipe workers, phone IMU receiver and
gamepad polling thread) publish immutable snapshots. The 250 Hz fusion worker
and the UI read a consistent view without blocking the producers for long.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from .markers import MarkerDetection
from .mathutil import clamp, normalize
from .pose_tracking import BodyObservation, HandObservation
from .protocol import Buttons, ControllerInput, DeviceId
from .types import PhoneSample, Pose, RuntimeStatus, TrackingSnapshot

GRIP_CLICK_THRESHOLD = 0.6


def _pose(
    position, rotation, timestamp: float, source: str, orientation_only: bool = False
) -> Pose:
    return Pose(
        position=np.asarray(position, dtype=np.float64).reshape(3).copy(),
        rotation=normalize(rotation),
        timestamp=float(timestamp),
        valid=True,
        orientation_only=bool(orientation_only),
        source=source,
    )


class SharedState:
    """Latest observations from every producer."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._markers: dict[int, MarkerDetection] = {}
        self._marker_time = 0.0
        self._board_tracker = None
        self._hands: dict[str, HandObservation] = {}
        self._hand_time = 0.0
        self._hand_inputs_enabled = True
        self._hand_trigger_pinch = 0.35
        self._body: Optional[BodyObservation] = None
        self._body_time = 0.0
        self._phone: Optional[PhoneSample] = None
        self._phone_pose: Optional[Pose] = None
        self._gamepads: dict[int, ControllerInput] = {}
        self._gamepad_names: dict[int, str] = {}
        self._gamepad_time = 0.0
        self._preview: Optional[np.ndarray] = None
        self._preview_time = 0.0
        self._preview_sequence = 0
        self._status = RuntimeStatus()
        self._recenter_requested = False
        self._recenter_count = 0

    # ------------------------------------------------------------------
    # Producers
    # ------------------------------------------------------------------
    def set_markers(
        self, detections: dict[int, MarkerDetection], timestamp: Optional[float] = None
    ) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._markers = {int(key): value for key, value in detections.items()}
            self._marker_time = stamp
            self._status.marker_count = len(self._markers)

    def set_marker_tracker(self, tracker) -> None:
        """Attach the live board tracker so snapshots can sample it at 250 Hz.

        The board tracker is only asked for its state at the snapshot time;
        the configured prediction horizon belongs to the driver so it is never
        applied twice.
        """
        with self._lock:
            self._board_tracker = tracker

    def clear_markers(self) -> None:
        with self._lock:
            self._markers = {}
            self._marker_time = 0.0
            self._board_tracker = None
            self._status.marker_count = 0

    def marker_details(self) -> dict[str, dict]:
        """Per-board reprojection, state and uncertainty evidence."""
        with self._lock:
            tracker = self._board_tracker
            detections = dict(self._markers)
        result: dict[str, dict] = {
            str(int(marker_id)): {
                "position": [round(float(value), 5) for value in detection.position],
                "rotation": [round(float(value), 5) for value in detection.rotation],
                "reprojection_error": round(float(detection.reprojection_error), 4),
                "distance": round(float(detection.distance), 4),
                "valid": bool(detection.valid),
                "state": detection.state.value,
                "confidence": round(float(detection.confidence), 3),
                "marker_count": int(detection.marker_count),
                "board": detection.board,
            }
            for marker_id, detection in detections.items()
        }
        if tracker is not None:
            for marker_id, tracker_object in tracker.object_states().items():
                status = tracker_object.status()
                entry = result.setdefault(str(int(marker_id)), {})
                entry.update(status)
        return result

    def set_hands(
        self, observations: dict[str, HandObservation], timestamp: Optional[float] = None
    ) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._hands = dict(observations)
            self._hand_time = stamp
            self._status.hands_visible = bool(observations)

    def clear_hands(self) -> None:
        with self._lock:
            self._hands = {}
            self._hand_time = 0.0
            self._status.hands_visible = False

    def set_hand_input_options(self, enabled: bool, trigger_pinch: float) -> None:
        with self._lock:
            self._hand_inputs_enabled = bool(enabled)
            self._hand_trigger_pinch = clamp(float(trigger_pinch), 0.05, 0.95)

    def set_body(
        self, observation: Optional[BodyObservation], timestamp: Optional[float] = None
    ) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._body = observation
            self._body_time = stamp
            self._status.body_visible = observation is not None

    def set_phone(self, sample: Optional[PhoneSample]) -> None:
        with self._lock:
            self._phone = sample.copy() if sample is not None else None

    def set_phone_pose(self, pose: Optional[Pose]) -> None:
        with self._lock:
            self._phone_pose = pose.copy() if pose is not None else None

    def set_gamepads(
        self,
        inputs: dict[int, ControllerInput],
        names: Optional[dict[int, str]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._gamepads = {int(key): value for key, value in inputs.items()}
            if names is not None:
                self._gamepad_names = {
                    int(key): str(value) for key, value in names.items()
                }
            self._gamepad_time = stamp
            self._status.gamepads = [
                self._gamepad_names.get(int(key), f"gamepad {key}")
                for key in sorted(self._gamepads)
            ]

    def set_preview(self, image: Optional[np.ndarray], timestamp: Optional[float] = None) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._preview = image
            self._preview_time = stamp
            self._preview_sequence += 1

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def set_status(self, **values) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self._status, key):
                    setattr(self._status, key, value)

    def add_note(self, note: str) -> None:
        with self._lock:
            if note and note not in self._status.notes:
                self._status.notes.append(note)
                del self._status.notes[:-12]

    def status(self) -> RuntimeStatus:
        with self._lock:
            return self._status.copy()

    # ------------------------------------------------------------------
    # Recenter
    # ------------------------------------------------------------------
    def request_recenter(self) -> None:
        with self._lock:
            self._recenter_requested = True
            self._recenter_count += 1

    def take_recenter(self) -> bool:
        with self._lock:
            requested = self._recenter_requested
            self._recenter_requested = False
            return requested

    @property
    def recenter_count(self) -> int:
        with self._lock:
            return self._recenter_count

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------
    def preview(self) -> tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            return self._preview, self._preview_time, self._preview_sequence

    def phone(self) -> Optional[PhoneSample]:
        with self._lock:
            return self._phone.copy() if self._phone is not None else None

    def game_inputs(self):
        """Return the phone quaternion and left-pad steering for the demo game."""
        with self._lock:
            orientation = (
                np.array(self._phone.quaternion, copy=True)
                if self._phone is not None
                else None
            )
            left = self._gamepads.get(int(DeviceId.LEFT))
            return (
                orientation,
                float(left.joystick_x) if left is not None else 0.0,
                float(left.joystick_y) if left is not None else 0.0,
                float(left.trigger) if left is not None else 0.0,
            )

    def snapshot(self, now: Optional[float] = None) -> TrackingSnapshot:
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            detections = dict(self._markers)
            marker_time = self._marker_time
            board_tracker = self._board_tracker
            hands_observations = dict(self._hands)
            hand_time = self._hand_time
            hand_inputs_enabled = self._hand_inputs_enabled
            hand_trigger_pinch = self._hand_trigger_pinch
            body_observation = self._body
            body_time = self._body_time
            phone = self._phone.copy() if self._phone is not None else None
            phone_pose = (
                self._phone_pose.copy() if self._phone_pose is not None else None
            )
            gamepads = dict(self._gamepads)
            gamepad_names = dict(self._gamepad_names)

        markers: dict[int, Pose] = {
            marker_id: _pose(
                detection.position,
                detection.rotation,
                marker_time,
                "markers",
            )
            for marker_id, detection in detections.items()
            if detection.valid
        }
        tracked_objects = {}
        if board_tracker is not None:
            # Predict outside the shared-state lock: the tracking thread takes
            # the tracker lock before publishing detections, so holding both in
            # the opposite order would deadlock.
            tracked_objects = {}
            markers = {}
            for marker_id, tracker_object in board_tracker.object_states().items():
                pose = tracker_object.pose_at(stamp)
                if pose is None:
                    continue
                tracked_objects[int(marker_id)] = pose
                if not pose.valid:
                    continue
                markers[int(marker_id)] = Pose(
                    position=np.array(pose.position, copy=True),
                    rotation=np.array(pose.rotation, copy=True),
                    velocity=np.array(pose.velocity, copy=True),
                    angular_velocity=np.array(pose.angular_velocity, copy=True),
                    timestamp=float(pose.timestamp),
                    valid=True,
                    source="markers",
                )

        hands: dict[int, Pose] = {}
        hand_inputs: dict[int, ControllerInput] = {}
        for side, observation in hands_observations.items():
            if not observation.valid:
                continue
            device_id = int(DeviceId.LEFT if side == "left" else DeviceId.RIGHT)
            hands[device_id] = _pose(
                observation.position,
                observation.rotation,
                hand_time,
                "hands",
            )
            if hand_inputs_enabled:
                hand_inputs[device_id] = self._hand_input(
                    device_id, observation, hand_trigger_pinch
                )
        body: dict[str, Pose] = {}
        if body_observation is not None and body_observation.valid:
            for role, position in body_observation.joints.items():
                rotation = body_observation.rotations.get(
                    role, (0.0, 0.0, 0.0, 1.0)
                )
                body[role] = _pose(position, rotation, body_time, "body")
        return TrackingSnapshot(
            markers=markers,
            tracked_objects=tracked_objects,
            hands=hands,
            hand_inputs=hand_inputs,
            body=body,
            phone=phone,
            phone_pose=phone_pose,
            gamepads=gamepads,
            gamepad_names=gamepad_names,
        )

    def _hand_input(
        self, device_id: int, observation: HandObservation, trigger_pinch: float
    ) -> ControllerInput:
        pinch = clamp(float(observation.pinch), 0.0, 1.0)
        grip = clamp(float(observation.grip), 0.0, 1.0)
        buttons = Buttons(0)
        trigger = 0.0
        if pinch >= trigger_pinch:
            buttons |= Buttons.TRIGGER_CLICK
            trigger = pinch
        if grip >= GRIP_CLICK_THRESHOLD:
            buttons |= Buttons.GRIP_CLICK
        return ControllerInput(
            device_id=device_id,
            buttons=buttons,
            trigger=trigger,
            grip=grip,
        )
