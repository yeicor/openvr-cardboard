import numpy as np
import pytest

from openvrcardboard.markers import MarkerDetection
from openvrcardboard.mathutil import IDENTITY
from openvrcardboard.pose_tracking import BodyObservation, HandObservation
from openvrcardboard.protocol import Buttons, ControllerInput, DeviceId
from openvrcardboard.state import SharedState


def detection(marker_id=1, position=(0.1, 1.2, -0.4)):
    return MarkerDetection(
        marker_id=marker_id,
        position=np.array(position, dtype=np.float64),
        rotation=IDENTITY.copy(),
        optical_position=np.array([0.0, 0.0, 1.0]),
        optical_rotation=IDENTITY.copy(),
        reprojection_error=0.5,
        distance=1.0,
    )


def hand(side="left", pinch=0.8, grip=0.7):
    return HandObservation(
        side=side,
        position=np.array([-0.2, 1.2, -0.3]),
        rotation=IDENTITY.copy(),
        pinch=pinch,
        grip=grip,
        landmarks=np.zeros((21, 2)),
    )


def test_snapshot_converts_markers_and_hands():
    state = SharedState()
    state.set_hand_input_options(True, 0.35)
    state.set_markers({1: detection(), 2: detection(2, (0.3, 1.1, -0.4))}, timestamp=5.0)
    state.set_hands({"left": hand("left"), "right": hand(side="right")}, timestamp=5.0)
    snapshot = state.snapshot(now=5.2)

    assert set(snapshot.markers) == {1, 2}
    assert snapshot.markers[1].position == pytest.approx([0.1, 1.2, -0.4])
    assert snapshot.markers[1].source == "markers"
    assert snapshot.markers[1].timestamp == pytest.approx(5.0)
    assert set(snapshot.hands) == {int(DeviceId.LEFT), int(DeviceId.RIGHT)}
    left = snapshot.hand_inputs[int(DeviceId.LEFT)]
    assert left.trigger == pytest.approx(0.8)
    assert int(left.buttons) & int(Buttons.TRIGGER_CLICK)
    assert int(left.buttons) & int(Buttons.GRIP_CLICK)
    assert left.grip == pytest.approx(0.7)


def test_hand_inputs_can_be_disabled_and_threshold_respected():
    state = SharedState()
    state.set_hands({"left": hand("left", pinch=0.2, grip=0.1)})
    snapshot = state.snapshot()
    left = snapshot.hand_inputs[int(DeviceId.LEFT)]
    assert left.trigger == pytest.approx(0.0)
    assert left.buttons == Buttons(0)
    assert left.grip == pytest.approx(0.1)

    state.set_hand_input_options(False, 0.2)
    snapshot = state.snapshot()
    assert snapshot.hand_inputs == {}


def test_snapshot_converts_body_joints():
    state = SharedState()
    state.set_body(
        BodyObservation(
            joints={
                "hips": np.array([0.0, 1.0, 0.0]),
                "left_foot": np.array([-0.2, 0.1, 0.0]),
            },
            rotations={"hips": IDENTITY.copy(), "left_foot": IDENTITY.copy()},
        ),
        timestamp=7.0,
    )
    snapshot = state.snapshot()
    assert snapshot.body["hips"].position == pytest.approx([0.0, 1.0, 0.0])
    assert snapshot.body["hips"].timestamp == pytest.approx(7.0)
    assert state.status().body_visible


def test_gamepad_and_status_updates():
    state = SharedState()
    state.set_gamepads(
        {
            int(DeviceId.LEFT): ControllerInput(DeviceId.LEFT, trigger=0.4),
            int(DeviceId.RIGHT): ControllerInput(DeviceId.RIGHT, trigger=0.9),
        },
        {int(DeviceId.LEFT): "Left XInput", int(DeviceId.RIGHT): "Right XInput"},
        timestamp=3.0,
    )
    snapshot = state.snapshot()
    assert snapshot.gamepads[int(DeviceId.RIGHT)].trigger == pytest.approx(0.9)
    assert snapshot.gamepad_names[int(DeviceId.LEFT)] == "Left XInput"
    assert state.status().gamepads == ["Left XInput", "Right XInput"]

    state.set_status(camera_fps=59.5, camera_error="")
    state.add_note("test note")
    status = state.status()
    assert status.camera_fps == pytest.approx(59.5)
    assert "test note" in status.notes


def test_recenter_is_a_one_shot_event():
    state = SharedState()
    assert not state.take_recenter()
    state.request_recenter()
    state.request_recenter()
    assert state.recenter_count == 2
    assert state.take_recenter()
    assert not state.take_recenter()


def test_preview_and_phone_are_shared():
    state = SharedState()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    state.set_preview(image, timestamp=2.0)
    preview, stamp, sequence = state.preview()
    assert preview is image
    assert stamp == pytest.approx(2.0)
    assert sequence == 1

    state.set_phone(None)
    assert state.phone() is None
    assert state.snapshot().phone is None


def test_invalid_markers_are_not_snapshotted():
    state = SharedState()
    marker = detection()
    marker.valid = False
    state.set_markers({1: marker})
    assert state.snapshot().markers == {}
