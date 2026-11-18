"""Xbox-style gamepad polling and VR controller input mapping.

Each physical gamepad drives one virtual controller: the left one by default
feeds the left hand and the right one the right hand. Inputs are published to
:class:`SharedState`, which the fusion worker merges under webcam/hand inputs
(real gamepads win). Driver haptics are routed back to the matching pad.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .config import GamepadConfig
from .mathutil import apply_deadzone, clamp
from .protocol import Buttons, ControllerInput, DeviceId
from .state import SharedState

LOGGER = logging.getLogger(__name__)


class GamepadUnavailable(RuntimeError):
    """Raised when pygame is not installed."""


@dataclass(slots=True)
class GamepadInfo:
    index: int
    name: str
    side: str = ""
    guid: str = ""


@dataclass(slots=True)
class _Pad:
    index: int
    side: str
    joystick: object
    name: str = ""


@dataclass(slots=True)
class SyntheticRumble:
    """One rumble request received by a synthetic controller."""

    time: float
    side: str
    amplitude: float
    duration_ms: int


class SyntheticJoystick:
    """Minimal joystick object used by the deterministic gamepad script."""

    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 2
    AXIS_RIGHT_Y = 3
    AXIS_TRIGGER_LEFT = 4
    AXIS_TRIGGER_RIGHT = 5
    AXIS_COUNT = 6
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LEFT_SHOULDER = 9
    BUTTON_RIGHT_SHOULDER = 10
    BUTTON_LEFT_STICK = 7
    BUTTON_RIGHT_STICK = 8
    BUTTON_COUNT = 16

    def __init__(self, side: str, index: int):
        self.side = side
        self.index = int(index)
        self.name = f"Synthetic Xbox Controller ({side})"
        self.guid = "ovrc-synthetic-xbox"
        self.axes = [0.0] * self.AXIS_COUNT
        self.buttons = [False] * self.BUTTON_COUNT
        self.rumble_events: list[SyntheticRumble] = []
        self._lock = threading.Lock()

    def init(self) -> None:  # pygame compatibility
        pass

    def get_name(self) -> str:
        return self.name

    def get_guid(self) -> str:
        return self.guid

    def get_axis(self, index: int) -> float:
        with self._lock:
            if 0 <= int(index) < len(self.axes):
                return float(self.axes[int(index)])
        return 0.0

    def get_button(self, index: int) -> bool:
        with self._lock:
            if 0 <= int(index) < len(self.buttons):
                return bool(self.buttons[int(index)])
        return False

    def set_state(self, axes: list[float], buttons: set[int]) -> None:
        with self._lock:
            self.axes = [float(value) for value in axes]
            self.buttons = [index in buttons for index in range(self.BUTTON_COUNT)]

    def rumble(self, low: float, high: float, duration_ms: int) -> bool:
        self.record_rumble(max(float(low), float(high)), duration_ms)
        return True

    def rumble_play(self, amplitude: float, duration_ms: int) -> None:
        self.record_rumble(float(amplitude), duration_ms)

    def record_rumble(self, amplitude: float, duration_ms: int) -> None:
        event = SyntheticRumble(
            time=time.monotonic(),
            side=self.side,
            amplitude=clamp(amplitude, 0.0, 1.0),
            duration_ms=int(duration_ms),
        )
        with self._lock:
            self.rumble_events.append(event)
            del self.rumble_events[:-200]

    def stop_rumble(self) -> None:
        pass


SYNTHETIC_CYCLE_S = 8.0


def _pulse(phase: float, start: float, length: float) -> float:
    """Smooth pulse between 0 and 1 with a soft top."""
    if phase < start or phase > start + length:
        return 0.0
    position = (phase - start) / max(1e-6, length)
    return math.sin(math.pi * max(0.0, min(1.0, position))) ** 0.25


def synthetic_gamepad_state(elapsed: float, side: str) -> dict:
    """Deterministic Xbox controller script: sticks, buttons and deep triggers.

    Each 8 s cycle moves both sticks and presses buttons. The trigger pulses
    stay above 0.75 for about half a second, which is the threshold the
    ``openxr-simple-example`` demo uses to request haptic feedback.
    """
    t = float(elapsed)
    phase = t % SYNTHETIC_CYCLE_S
    offset = 0.0 if side == "left" else 4.0
    local = (phase + offset) % SYNTHETIC_CYCLE_S

    left_stick = (
        0.62 * math.sin(1.10 * t + (0.0 if side == "left" else 0.9)),
        0.48 * math.sin(0.70 * t + (0.4 if side == "left" else 1.8)),
    )
    right_stick = (
        0.55 * math.sin(0.90 * t + 1.2 + (0.0 if side == "left" else 1.1)),
        0.60 * math.cos(0.50 * t + (0.0 if side == "left" else 1.4)),
    )
    trigger = max(
        _pulse(local, 0.6, 1.4),
        0.0 if side == "left" else _pulse(local, 2.2, 0.9) * 0.4,
    )
    grip = 0.7 if 5.4 <= local <= 5.9 else 0.0

    pressed: set[str] = set()
    if 2.8 <= local <= 3.3:
        pressed.add("a")
    if 6.2 <= local <= 6.6:
        pressed.add("b")
    if 1.8 <= local <= 2.2:
        pressed.add("x")
    if 3.7 <= local <= 4.1:
        pressed.add("y")
    if 5.4 <= local <= 5.9:
        pressed.add("left_shoulder" if side == "left" else "right_shoulder")
    if 7.0 <= local <= 7.3:
        pressed.add("dpad_left" if side == "left" else "dpad_right")
    if 4.4 <= local <= 4.6:
        pressed.add("start")

    return {
        "left_stick": left_stick,
        "right_stick": right_stick,
        "trigger": trigger,
        "grip": grip,
        "pressed": pressed,
    }


def normalize_trigger(value: float) -> float:
    """Map SDL trigger axes (-1..1 rest or 0..1 rest) to 0..1."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return clamp((value + 1.0) / 2.0, 0.0, 1.0)
    return clamp(value, 0.0, 1.0)


def map_gamepad(
    device_id: int,
    config: GamepadConfig,
    left_stick: tuple[float, float],
    right_stick: tuple[float, float],
    trigger: float,
    grip: float,
    pressed: set[str],
) -> ControllerInput:
    """Convert semantic gamepad values to a driver controller input."""
    deadzone = clamp(float(config.deadzone), 0.0, 0.6)
    invert_y = config.invert_left_y if device_id == DeviceId.LEFT else config.invert_right_y
    stick_x = apply_deadzone(left_stick[0], deadzone)
    stick_y = apply_deadzone(left_stick[1], deadzone)
    if invert_y:
        stick_y = -stick_y
    pad_x = apply_deadzone(right_stick[0], deadzone)
    pad_y = apply_deadzone(right_stick[1], deadzone)
    if invert_y:
        pad_y = -pad_y

    trigger_value = clamp(float(trigger), 0.0, 1.0)
    grip_value = clamp(float(grip), 0.0, 1.0)
    buttons = Buttons(0)
    if trigger_value >= clamp(float(config.trigger_threshold), 0.0, 1.0):
        buttons |= Buttons.TRIGGER_CLICK
    if grip_value >= clamp(float(config.trigger_threshold), 0.0, 1.0):
        buttons |= Buttons.GRIP_CLICK
    if trigger_value > 0.0:
        buttons |= Buttons.TRIGGER_TOUCH
    if grip_value > 0.0:
        buttons |= Buttons.GRIP_TOUCH

    button_map = {
        "a": Buttons.A,
        "b": Buttons.B,
        "x": Buttons.X,
        "y": Buttons.Y,
        "left_shoulder": Buttons.GRIP_CLICK,
        "right_shoulder": Buttons.GRIP_CLICK,
        "left_stick": Buttons.JOYSTICK_CLICK,
        "right_stick": Buttons.TRACKPAD_CLICK,
        "dpad_up": Buttons.DPAD_UP,
        "dpad_down": Buttons.DPAD_DOWN,
        "dpad_left": Buttons.DPAD_LEFT,
        "dpad_right": Buttons.DPAD_RIGHT,
        "start": Buttons.START,
        "back": Buttons.SELECT,
        "guide": Buttons.SYSTEM,
        "menu": Buttons.APPLICATION_MENU,
    }
    for name in pressed:
        buttons |= button_map.get(name.lower(), Buttons(0))

    return ControllerInput(
        device_id=device_id,
        buttons=buttons,
        trigger=trigger_value,
        grip=grip_value,
        joystick_x=stick_x,
        joystick_y=stick_y,
        trackpad_x=pad_x,
        trackpad_y=pad_y,
    )


class GamepadManager:
    """Polls pygame joysticks and publishes controller inputs."""

    def __init__(
        self,
        state: SharedState,
        config: GamepadConfig,
        poll_hz: Optional[int] = None,
    ):
        self.state = state
        self.config = config
        if poll_hz is None:
            poll_hz = int(getattr(config, "poll_hz", 250))
        self.poll_hz = max(20, int(poll_hz))
        self.hotplug_interval_s = max(
            0.25, float(getattr(config, "hotplug_interval_s", 1.0))
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pygame = None
        self._pads: dict[str, _Pad] = {}
        self._all: list[GamepadInfo] = []
        self._error = ""
        self._last_scan = 0.0
        self._synthetic_started = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def error(self) -> str:
        return self._error

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        self._last_scan = 0.0
        self._thread = threading.Thread(
            target=self._loop, name="openvrcardboard-gamepad", daemon=True
        )
        self._thread.start()
        LOGGER.info(
            "gamepad manager started (%s)",
            "synthetic controllers" if self.config.synthetic else "pygame joysticks",
        )

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._pads.clear()
        try:
            self.state.set_gamepads({}, {}, timestamp=time.monotonic())
        except Exception:
            pass
        LOGGER.info("gamepad manager stopped")

    def __enter__(self) -> "GamepadManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # pygame plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pygame():
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise GamepadUnavailable(
                "gamepad support needs the 'pygame' package"
            ) from exc
        return pygame

    def _scan(self) -> None:
        pygame = self._pygame
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
        except Exception as exc:
            self._error = f"could not initialise joysticks: {exc}"
            return

        count = 0
        try:
            count = int(pygame.joystick.get_count())
        except Exception:
            count = 0

        infos: list[GamepadInfo] = []
        pads: dict[str, _Pad] = {}
        for index in range(count):
            try:
                joystick = pygame.joystick.Joystick(index)
                joystick.init()
            except Exception as exc:
                LOGGER.debug("could not open gamepad %d: %s", index, exc)
                continue
            name = ""
            guid = ""
            try:
                name = joystick.get_name()
                guid = joystick.get_guid()
            except Exception:
                pass
            infos.append(GamepadInfo(index=index, name=name or f"gamepad {index}", guid=guid))
        self._all = infos

        wanted = {
            "left": int(self.config.left_index),
            "right": int(self.config.right_index),
        }
        for side, wanted_index in wanted.items():
            chosen: Optional[GamepadInfo] = None
            for info in infos:
                if info.index == wanted_index:
                    chosen = info
                    break
            if chosen is None:
                # Use the first pad not already bound to the other side.
                used = {pad.index for pad in pads.values()}
                available = [info for info in infos if info.index not in used]
                if available:
                    chosen = available[0]
            if chosen is None:
                continue
            try:
                joystick = pygame.joystick.Joystick(chosen.index)
                joystick.init()
            except Exception as exc:
                self._error = f"could not open gamepad {chosen.index}: {exc}"
                continue
            pads[side] = _Pad(index=chosen.index, side=side, joystick=joystick, name=chosen.name)
            chosen.side = side

        changed = {side: pad.index for side, pad in pads.items()} != {
            side: pad.index for side, pad in self._pads.items()
        }
        self._pads = pads
        if changed:
            names = {pad.index: pad.name for pad in pads.values()}
            self.state.set_gamepads({}, names, timestamp=time.monotonic())
            LOGGER.info(
                "gamepads: %s",
                ", ".join(f"{side}={pad.name}" for side, pad in pads.items()) or "none",
            )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def _synthetic_loop(self) -> None:
        left_index = int(self.config.left_index)
        right_index = int(self.config.right_index)
        self._pads = {
            "left": _Pad(
                index=left_index,
                side="left",
                joystick=SyntheticJoystick("left", left_index),
                name=f"Synthetic Xbox Controller ({left_index})",
            ),
            "right": _Pad(
                index=right_index,
                side="right",
                joystick=SyntheticJoystick("right", right_index),
                name=f"Synthetic Xbox Controller ({right_index})",
            ),
        }
        self._all = [
            GamepadInfo(index=left_index, name=self._pads["left"].name, side="left", guid="ovrc-synthetic-xbox"),
            GamepadInfo(index=right_index, name=self._pads["right"].name, side="right", guid="ovrc-synthetic-xbox"),
        ]
        names = {
            int(DeviceId.LEFT): self._pads["left"].name,
            int(DeviceId.RIGHT): self._pads["right"].name,
        }
        self._error = ""
        self.state.set_gamepads({}, names, timestamp=time.monotonic())
        LOGGER.info("synthetic Xbox controllers active")

        self._synthetic_started = time.monotonic()
        period = 1.0 / self.poll_hz
        next_time = self._synthetic_started
        button_ids = {
            "a": SyntheticJoystick.BUTTON_A,
            "b": SyntheticJoystick.BUTTON_B,
            "x": SyntheticJoystick.BUTTON_X,
            "y": SyntheticJoystick.BUTTON_Y,
            "left_stick": SyntheticJoystick.BUTTON_LEFT_STICK,
            "right_stick": SyntheticJoystick.BUTTON_RIGHT_STICK,
        }
        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - self._synthetic_started
            inputs: dict[int, ControllerInput] = {}
            for side, pad in self._pads.items():
                scripted = synthetic_gamepad_state(elapsed, side)
                shoulder = (
                    SyntheticJoystick.BUTTON_LEFT_SHOULDER
                    if side == "left"
                    else SyntheticJoystick.BUTTON_RIGHT_SHOULDER
                )
                pressed_ids = {
                    button_ids[name]
                    for name in scripted["pressed"]
                    if name in button_ids
                }
                if scripted["grip"] > 0.0:
                    pressed_ids.add(shoulder)
                pad.joystick.set_state(
                    [
                        scripted["left_stick"][0],
                        scripted["left_stick"][1],
                        scripted["right_stick"][0],
                        scripted["right_stick"][1],
                        scripted["trigger"],
                        scripted["trigger"],
                    ],
                    pressed_ids,
                )
                device_id = int(
                    DeviceId.LEFT if side == "left" else DeviceId.RIGHT
                )
                inputs[device_id] = map_gamepad(
                    device_id,
                    self.config,
                    scripted["left_stick"],
                    scripted["right_stick"],
                    scripted["trigger"],
                    scripted["grip"],
                    scripted["pressed"],
                )
            self.state.set_gamepads(inputs, names, timestamp=now)

            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_time = time.monotonic()

    def _loop(self) -> None:
        if self.config.synthetic:
            self._synthetic_loop()
            return
        try:
            pygame = self._load_pygame()
        except GamepadUnavailable as exc:
            self._error = str(exc)
            self.state.add_note(str(exc))
            LOGGER.warning("%s", exc)
            self._running = False
            return
        self._pygame = pygame
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as exc:
            self._error = f"could not initialise pygame: {exc}"
            self.state.add_note(self._error)
            LOGGER.warning("%s", exc)
            self._running = False
            return

        period = 1.0 / self.poll_hz
        next_time = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - self._last_scan > self.hotplug_interval_s or not self._pads:
                self._last_scan = now
                try:
                    self._scan()
                except Exception as exc:
                    self._error = f"gamepad scan failed: {exc}"
                    LOGGER.debug("gamepad scan failed: %s", exc)
            try:
                pygame.event.pump()
            except Exception:
                pass

            inputs: dict[int, ControllerInput] = {}
            names: dict[int, str] = {}
            for side, pad in list(self._pads.items()):
                device_id = int(DeviceId.LEFT if side == "left" else DeviceId.RIGHT)
                try:
                    inputs[device_id] = self._read_pad(pad, device_id)
                except Exception as exc:
                    LOGGER.debug("gamepad %s read failed: %s", pad.name, exc)
                names[device_id] = pad.name
            self.state.set_gamepads(inputs, names, timestamp=now)

            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_time = time.monotonic()

    def _read_pad(self, pad: _Pad, device_id: int) -> ControllerInput:
        pygame = self._pygame
        joystick = pad.joystick
        axes = pygame.CONTROLLER_AXIS
        buttons = pygame.CONTROLLER_BUTTON

        def axis(name: str, fallback: int) -> float:
            value = getattr(axes, name, None)
            if value is None:
                try:
                    return float(joystick.get_axis(fallback))
                except Exception:
                    return 0.0
            try:
                return float(joystick.get_axis(value))
            except Exception:
                return 0.0

        left_stick = (axis("LEFTX", 0), axis("LEFTY", 1))
        right_stick = (axis("RIGHTX", 2), axis("RIGHTY", 3))
        trigger = normalize_trigger(axis("TRIGGERLEFT" if pad.side == "left" else "TRIGGERRIGHT", 4 if pad.side == "left" else 5))
        shoulder_name = "LEFTSHOULDER" if pad.side == "left" else "RIGHTSHOULDER"
        shoulder_button = getattr(buttons, shoulder_name, None)

        pressed: set[str] = set()

        def pressed_button(name: str, fallback: int) -> bool:
            value = getattr(buttons, name, None)
            if value is None:
                value = fallback
            try:
                return bool(joystick.get_button(value))
            except Exception:
                return False

        if pressed_button("A", 0):
            pressed.add("a")
        if pressed_button("B", 1):
            pressed.add("b")
        if pressed_button("X", 2):
            pressed.add("x")
        if pressed_button("Y", 3):
            pressed.add("y")
        if shoulder_button is not None and pressed_button(shoulder_name, 9 if pad.side == "left" else 10):
            pressed.add(shoulder_name.lower())
        # The other shoulder and the stick clicks are still useful bindings.
        other_shoulder = "RIGHTSHOULDER" if pad.side == "left" else "LEFTSHOULDER"
        if pressed_button(other_shoulder, 10 if pad.side == "left" else 9):
            pressed.add(other_shoulder.lower())
        if pressed_button("LEFTSTICK", 7):
            pressed.add("left_stick")
        if pressed_button("RIGHTSTICK", 8):
            pressed.add("right_stick")
        for name in (
            "DPAD_UP",
            "DPAD_DOWN",
            "DPAD_LEFT",
            "DPAD_RIGHT",
            "START",
            "BACK",
            "GUIDE",
        ):
            if pressed_button(name, -1):
                pressed.add(name.lower())

        grip = 1.0 if shoulder_name.lower() in pressed else 0.0
        return map_gamepad(
            device_id,
            self.config,
            left_stick,
            right_stick,
            trigger,
            grip,
            pressed,
        )

    # ------------------------------------------------------------------
    # Haptics
    # ------------------------------------------------------------------
    def rumble(self, device_id: int, amplitude: float, frequency: float = 0.0, duration: float = 0.05) -> bool:
        side = "left" if int(device_id) == int(DeviceId.LEFT) else "right"
        pad = self._pads.get(side)
        if pad is None or not self.config.rumble:
            return False
        amplitude = clamp(float(amplitude), 0.0, 1.0)
        duration_ms = int(max(10.0, min(2000.0, float(duration) * 1000.0)))
        try:
            if hasattr(pad.joystick, "rumble"):
                return bool(pad.joystick.rumble(amplitude, amplitude, duration_ms))
            if hasattr(pad.joystick, "rumble_play"):  # very old pygame
                pad.joystick.rumble_play(amplitude, duration_ms)
                return True
        except Exception as exc:
            LOGGER.debug("gamepad rumble failed: %s", exc)
        return False

    def stop_rumble(self) -> None:
        for pad in self._pads.values():
            try:
                if hasattr(pad.joystick, "stop_rumble"):
                    pad.joystick.stop_rumble()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def devices(self) -> list[GamepadInfo]:
        return [
            GamepadInfo(index=info.index, name=info.name, side=info.side, guid=info.guid)
            for info in self._all
        ]

    def status(self) -> dict:
        haptics = self._haptic_status()
        return {
            "running": self._running,
            "error": self._error,
            "synthetic": bool(self.config.synthetic),
            "pads": {
                side: {"index": pad.index, "name": pad.name}
                for side, pad in self._pads.items()
            },
            "haptics": haptics,
        }

    def _haptic_status(self) -> dict:
        events: list[SyntheticRumble] = []
        for pad in self._pads.values():
            joystick = pad.joystick
            if isinstance(joystick, SyntheticJoystick):
                events.extend(joystick.rumble_events)
        events.sort(key=lambda item: item.time)
        counts = {
            "left": sum(1 for event in events if event.side == "left"),
            "right": sum(1 for event in events if event.side == "right"),
        }
        last = events[-1] if events else None
        return {
            "supported": bool(self.config.rumble),
            "count": len(events),
            "left": counts["left"],
            "right": counts["right"],
            "last": (
                {
                    "side": last.side,
                    "amplitude": round(last.amplitude, 3),
                    "duration_ms": last.duration_ms,
                    "age_s": round(max(0.0, time.monotonic() - last.time), 3),
                }
                if last is not None
                else None
            ),
            "events": [
                {
                    "side": event.side,
                    "amplitude": round(event.amplitude, 3),
                    "duration_ms": event.duration_ms,
                }
                for event in events[-20:]
            ],
        }
