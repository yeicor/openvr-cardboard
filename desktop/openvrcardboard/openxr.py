"""Monado OpenXR bridge and eye-capture receiver.

This module is the glue between the desktop app and the OpenXR application used
by the recorded end-to-end test. Two independent local links are involved:

* :class:`MonadoRemoteLink` is a TCP client for Monado's ``remote`` driver. It
  converts the composed poses and controller inputs into the remote driver's
  fixed-size ``r_remote_data`` packet, so an OpenXR application sees the same
  6DoF tracking and gamepad input as SteamVR through the OpenVR driver.
* :class:`OpenXRCaptureServer` accepts the side-by-side JPEG stream produced by
  ``libovrc_xr_capture.so``. The shim reports the per-eye textures, grab action
  values and haptic feedback of the real OpenXR application.

The wire layouts are deliberately self-contained; see ``shim/`` for the C side
and ``docs/PROTOCOL.md`` for the documented packet layout.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import default_config_dir

from .fusion import ComposedFrame
from .mathutil import normalize
from .monado_ipc import monado_socket_fits
from .protocol import Buttons, ControllerInput, DeviceId

LOGGER = logging.getLogger(__name__)

REMOTE_HEADER = struct.unpack("<Q", b"mndrmt3\0")[0]
REMOTE_PACKET_SIZE = 376

CAPTURE_MAGIC = b"OVXS"
CAPTURE_HEADER = struct.Struct("<4sB3xI")
CAPTURE_FRAME_HEADER = struct.Struct("<IdII")
CAPTURE_MAX_PAYLOAD = 16 * 1024 * 1024

_CAPTURE_VIEW = struct.Struct("<16s28sI")
_CAPTURE_POSE = struct.Struct("<7f")
_CAPTURE_HEAD_TRAILER = struct.Struct("<B3x")
_CAPTURE_CONTROLLER = struct.Struct("<7f3f3f5ffff2ff2f16B")

assert _CAPTURE_VIEW.size == 48
assert _CAPTURE_POSE.size == 28
assert _CAPTURE_CONTROLLER.size == 120


def pack_remote_data(frame: ComposedFrame) -> bytes:
    """Serialise a composed frame as one Monado ``r_remote_data`` packet.

    Positions are sent in tracking (stage) space, the same coordinates the
    OpenVR driver receives. Monado's remote builder places its devices in that
    space and maps them into ``LOCAL`` for the application itself, so no
    height adjustment belongs on this side of the bridge.
    """
    hmd = frame.hmd
    if hmd is not None:
        position = np.asarray(hmd.position, dtype=np.float64).reshape(3)
        rotation = normalize(hmd.rotation)
    else:
        position = np.zeros(3, dtype=np.float64)
        rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    # Per-view data is optional; Monado fills in the eyes from the head pose
    # and the configured IPD, which keeps the bridge simple and robust.
    views = _CAPTURE_VIEW.pack(b"\x00" * 16, b"\x00" * 28, 0) * 2
    center = _CAPTURE_POSE.pack(
        float(rotation[0]),
        float(rotation[1]),
        float(rotation[2]),
        float(rotation[3]),
        float(position[0]),
        float(position[1]),
        float(position[2]),
    )
    head = views + center + _CAPTURE_HEAD_TRAILER.pack(0)

    controllers = b"".join(
        _pack_controller(
            frame.left_controller if side == "left" else frame.right_controller,
            frame.inputs.get(int(DeviceId.LEFT if side == "left" else DeviceId.RIGHT)),
        )
        for side in ("left", "right")
    )

    packet = struct.pack("<Q", REMOTE_HEADER) + head + controllers
    if len(packet) != REMOTE_PACKET_SIZE:  # pragma: no cover - layout guard
        raise RuntimeError(
            f"remote packet is {len(packet)} bytes, expected {REMOTE_PACKET_SIZE}"
        )
    return packet


def _pack_controller(pose, control: Optional[ControllerInput]) -> bytes:
    active = pose is not None and bool(getattr(pose, "valid", False))
    if pose is not None and active:
        position = np.asarray(pose.position, dtype=np.float64).reshape(3)
        rotation = normalize(pose.rotation)
        velocity = np.asarray(pose.velocity, dtype=np.float64).reshape(3)
        angular = np.asarray(pose.angular_velocity, dtype=np.float64).reshape(3)
    else:
        position = np.zeros(3, dtype=np.float64)
        rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        velocity = np.zeros(3, dtype=np.float64)
        angular = np.zeros(3, dtype=np.float64)

    control = control or ControllerInput(device_id=DeviceId.LEFT)
    buttons = Buttons(control.buttons)
    trigger = max(0.0, min(1.0, float(control.trigger)))
    grip = max(0.0, min(1.0, float(control.grip)))
    thumb_x = max(-1.0, min(1.0, float(control.joystick_x)))
    thumb_y = max(-1.0, min(1.0, float(control.joystick_y)))
    pad_x = max(-1.0, min(1.0, float(control.trackpad_x)))
    pad_y = max(-1.0, min(1.0, float(control.trackpad_y)))

    booleans = [
        False,  # hand_tracking_active
        active,
        bool(buttons & Buttons.SYSTEM),
        False,  # system_touch
        bool(buttons & Buttons.A),
        False,  # a_touch
        bool(buttons & Buttons.B),
        False,  # b_touch
        bool(buttons & Buttons.TRIGGER_CLICK),
        bool(buttons & Buttons.TRIGGER_TOUCH) or trigger > 0.01,
        bool(buttons & Buttons.JOYSTICK_CLICK),
        bool(buttons & Buttons.JOYSTICK_TOUCH) or abs(thumb_x) + abs(thumb_y) > 0.02,
        bool(buttons & Buttons.TRACKPAD_TOUCH) or abs(pad_x) + abs(pad_y) > 0.02,
        False,
        False,
        False,
    ]

    return _CAPTURE_CONTROLLER.pack(
        float(rotation[0]),
        float(rotation[1]),
        float(rotation[2]),
        float(rotation[3]),
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(velocity[0]),
        float(velocity[1]),
        float(velocity[2]),
        float(angular[0]),
        float(angular[1]),
        float(angular[2]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        trigger,
        grip,
        grip,
        thumb_x,
        thumb_y,
        0.0,
        pad_x,
        pad_y,
        *booleans,
    )


@dataclass(slots=True)
class MonadoServiceStats:
    running: bool = False
    managed: bool = False
    external: bool = False
    pid: int = 0
    service: str = ""
    log: str = ""
    last_error: str = ""


class MonadoServiceError(OSError):
    """Raised when the bundled Monado service cannot be started."""


def find_monado_service() -> Optional[Path]:
    """Locate ``monado-service`` in a portable build, the repo or ``PATH``."""
    candidates: list[Path] = []

    def add(prefix: Optional[Path]) -> None:
        if prefix is not None:
            candidates.append(Path(prefix) / "bin" / "monado-service")

    configured = os.environ.get("OVRC_OPENXR_ROOT")
    if configured:
        add(Path(configured).expanduser())
    try:
        from .integration import bundled_paths

        _, bundled_openxr = bundled_paths()
        add(bundled_openxr)
    except Exception:  # pragma: no cover - import guard for partial installs
        pass
    repository = Path(__file__).resolve().parents[2]
    add(repository / "out" / "openxr")
    add(Path("/opt/openxr"))
    found = shutil.which("monado-service")
    if found:
        candidates.append(Path(found))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _port_is_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class MonadoService:
    """Start and own ``monado-service`` so other OpenXR apps can connect.

    The installer registers the bundled runtime and writes a Monado ``remote``
    configuration. The runtime client library only talks to a service process;
    this supervisor starts that process for the lifetime of the desktop app and
    leaves an already-running service (for example one started manually)
    untouched.
    """

    def __init__(
        self,
        service: Optional[Path | str] = None,
        host: str = "127.0.0.1",
        remote_port: int = 4242,
        log_path: Optional[Path | str] = None,
        startup_timeout: float = 10.0,
    ):
        self.service = Path(service).expanduser() if service else None
        self.host = str(host)
        self.remote_port = int(remote_port)
        self.log_path = (
            Path(log_path).expanduser()
            if log_path
            else default_config_dir() / "monado-service.log"
        )
        self.startup_timeout = max(0.5, float(startup_timeout))
        self._process: Optional[subprocess.Popen] = None
        self._external = False
        self._temporary_runtime_dir: Optional[Path] = None
        self._error = ""

    @property
    def running(self) -> bool:
        return self._external or self._process is not None

    def note_error(self, message: str) -> None:
        self._error = str(message)

    def stats(self) -> MonadoServiceStats:
        return MonadoServiceStats(
            running=self.running,
            managed=self._process is not None,
            external=self._external,
            pid=self._process.pid if self._process is not None else 0,
            service=str(self.service) if self.service else "",
            log=str(self.log_path),
            last_error=self._error,
        )

    def _tail(self, lines: int = 30) -> str:
        try:
            content = self.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return ""
        return "\n".join(content[-lines:])

    def _runtime_directory(self, cache_home: Path) -> Path:
        """Return a runtime dir where Monado's IPC socket path fits.

        Monado appends its fixed socket filename to ``XDG_RUNTIME_DIR``. When
        the configured directory makes that path exceed
        ``sockaddr_un.sun_path``, start the service with a short temporary
        directory instead; ``_terminate`` removes it.
        """
        runtime_dir = Path(
            os.environ.get("XDG_RUNTIME_DIR")
            or (cache_home / "openvrcardboard" / "runtime")
        ).expanduser()
        if monado_socket_fits(runtime_dir):
            return runtime_dir
        temporary = Path(tempfile.mkdtemp(prefix="openvrcardboard-monado-"))
        self._temporary_runtime_dir = temporary
        LOGGER.info(
            "using %s for Monado's IPC socket because %s is too long",
            temporary,
            runtime_dir,
        )
        return temporary

    def start(self) -> None:
        if self.running:
            return
        self._error = ""
        if _port_is_open(self.host, self.remote_port):
            self._external = True
            LOGGER.info(
                "using an already running Monado service on %s:%d",
                self.host,
                self.remote_port,
            )
            return
        if self.service is None:
            self.service = find_monado_service()
        if self.service is None:
            raise MonadoServiceError(
                "monado-service was not found; package the OpenXR runtime or "
                "set OVRC_OPENXR_ROOT"
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        env = os.environ.copy()
        # Monado writes a pidfile into XDG_RUNTIME_DIR and falls back to a
        # literal "~/.cache" (which it cannot open) when neither variable is
        # set, for example under a service manager or a bare container.
        cache_home = Path(
            env.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        ).expanduser()
        runtime_dir = self._runtime_directory(cache_home)
        cache_home.mkdir(parents=True, exist_ok=True)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        try:
            runtime_dir.chmod(0o700)
        except OSError:
            pass
        env["XDG_CACHE_HOME"] = str(cache_home)
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
        env["P_OVERRIDE_ACTIVE_CONFIG"] = "remote"
        env["XRT_COMPOSITOR_NULL"] = "1"
        env.setdefault("XRT_COMPOSITOR_NULL_FPS", "90")
        if not env.get("VK_ICD_FILENAMES"):
            lavapipe = Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
            if lavapipe.exists():
                env["VK_ICD_FILENAMES"] = str(lavapipe)
        try:
            self._process = subprocess.Popen(
                [str(self.service)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=not sys.platform.startswith("win"),
            )
        except OSError as exc:
            self._process = None
            raise MonadoServiceError(
                f"could not start {self.service}: {exc}"
            ) from exc
        finally:
            log.close()

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._process = None
                raise MonadoServiceError(
                    "monado-service exited during startup:\n" + self._tail()
                )
            if _port_is_open(self.host, self.remote_port):
                LOGGER.info(
                    "started monado-service (pid %d) on %s:%d",
                    self._process.pid,
                    self.host,
                    self.remote_port,
                )
                return
            time.sleep(0.1)
        self._terminate()
        raise MonadoServiceError(
            "monado-service did not open its remote port within "
            f"{self.startup_timeout:.1f}s:\n{self._tail()}"
        )

    def _terminate(self) -> None:
        process = self._process
        self._process = None
        self._external = False
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        temporary = self._temporary_runtime_dir
        self._temporary_runtime_dir = None
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    def stop(self) -> None:
        self._terminate()


@dataclass(slots=True)
class RemoteLinkStats:
    running: bool = False
    connected: bool = False
    packets: int = 0
    send_errors: int = 0
    connect_attempts: int = 0
    last_error: str = ""


class MonadoRemoteLink:
    """TCP client feeding Monado's ``remote`` driver at a fixed rate."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4242,
        rate_hz: int = 100,
    ):
        self.host = str(host)
        self.port = int(port)
        self.rate_hz = max(1, int(rate_hz))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[ComposedFrame] = None
        self._running = False
        self._connected = False
        self._socket: Optional[socket.socket] = None
        self._stats = RemoteLinkStats()

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        self._stats.running = True
        self._thread = threading.Thread(
            target=self._loop, name="openvrcardboard-monado", daemon=True
        )
        self._thread.start()
        LOGGER.info("Monado remote link targeting %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._running = False
        self._stats.running = False
        self._stop.set()
        self._close_socket()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def update_frame(self, frame: ComposedFrame, now: Optional[float] = None) -> None:
        snapshot = frame.copy()
        with self._lock:
            self._frame = snapshot

    def stats(self) -> RemoteLinkStats:
        stats = RemoteLinkStats(
            running=self._running,
            connected=self._connected,
            packets=self._stats.packets,
            send_errors=self._stats.send_errors,
            connect_attempts=self._stats.connect_attempts,
            last_error=self._stats.last_error,
        )
        return stats

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        self._connected = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _connect(self) -> bool:
        self._stats.connect_attempts += 1
        try:
            sock = socket.create_connection((self.host, self.port), timeout=0.5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.2)
            self._socket = sock
            self._connected = True
            self._stats.last_error = ""
            LOGGER.info("connected to Monado remote driver on %s:%d", self.host, self.port)
            return True
        except OSError as exc:
            self._stats.last_error = str(exc)
            return False

    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_time = time.monotonic()
        while not self._stop.is_set():
            if self._socket is None:
                if not self._connect():
                    self._stop.wait(0.5)
                    next_time = time.monotonic()
                    continue

            with self._lock:
                frame = self._frame
            if frame is None:
                self._stop.wait(0.01)
                continue

            try:
                packet = pack_remote_data(frame)
                self._socket.sendall(packet)
                self._stats.packets += 1
            except (OSError, RuntimeError) as exc:
                self._stats.send_errors += 1
                self._stats.last_error = str(exc)
                LOGGER.debug("Monado remote send failed: %s", exc)
                self._close_socket()
                self._stop.wait(0.25)
                next_time = time.monotonic()
                continue

            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_time = time.monotonic()


@dataclass(slots=True)
class CaptureFrame:
    sequence: int
    timestamp: float
    width: int
    height: int
    jpeg: bytes
    received_at: float = 0.0


@dataclass(slots=True)
class CaptureStats:
    running: bool = False
    connected: bool = False
    frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    haptic_events: int = 0
    grab_left: float = 0.0
    grab_right: float = 0.0
    grab_active_left: bool = False
    grab_active_right: bool = False
    last_haptic: Optional[dict] = None
    last_error: str = ""
    client: str = ""


class OpenXRCaptureServer:
    """Receive the shim's frame/event stream over loopback TCP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 39102,
        on_haptic: Optional[Callable[[str, float, float], None]] = None,
        event_history: int = 400,
    ):
        self.host = str(host)
        self.port = int(port)
        self.on_haptic = on_haptic
        self._event_history = max(1, int(event_history))
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._running = False
        self._latest: Optional[CaptureFrame] = None
        self._frames = 0
        self._fps = 0.0
        self._last_frame_time = 0.0
        self._haptic_events = 0
        self._events: list[dict] = []
        self._grabs = {"left": 0.0, "right": 0.0}
        self._grab_active = {"left": False, "right": False}
        self._last_haptic: Optional[dict] = None
        self._error = ""
        self._connected = False
        self._client_address = ""

    @property
    def running(self) -> bool:
        return self._running

    @property
    def local_port(self) -> int:
        """Bound port, useful when started with port 0 in tests."""
        if self._server is not None:
            return int(self._server.getsockname()[1])
        return int(self.port)

    def start(self) -> None:
        if self._running:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(2)
        server.settimeout(0.5)
        self._server = server
        self._running = True
        self._error = ""
        self._thread = threading.Thread(
            target=self._accept_loop, name="openvrcardboard-xr-capture", daemon=True
        )
        self._thread.start()
        LOGGER.info("OpenXR capture server listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._connected = False

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def latest(self, after_sequence: Optional[int] = None, timeout: float = 2.0) -> Optional[CaptureFrame]:
        """Return the newest frame, optionally waiting for a new sequence."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                frame = self._latest
                if frame is not None and (
                    after_sequence is None or frame.sequence != int(after_sequence)
                ):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or not self._running:
                    return frame
                self._condition.wait(min(0.5, remaining))

    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(
                running=self._running,
                connected=self._connected,
                frames=self._frames,
                fps=self._fps,
                width=self._latest.width if self._latest else 0,
                height=self._latest.height if self._latest else 0,
                haptic_events=self._haptic_events,
                grab_left=self._grabs["left"],
                grab_right=self._grabs["right"],
                grab_active_left=self._grab_active["left"],
                grab_active_right=self._grab_active["right"],
                last_haptic=dict(self._last_haptic) if self._last_haptic else None,
                last_error=self._error,
                client=self._client_address,
            )

    def events(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._events]

    # ------------------------------------------------------------------
    # Socket handling
    # ------------------------------------------------------------------
    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, address = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(1.0)
            with self._lock:
                previous = self._client
                self._client = client
                self._connected = True
                self._client_address = f"{address[0]}:{address[1]}"
            if previous is not None:
                try:
                    previous.close()
                except OSError:
                    pass
            LOGGER.info("OpenXR shim connected from %s", self._client_address)
            try:
                self._read_loop(client)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                if not self._stop.is_set():
                    self._error = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning("OpenXR capture stream ended: %s", exc)
            finally:
                with self._lock:
                    if self._client is client:
                        self._client = None
                        self._connected = False
                try:
                    client.close()
                except OSError:
                    pass

    def _read_exact(self, client: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            data = client.recv(size - len(chunks))
            if not data:
                raise ConnectionError("capture connection closed")
            chunks.extend(data)
        return bytes(chunks)

    def _read_loop(self, client: socket.socket) -> None:
        while not self._stop.is_set():
            header = self._read_exact(client, CAPTURE_HEADER.size)
            magic, kind, length = CAPTURE_HEADER.unpack(header)
            if magic != CAPTURE_MAGIC:
                raise ValueError(f"bad capture magic {magic!r}")
            if length > CAPTURE_MAX_PAYLOAD:
                raise ValueError(f"capture payload too large ({length} bytes)")
            payload = self._read_exact(client, length) if length else b""
            if kind == 1:
                self._handle_frame(payload)
            elif kind == 2:
                self._handle_event(payload)

    def _handle_frame(self, payload: bytes) -> None:
        if len(payload) < CAPTURE_FRAME_HEADER.size:
            raise ValueError("short capture frame")
        sequence, timestamp, width, height = CAPTURE_FRAME_HEADER.unpack_from(payload, 0)
        jpeg = payload[CAPTURE_FRAME_HEADER.size :]
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("capture frame is not JPEG")
        now = time.monotonic()
        with self._lock:
            self._latest = CaptureFrame(
                sequence=int(sequence),
                timestamp=float(timestamp),
                width=int(width),
                height=int(height),
                jpeg=jpeg,
                received_at=now,
            )
            self._frames += 1
            if self._last_frame_time > 0.0:
                dt = now - self._last_frame_time
                if 1e-5 < dt < 2.0:
                    instant = 1.0 / dt
                    self._fps = instant if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * instant
            self._last_frame_time = now
        with self._condition:
            self._condition.notify_all()

    def _handle_event(self, payload: bytes) -> None:
        event = json.loads(payload.decode("utf-8"))
        if not isinstance(event, dict):
            return
        kind = str(event.get("type", ""))
        now = time.monotonic()
        haptic = None
        with self._lock:
            self._events.append({"time": now, **event})
            del self._events[: max(0, len(self._events) - self._event_history)]
            if kind == "haptic":
                self._haptic_events += 1
                self._last_haptic = {"time": now, **event}
                haptic = event
            elif kind == "grab":
                hand = str(event.get("hand", ""))
                if hand in self._grabs:
                    self._grabs[hand] = float(event.get("value", 0.0))
                    self._grab_active[hand] = bool(event.get("active", False))
            elif kind == "ready":
                LOGGER.info(
                    "OpenXR shim ready: %sx%s at %s fps",
                    event.get("width"),
                    event.get("height"),
                    event.get("fps"),
                )
        if haptic is not None and self.on_haptic is not None:
            hand = str(haptic.get("hand", ""))
            if hand not in ("left", "right"):
                return
            amplitude = max(0.0, min(1.0, float(haptic.get("amplitude", 0.0))))
            duration_ms = float(haptic.get("duration_ms", 25.0))
            if duration_ms <= 0.0:
                # XR_MIN_HAPTIC_DURATION asks the runtime to pick a length;
                # the synthetic gamepad needs a concrete value.
                duration_ms = 25.0
            if amplitude <= 0.0:
                return
            try:
                self.on_haptic(hand, amplitude, duration_ms / 1000.0)
            except Exception:
                LOGGER.debug("haptic callback failed", exc_info=True)


class OpenXRManager:
    """Own the Monado link and the capture receiver."""

    def __init__(
        self,
        config,
        on_haptic: Optional[Callable[[str, float, float], None]] = None,
    ):
        self.config = config
        self.capture = OpenXRCaptureServer(
            host=str(getattr(config, "capture_host", "127.0.0.1")),
            port=int(getattr(config, "capture_port", 39102)),
            on_haptic=on_haptic,
        )
        self.remote = MonadoRemoteLink(
            host=str(getattr(config, "remote_host", "127.0.0.1")),
            port=int(getattr(config, "remote_port", 4242)),
            rate_hz=int(getattr(config, "send_rate_hz", 100)),
        )
        self.enabled = bool(getattr(config, "enabled", False))
        self.manage_service = bool(getattr(config, "manage_service", True))
        self.service = MonadoService(
            host=str(getattr(config, "remote_host", "127.0.0.1")),
            remote_port=int(getattr(config, "remote_port", 4242)),
        )

    @property
    def running(self) -> bool:
        return self.capture.running or self.remote.running or self.service.running

    def start(self) -> None:
        if not self.enabled:
            return
        service_error = ""
        if self.manage_service:
            try:
                self.service.start()
            except (MonadoServiceError, OSError) as exc:
                # An external service may still appear later; keep the capture
                # and remote links available and surface the problem in stats.
                service_error = str(exc)
                LOGGER.warning("Monado service: %s", exc)
        if service_error:
            self.service.note_error(service_error)
        self.capture.start()
        self.remote.start()

    def stop(self) -> None:
        self.remote.stop()
        self.capture.stop()
        self.service.stop()

    def update_frame(self, frame: ComposedFrame, now: Optional[float] = None) -> None:
        if self.enabled:
            self.remote.update_frame(frame, now)

    def stats(self) -> dict:
        capture = self.capture.stats()
        remote = self.remote.stats()
        service = self.service.stats()
        return {
            "enabled": self.enabled,
            "service": {
                "running": service.running,
                "managed": service.managed,
                "external": service.external,
                "pid": service.pid,
                "path": service.service,
                "log": service.log,
                "last_error": service.last_error,
            },
            "capture": {
                "running": capture.running,
                "connected": capture.connected,
                "frames": capture.frames,
                "fps": round(capture.fps, 2),
                "width": capture.width,
                "height": capture.height,
                "haptic_events": capture.haptic_events,
                "grab_left": round(capture.grab_left, 3),
                "grab_right": round(capture.grab_right, 3),
                "grab_active_left": capture.grab_active_left,
                "grab_active_right": capture.grab_active_right,
                "last_haptic": capture.last_haptic,
                "last_error": capture.last_error,
                "client": capture.client,
            },
            "remote": {
                "running": remote.running,
                "connected": remote.connected,
                "packets": remote.packets,
                "send_errors": remote.send_errors,
                "connect_attempts": remote.connect_attempts,
                "last_error": remote.last_error,
            },
        }
