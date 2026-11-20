"""UDP link between the desktop app and the OpenVR Cardboard OpenVR driver.

The driver listens on loopback for STATE packets and answers with HAPTIC and
STATUS packets to the address that sent the last STATE. This module owns that
socket, sends the newest composed frame at a fixed rate from a dedicated thread
and dispatches driver events to callbacks.
"""

from __future__ import annotations

import logging
import select
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .fusion import ComposedFrame
from .mathutil import normalize
from .netutil import PacketRateLimiter
from .protocol import (
    DeviceFlags,
    DevicePose,
    DriverStatus,
    HapticEvent,
    PacketType,
    ProtocolError,
    pack_state,
    parse_packet,
)

LOGGER = logging.getLogger(__name__)

HapticCallback = Callable[[HapticEvent], None]
StatusCallback = Callable[[DriverStatus], None]


@dataclass(slots=True)
class DriverLinkStats:
    """Snapshot of link counters for the UI."""

    running: bool = False
    connected: bool = False
    send_packets: int = 0
    send_errors: int = 0
    receive_errors: int = 0
    status_packets: int = 0
    haptic_packets: int = 0
    parse_errors: int = 0
    rate_limited: int = 0
    last_status: Optional[DriverStatus] = None
    last_status_time: float = 0.0
    last_haptic: Optional[HapticEvent] = None
    last_send_time: float = 0.0
    # Prediction horizon the sender applies through the STATE packet.  Zero
    # means the driver consumes poses exactly as measured.
    prediction_ms: float = 0.0


class DriverLink:
    """Bidirectional UDP link to the OpenVR driver.

    ``update_frame`` is thread-safe and only stores a copy of the newest frame;
    the sender thread serialises it at ``rate_hz``. Call ``start`` to run the
    receiver and sender threads, or call ``send_once`` manually in tests.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 39100,
        rate_hz: int = 250,
        stale_after_s: float = 1.0,
        prediction_ms: float = 0.0,
        on_haptic: Optional[HapticCallback] = None,
        on_status: Optional[StatusCallback] = None,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ):
        self.host = host
        self.port = int(port)
        self.rate_hz = max(1, int(rate_hz))
        self.stale_after_s = max(0.0, float(stale_after_s))
        self.prediction_ms = max(0.0, min(1000.0, float(prediction_ms)))
        self.on_haptic = on_haptic
        self.on_status = on_status
        self._socket_factory = socket_factory

        self._target = (socket.gethostbyname(host), self.port)
        self._socket = self._open_socket()

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._frame: Optional[ComposedFrame] = None
        self._frame_time = 0.0
        self._sequence = 0

        self._stats = DriverLinkStats()
        self._stats_lock = threading.Lock()
        # The driver sends a few hundred packets per second at most.
        self._limiter = PacketRateLimiter(rate_hz=2000.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _open_socket(self) -> socket.socket:
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.bind(("0.0.0.0", 0))
        return sock

    @property
    def local_port(self) -> int:
        try:
            return int(self._socket.getsockname()[1])
        except OSError:
            return 0

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        if self._socket.fileno() < 0:
            self._socket = self._open_socket()
        self._running = True
        self._stop.clear()
        self._stats.running = True
        self._threads = [
            threading.Thread(target=self._send_loop, name="openvrcardboard-send", daemon=True),
            threading.Thread(target=self._receive_loop, name="openvrcardboard-recv", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        LOGGER.info("driver link sending to %s:%d from UDP port %d", self.host, self.port, self.local_port)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stats.running = False
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        try:
            self._socket.close()
        except OSError:
            pass
        LOGGER.info("driver link stopped")

    def __enter__(self) -> "DriverLink":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------
    def update_frame(self, frame: ComposedFrame, now: Optional[float] = None) -> None:
        """Publish the newest composed frame to the sender thread."""
        stamp = time.monotonic() if now is None else float(now)
        snapshot = frame.copy()
        with self._lock:
            self._frame = snapshot
            self._frame_time = stamp

    def clear_frame(self) -> None:
        with self._lock:
            self._frame = None
            self._frame_time = 0.0

    def send_once(self, now: Optional[float] = None) -> bool:
        """Serialise and send the current frame. Returns True on success."""
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            frame = self._frame
            frame_time = self._frame_time
            sequence = self._sequence
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        if frame is None:
            return False
        if self.stale_after_s > 0.0 and stamp - frame_time > self.stale_after_s:
            return False

        try:
            packet = build_state_packet(frame, sequence, stamp, self.prediction_ms)
        except (TypeError, ValueError, OverflowError) as exc:
            LOGGER.warning("could not serialise driver state: %s", exc)
            self._bump("send_errors")
            return False

        try:
            self._socket.sendto(packet, self._target)
        except OSError as exc:
            # A missing driver is normal while SteamVR is not running.
            self._bump("send_errors")
            LOGGER.debug("driver state send failed: %s", exc)
            return False

        with self._stats_lock:
            self._stats.send_packets += 1
            self._stats.last_send_time = stamp
        return True

    def _send_loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_time = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            self.send_once(now)
            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                # We fell behind (suspended laptop, loaded CPU); resynchronise
                # instead of trying to catch up with a burst.
                next_time = time.monotonic()

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------
    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._socket], [], [], 0.1)
            except (OSError, ValueError):
                break
            if not readable:
                continue
            for _ in range(32):
                try:
                    data, address = self._socket.recvfrom(4096)
                except BlockingIOError:
                    break
                except OSError as exc:
                    if self._stop.is_set():
                        break
                    # Linux reports ICMP port-unreachable on the next recv.
                    # The driver simply is not running yet.
                    LOGGER.debug("driver packet receive failed: %s", exc)
                    self._bump("receive_errors")
                    break
                if not self._limiter.allow(address, time.monotonic()):
                    self._bump("rate_limited")
                    continue
                self._handle_datagram(data, address)
                try:
                    readable, _, _ = select.select([self._socket], [], [], 0.0)
                except (OSError, ValueError):
                    break
                if not readable:
                    break

    def _handle_datagram(self, data: bytes, address) -> None:
        try:
            kind, payload = parse_packet(data)
        except ProtocolError as exc:
            LOGGER.debug("ignoring malformed driver packet from %s: %s", address, exc)
            self._bump("parse_errors")
            return

        if kind == PacketType.STATUS:
            with self._stats_lock:
                self._stats.status_packets += 1
                self._stats.last_status = payload
                self._stats.last_status_time = time.monotonic()
            self._invoke(self.on_status, payload)
        elif kind == PacketType.HAPTIC:
            with self._stats_lock:
                self._stats.haptic_packets += 1
                self._stats.last_haptic = payload
            self._invoke(self.on_haptic, payload)
        else:
            LOGGER.debug("ignoring unexpected packet type %s from driver", kind)

    @staticmethod
    def _invoke(callback, payload) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:  # pragma: no cover - callback errors must not kill the link
            LOGGER.exception("driver link callback failed")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _bump(self, name: str) -> None:
        with self._stats_lock:
            setattr(self._stats, name, getattr(self._stats, name) + 1)

    def stats(self, now: Optional[float] = None) -> DriverLinkStats:
        stamp = time.monotonic() if now is None else float(now)
        with self._stats_lock:
            result = DriverLinkStats(
                running=self._running,
                connected=False,
                send_packets=self._stats.send_packets,
                send_errors=self._stats.send_errors,
                receive_errors=self._stats.receive_errors,
                status_packets=self._stats.status_packets,
                haptic_packets=self._stats.haptic_packets,
                parse_errors=self._stats.parse_errors,
                rate_limited=self._stats.rate_limited,
                last_status=self._stats.last_status,
                last_status_time=self._stats.last_status_time,
                last_haptic=self._stats.last_haptic,
                last_send_time=self._stats.last_send_time,
                prediction_ms=self.prediction_ms,
            )
        if result.last_status_time > 0.0:
            result.connected = (stamp - result.last_status_time) <= 2.5
        return result


def build_state_packet(
    frame: ComposedFrame, sequence: int, timestamp: float, prediction_ms: float = 0.0
) -> bytes:
    """Convert a composed frame to a STATE datagram."""
    devices: list[DevicePose] = []
    for device_id, pose in frame.devices():
        position = _finite3(pose.position)
        rotation = normalize(_finite4(pose.rotation, (0.0, 0.0, 0.0, 1.0)))
        flags = DeviceFlags.CONNECTED
        if pose.valid:
            flags |= DeviceFlags.POSE_VALID
        if pose.orientation_only:
            flags |= DeviceFlags.ORIENTATION_ONLY
        devices.append(
            DevicePose(
                device_id=int(device_id),
                position=tuple(float(v) for v in position),
                rotation=tuple(float(v) for v in rotation),
                velocity=tuple(float(v) for v in _finite3(pose.velocity)),
                angular_velocity=tuple(float(v) for v in _finite3(pose.angular_velocity)),
                flags=flags,
            )
        )
    inputs = list(frame.inputs.values())
    return pack_state(sequence, float(timestamp), devices, inputs, prediction_ms)


def _finite3(value) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return np.zeros(3, dtype=np.float64)
    return np.where(np.isfinite(result), result, 0.0)


def _finite4(value, fallback=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return np.asarray(fallback, dtype=np.float64)
    if not np.all(np.isfinite(result)) or float(np.linalg.norm(result)) < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return result
