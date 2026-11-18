"""Phone IMU receiver and clock-synchronisation responder.

The Android app streams rotation-vector, gyroscope and linear-acceleration
samples over UDP. This module validates and orders those samples, keeps the
latest one for the pose composer and answers TIME_PING packets so the phone can
estimate the desktop clock offset.
"""

from __future__ import annotations

import logging
import math
import os
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .mathutil import normalize
from .protocol import (
    PacketType,
    PhoneImuPacket,
    ProtocolError,
    TimePong,
    pack_time_pong,
    parse_packet,
)
from .types import PhoneSample

LOGGER = logging.getLogger(__name__)

SampleCallback = Callable[[PhoneSample], None]


@dataclass(slots=True)
class PhoneImuStats:
    running: bool = False
    address: str = ""
    last_seen: float = 0.0
    age_s: float = 0.0
    rate_hz: float = 0.0
    samples: int = 0
    parse_errors: int = 0
    pings: int = 0
    last_accuracy: int = 0
    last_sequence: int = 0
    screen_rotation: int = 0
    accurate: bool = False
    clock_offset_ms: float = 0.0
    dropped_samples: int = 0
    buffered_samples: int = 0

    @property
    def connected(self) -> bool:
        return self.address != "" and self.age_s <= 2.0


class PhoneImuReceiver:
    """UDP server for phone IMU and clock-sync packets."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 39101,
        on_sample: Optional[SampleCallback] = None,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ):
        self.host = host
        self.port = int(port)
        self.on_sample = on_sample
        self._socket_factory = socket_factory
        self._socket = self._open_socket()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._latest: Optional[PhoneSample] = None
        self._address = ""
        self._last_sequence: Optional[int] = None
        self._last_received = 0.0
        self._rate_hz = 0.0
        self._samples = 0
        self._parse_errors = 0
        self._pings = 0
        self._last_accuracy = 0
        self._screen_rotation = 0

        # Phone-to-desktop clock offset.  The minimum observed one-way delay
        # (receive time minus phone timestamp) is the best estimate short of a
        # hardware clock: every observed value is offset plus queueing delay,
        # so the smallest value is closest to the true offset.
        self._clock_offset: Optional[float] = None
        self._clock_candidates: deque[float] = deque(maxlen=256)
        # A one-sample reorder buffer lets a packet that overtook its
        # neighbour on the network be emitted in sensor order.  The timeout
        # flushes the last sample of a burst so a stopped stream leaves one
        # final pose available.
        self._reorder_samples = 1
        self._reorder_deadline_s = 0.020
        self._pending: list[PhoneSample] = []
        self._last_emitted_timestamp = 0.0
        self._dropped_samples = 0

    def _open_socket(self) -> socket.socket:
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.setblocking(False)
        return sock

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
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
        self._thread = threading.Thread(
            target=self._receive_loop, name="openvrcardboard-phone-imu", daemon=True
        )
        self._thread.start()
        LOGGER.info("phone IMU listening on %s:%d", self.host, self.local_port)

    def stop(self) -> None:
        was_running = self._running
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._socket.close()
        except OSError:
            pass
        if was_running:
            LOGGER.info("phone IMU receiver stopped")

    def __enter__(self) -> "PhoneImuReceiver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._socket], [], [], 0.004)
            except (OSError, ValueError):
                break
            now = time.monotonic()
            self.flush(now)
            if not readable:
                continue
            try:
                data, address = self._socket.recvfrom(4096)
            except BlockingIOError:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    break
                LOGGER.debug("phone IMU receive failed: %s", exc)
                continue
            self.handle_datagram(data, address, time.monotonic())

    # ------------------------------------------------------------------
    # Datagram handling
    # ------------------------------------------------------------------
    def handle_datagram(self, data: bytes, address, now: Optional[float] = None) -> bool:
        """Process one datagram. Returns True when a sample was accepted."""
        now = time.monotonic() if now is None else float(now)
        try:
            kind, payload = parse_packet(data)
        except ProtocolError as exc:
            with self._lock:
                self._parse_errors += 1
            LOGGER.debug("ignoring malformed phone packet from %s: %s", address, exc)
            return False

        if kind == PacketType.TIME_PING:
            with self._lock:
                self._pings += 1
            # The client timestamp uses the phone's monotonic clock, so a ping
            # that waits in the socket queue only overestimates the offset.
            self._note_clock_candidate(now - payload.client_time)
            pong = pack_time_pong(payload.sequence, payload.client_time, now)
            try:
                self._socket.sendto(pong, address)
            except OSError as exc:
                LOGGER.debug("could not send TIME_PONG to %s: %s", address, exc)
            return False

        if kind != PacketType.IMU:
            LOGGER.debug("ignoring unexpected phone packet type %s", kind)
            return False

        source = self._source_address(address)
        with self._lock:
            if source != self._address:
                # A different phone took over. Do not mix its sequence
                # numbers, clock offset or rate estimate with the previous
                # device.  Reset before converting this sample's timestamp.
                self._address = source
                self._last_sequence = None
                self._rate_hz = 0.0
                self._last_received = 0.0
                self._clock_offset = None
                self._clock_candidates.clear()
                self._pending.clear()
                self._last_emitted_timestamp = 0.0
                if source:
                    LOGGER.info("phone IMU samples from %s", source)

        sample = self._validate_imu(payload, address, now)
        if sample is None:
            return False

        emitted: list[PhoneSample] = []
        with self._lock:
            if self._last_sequence is not None:
                delta = (sample.sequence - self._last_sequence) & 0xFFFFFFFF
                if delta == 0:
                    self._dropped_samples += 1
                    return False
                if delta >= 0x80000000:
                    # A small backwards step is likely network reordering and
                    # goes through the reorder buffer; anything older is
                    # stale.  The timestamp check below makes the final call.
                    if (0x100000000 - delta) > 8:
                        self._dropped_samples += 1
                        return False
                else:
                    self._last_sequence = sample.sequence
            else:
                self._last_sequence = sample.sequence
            self._last_received = sample.received_at
            self._last_accuracy = sample.accuracy
            self._screen_rotation = sample.screen_rotation

            if sample.timestamp <= self._last_emitted_timestamp + 1e-9:
                self._dropped_samples += 1
                return False
            self._pending.append(sample)
            self._pending.sort(key=lambda item: item.timestamp)
            while len(self._pending) > self._reorder_samples:
                self._emit_locked(self._pending.pop(0), emitted)

        self._dispatch(emitted)
        return True

    def flush(self, now: Optional[float] = None) -> None:
        """Emit buffered samples whose reorder deadline has passed."""
        stamp = time.monotonic() if now is None else float(now)
        emitted: list[PhoneSample] = []
        with self._lock:
            while self._pending:
                oldest = self._pending[0]
                if stamp - oldest.received_at < self._reorder_deadline_s:
                    break
                self._pending.pop(0)
                self._emit_locked(oldest, emitted)
        self._dispatch(emitted)

    def _note_clock_candidate(self, candidate: float) -> None:
        if not math.isfinite(candidate):
            return
        with self._lock:
            if (
                self._clock_offset is None
                or abs(candidate - self._clock_offset) > 2.0
            ):
                # First packet, phone reboot or a different device: start a
                # fresh window instead of averaging unrelated clocks.
                self._clock_offset = candidate
                self._clock_candidates.clear()
            self._clock_candidates.append(candidate)
            self._clock_offset = min(self._clock_candidates)

    def _emit_locked(self, sample: PhoneSample, emitted: list[PhoneSample]) -> None:
        if sample.timestamp <= self._last_emitted_timestamp + 1e-9:
            self._dropped_samples += 1
            return
        previous = self._last_emitted_timestamp
        self._last_emitted_timestamp = sample.timestamp
        if previous > 0.0:
            dt = sample.timestamp - previous
            if 1e-4 < dt < 0.5:
                instant = 1.0 / dt
                self._rate_hz = (
                    instant
                    if self._rate_hz <= 0.0
                    else 0.9 * self._rate_hz + 0.1 * instant
                )
        self._latest = sample
        self._samples += 1
        emitted.append(sample)

    def _dispatch(self, samples: list[PhoneSample]) -> None:
        callback = self.on_sample
        if callback is None:
            return
        for sample in samples:
            try:
                callback(sample)
            except Exception:  # pragma: no cover - callback errors must not kill the receiver
                LOGGER.exception("phone IMU sample callback failed")

    def _validate_imu(
        self, payload: PhoneImuPacket, address, now: float
    ) -> Optional[PhoneSample]:
        quaternion = np.asarray(payload.quaternion, dtype=np.float64)
        gyroscope = np.asarray(payload.gyroscope, dtype=np.float64)
        acceleration = np.asarray(payload.acceleration, dtype=np.float64)
        if (
            quaternion.shape != (4,)
            or gyroscope.shape != (3,)
            or acceleration.shape != (3,)
            or not np.all(np.isfinite(quaternion))
            or not np.all(np.isfinite(gyroscope))
            or not np.all(np.isfinite(acceleration))
            or not math.isfinite(payload.timestamp)
        ):
            with self._lock:
                self._parse_errors += 1
            LOGGER.debug("ignoring IMU packet with non-finite values from %s", address)
            return None
        if float(np.linalg.norm(quaternion)) < 1e-8:
            with self._lock:
                self._parse_errors += 1
            return None

        phone_time = float(payload.timestamp)
        if phone_time > 0.0:
            self._note_clock_candidate(now - phone_time)
            offset = self._clock_offset if self._clock_offset is not None else 0.0
            desktop_time = phone_time + offset
        else:
            desktop_time = now
        # A phone that rebooted or restored the wrong clock must not inject a
        # pose seconds away from the frame that delivered it.
        if (
            not math.isfinite(desktop_time)
            or desktop_time > now + 0.05
            or now - desktop_time > 0.5
        ):
            with self._lock:
                self._dropped_samples += 1
            return None

        try:
            source = f"{address[0]}:{address[1]}"
        except (IndexError, TypeError):
            source = str(address)
        return PhoneSample(
            sequence=int(payload.sequence),
            timestamp=float(desktop_time),
            quaternion=normalize(quaternion),
            gyroscope=gyroscope,
            acceleration=acceleration,
            accuracy=int(payload.accuracy) & 0x03,
            received_at=now,
            source_address=source,
            screen_rotation=int(payload.screen_rotation) & 0x03,
        )

    @staticmethod
    def _source_address(address) -> str:
        try:
            return f"{address[0]}:{address[1]}"
        except (IndexError, TypeError):
            return str(address)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    def latest(self) -> Optional[PhoneSample]:
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def stats(self, now: Optional[float] = None) -> PhoneImuStats:
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            return PhoneImuStats(
                running=self._running,
                address=self._address,
                last_seen=self._last_received,
                age_s=(stamp - self._last_received) if self._last_received > 0.0 else math.inf,
                rate_hz=self._rate_hz,
                samples=self._samples,
                parse_errors=self._parse_errors,
                pings=self._pings,
                last_accuracy=self._last_accuracy,
                last_sequence=self._last_sequence or 0,
                screen_rotation=self._screen_rotation,
                accurate=self._last_accuracy >= 2,
                clock_offset_ms=(self._clock_offset or 0.0) * 1000.0,
                dropped_samples=self._dropped_samples,
                buffered_samples=len(self._pending),
            )
