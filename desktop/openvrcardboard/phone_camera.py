"""Phone-as-webcam WebSocket receiver and low-latency H.264 decoder.

The Android app captures its rear camera, encodes 720p H.264 with MediaCodec
and uploads Annex-B access units to ``/camera`` on ``phone.camera_port``. The
desktop decodes them with the same bundled ffmpeg used by the display stream
and publishes the newest BGR frame through the same interface as
:class:`openvrcardboard.camera.CameraCapture`.

Only one phone is accepted at a time. The token is the existing
``phone.token`` and is checked in the URL, so camera mode reuses the phone's
normal connection settings. A phone frame becomes stale after
``phone.camera_timeout_s``; :class:`CameraRouter` then falls back to the local
camera until frames arrive again.
"""

from __future__ import annotations

import json
import logging
import queue
import secrets
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Optional, cast
from urllib.parse import parse_qs, urlparse

import numpy as np

from .camera import CameraFrame, CameraStats
from .config import PhoneConfig
from .gpu import find_ffmpeg

LOGGER = logging.getLogger(__name__)

CAMERA_MAGIC = b"OVRC"
CAMERA_VERSION = 1
CAMERA_TYPE = 0x30
CAMERA_FLAG_KEYFRAME = 0x01
CAMERA_HEADER = struct.Struct("<4sBBHIdII")
CAMERA_HEADER_SIZE = CAMERA_HEADER.size
assert CAMERA_HEADER_SIZE == 28

_MAX_FRAME_BYTES = 8 << 20
_QUEUE_DEPTH = 120


def camera_key_for_info(info: dict) -> str:
    """Build a stable calibration key from a ``camera_hello`` payload.

    The Camera2 ID is only unique within one phone, so the key includes the
    device name reported by the app. Older apps that do not send a camera ID
    return an empty key and keep using the FOV estimate.
    """
    camera_id = str(info.get("camera_id", "") or "").strip()
    if not camera_id:
        return ""
    device = str(info.get("device", "") or info.get("name", "") or "phone").strip()
    return f"phone:{device}:{camera_id}"


class PhoneCameraError(RuntimeError):
    """Raised when the phone camera server cannot start."""


def pack_camera_frame(
    sequence: int,
    timestamp: float,
    width: int,
    height: int,
    payload: bytes,
    keyframe: bool = False,
) -> bytes:
    """Build one binary WebSocket camera message."""
    flags = CAMERA_FLAG_KEYFRAME if keyframe else 0
    return (
        CAMERA_HEADER.pack(
            CAMERA_MAGIC,
            CAMERA_VERSION,
            CAMERA_TYPE,
            flags,
            int(sequence) & 0xFFFFFFFF,
            float(timestamp),
            max(0, int(width)),
            max(0, int(height)),
        )
        + bytes(payload)
    )


def parse_camera_frame(data: bytes) -> tuple[int, float, int, int, bytes, bool]:
    """Return ``(sequence, timestamp, width, height, payload, keyframe)``."""
    if len(data) < CAMERA_HEADER_SIZE:
        raise ValueError("camera frame is shorter than its header")
    magic, version, kind, flags, sequence, timestamp, width, height = CAMERA_HEADER.unpack_from(
        data
    )
    if magic != CAMERA_MAGIC:
        raise ValueError("camera frame has a bad magic")
    if version != CAMERA_VERSION:
        raise ValueError(f"unsupported camera protocol version {version}")
    if kind != CAMERA_TYPE:
        raise ValueError(f"unexpected camera packet type {kind}")
    if not 16 <= width <= 7680 or not 16 <= height <= 4320:
        raise ValueError(f"implausible camera dimensions {width}x{height}")
    return (
        int(sequence),
        float(timestamp),
        int(width),
        int(height),
        bytes(data[CAMERA_HEADER_SIZE:]),
        bool(flags & CAMERA_FLAG_KEYFRAME),
    )


class _H264Decoder:
    """ffmpeg child process fed one Annex-B access unit at a time."""

    def __init__(
        self,
        width: int,
        height: int,
        on_frame: Callable[[np.ndarray, float], None],
        on_error: Optional[Callable[[str], None]] = None,
        on_keyframe_request: Optional[Callable[[], None]] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.on_frame = on_frame
        self.on_error = on_error
        self.on_keyframe_request = on_keyframe_request
        self._queue: queue.Queue[tuple[bytes, bool]] = queue.Queue(_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._waiting_keyframe = False
        self._last_keyframe_request = 0.0
        self._error = ""
        self._failed = False
        self.dropped = 0
        self.frames = 0
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._threads: list[threading.Thread] = []

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise PhoneCameraError(
                "phone camera decoding needs ffmpeg (install it or the "
                "'video' extra)"
            )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-flags",
            "low_delay",
            "-probesize",
            "200000",
            "-analyzeduration",
            "0",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "-flush_packets",
            "1",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise PhoneCameraError(f"could not start ffmpeg: {exc}") from exc

        writer = threading.Thread(
            target=self._write_loop, name="openvrcardboard-camera-h264-write", daemon=True
        )
        reader = threading.Thread(
            target=self._read_loop, name="openvrcardboard-camera-h264-read", daemon=True
        )
        stderr = threading.Thread(
            target=self._stderr_loop, name="openvrcardboard-camera-h264-log", daemon=True
        )
        self._threads = [writer, reader, stderr]
        for thread in self._threads:
            thread.start()

    @property
    def error(self) -> str:
        return self._error

    @property
    def failed(self) -> bool:
        return self._failed

    def feed(self, data: bytes, keyframe: bool) -> None:
        if not data:
            return
        with self._lock:
            if self._stop.is_set():
                return
            if self._waiting_keyframe and not keyframe:
                self.dropped += 1
                return
            if keyframe:
                self._waiting_keyframe = False
            try:
                self._queue.put_nowait((bytes(data), bool(keyframe)))
            except queue.Full:
                # The decoder or the network fell behind. Anything after a
                # lost reference frame is undecodable, so request a fresh
                # keyframe and drop until it arrives.
                self._waiting_keyframe = True
                self.dropped += 1
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                now = time.monotonic()
                if (
                    self.on_keyframe_request is not None
                    and now - self._last_keyframe_request > 0.25
                ):
                    self._last_keyframe_request = now
                    try:
                        self.on_keyframe_request()
                    except Exception:
                        pass

    def _write_loop(self) -> None:
        process = self._process
        stream = process.stdin if process is not None else None
        while not self._stop.is_set():
            try:
                data, _keyframe = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if stream is None:
                return
            try:
                stream.write(data)
            except (BrokenPipeError, OSError) as exc:
                self._fail(f"ffmpeg input closed: {exc}")
                return

    def _read_loop(self) -> None:
        process = self._process
        stream = cast(Any, process.stdout) if process is not None else None
        if stream is None:
            return
        frame_bytes = self.width * self.height * 3
        while not self._stop.is_set():
            image = np.empty((self.height, self.width, 3), dtype=np.uint8)
            view = memoryview(image).cast("B")
            offset = 0
            started = time.perf_counter()
            while offset < frame_bytes:
                try:
                    count = stream.readinto(view[offset:])
                except (BrokenPipeError, OSError, ValueError) as exc:
                    self._fail(f"ffmpeg output closed: {exc}")
                    return
                if not count:
                    if not self._stop.is_set():
                        self._fail(self._error or "ffmpeg stopped producing frames")
                    return
                offset += int(count)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self._stop.is_set():
                return
            self.frames += 1
            try:
                self.on_frame(image, elapsed_ms)
            except Exception:
                LOGGER.debug("phone camera frame callback failed", exc_info=True)

    def _stderr_loop(self) -> None:
        process = self._process
        stream = process.stderr if process is not None else None
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._error = line[-500:]
                LOGGER.debug("phone camera ffmpeg: %s", line)

    def _fail(self, message: str) -> None:
        if self._error:
            message = f"{message} ({self._error})"
        self._error = message
        self._failed = True
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                pass

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        process = self._process
        if process is not None:
            # Kill ffmpeg before touching stdin. A writer thread may be blocked
            # in a pipe write (notably on Windows when ffmpeg stopped reading),
            # and closing stdin first can then deadlock with that write. Once
            # the child is gone the blocked write fails and the pipe can be
            # closed safely.
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._threads = []


class PhoneCameraReceiver:
    """WebSocket server that turns a phone's H.264 stream into camera frames."""

    def __init__(self, config: PhoneConfig, host: str = "0.0.0.0"):
        self.config = config
        self.host = host
        self.token = str(getattr(config, "token", "") or "")
        self._requested_width = 0
        self._requested_height = 0
        self._requested_fps = 0
        self._bitrate_mbps = float(getattr(config, "camera_bitrate_mbps", 40.0))

        self._running = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._start_error = ""
        self._server: Optional[Any] = None
        self._server_thread: Optional[threading.Thread] = None
        self._local_port = int(getattr(config, "camera_port", 8766))
        self._session = secrets.token_hex(2)

        self._lock = threading.Lock()
        self._frame: Optional[CameraFrame] = None
        self._frames = 0
        self._sequence = 0
        self._fps = 0.0
        self._frame_ms = 0.0
        self._last_frame_time = 0.0
        self._last_error = ""
        self._connected = False
        self._address = ""
        self._info: dict = {}
        self._horizontal_fov_deg = 0.0
        self._camera_id = ""
        self._camera_key = ""
        self._camera_name = ""

        self._client_lock = threading.Lock()
        self._connection: Optional[Any] = None
        self._send_lock = threading.Lock()

        self._decoder_lock = threading.Lock()
        self._decoder: Optional[_H264Decoder] = None
        self._decoder_size = (0, 0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def local_port(self) -> int:
        return self._local_port

    def horizontal_fov(self) -> float:
        """Reported phone camera horizontal FOV in degrees (0 when unknown)."""
        with self._lock:
            return float(self._horizontal_fov_deg)

    def camera_id(self) -> str:
        """Camera2 ID reported by the connected phone (empty when unknown)."""
        with self._lock:
            return self._camera_id

    def camera_key(self) -> str:
        """Stable calibration key for the connected phone camera."""
        with self._lock:
            return self._camera_key

    def camera_name(self) -> str:
        """Human-readable name for the connected phone camera."""
        with self._lock:
            return self._camera_name

    def start(self, timeout: float = 5.0) -> None:
        if self._running:
            return
        self._stop.clear()
        self._ready.clear()
        self._start_error = ""
        self._running = True
        self._server_thread = threading.Thread(
            target=self._server_loop,
            name="openvrcardboard-phone-camera",
            daemon=True,
        )
        self._server_thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise PhoneCameraError("phone camera server did not start in time")
        if self._start_error:
            error = self._start_error
            self.stop()
            raise PhoneCameraError(error)
        LOGGER.info(
            "phone camera listening on port %d (token %s)",
            self._local_port,
            "required" if self.token else "not set",
        )

    def stop(self) -> None:
        if (
            not self._running
            and self._server is None
            and self._server_thread is None
        ):
            return
        self._running = False
        self._stop.set()
        self._close_client()
        self._close_decoder()
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        self._server = None
        with self._lock:
            self._frame = None
            self._connected = False
        LOGGER.info("phone camera stopped")

    def __enter__(self) -> "PhoneCameraReceiver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Server thread
    # ------------------------------------------------------------------
    def _server_loop(self) -> None:
        try:
            from websockets.sync.server import serve
        except ImportError:
            self._start_error = "phone camera needs the 'websockets' package"
            self._ready.set()
            return
        listener = None
        try:
            listener = socket.create_server(
                (self.host, int(getattr(self.config, "camera_port", 8766))),
                backlog=8,
            )
            self._local_port = int(listener.getsockname()[1])
            with serve(
                self._handle_client,
                sock=listener,
                max_size=_MAX_FRAME_BYTES,
                compression=None,
                ping_interval=10.0,
                ping_timeout=10.0,
            ) as server:
                self._server = server
                self._ready.set()
                server.serve_forever()
        except Exception as exc:
            self._start_error = (
                f"could not start phone camera on port "
                f"{getattr(self.config, 'camera_port', 8766)}: {exc}"
            )
            self._ready.set()
            LOGGER.warning("%s", self._start_error)
        finally:
            if listener is not None:
                try:
                    listener.close()
                except OSError:
                    pass
            if not self._running:
                self._ready.set()

    def _handle_client(self, connection: Any) -> None:
        path = ""
        try:
            path = connection.request.path
        except Exception:
            pass
        parsed = urlparse(path)
        if parsed.path not in ("/camera", "/webcam", "/"):
            self._send_json({"type": "error", "message": "unknown path"}, connection)
            connection.close()
            return
        if self.token:
            query = parse_qs(parsed.query)
            supplied = (query.get("token") or [""])[0]
            if not secrets.compare_digest(supplied, self.token):
                self._send_json(
                    {"type": "error", "message": "invalid pairing token"}, connection
                )
                connection.close()
                return

        previous = None
        with self._client_lock:
            previous = self._connection
            self._connection = connection
        if previous is not None and previous is not connection:
            try:
                previous.close(1012, "replaced by another phone")
            except Exception:
                pass

        address = str(getattr(connection, "remote_address", "") or "")
        with self._lock:
            self._connected = True
            self._address = address
            self._last_error = ""
            self._frame = None
            self._fps = 0.0
            self._last_frame_time = 0.0
            self._info = {}
            self._horizontal_fov_deg = 0.0
            self._camera_id = ""
            self._camera_key = ""
            self._camera_name = ""
        self._send_json(
            {
                "type": "camera_welcome",
                "protocol": 1,
                "session": self._session,
                "codec": "h264",
                "bitrate_mbps": self._bitrate_mbps,
                "prefer_width": 1280,
                "prefer_height": 720,
                "prefer_fps": 120,
            },
            connection,
        )
        LOGGER.info("phone camera connected: %s", address)

        try:
            while self._running:
                try:
                    message = connection.recv(timeout=0.5)
                except TimeoutError:
                    continue
                except Exception:
                    break
                if message is None:
                    break
                if isinstance(message, bytes):
                    self._handle_binary(message, connection)
                else:
                    self._handle_text(message)
        finally:
            with self._client_lock:
                current = self._connection is connection
                if current:
                    self._connection = None
            if current:
                self._close_decoder()
                with self._lock:
                    self._connected = False
                    self._frame = None
                    self._info = {}
                    self._horizontal_fov_deg = 0.0
                    self._camera_id = ""
                    self._camera_key = ""
                    self._camera_name = ""
            try:
                connection.close()
            except Exception:
                pass
            LOGGER.info("phone camera disconnected: %s", address)

    def _handle_text(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind in ("camera_hello", "hello"):
            info = dict(payload)
            try:
                fov = float(payload.get("hfov_deg", 0.0) or 0.0)
            except (TypeError, ValueError):
                fov = 0.0
            if not 10.0 <= fov <= 150.0:
                fov = 0.0

            def as_int(key: str) -> int:
                try:
                    return int(payload.get(key, 0) or 0)
                except (TypeError, ValueError):
                    return 0

            with self._lock:
                self._info = info
                self._horizontal_fov_deg = fov
                self._camera_id = str(payload.get("camera_id", "") or "").strip()
                self._camera_key = camera_key_for_info(info)
                self._camera_name = str(
                    payload.get("camera_name", "")
                    or payload.get("name", "")
                    or "phone camera"
                ).strip()
                self._requested_width = as_int("width")
                self._requested_height = as_int("height")
                self._requested_fps = as_int("fps")
            LOGGER.info(
                "phone camera hello: %s %sx%s @ %s fps (%s, hfov %.1f)",
                payload.get("name", "phone"),
                payload.get("width", "?"),
                payload.get("height", "?"),
                payload.get("fps", "?"),
                payload.get("codec", "h264"),
                fov,
            )
        elif kind == "camera_stats":
            LOGGER.debug("phone camera stats: %s", payload)
        elif kind == "ping":
            self._send_current_json({"type": "pong", "t": payload.get("t")})

    def _handle_binary(self, data: bytes, connection: Any) -> None:
        try:
            _sequence, _timestamp, width, height, payload, keyframe = parse_camera_frame(
                data
            )
        except ValueError as exc:
            with self._lock:
                self._last_error = str(exc)
            return
        if not payload:
            return
        decoder = self._ensure_decoder(width, height)
        if decoder is None:
            return
        decoder.feed(payload, keyframe)

    def _ensure_decoder(self, width: int, height: int) -> Optional[_H264Decoder]:
        with self._decoder_lock:
            if (
                self._decoder is not None
                and self._decoder_size == (width, height)
                and not self._decoder.failed
            ):
                return self._decoder
            old = self._decoder
            self._decoder = None
            self._decoder_size = (0, 0)
            if old is not None:
                old.close()
            try:
                decoder = _H264Decoder(
                    width,
                    height,
                    self._publish,
                    on_error=self._decoder_error,
                    on_keyframe_request=self._request_keyframe,
                )
            except PhoneCameraError as exc:
                with self._lock:
                    self._last_error = str(exc)
                return None
            self._decoder = decoder
            self._decoder_size = (width, height)
            return decoder

    def _close_decoder(self) -> None:
        with self._decoder_lock:
            decoder = self._decoder
            self._decoder = None
            self._decoder_size = (0, 0)
        if decoder is not None:
            decoder.close()

    def _decoder_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
        LOGGER.warning("phone camera decoder: %s", message)
        # This callback runs on a decoder thread, so it must not close the
        # decoder: close() joins those threads and would wait on itself.
        # The decoder marks itself failed; the connection thread replaces it
        # when the next frame arrives.
        self._request_keyframe()

    def _request_keyframe(self) -> None:
        self._send_current_json({"type": "camera_request_keyframe"})

    def _send_current_json(self, payload: dict) -> None:
        with self._client_lock:
            connection = self._connection
        if connection is not None:
            self._send_json(payload, connection)

    def _send_json(self, payload: dict, connection: Any) -> None:
        try:
            with self._send_lock:
                connection.send(json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Frame publication
    # ------------------------------------------------------------------
    def _publish(self, image: np.ndarray, elapsed_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._frames += 1
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            self._frame_ms = (
                0.9 * self._frame_ms + 0.1 * float(elapsed_ms)
                if self._frame_ms
                else float(elapsed_ms)
            )
            if self._last_frame_time > 0.0:
                dt = now - self._last_frame_time
                if 1e-5 < dt < 2.0:
                    instant = 1.0 / dt
                    self._fps = (
                        instant if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * instant
                    )
            self._last_frame_time = now
            height, width = image.shape[:2]
            self._frame = CameraFrame(
                image=image,
                timestamp=now,
                sequence=self._sequence,
                width=int(width),
                height=int(height),
                source="phone",
            )
            self._last_error = ""

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def active(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            if self._frame is None or not self._connected:
                return False
            limit = (
                max(0.05, float(timeout))
                if timeout is not None
                else max(0.05, float(getattr(self.config, "camera_timeout_s", 0.5)))
            )
            return time.monotonic() - self._frame.timestamp <= limit

    def latest(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._frame

    def latest_copy(self) -> Optional[CameraFrame]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        return CameraFrame(
            image=frame.image.copy(),
            timestamp=frame.timestamp,
            sequence=frame.sequence,
            width=frame.width,
            height=frame.height,
            source=frame.source,
        )

    def frame_if_active(self, timeout: Optional[float] = None) -> Optional[CameraFrame]:
        with self._lock:
            frame = self._frame
            if frame is None or not self._connected:
                return None
            limit = (
                max(0.05, float(timeout))
                if timeout is not None
                else max(0.05, float(getattr(self.config, "camera_timeout_s", 0.5)))
            )
            if time.monotonic() - frame.timestamp > limit:
                return None
            return frame

    def stats(self) -> CameraStats:
        with self._lock:
            decoder = self._decoder
            dropped = int(decoder.dropped) if decoder is not None else 0
            width = self._frame.width if self._frame is not None else 0
            height = self._frame.height if self._frame is not None else 0
            return CameraStats(
                running=self._running,
                opened=self._connected and self._frame is not None,
                fps=self._fps,
                frame_ms=self._frame_ms,
                width=width,
                height=height,
                frames=self._frames,
                read_errors=0,
                last_error=self._last_error,
                device=(
                    f"phone camera {self._address}".strip()
                    if self._connected
                    else "phone camera (waiting)"
                ),
                fourcc="H264",
                requested_width=self._requested_width,
                requested_height=self._requested_height,
                requested_fps=self._requested_fps,
                mode_ok=True,
                backend="phone",
                timestamp_source="arrival",
                source="phone",
                phone_connected=self._connected,
                phone_active=self._frame is not None,
                dropped_frames=dropped,
            )

    def status(self) -> dict:
        with self._lock:
            decoder = self._decoder
            connected = self._connected
            address = self._address
            width = self._frame.width if self._frame is not None else 0
            height = self._frame.height if self._frame is not None else 0
            return {
                "enabled": bool(getattr(self.config, "camera_enabled", True)),
                "running": self._running,
                "connected": connected,
                "address": address,
                "camera_id": self._camera_id,
                "camera_key": self._camera_key,
                "camera_name": self._camera_name,
                "port": self._local_port,
                "width": width,
                "height": height,
                "fps": self._fps,
                "frames": self._frames,
                "dropped": int(decoder.dropped) if decoder is not None else 0,
                "decoder_frames": int(decoder.frames) if decoder is not None else 0,
                "error": self._last_error,
                "info": dict(self._info),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _close_client(self) -> None:
        with self._client_lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            try:
                connection.close(1001, "stopping")
            except Exception:
                pass
