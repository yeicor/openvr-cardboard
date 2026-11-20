"""WebSocket server that streams encoded phone video frames.

The capture and encoding sources live in :mod:`openvrcardboard.video`; this
module owns the client protocol, per-client queues and the capture thread.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import secrets
import socket
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .config import PhoneConfig
from .encoder import create_stream_encoder, parameter_sets
from .protocol import pack_video_frame
from .video import (
    EncodedFrame,
    VideoError,
    VideoSource,
    VideoStreamStats,
    _fit,
    create_source,
    encode_jpeg,
    frame_dimensions,
    layout_value,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------
@dataclass
class _Client:
    connection: object
    queue: queue.Queue
    send_lock: threading.Lock
    address: str = ""
    info: dict = None
    active: bool = True


class VideoStreamServer:
    """Capture/encode thread plus a WebSocket server for the phone app."""

    def __init__(
        self,
        config: PhoneConfig,
        host: str = "0.0.0.0",
        view_provider=None,
        capture_provider=None,
    ):
        self.config = config
        self.host = host
        self.view_provider = view_provider
        self.capture_provider = capture_provider
        self.width = max(16, int(config.width))
        self.height = max(16, int(config.height))
        self.layout = config.layout if config.layout in ("mono", "sbs", "top_bottom") else "sbs"
        self.frame_width, self.frame_height = frame_dimensions(
            self.layout, self.width, self.height
        )
        self.token = str(config.token or "")

        self._running = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._video_ready = threading.Event()
        self._start_error = ""
        self._capture_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server = None
        self._source: Optional[VideoSource] = None

        self._clients: list[_Client] = []
        self._clients_lock = threading.Lock()
        self._sequence = 0
        self._frames = 0
        self._fps = 0.0
        self._jpeg_bytes = 0
        self._last_error = ""
        self._local_port = int(config.video_port)
        self._session = secrets.token_hex(2)
        self._raw_frames = 0
        self._composed_frames = 0
        self._prepare_ms = 0.0
        self._encoder = None
        self._codec = "jpeg"
        self._encoded_bytes = 0
        self._keyframes = 0
        self._encoder_error = ""
        self._csd = b""
        self._last_keyframe: Optional[bytes] = None
        self._last_encode_time = 0.0
        self._fallback_requested = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def local_port(self) -> int:
        return self._local_port

    def start(self, timeout: float = 5.0) -> None:
        if self._running:
            return
        self._stop.clear()
        self._ready.clear()
        self._video_ready.clear()
        self._start_error = ""
        self._last_error = ""
        self._fallback_requested = False
        self._running = True

        self._server_thread = threading.Thread(
            target=self._server_loop, name="openvrcardboard-video-server", daemon=True
        )
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="openvrcardboard-video-capture", daemon=True
        )
        self._server_thread.start()
        self._capture_thread.start()

        if not self._ready.wait(timeout):
            self.stop()
            raise VideoError("video server did not start in time")
        if self._start_error:
            error = self._start_error
            self.stop()
            raise VideoError(error)
        LOGGER.info(
            "video streaming %s %dx%d (per eye %dx%d) on port %d",
            self.layout,
            self.frame_width,
            self.frame_height,
            self.width,
            self.height,
            self._local_port,
        )

    def stop(self) -> None:
        if (
            not self._running
            and self._server is None
            and self._server_thread is None
            and self._capture_thread is None
        ):
            return
        self._running = False
        self._stop.set()

        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            client.active = False
            try:
                client.connection.close()
            except Exception:
                pass

        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
        for thread in (self._server_thread, self._capture_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._server_thread = None
        self._capture_thread = None
        self._server = None
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
            self._source = None
        if self._encoder is not None:
            try:
                self._encoder.close()
            except Exception:
                pass
            self._encoder = None
        self._codec = "jpeg"
        self._csd = b""
        self._last_keyframe = None
        with self._clients_lock:
            self._clients.clear()
        LOGGER.info("video stream stopped")

    def __enter__(self) -> "VideoStreamServer":
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
            self._start_error = "video streaming needs the 'websockets' package"
            self._ready.set()
            return
        listener = None
        try:
            listener = socket.create_server(
                (self.host, int(self.config.video_port)), backlog=16
            )
            self._local_port = int(listener.getsockname()[1])
            with serve(
                self._handle_client,
                sock=listener,
                max_size=1 << 20,
                compression=None,
                ping_interval=20.0,
                ping_timeout=20.0,
            ) as server:
                self._server = server
                self._ready.set()
                server.serve_forever()
        except Exception as exc:
            self._start_error = f"could not start video server on port {self.config.video_port}: {exc}"
            self._ready.set()
            LOGGER.warning(self._start_error)
        finally:
            if listener is not None:
                try:
                    listener.close()
                except OSError:
                    pass
            if not self._running:
                self._ready.set()

    def _handle_client(self, connection) -> None:
        path = ""
        try:
            path = connection.request.path
        except Exception:
            pass
        parsed = urlparse(path)
        if parsed.path not in ("/video", "/"):
            self._control_send(connection, {"type": "error", "message": "unknown path"})
            connection.close()
            return
        if self.token:
            query = parse_qs(parsed.query)
            supplied = (query.get("token") or [""])[0]
            if not secrets.compare_digest(supplied, self.token):
                self._control_send(connection, {"type": "error", "message": "invalid pairing token"})
                connection.close()
                return

        client = _Client(
            connection=connection,
            queue=queue.Queue(maxsize=1),
            send_lock=threading.Lock(),
            address=str(getattr(connection, "remote_address", "")),
            info={},
        )
        with self._clients_lock:
            self._clients.append(client)
        LOGGER.info("phone video client connected: %s", client.address)

        # Wait briefly for the capture thread to decide between JPEG and a
        # hardware codec, otherwise the phone would configure the wrong
        # decoder and then see a different payload format.
        if not self._video_ready.is_set():
            self._video_ready.wait(timeout=2.0)

        self._client_send_text(
            client,
            {
                "type": "welcome",
                "protocol": 2,
                "session": self._session,
                "video": {
                    "format": self._codec,
                    "codec": self._codec,
                    "layout": self.layout,
                    "width": self.frame_width,
                    "height": self.frame_height,
                    "per_eye_width": self.width,
                    "per_eye_height": self.height,
                    "fps": int(self.config.fps),
                    "quality": int(self.config.jpeg_quality),
                    "bitrate_mbps": float(self.config.bitrate_mbps),
                    "csd": base64.b64encode(self._csd).decode("ascii") if self._csd else "",
                    "lens": asdict(self.config.lens),
                },
            },
        )
        if self._last_keyframe is not None:
            try:
                client.queue.put_nowait(self._last_keyframe)
            except queue.Full:
                pass

        sender = threading.Thread(
            target=self._client_sender, args=(client,), name="openvrcardboard-video-client", daemon=True
        )
        sender.start()
        try:
            while self._running and client.active:
                try:
                    message = connection.recv(timeout=0.5)
                except TimeoutError:
                    continue
                except Exception:
                    break
                if message is None:
                    break
                if isinstance(message, bytes):
                    continue
                self._handle_text(client, message)
        finally:
            client.active = False
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            try:
                connection.close()
            except Exception:
                pass
            sender.join(timeout=1.0)
            LOGGER.info("phone video client disconnected: %s", client.address)

    def _handle_text(self, client: _Client, message: str) -> None:
        try:
            payload = json.loads(message)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind == "hello":
            client.info = payload
            LOGGER.info(
                "phone says hello: %s %sx%s sensor %s Hz",
                payload.get("name", "phone"),
                payload.get("width", "?"),
                payload.get("height", "?"),
                payload.get("sensor_hz", "?"),
            )
        elif kind == "ping":
            self._client_send_text(client, {"type": "pong", "t": payload.get("t")})
        elif kind == "stats":
            LOGGER.debug("phone stats: %s", payload)
        elif kind == "video_fallback":
            # The phone has no decoder for the negotiated codec (or its
            # hardware decoder rejected the stream). Switch the whole session
            # to JPEG so every connected client keeps working.
            if self._encoder is not None:
                self._fallback_requested = True
                LOGGER.warning(
                    "phone requested JPEG fallback: %s",
                    payload.get("reason", "no compatible decoder"),
                )

    def _control_send(self, connection, payload: dict) -> None:
        try:
            connection.send(json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    def _client_send_text(self, client: _Client, payload: dict) -> None:
        try:
            with client.send_lock:
                client.connection.send(json.dumps(payload, separators=(",", ":")))
        except Exception:
            client.active = False

    def _client_sender(self, client: _Client) -> None:
        while self._running and client.active:
            try:
                packet = client.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                with client.send_lock:
                    client.connection.send(packet)
            except Exception:
                client.active = False
                break

    # ------------------------------------------------------------------
    # Capture thread
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        source_config = self.config
        source: Optional[VideoSource] = None
        retry_time = 0.0
        period = 1.0 / max(1, int(source_config.fps))
        next_frame = time.monotonic()

        while not self._stop.is_set():
            if self._fallback_requested and self._encoder is not None:
                self._fallback_requested = False
                self._encoder_error = "phone requested JPEG fallback"
                try:
                    self._encoder.close()
                except Exception:
                    pass
                self._encoder = None
                self._codec = "jpeg"
                self._csd = b""
                self._last_keyframe = None
                self.broadcast_control(
                    {
                        "type": "video_config",
                        "codec": "jpeg",
                        "csd": "",
                    }
                )
                LOGGER.warning("phone video switched to JPEG on phone request")
            now = time.monotonic()
            if source is None and now >= retry_time:
                try:
                    source = create_source(
                        source_config, self.view_provider, self.capture_provider
                    )
                    source.open()
                    self._source = source
                    self._last_error = ""
                    self._encoder = create_stream_encoder(
                        source_config, self.frame_width, self.frame_height
                    )
                    self._codec = self._encoder.codec if self._encoder is not None else "jpeg"
                    self._encoder_error = ""
                    self._csd = b""
                    self._last_keyframe = None
                    self._video_ready.set()
                except Exception as exc:
                    if source is not None:
                        try:
                            source.close()
                        except Exception:
                            pass
                    source = None
                    self._source = None
                    self._last_error = str(exc)
                    LOGGER.warning("video source unavailable: %s", exc)
                    retry_time = now + 1.0
                    self._stop.wait(0.2)
                    continue

            if source is None:
                self._stop.wait(0.1)
                continue

            try:
                started = time.perf_counter()
                if self._encoder is not None:
                    image = source.read_frame(
                        self.layout,
                        self.width,
                        self.height,
                        annotate_eyes=source.name == "test",
                    )
                    packets = self._encoder.encode(image, timeout=0.05)
                    finish = time.perf_counter()
                    self._composed_frames += 1
                    self._prepare_ms = 0.9 * self._prepare_ms + 0.1 * (
                        (finish - started) * 1000.0
                    )
                    if packets:
                        for packet in packets:
                            self._publish_encoded(
                                packet.data,
                                packet.width,
                                packet.height,
                                packet.codec,
                                packet.keyframe,
                                packet.timestamp,
                            )
                    else:
                        # The encoder pipeline is a few frames deep. Keep
                        # capturing; packets arrive on the next iterations.
                        pass
                else:
                    if hasattr(source, "read_encoded"):
                        encoded = source.read_encoded(
                            self.layout,
                            self.width,
                            self.height,
                            source_config.jpeg_quality,
                            annotate_eyes=source.name == "test",
                        )
                    else:
                        # Legacy providers such as StereoGameSource render in
                        # the requested layout themselves and only expose read().
                        image = source.read()
                        output = _fit(image, (self.frame_width, self.frame_height))
                        encoded = EncodedFrame(
                            jpeg=encode_jpeg(output, source_config.jpeg_quality),
                            width=output.width,
                            height=output.height,
                        )
                    finish = time.perf_counter()
                    jpeg = encoded.jpeg
                    if encoded.raw:
                        self._raw_frames += 1
                    else:
                        self._composed_frames += 1
                        self._prepare_ms = 0.9 * self._prepare_ms + 0.1 * (
                            (finish - started) * 1000.0
                        )
                    self._publish_encoded(
                        jpeg,
                        encoded.width,
                        encoded.height,
                        "jpeg",
                        True,
                        time.monotonic(),
                    )
            except Exception as exc:
                if self._encoder is not None:
                    # A GPU/encoder failure must never stop the phone stream:
                    # close it and continue on the always-available JPEG path.
                    self._encoder_error = str(exc)
                    LOGGER.warning(
                        "hardware stream failed, falling back to JPEG: %s", exc
                    )
                    try:
                        self._encoder.close()
                    except Exception:
                        pass
                    self._encoder = None
                    self._codec = "jpeg"
                    self._csd = b""
                    self._last_keyframe = None
                else:
                    self._last_error = str(exc)
                    LOGGER.warning("video capture failed: %s", exc)
                    try:
                        source.close()
                    except Exception:
                        pass
                    source = None
                    self._source = None
                    retry_time = time.monotonic() + 1.0
                    continue

            next_frame += period
            delay = next_frame - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_frame = time.monotonic()

        if source is not None:
            source.close()
        self._source = None
        if self._encoder is not None:
            try:
                self._encoder.close()
            except Exception:
                pass
        self._encoder = None

    def _publish_encoded(
        self,
        payload: bytes,
        width: int,
        height: int,
        codec: str,
        keyframe: bool,
        timestamp: float,
    ) -> None:
        sequence = self._sequence
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        packet = pack_video_frame(
            sequence,
            timestamp,
            layout_value(self.layout),
            width,
            height,
            payload,
            codec=codec,
            keyframe=keyframe,
        )
        self._frames += 1
        self._jpeg_bytes = len(payload)
        self._encoded_bytes += len(payload)
        if keyframe:
            self._keyframes += 1
            if codec in ("hevc", "h264") and not self._csd:
                csd = parameter_sets(codec, payload)
                if csd:
                    self._csd = csd
                    self.broadcast_control(
                        {
                            "type": "video_config",
                            "codec": codec,
                            "csd": base64.b64encode(csd).decode("ascii"),
                        }
                    )
            self._last_keyframe = packet
        now = time.monotonic()
        if self._last_encode_time > 0.0:
            dt = now - self._last_encode_time
            if 1e-5 < dt < 2.0:
                instant = 1.0 / dt
                self._fps = (
                    instant
                    if self._fps <= 0.0
                    else 0.9 * self._fps + 0.1 * instant
                )
        self._last_encode_time = now
        self._broadcast(packet)

    def _broadcast(self, packet: bytes) -> None:
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            if not client.active:
                continue
            try:
                client.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                client.queue.put_nowait(packet)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Control and stats
    # ------------------------------------------------------------------
    def broadcast_control(self, payload: dict) -> int:
        """Send a JSON control message to every connected phone."""
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            self._client_send_text(client, payload)
        return len(clients)

    def recenter(self) -> int:
        return self.broadcast_control({"type": "recenter"})

    def stats(self) -> VideoStreamStats:
        with self._clients_lock:
            clients = len(self._clients)
        return VideoStreamStats(
            running=self._running,
            clients=clients,
            frames=self._frames,
            fps=self._fps,
            jpeg_bytes=self._jpeg_bytes,
            last_error=self._last_error,
            source=self.config.capture_source,
            layout=self.layout,
            width=self.frame_width,
            height=self.frame_height,
            per_eye_width=self.width,
            per_eye_height=self.height,
            local_port=self._local_port,
            raw_frames=self._raw_frames,
            composed_frames=self._composed_frames,
            prepare_ms=self._prepare_ms,
            codec=self._codec,
            encoded_bytes=self._encoded_bytes,
            keyframes=self._keyframes,
            encoder_error=self._encoder_error,
        )
