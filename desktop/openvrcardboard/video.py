"""Phone video capture, JPEG encoding and WebSocket streaming.

The desktop renders a mono or stereo frame, JPEG-encodes it and streams binary
frames to the Android app. Capture can come from the desktop (mss), a camera
(OpenCV) or a built-in animated test pattern, which is useful before SteamVR is
running.

``PhoneConfig.width`` / ``height`` are **per-eye** dimensions for stereo
layouts; the encoded frame is twice as wide (side-by-side) or twice as tall
(top-bottom). For mono they are the full frame dimensions.
"""

from __future__ import annotations

import base64
import io
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import PhoneConfig
from .encoder import create_stream_encoder, parameter_sets
from .protocol import VideoLayout, pack_video_frame

LOGGER = logging.getLogger(__name__)

_CV2 = None


def _cv2_encoder():
    """Return OpenCV when available (much faster JPEG encoding than Pillow)."""
    global _CV2
    if _CV2 is None:
        try:
            import cv2 as _module
        except ImportError:
            _CV2 = False
        else:
            _CV2 = _module
    return _CV2 or None


class VideoError(RuntimeError):
    """Raised when the stream cannot be started or a source fails."""


class VideoSourceError(VideoError):
    """Raised when a capture source cannot open or read a frame."""


@dataclass(slots=True)
class VideoStreamStats:
    running: bool = False
    clients: int = 0
    frames: int = 0
    fps: float = 0.0
    jpeg_bytes: int = 0
    last_error: str = ""
    source: str = ""
    layout: str = ""
    width: int = 0
    height: int = 0
    per_eye_width: int = 0
    per_eye_height: int = 0
    local_port: int = 0
    raw_frames: int = 0
    composed_frames: int = 0
    prepare_ms: float = 0.0
    codec: str = "jpeg"
    encoded_bytes: int = 0
    keyframes: int = 0
    encoder_error: str = ""


# ---------------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------------
def frame_dimensions(layout: str, width: int, height: int) -> tuple[int, int]:
    """Return the encoded ``(width, height)`` for a layout and per-eye size."""
    width = max(16, int(width))
    height = max(16, int(height))
    if layout == "sbs":
        return width * 2, height
    if layout == "top_bottom":
        return width, height * 2
    return width, height


def layout_value(layout: str) -> VideoLayout:
    if layout == "sbs":
        return VideoLayout.SBS
    if layout == "top_bottom":
        return VideoLayout.TOP_BOTTOM
    return VideoLayout.MONO


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return ImageOps.pad(
        image.convert("RGB"),
        size,
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


def compose_image(
    source: Image.Image,
    layout: str,
    width: int,
    height: int,
    annotate_eyes: bool = False,
) -> Image.Image:
    """Resize a mono source and duplicate it into the requested stereo layout."""
    eye = _fit(source, (max(16, int(width)), max(16, int(height))))
    if layout == "sbs":
        output = Image.new("RGB", (eye.width * 2, eye.height))
        output.paste(eye, (0, 0))
        output.paste(eye, (eye.width, 0))
    elif layout == "top_bottom":
        output = Image.new("RGB", (eye.width, eye.height * 2))
        output.paste(eye, (0, 0))
        output.paste(eye, (0, eye.height))
    else:
        output = eye

    if annotate_eyes and layout != "mono":
        draw = ImageDraw.Draw(output)
        font = _load_font(max(18, output.height // 12), bold=True)
        if layout == "sbs":
            draw.text((8, 8), "L", font=font, fill=(255, 220, 0))
            draw.text((eye.width + 8, 8), "R", font=font, fill=(255, 220, 0))
        else:
            draw.text((8, 8), "L", font=font, fill=(255, 220, 0))
            draw.text((8, eye.height + 8), "R", font=font, fill=(255, 220, 0))
    return output


def encode_jpeg(image: Image.Image, quality: int = 70) -> bytes:
    quality = max(20, min(95, int(quality)))
    cv2 = _cv2_encoder()
    if cv2 is not None:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            bgr = np.ascontiguousarray(array[:, :, ::-1])
            ok, buffer = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if ok:
                return buffer.tobytes()
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


@dataclass(slots=True)
class EncodedFrame:
    """One JPEG frame ready for the wire, with its encoded dimensions."""

    jpeg: bytes
    width: int
    height: int
    raw: bool = False


# ---------------------------------------------------------------------------
# Capture sources
# ---------------------------------------------------------------------------
class VideoSource:
    name = "none"
    stereo = False

    def open(self) -> None:
        pass

    def read(self) -> Image.Image:
        raise NotImplementedError

    def read_frame(
        self,
        layout: str,
        eye_width: int,
        eye_height: int,
        annotate_eyes: bool = False,
    ) -> Image.Image:
        """Return a complete RGB frame for the requested layout."""
        image = self.read()
        if self.stereo:
            return _fit(image, frame_dimensions(layout, eye_width, eye_height))
        return compose_image(
            image,
            layout,
            eye_width,
            eye_height,
            annotate_eyes=annotate_eyes,
        )

    def read_encoded(
        self,
        layout: str,
        eye_width: int,
        eye_height: int,
        quality: int,
        annotate_eyes: bool = False,
    ) -> EncodedFrame:
        """Capture and encode one frame for the phone.

        Sources that already produce a suitable JPEG (the OpenXR shim) override
        this to pass the compressed bytes through untouched.
        """
        output = self.read_frame(layout, eye_width, eye_height, annotate_eyes)
        return EncodedFrame(
            jpeg=encode_jpeg(output, quality),
            width=output.width,
            height=output.height,
        )

    def close(self) -> None:
        pass


class TestPatternSource(VideoSource):
    """Animated test card with a moving sweep and timing information."""

    __test__ = False
    name = "test"

    def __init__(self, width: int, height: int, pattern: str = "sbs"):
        self.width = max(32, int(width))
        self.height = max(32, int(height))
        self.pattern = pattern if pattern in ("sbs", "grid", "bars") else "sbs"
        self._font = _load_font(max(14, self.height // 24), bold=True)
        self._small = _load_font(max(10, self.height // 40))
        self._base = self._make_base()
        self._frames = 0

    def _make_base(self) -> Image.Image:
        width, height = self.width, self.height
        image = Image.new("RGB", (width, height), (12, 14, 20))
        draw = ImageDraw.Draw(image)
        if self.pattern == "bars":
            colors = [
                (192, 192, 192),
                (192, 192, 0),
                (0, 192, 192),
                (0, 192, 0),
                (192, 0, 192),
                (192, 0, 0),
                (0, 0, 192),
            ]
            bar_width = width / len(colors)
            for index, color in enumerate(colors):
                draw.rectangle(
                    [int(index * bar_width), 0, int((index + 1) * bar_width), int(height * 0.72)],
                    fill=color,
                )
            draw.rectangle([0, int(height * 0.72), width, height], fill=(20, 20, 20))
        else:
            # Subtle vertical gradient.
            for y in range(height):
                value = int(18 + 28 * y / max(1, height - 1))
                draw.line([(0, y), (width, y)], fill=(value, value + 2, value + 8))
            step = max(40, width // 16)
            for x in range(0, width, step):
                draw.line([(x, 0), (x, height)], fill=(40, 60, 85), width=1)
            for y in range(0, height, step):
                draw.line([(0, y), (width, y)], fill=(40, 60, 85), width=1)

        center = (width // 2, height // 2)
        radius = max(20, min(width, height) // 6)
        draw.ellipse(
            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
            outline=(255, 210, 0),
            width=max(2, height // 240),
        )
        draw.line([(center[0] - radius, center[1]), (center[0] + radius, center[1])], fill=(255, 210, 0), width=2)
        draw.line([(center[0], center[1] - radius), (center[0], center[1] + radius)], fill=(255, 210, 0), width=2)

        margin = max(8, height // 40)
        draw.text((margin, margin), "OpenVR Cardboard", font=self._font, fill=(245, 245, 245))
        draw.text(
            (margin, margin + self._font.size + 4),
            f"{self.width}x{self.height}  {self.pattern}",
            font=self._small,
            fill=(160, 200, 255),
        )
        return image

    def read(self) -> Image.Image:
        image = self._base.copy()
        draw = ImageDraw.Draw(image)
        sweep_width = max(6, self.width // 90)
        position = int((time.monotonic() * 160.0) % (self.width + sweep_width * 4)) - sweep_width * 2
        draw.rectangle(
            [position, 0, position + sweep_width, self.height], fill=(0, 220, 120)
        )
        draw.text(
            (max(8, self.height // 40), self.height - max(18, self.height // 22)),
            f"frame {self._frames}  {time.strftime('%H:%M:%S')}",
            font=self._small,
            fill=(255, 255, 255),
        )
        self._frames += 1
        return image


class ScreenSource(VideoSource):
    name = "screen"

    def __init__(self, monitor: int = 1, region: Optional[list[int]] = None):
        self.monitor = int(monitor)
        self.region = list(region) if region and len(region) == 4 else None
        self._mss = None
        self._capture = None

    def open(self) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VideoSourceError("screen capture needs the 'mss' package") from exc
        # mss 10 renamed the factory to MSS and kept mss() as a deprecated alias.
        factory = getattr(mss, "MSS", None) or getattr(mss, "mss")
        self._mss = factory()
        monitors = self._mss.monitors
        if self.region:
            x, y, width, height = (int(v) for v in self.region)
            if width <= 0 or height <= 0:
                raise VideoSourceError("screen capture region has an invalid size")
            self._capture = {"left": x, "top": y, "width": width, "height": height}
        else:
            index = self.monitor
            if index < 1 or index >= len(monitors):
                index = 1 if len(monitors) > 1 else 0
            self._capture = monitors[index]

    def read(self) -> Image.Image:
        if self._mss is None or self._capture is None:
            raise VideoSourceError("screen capture is not open")
        shot = self._mss.grab(self._capture)
        array = np.asarray(shot, dtype=np.uint8)
        # mss returns BGRA; drop alpha and swap blue/red.
        return Image.fromarray(np.ascontiguousarray(array[:, :, [2, 1, 0]]), mode="RGB")

    def close(self) -> None:
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
        self._mss = None
        self._capture = None


class CameraSource(VideoSource):
    name = "camera"

    def __init__(self, index: int = 1, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = int(index)
        self.width = max(160, int(width))
        self.height = max(120, int(height))
        self.fps = max(1, int(fps))
        self._capture = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VideoSourceError("camera capture needs the 'opencv-python' package") from exc
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"could not open camera index {self.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture

    def read(self) -> Image.Image:
        if self._capture is None:
            raise VideoSourceError("camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise VideoSourceError("camera returned no frame")
        return Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]), mode="RGB")

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None


class OpenXRCaptureSource(VideoSource):
    """Live SBS eye textures received from the OpenXR capture shim.

    The shim already sends a complete JPEG. When the requested phone frame has
    the same dimensions, that JPEG is forwarded byte-for-byte: no decode, no
    resize and no second JPEG generation. This is what keeps the 1200x1080
    per-eye stream comfortably above 90 FPS on a CPU.
    """

    name = "openxr"
    stereo = True

    def __init__(self, capture, timeout: float = 2.0):
        self.capture = capture
        self.timeout = max(0.1, float(timeout))
        self._sequence: Optional[int] = None

    def open(self) -> None:
        if self.capture is None:
            raise VideoSourceError("OpenXR capture is not configured")
        frame = self.capture.latest(timeout=self.timeout)
        if frame is None:
            raise VideoSourceError(
                "waiting for the OpenXR application to connect to the capture shim"
            )
        self._sequence = int(frame.sequence)

    def _next_frame(self):
        frame = self.capture.latest(
            after_sequence=self._sequence, timeout=self.timeout
        )
        if frame is None:
            raise VideoSourceError("OpenXR capture has no frames")
        if self._sequence is not None and int(frame.sequence) == self._sequence:
            raise VideoSourceError("OpenXR capture stopped producing frames")
        self._sequence = int(frame.sequence)
        return frame

    def read(self) -> Image.Image:
        frame = self._next_frame()
        image = Image.open(io.BytesIO(frame.jpeg))
        image.load()
        return image.convert("RGB")

    def read_encoded(
        self,
        layout: str,
        eye_width: int,
        eye_height: int,
        quality: int,
        annotate_eyes: bool = False,
    ) -> EncodedFrame:
        frame = self._next_frame()
        target = frame_dimensions(layout, eye_width, eye_height)
        if (
            layout == "sbs"
            and int(frame.width) == target[0]
            and int(frame.height) == target[1]
        ):
            return EncodedFrame(
                jpeg=frame.jpeg,
                width=target[0],
                height=target[1],
                raw=True,
            )
        image = Image.open(io.BytesIO(frame.jpeg))
        image.load()
        output = _fit(image.convert("RGB"), target)
        return EncodedFrame(
            jpeg=encode_jpeg(output, quality),
            width=output.width,
            height=output.height,
        )

    def close(self) -> None:
        self._sequence = None


def create_source(
    config: PhoneConfig,
    view_provider=None,
    capture_provider=None,
) -> VideoSource:
    if config.capture_source == "screen":
        return ScreenSource(config.capture_monitor, config.capture_region)
    if config.capture_source == "camera":
        return CameraSource(config.camera_index, config.width, config.height, config.fps)
    if config.capture_source == "openxr":
        return OpenXRCaptureSource(capture_provider)
    if config.capture_source == "game":
        from .game import StereoGameSource

        return StereoGameSource(
            config.layout,
            config.width,
            config.height,
            view_provider=view_provider,
        )
    return TestPatternSource(config.width, config.height, config.test_pattern)


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
