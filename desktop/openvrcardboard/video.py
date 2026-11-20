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

import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import PhoneConfig
from .protocol import VideoLayout

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
