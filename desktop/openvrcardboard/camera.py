"""Webcam capture thread feeding marker and MediaPipe tracking.

Capture runs in its own thread with a one-frame buffer so tracking can never
fall behind a stale backlog. If the camera disappears (USB glitch, another app
taking it) the thread retries once per second and reports the error through
``CameraStats``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CameraConfig

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when OpenCV is unavailable or the camera cannot be opened."""


@dataclass(slots=True)
class CameraFrame:
    image: np.ndarray
    timestamp: float
    sequence: int
    width: int
    height: int


@dataclass(slots=True)
class CameraStats:
    running: bool = False
    opened: bool = False
    fps: float = 0.0
    frame_ms: float = 0.0
    width: int = 0
    height: int = 0
    frames: int = 0
    read_errors: int = 0
    last_error: str = ""
    device: str = ""
    fourcc: str = ""
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: int = 0
    mode_ok: bool = False
    backend: str = ""
    # Live camera-control evidence.  These are the values the marker-aware
    # loop actually applied, not the configured requests.
    timestamp_source: str = "read"
    exposure: float = -1.0
    gain: float = -1.0
    brightness: float = 0.0
    detected_markers: int = 0
    visible_markers: int = 0
    auto_control: bool = False
    auto_actions: int = 0


def _fourcc_text(cv2, value: int) -> str:
    value = int(value or 0)
    if value <= 0:
        return ""
    text = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
    return "".join(char if 32 <= ord(char) < 127 else "." for char in text)


def _fourcc_value(cv2, text: str) -> int:
    text = (text or "").strip()
    if len(text) != 4:
        return 0
    value = 0
    for index, char in enumerate(text[:4]):
        # OpenCV uses the same packing as VideoWriter::fourcc: the first
        # character occupies the least significant byte.
        value |= ord(char) << 8 * index
    return value


class CameraCapture:
    """Owns a ``cv2.VideoCapture`` and publishes the newest BGR frame."""

    def __init__(
        self,
        config: CameraConfig,
        marker_ids: Optional[tuple[int, int, int]] = None,
        boards=None,
    ):
        self.config = config
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._marker_ids = tuple(int(value) for value in (marker_ids or (0, 5, 10)))
        self._boards = boards

        self._lock = threading.Lock()
        self._frame: Optional[CameraFrame] = None
        self._stats = CameraStats()
        self._last_frame_time = 0.0
        self._running = False

        self._cv2 = None
        self._capture = None

        # POS_MSEC is a stream clock, not a wall clock.  One anchor converts
        # its deltas to monotonic time; the first frame's transport delay is
        # removed by the calibrated offset if the user sets one.
        self._pos_anchor_value: Optional[float] = None
        self._pos_anchor_time = 0.0
        self._timestamp_source = "read"

        # Marker-aware exposure/gain loop.  Detection results arrive from the
        # tracking thread; exposure changes are rate-limited so the camera and
        # the detector can settle between steps.
        self._detection_lock = threading.Lock()
        self._detected_markers = 0
        self._visible_markers = 0
        self._last_detection_time = 0.0
        self._last_visible_time = 0.0
        self._last_auto_action = 0.0
        self._exposure_level: Optional[float] = None
        self._gain_level: Optional[float] = None
        self._auto_exposure_hint = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop.clear()
        with self._lock:
            self._frame = None
            self._stats.running = True
            self._stats.opened = False
            self._stats.last_error = ""
            self._last_frame_time = 0.0
        self._thread = threading.Thread(
            target=self._capture_loop, name="openvrcardboard-camera", daemon=True
        )
        self._thread.start()
        if getattr(self.config, "source", "webcam") == "synthetic":
            LOGGER.info("camera capture starting (synthetic scene)")
        else:
            LOGGER.info("camera capture starting (index %d)", self.config.index)

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()
        with self._lock:
            self._stats.running = False
            self._stats.opened = False
        LOGGER.info("camera capture stopped")

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _load_cv2(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CameraError(
                "webcam tracking needs the 'opencv-python' package"
            ) from exc
        self._cv2 = cv2
        return cv2

    def _open(self) -> bool:
        try:
            cv2 = self._load_cv2()
        except CameraError as exc:
            with self._lock:
                self._stats.last_error = str(exc)
            LOGGER.warning("%s", exc)
            return False

        backend_id = None
        backend = (self.config.backend or "auto").lower()
        if backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
            backend_id = cv2.CAP_V4L2
        elif backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
            backend_id = cv2.CAP_DSHOW
        elif backend == "msmf" and hasattr(cv2, "CAP_MSMF"):
            backend_id = cv2.CAP_MSMF
        elif backend == "any":
            backend_id = cv2.CAP_ANY

        requested = (
            str(self.config.fourcc or "").strip(),
            int(self.config.width),
            int(self.config.height),
            int(self.config.fps),
        )
        attempts = [requested]
        if self.config.allow_mode_fallback and self.config.verify_mode:
            # A mode the camera driver refuses must not silently leave the
            # app decoding 720p at 5 FPS. Prefer a lower resolution at the
            # requested frame rate, then the same resolution at a lower rate:
            # 1080p60, 720p60, 1080p30, 720p30, ...  A camera whose 1080p60
            # request negotiates 1080p30 is therefore retried at 1080p30
            # instead of settling for 720p30.
            attempts.extend(self._fallback_attempts(requested))

        capture = None
        for fourcc, width, height, fps in attempts:
            candidate = self._open_capture(cv2, backend_id, fourcc, width, height, fps)
            if candidate is None:
                continue
            mode = self._capture_mode(cv2, candidate)
            capture = candidate
            if not self.config.verify_mode or self._mode_matches(
                mode, (width, height, fps)
            ):
                break
            LOGGER.warning(
                "camera requested %s %dx%d@%d but negotiated %s %dx%d@%.0f",
                fourcc or "default",
                width,
                height,
                fps,
                mode[3],
                mode[0],
                mode[1],
                mode[2],
            )
            try:
                candidate.release()
            except Exception:
                pass
            capture = None

        if capture is None:
            with self._lock:
                self._stats.last_error = (
                    f"could not open camera index {self.config.index}"
                )
            return False

        self._capture = capture
        self._apply_camera_controls(cv2, capture)
        width, height, actual_fps, actual_fourcc = self._capture_mode(cv2, capture)
        requested_fps = int(self.config.fps)
        mode_ok = self._mode_matches(
            (width, height, actual_fps, actual_fourcc),
            (int(self.config.width), int(self.config.height), requested_fps),
        )
        with self._lock:
            self._stats.opened = True
            self._stats.width = width
            self._stats.height = height
            self._stats.last_error = ""
            self._stats.device = f"index {self.config.index}"
            self._stats.fourcc = actual_fourcc
            self._stats.requested_width = int(self.config.width)
            self._stats.requested_height = int(self.config.height)
            self._stats.requested_fps = requested_fps
            self._stats.mode_ok = mode_ok
            self._stats.backend = backend
        LOGGER.info(
            "camera opened: %s %dx%d @ %.1f fps (requested %s %dx%d@%d)",
            actual_fourcc or "default",
            width,
            height,
            actual_fps,
            self.config.fourcc or "default",
            self.config.width,
            self.config.height,
            requested_fps,
        )
        if not mode_ok:
            LOGGER.warning(
                "camera negotiated a lower mode; check USB bandwidth, cable, "
                "and whether %s is supported by the camera",
                self.config.fourcc or "the requested format",
            )
        return True

    def _open_capture(self, cv2, backend_id, fourcc, width, height, fps):
        try:
            if backend_id is None:
                capture = cv2.VideoCapture(int(self.config.index))
            else:
                capture = cv2.VideoCapture(int(self.config.index), backend_id)
        except Exception as exc:
            with self._lock:
                self._stats.last_error = f"could not open camera: {exc}"
            LOGGER.warning("could not open camera: %s", exc)
            return None
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            return None
        try:
            # Order matters: UVC negotiates its frame-size/frame-rate list from
            # the selected pixel format. Selecting MJPG first is what turns a
            # 1080p webcam from 5 FPS YUYV into 60 FPS MJPG.
            value = _fourcc_value(cv2, fourcc)
            if value and hasattr(cv2, "CAP_PROP_FOURCC"):
                capture.set(cv2.CAP_PROP_FOURCC, float(value))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            capture.set(cv2.CAP_PROP_FPS, float(fps))
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        except Exception as exc:
            with self._lock:
                self._stats.last_error = f"could not configure camera: {exc}"
            LOGGER.warning("could not configure camera: %s", exc)
            try:
                capture.release()
            except Exception:
                pass
            return None
        return capture

    @staticmethod
    def _capture_mode(cv2, capture) -> tuple[int, int, float, str]:
        def get(prop, fallback=0.0):
            try:
                return capture.get(prop)
            except Exception:
                return fallback

        width = int(get(cv2.CAP_PROP_FRAME_WIDTH, 0) or 0)
        height = int(get(cv2.CAP_PROP_FRAME_HEIGHT, 0) or 0)
        fps = float(get(cv2.CAP_PROP_FPS, 0.0) or 0.0)
        fourcc = ""
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            fourcc = _fourcc_text(cv2, int(get(cv2.CAP_PROP_FOURCC, 0) or 0))
        return width, height, fps, fourcc

    @staticmethod
    def _fallback_attempts(
        requested: tuple[str, int, int, int],
    ) -> list[tuple[str, int, int, int]]:
        """Return UVC fallback modes in least-surprising order.

        Resolution is capped at the requested size so a 720p request never
        silently opens 1080p.  Frame rates are capped too: a camera asked for
        30 FPS is not retried at 60 FPS.
        """
        fourcc, width, height, fps = requested
        if width <= 0 or height <= 0:
            return []
        # Keep the 1080p/720p ladder interleaved by frame rate:
        # 1080p60, 720p60, 1080p30, 720p30. 640x480 follows the ladder so a
        # camera is not dropped to 640p before its 1080p30 mode is tried.
        sizes = [(int(width), int(height))]
        for candidate in ((1920, 1080), (1280, 720)):
            if candidate[0] <= width and candidate[1] <= height:
                if candidate not in sizes:
                    sizes.append(candidate)
        fps_order = [int(fps)] if fps > 0 else [60, 30]
        for candidate in (60, 30):
            if candidate < fps_order[0] and candidate not in fps_order:
                fps_order.append(candidate)
        attempts: list[tuple[str, int, int, int]] = []
        for frame_rate in fps_order:
            for candidate_width, candidate_height in sizes:
                attempt = (
                    str(fourcc or ""),
                    candidate_width,
                    candidate_height,
                    frame_rate,
                )
                if attempt != requested and attempt not in attempts:
                    attempts.append(attempt)
        for frame_rate in fps_order:
            attempt = (str(fourcc or ""), 640, 480, frame_rate)
            if attempt != requested and attempt not in attempts:
                attempts.append(attempt)
        # Some UVC cameras expose a mode only with the backend's default
        # pixel format. Keep two modest fallbacks at the very end, but never
        # exceed the requested size or frame rate.
        for attempt in (("", 1280, 720, 30), ("", 640, 480, 30)):
            if attempt[1] > width or attempt[2] > height:
                continue
            if fps > 0 and attempt[3] > fps:
                continue
            if attempt not in attempts and attempt != requested:
                attempts.append(attempt)
        return attempts

    @staticmethod
    def _mode_matches(actual: tuple[int, int, float, str], requested: tuple[int, int, int]) -> bool:
        width, height, fps, _fourcc = actual
        req_width, req_height, req_fps = requested
        if width <= 0 or height <= 0:
            return False
        # Some backends only report the real frame rate once frames flow. Do
        # not reject a correctly sized mode because of an FPS placeholder.
        if width < req_width or height < req_height:
            return False
        if fps > 1.0 and req_fps > 1 and fps < req_fps * 0.8:
            return False
        return True

    def _apply_camera_controls(self, cv2, capture) -> None:
        config = self.config
        marker_control = bool(config.auto_marker_control) and float(
            config.camera_aggressiveness
        ) > 0.0
        controls: list[tuple[str, bool, float]] = []
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            controls.append(("CAP_PROP_AUTOFOCUS", True, 1.0 if config.autofocus else 0.0))
        if not config.autofocus and config.focus >= 0 and hasattr(cv2, "CAP_PROP_FOCUS"):
            controls.append(("CAP_PROP_FOCUS", False, config.focus))
        if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            # V4L2 uses 3 for automatic and 1 for manual exposure; DirectShow
            # ignores this property, so failures are harmless.  Marker-aware
            # control always takes ownership of exposure.
            automatic = bool(config.auto_exposure) and not marker_control
            controls.append(
                ("CAP_PROP_AUTO_EXPOSURE", True, 3.0 if automatic else 1.0)
            )
        if not config.auto_exposure or marker_control:
            initial = self._initial_exposure(cv2, capture)
            if initial is not None and hasattr(cv2, "CAP_PROP_EXPOSURE"):
                controls.append(("CAP_PROP_EXPOSURE", False, initial))
                self._exposure_level = initial
            if config.gain >= 0 and hasattr(cv2, "CAP_PROP_GAIN"):
                controls.append(("CAP_PROP_GAIN", False, config.gain))
                self._gain_level = config.gain
            elif marker_control and hasattr(cv2, "CAP_PROP_GAIN"):
                # Start from the camera's current gain instead of jumping to
                # the maximum.
                current = self._get_prop(cv2, capture, "CAP_PROP_GAIN", 0.0)
                if self._valid_prop(current):
                    controls.append(("CAP_PROP_GAIN", False, current))
                    self._gain_level = float(current)
        if hasattr(cv2, "CAP_PROP_AUTO_WB"):
            controls.append(("CAP_PROP_AUTO_WB", True, 1.0 if config.auto_white_balance else 0.0))
        if (
            not config.auto_white_balance
            and config.white_balance >= 0
            and hasattr(cv2, "CAP_PROP_WB_TEMPERATURE")
        ):
            controls.append(("CAP_PROP_WB_TEMPERATURE", False, config.white_balance))
        for name, always, value in controls:
            if not always and value < 0:
                continue
            try:
                capture.set(getattr(cv2, name), float(value))
            except Exception:
                pass
        with self._lock:
            self._stats.auto_control = marker_control
            self._stats.exposure = (
                float(self._exposure_level)
                if self._exposure_level is not None
                else -1.0
            )
            self._stats.gain = (
                float(self._gain_level) if self._gain_level is not None else -1.0
            )

    @staticmethod
    def _get_prop(cv2, capture, name: str, fallback: float = 0.0) -> float:
        prop = getattr(cv2, name, None)
        if prop is None:
            return float(fallback)
        try:
            return float(capture.get(prop))
        except Exception:
            return float(fallback)

    @staticmethod
    def _valid_prop(value: float) -> bool:
        return math.isfinite(value) and value >= 0.0

    def _initial_exposure(self, cv2, capture) -> Optional[float]:
        """Choose a manual starting exposure for the marker-aware loop.

        V4L2/UVC reports absolute exposure (often 100 us units), while
        DirectShow reports log2 seconds (negative).  Start from the user's
        manual value when present, otherwise from the current value, and keep
        the loop's steps within the convention the camera reports.
        """
        if hasattr(cv2, "CAP_PROP_EXPOSURE"):
            if float(self.config.exposure) >= 0.0:
                return float(self.config.exposure)
            current = self._get_prop(cv2, capture, "CAP_PROP_EXPOSURE", 0.0)
            if math.isfinite(current) and current != 0.0:
                return current
        backend = (self.config.backend or "auto").lower()
        if backend in ("dshow", "msmf"):
            return -6.0
        return 80.0

    def report_marker_detection(self, detected: int, visible: int) -> None:
        """Publish marker coverage for the automatic exposure controller.

        When a previously visible board disappears entirely, the last visible
        count is retained so the loop can lengthen exposure and recover it
        instead of freezing at a too-short setting.
        """
        with self._detection_lock:
            self._detected_markers = max(0, int(detected))
            if int(visible) > 0:
                self._visible_markers = int(visible)
                self._last_visible_time = time.monotonic()
            self._last_detection_time = time.monotonic()

    def _release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        with self._lock:
            self._stats.opened = False

    def _frame_timestamp(self, cv2, pos_msec: float, read_time: float) -> float:
        """Map the camera stream clock to monotonic time.

        ``CAP_PROP_POS_MSEC`` removes the jitter of the read call from frame
        intervals, which is what the Kalman filters integrate.  A backend that
        reports zero (or jumps) falls back to the read time.  The first frame
        anchors the stream clock, so no absolute clock sync is required.
        """
        source = str(getattr(self.config, "camera_timestamp_source", "auto") or "auto").lower()
        offset_s = float(getattr(self.config, "camera_timestamp_offset_ms", 0.0)) / 1000.0
        use_pos = math.isfinite(float(pos_msec)) and float(pos_msec) > 1e-3
        if source == "read":
            use_pos = False
        elif source == "pos_msec" and not use_pos:
            use_pos = False

        stamp = read_time
        self._timestamp_source = "read"
        if use_pos and hasattr(cv2, "CAP_PROP_POS_MSEC"):
            value = float(pos_msec) / 1000.0
            anchor_value = self._pos_anchor_value
            if (
                anchor_value is None
                or value < anchor_value - 1.0
                or value > anchor_value + 30.0
            ):
                self._pos_anchor_value = value
                self._pos_anchor_time = read_time
                anchor_value = value
            stamp = self._pos_anchor_time + (value - anchor_value)
            # A broken backend clock must not move a pose seconds away from
            # the frame that actually produced it.
            if abs(stamp - read_time) > 0.5:
                self._pos_anchor_value = value
                self._pos_anchor_time = read_time
                stamp = read_time
            self._timestamp_source = "pos_msec"
        return stamp - offset_s

    def _maybe_adapt_camera(self, cv2, image: np.ndarray, now: float) -> None:
        """Shorten exposure until marker coverage drops, then back off.

        Exposure is reduced while every currently visible board is detected
        and the image is bright enough, because that is the setting with the
        least motion blur.  Gain is raised first when the image is dark; if
        gain is already at the limit, exposure is lengthened.  Marker coverage
        that drops while the exposure is being shortened immediately restores
        the previous value.
        """
        config = self.config
        if not config.auto_marker_control or float(config.camera_aggressiveness) <= 0.0:
            return
        aggressiveness = max(0.0, min(1.0, float(config.camera_aggressiveness)))
        interval = 1.0 - 0.75 * aggressiveness
        if now - self._last_auto_action < interval:
            return
        with self._detection_lock:
            detected = self._detected_markers
            visible = self._visible_markers
            detection_age = now - self._last_detection_time
            visible_age = now - self._last_visible_time
        if detection_age > 1.0 or visible <= 0 or visible_age > 1.0:
            # Only recover markers that were on screen moments ago.  Without
            # this, pointing the camera at a wall would drive exposure to the
            # limit because the last seen board never expires.
            return

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray)) / 255.0
        except Exception:
            return
        self._last_auto_action = now
        target = max(0.1, min(0.9, float(config.camera_target_brightness)))
        coverage = detected / max(1, visible)
        changed = False

        exposure = self._exposure_level
        gain = self._gain_level
        if exposure is None:
            exposure = self._initial_exposure(cv2, self._capture)
        if gain is None:
            gain = max(0.0, float(config.gain) if config.gain >= 0 else 0.0)

        def apply(name: str, value: float) -> bool:
            prop = getattr(cv2, name, None)
            if prop is None or self._capture is None:
                return False
            try:
                self._capture.set(prop, float(value))
                return True
            except Exception:
                return False

        log2_exposure = exposure is not None and exposure <= 0.0
        step = 0.25 + 0.75 * aggressiveness
        if log2_exposure:
            exposure_step = step * 0.5
        else:
            exposure_step = max(1.0, (exposure or 1.0) * 0.12 * step)

        if coverage < 0.75:
            # Missing markers: undo a too-aggressive shortening.  If gain can
            # still rise, do that first so the extra light is blur-free.
            if gain < float(config.camera_max_gain) and brightness < target:
                gain = min(float(config.camera_max_gain), max(gain + 8.0 * step, 8.0))
                changed = apply("CAP_PROP_GAIN", gain)
            elif exposure is not None:
                exposure = self._step_exposure(exposure, +1, exposure_step, log2_exposure)
                changed = apply("CAP_PROP_EXPOSURE", exposure)
        elif brightness < target - 0.06:
            if gain < float(config.camera_max_gain):
                gain = min(float(config.camera_max_gain), max(gain + 8.0 * step, 8.0))
                changed = apply("CAP_PROP_GAIN", gain)
            elif exposure is not None:
                exposure = self._step_exposure(exposure, +1, exposure_step, log2_exposure)
                changed = apply("CAP_PROP_EXPOSURE", exposure)
        else:
            # Complete coverage and adequate brightness: trade gain for a
            # shorter exposure, which is the only way to remove motion blur.
            if brightness > target + 0.18 and gain > 0.0:
                gain = max(0.0, gain - 8.0 * step)
                changed = apply("CAP_PROP_GAIN", gain)
            elif exposure is not None:
                shorter = self._step_exposure(
                    exposure, -1, exposure_step, log2_exposure
                )
                if abs(shorter - exposure) > 1e-9:
                    exposure = shorter
                    changed = apply("CAP_PROP_EXPOSURE", exposure)
                if brightness < target - 0.10 and gain < float(config.camera_max_gain):
                    gain = min(float(config.camera_max_gain), max(gain + 4.0 * step, 4.0))
                    changed = apply("CAP_PROP_GAIN", gain) or changed

        self._exposure_level = exposure
        self._gain_level = gain
        with self._lock:
            self._stats.exposure = float(exposure) if exposure is not None else -1.0
            self._stats.gain = float(gain)
            self._stats.brightness = brightness
            self._stats.detected_markers = detected
            self._stats.visible_markers = visible
            if changed:
                self._stats.auto_actions += 1

    @staticmethod
    def _step_exposure(current: float, direction: int, step: float, log2: bool) -> float:
        # direction +1 = longer exposure (more light); -1 = shorter.  For
        # DirectShow's log2-seconds scale, larger values are longer too.
        if log2:
            return max(-13.0, min(-1.0, float(current) + direction * step))
        return max(1.0, min(10000.0, float(current) + direction * step))

    def _publish_frame(self, image: np.ndarray, elapsed_ms: float, stamp: float) -> None:
        image = np.asarray(image)
        if self.config.mirror_preview and image.ndim == 3:
            image = np.ascontiguousarray(image[:, ::-1])
        with self._lock:
            self._stats.frames += 1
            self._stats.frame_ms = (
                0.9 * self._stats.frame_ms + 0.1 * elapsed_ms
                if self._stats.frame_ms
                else elapsed_ms
            )
            if self._last_frame_time > 0.0:
                dt = stamp - self._last_frame_time
                if 1e-5 < dt < 2.0:
                    instant = 1.0 / dt
                    self._stats.fps = (
                        instant
                        if self._stats.fps <= 0.0
                        else 0.9 * self._stats.fps + 0.1 * instant
                    )
            self._last_frame_time = stamp
            self._stats.width = int(image.shape[1]) if image.ndim >= 2 else 0
            self._stats.height = int(image.shape[0]) if image.ndim >= 2 else 0
            sequence = self._stats.frames
            self._frame = CameraFrame(
                image=image,
                timestamp=stamp,
                sequence=sequence,
                width=self._stats.width,
                height=self._stats.height,
            )
            self._stats.last_error = ""

    def _synthetic_loop(self) -> None:
        from .synthetic import SyntheticError, SyntheticScene

        try:
            scene = SyntheticScene(
                width=self.config.width,
                height=self.config.height,
                horizontal_fov_deg=self.config.horizontal_fov_deg,
                dictionary=self.config.dictionary,
                boards=self._boards,
            )
        except (SyntheticError, RuntimeError, ValueError) as exc:
            with self._lock:
                self._stats.last_error = f"synthetic camera unavailable: {exc}"
            LOGGER.warning("synthetic camera unavailable: %s", exc)
            self._running = False
            return

        # Marker identifiers are filled by the tracking configuration; the
        # scene's defaults match the documented 0/1/2 layout.
        with self._lock:
            self._stats.opened = True
            self._stats.device = "synthetic scene"
            self._stats.width = scene.width
            self._stats.height = scene.height
            self._stats.last_error = ""
        LOGGER.info(
            "synthetic camera running: %dx%d at %.1f fps",
            scene.width,
            scene.height,
            float(self.config.fps),
        )

        start = time.monotonic()
        period = 1.0 / max(1, int(self.config.fps))
        next_frame = start
        while not self._stop.is_set():
            stamp = time.monotonic()
            started = time.perf_counter()
            try:
                image = scene.render(stamp - start)
            except Exception as exc:
                with self._lock:
                    self._stats.last_error = f"synthetic render failed: {exc}"
                LOGGER.warning("synthetic render failed: %s", exc)
                self._stop.wait(0.2)
                continue
            elapsed = (time.perf_counter() - started) * 1000.0
            self._publish_frame(image, elapsed, stamp)

            next_frame += period
            delay = next_frame - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_frame = time.monotonic()

        with self._lock:
            self._stats.opened = False

    def _capture_loop(self) -> None:
        if getattr(self.config, "source", "webcam") == "synthetic":
            self._synthetic_loop()
            return
        cv2 = None
        retry_at = 0.0
        errors = 0
        period = 1.0 / max(1, int(self.config.fps))
        next_frame = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            if self._capture is None:
                if now < retry_at:
                    self._stop.wait(0.1)
                    continue
                if not self._open():
                    retry_at = now + 1.0
                    self._stop.wait(0.2)
                    continue
                cv2 = self._cv2
                errors = 0
                next_frame = time.monotonic()

            started = time.perf_counter()
            frame = None
            try:
                ok, frame = self._capture.read()
            except Exception as exc:
                ok, frame = False, None
                with self._lock:
                    self._stats.last_error = str(exc)
                LOGGER.debug("camera read raised: %s", exc)
            elapsed = (time.perf_counter() - started) * 1000.0
            read_time = time.monotonic()
            pos_msec = self._get_prop(cv2, self._capture, "CAP_PROP_POS_MSEC", 0.0)

            if not ok or frame is None:
                errors += 1
                with self._lock:
                    self._stats.read_errors += 1
                    self._stats.last_error = "camera returned no frame"
                if errors >= 5:
                    LOGGER.warning("camera stopped returning frames; reopening")
                    self._release()
                    retry_at = time.monotonic() + 0.5
                else:
                    self._stop.wait(0.005)
                continue
            errors = 0

            image = np.asarray(frame)
            stamp = self._frame_timestamp(cv2, pos_msec, read_time)
            self._publish_frame(image, elapsed, stamp)
            self._maybe_adapt_camera(cv2, image, read_time)
            with self._lock:
                self._stats.timestamp_source = self._timestamp_source

            # Pace reads to the configured rate. The capture thread keeps only
            # the newest frame, so this only prevents a busy loop when the
            # driver returns buffered frames faster than the sensor rate.
            next_frame += period
            delay = next_frame - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_frame = time.monotonic()

        self._release()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def latest(self) -> Optional[CameraFrame]:
        """Return the newest frame. The image array must not be mutated."""
        with self._lock:
            return self._frame

    def latest_copy(self) -> Optional[CameraFrame]:
        frame = self.latest()
        if frame is None:
            return None
        return CameraFrame(
            image=frame.image.copy(),
            timestamp=frame.timestamp,
            sequence=frame.sequence,
            width=frame.width,
            height=frame.height,
        )

    def stats(self) -> CameraStats:
        with self._lock:
            return CameraStats(
                running=self._stats.running,
                opened=self._stats.opened,
                fps=self._stats.fps,
                frame_ms=self._stats.frame_ms,
                width=self._stats.width,
                height=self._stats.height,
                frames=self._stats.frames,
                read_errors=self._stats.read_errors,
                last_error=self._stats.last_error,
                device=self._stats.device,
                fourcc=self._stats.fourcc,
                requested_width=self._stats.requested_width,
                requested_height=self._stats.requested_height,
                requested_fps=self._stats.requested_fps,
                mode_ok=self._stats.mode_ok,
                backend=self._stats.backend,
                timestamp_source=self._stats.timestamp_source,
                exposure=self._stats.exposure,
                gain=self._stats.gain,
                brightness=self._stats.brightness,
                detected_markers=self._stats.detected_markers,
                visible_markers=self._stats.visible_markers,
                auto_control=self._stats.auto_control,
                auto_actions=self._stats.auto_actions,
            )
