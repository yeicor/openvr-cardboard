"""Tracking loop and 250 Hz pose fusion worker.

``TrackingWorker`` owns the webcam and the optional marker/MediaPipe trackers.
``FusionWorker`` consumes the shared state, advances the phone pose estimator,
composes driver poses with :class:`PoseComposer` and feeds the UDP driver link.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from .calibration import CameraModel
from .camera import CameraCapture
from .config import AppConfig
from .driver_link import DriverLink
from .fusion import ComposedFrame, PhonePoseEstimator, PoseComposer
from .markers import BoardTracker
from .boards import boards_from_config
from .pose_tracking import (
    BodyTracker,
    HandTracker,
    TrackingUnavailable,
)
from .state import SharedState

LOGGER = logging.getLogger(__name__)


def _resize_for_inference(image, max_width: int, max_height: int):
    """Downscale a camera frame before MediaPipe inference.

    MediaPipe's hand and pose networks internally crop to a small square, so
    feeding a 1080p frame only adds RGB conversion and cache pressure. Marker
    detection still runs on the full-resolution frame. ``max_width=0`` and
    ``max_height=0`` leave the frame untouched; a zero side preserves the
    camera aspect ratio.
    """
    if max_width <= 0 and max_height <= 0:
        return image
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image
    scale = 1.0
    if max_width > 0:
        scale = min(scale, float(max_width) / width)
    if max_height > 0:
        scale = min(scale, float(max_height) / height)
    if scale >= 1.0:
        return image
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    try:
        import cv2
    except ImportError:
        # Nearest-neighbour fallback: slower than cv2 but always available.
        rows = np.linspace(0, height - 1, target_height).astype(np.int32)
        columns = np.linspace(0, width - 1, target_width).astype(np.int32)
        return image[rows][:, columns]
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


class _InferenceWorker:
    """One-slot MediaPipe worker, decoupled from marker detection.

    Marker detection is the latency-critical path for the head pose.  Running
    hand and body inference on the same thread made a slow pose model delay
    every camera frame.  This worker keeps only the newest submitted frame, so
    inference can never build a backlog, and the tracking loop continues to
    detect markers while it runs.
    """

    def __init__(self, state: SharedState, config: AppConfig,
                 hands: Optional[HandTracker], body: Optional[BodyTracker]):
        self.state = state
        self.config = config
        self.hands = hands
        self.body = body
        self.inference_ms = 0.0
        self._lock = threading.Lock()
        self._job: Optional[tuple[np.ndarray, float]] = None
        self._results: Optional[tuple[float, dict, object]] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="openvrcardboard-mediapipe", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._ready.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, image: np.ndarray, timestamp: float) -> None:
        with self._lock:
            self._job = (image, float(timestamp))
        self._ready.set()

    def latest(self) -> Optional[tuple[float, dict, object]]:
        with self._lock:
            return self._results

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._ready.wait(0.1)
            self._ready.clear()
            if self._stop.is_set():
                return
            with self._lock:
                job = self._job
                self._job = None
            if job is None:
                continue
            image, timestamp = job
            started = time.perf_counter()
            hands: dict = {}
            body = None
            inference_image = _resize_for_inference(
                image,
                self.config.tracking.inference_width,
                self.config.tracking.inference_height,
            )
            if self.hands is not None:
                try:
                    hands = self.hands.process(inference_image, timestamp)
                    # A throttled MediaPipe result still describes the frame
                    # it was inferred from.  Publishing it with the camera
                    # timestamp would make a stale pose look fresh.
                    hand_time = max(
                        (observation.timestamp for observation in hands.values()),
                        default=timestamp,
                    )
                    self.state.set_hands(hands, hand_time)
                except Exception as exc:
                    LOGGER.debug("hand tracking failed: %s", exc)
            if self.body is not None:
                try:
                    body = self.body.process(inference_image, timestamp)
                    body_time = (
                        body.timestamp
                        if body is not None and body.timestamp > 0.0
                        else timestamp
                    )
                    self.state.set_body(body, body_time)
                except Exception as exc:
                    LOGGER.debug("body tracking failed: %s", exc)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self.inference_ms = elapsed_ms
                self._results = (timestamp, hands, body)


class TrackingWorker:
    """Camera capture plus marker, hand and body inference."""

    def __init__(
        self,
        state: SharedState,
        config: AppConfig,
        model: CameraModel,
        camera_position,
        camera_rotation,
        camera: Optional[CameraCapture] = None,
    ):
        self.state = state
        self.config = config
        self.model = model
        self.camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        self.camera_rotation = np.asarray(camera_rotation, dtype=np.float64).reshape(4)
        self.boards = boards_from_config(config.tracking)
        self.camera = camera or CameraCapture(
            config.camera,
            marker_ids=(
                config.tracking.hmd_marker_id,
                config.tracking.left_marker_id,
                config.tracking.right_marker_id,
            ),
            boards=self.boards,
        )

        self.markers: Optional[BoardTracker] = None
        self.hands: Optional[HandTracker] = None
        self.body: Optional[BodyTracker] = None
        self.inference: Optional[_InferenceWorker] = None
        self._init_errors: list[str] = []

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._init_trackers()
        self.state.set_hand_input_options(
            self.config.tracking.hand_inputs,
            self.config.tracking.hand_trigger_pinch,
        )
        for note in self._init_errors:
            self.state.add_note(note)
        if self.hands is not None or self.body is not None:
            self.inference = _InferenceWorker(
                self.state, self.config, self.hands, self.body
            )
            self.inference.start()
        self.camera.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="openvrcardboard-tracking", daemon=True
        )
        self._thread.start()
        LOGGER.info("tracking worker started")

    def _init_trackers(self) -> None:
        tracking = self.config.tracking
        if tracking.markers_enabled and self.markers is None:
            try:
                self.markers = BoardTracker(
                    self.model,
                    self.camera_position,
                    self.camera_rotation,
                    boards=self.boards,
                    dictionary=self.config.camera.dictionary,
                    detection_width=self.config.camera.marker_detection_width,
                    refine_pose=self.config.camera.marker_refine_pose,
                    max_reprojection_error=self.config.camera.max_reprojection_error,
                    hold_time_s=tracking.marker_hold_s,
                    lost_timeout_s=tracking.marker_lost_s,
                    reacquire_frames=tracking.marker_reacquire_frames,
                    process_noise=tracking.marker_process_noise,
                    angular_process_noise=tracking.marker_angular_noise,
                    adaptive_noise=tracking.marker_adaptive_noise,
                    static_noise_scale=tracking.marker_static_noise_scale,
                    exposure_s=tracking.marker_exposure_s,
                    min_markers_tracking=tracking.marker_min_markers,
                    min_confidence=tracking.marker_min_confidence,
                    lost_position_sigma_m=tracking.marker_lost_position_sigma_m,
                    lost_orientation_sigma_deg=(
                        tracking.marker_lost_orientation_sigma_deg
                    ),
                    reinitialize_gap_s=tracking.marker_reinitialize_gap_s,
                )
                # Fail early when OpenCV is missing or the dictionary is bad.
                self.markers._load_cv2()
                self.markers._make_detector()
                self.state.set_marker_tracker(self.markers)
            except Exception as exc:
                self._init_errors.append(f"marker tracking unavailable: {exc}")
                LOGGER.warning("marker tracking unavailable: %s", exc)
                self.markers = None
                self.state.set_marker_tracker(None)
        if tracking.hands_enabled and self.hands is None:
            try:
                self.hands = HandTracker(
                    self.model,
                    self.camera_position,
                    self.camera_rotation,
                    max_fps=tracking.mediapipe_max_fps,
                    swap_hands=tracking.hand_swap,
                    landmark_cutoff=tracking.landmark_cutoff,
                    landmark_beta=tracking.landmark_beta,
                    delegate=tracking.mediapipe_delegate,
                    model_dir=tracking.model_dir or None,
                )
                self.hands._ensure()
            except TrackingUnavailable as exc:
                self._init_errors.append(f"hand tracking unavailable: {exc}")
                LOGGER.warning("hand tracking unavailable: %s", exc)
                self.hands = None
        if tracking.body_enabled and self.body is None:
            try:
                self.body = BodyTracker(
                    self.model,
                    self.camera_position,
                    self.camera_rotation,
                    body_scale=tracking.body_scale,
                    position_offset=tracking.body_position_offset,
                    max_fps=tracking.mediapipe_max_fps,
                    roles=tracking.tracker_roles[: self.config.driver.tracker_count],
                    pose_model=tracking.body_model,
                    min_visibility=tracking.body_min_visibility,
                    landmark_cutoff=tracking.landmark_cutoff,
                    landmark_beta=tracking.landmark_beta,
                    delegate=tracking.mediapipe_delegate,
                    model_dir=tracking.model_dir or None,
                )
                self.body._ensure()
            except TrackingUnavailable as exc:
                self._init_errors.append(f"body tracking unavailable: {exc}")
                LOGGER.warning("body tracking unavailable: %s", exc)
                self.body = None

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.camera.stop()
        if self.inference is not None:
            self.inference.stop()
            self.inference = None
        for tracker in (self.hands, self.body, self.markers):
            if tracker is not None:
                tracker.close()
        self.hands = None
        self.body = None
        self.markers = None
        self.state.clear_markers()
        LOGGER.info("tracking worker stopped")

    def __enter__(self) -> "TrackingWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        last_sequence = -1
        while not self._stop.is_set():
            frame = self.camera.latest()
            if frame is None:
                self._stop.wait(0.01)
                continue
            if frame.sequence == last_sequence:
                self._stop.wait(0.002)
                continue
            last_sequence = frame.sequence

            started = time.perf_counter()
            image = frame.image
            timestamp = frame.timestamp
            detections = {}

            if self.markers is not None:
                try:
                    detections = self.markers.detect(image, timestamp)
                    self.state.set_markers(detections, timestamp)
                    # Feed the camera's exposure loop with how complete the
                    # currently visible boards are.  A board that produced no
                    # markers at all is not counted as expected, so pointing
                    # the camera at a wall does not drive exposure.
                    detected_markers = sum(
                        len(detection.marker_ids) for detection in detections.values()
                    )
                    visible_markers = sum(
                        len(self.boards[name].markers)
                        for name, detection in detections.items()
                        if name in self.boards
                    )
                    self.camera.report_marker_detection(
                        detected_markers, visible_markers
                    )
                except Exception as exc:  # keep tracking alive across OpenCV hiccups
                    LOGGER.debug("marker tracking failed: %s", exc)
            if self.inference is not None:
                self.inference.submit(image, timestamp)
            results = self.inference.latest() if self.inference is not None else None
            hands = {}
            body = None
            if results is not None and abs(results[0] - timestamp) <= 0.12:
                hands = results[1]
                body = results[2]
            inference_ms = (
                self.inference.inference_ms if self.inference is not None else 0.0
            )

            overlay = image.copy()
            if self.markers is not None and detections:
                try:
                    self.markers.draw(overlay, detections)
                except Exception:
                    pass
            if self.hands is not None and hands:
                try:
                    self.hands.draw(overlay, hands)
                except Exception:
                    pass
            if self.body is not None and body is not None:
                try:
                    self.body.draw(overlay, body)
                except Exception:
                    pass
            self.state.set_preview(overlay, timestamp)

            stats = self.camera.stats()
            self.state.set_status(
                camera_fps=stats.fps,
                camera_frame_ms=stats.frame_ms,
                camera_resolution=(stats.width, stats.height),
                camera_error=stats.last_error,
                inference_ms=inference_ms,
            )
            elapsed = time.perf_counter() - started
            # If inference took longer than a frame, drop the pending frame and
            # pick up the newest one instead of building a backlog.
            if elapsed < 0.001:
                self._stop.wait(0.001)


class FusionWorker:
    """Runs pose composition at a fixed rate and feeds the driver link."""

    def __init__(
        self,
        state: SharedState,
        config: AppConfig,
        composer: PoseComposer,
        estimator: PhonePoseEstimator,
        driver: DriverLink,
        rate_hz: int = 250,
        on_frame: Optional[Callable[[ComposedFrame, float], None]] = None,
    ):
        self.state = state
        self.config = config
        self.composer = composer
        self.estimator = estimator
        self.driver = driver
        self.on_frame = on_frame
        self.rate_hz = max(30, int(rate_hz))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_phone_sequence: Optional[int] = None
        self._last_phone_time = 0.0
        self._last_frame: Optional[ComposedFrame] = None
        self._frames = 0
        self._started_at = time.monotonic()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_frame(self) -> Optional[ComposedFrame]:
        return self._last_frame

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name="openvrcardboard-fusion", daemon=True
        )
        self._thread.start()
        LOGGER.info("fusion worker started at %d Hz", self.rate_hz)

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        LOGGER.info("fusion worker stopped")

    def step(self, now: Optional[float] = None) -> ComposedFrame:
        """Compose and publish one frame (also used by headless/tests)."""
        stamp = time.monotonic() if now is None else float(now)
        if self.state.take_recenter():
            self.estimator.recenter(stamp)
            LOGGER.info("phone HMD recentered")

        sample = self.state.phone()
        if sample is not None:
            if (
                sample.received_at != self._last_phone_time
                or sample.sequence != self._last_phone_sequence
            ):
                # Phone timestamps were converted to desktop monotonic time by
                # the receiver's clock-offset estimate.  Feeding the fusion
                # tick instead would add socket and scheduler jitter to every
                # filter step.
                update_time = sample.timestamp if sample.timestamp > 0.0 else stamp
                self.estimator.update(sample, update_time)
                self._last_phone_time = sample.received_at
                self._last_phone_sequence = sample.sequence
            self.state.set_phone_pose(self.estimator.pose(stamp))
        else:
            self.state.set_phone_pose(self.estimator.pose(stamp))

        snapshot = self.state.snapshot(stamp)
        frame = self.composer.update(snapshot, stamp)
        self._last_frame = frame
        self._frames += 1
        self.driver.update_frame(frame, stamp)
        if self.on_frame is not None:
            try:
                self.on_frame(frame, stamp)
            except Exception:
                LOGGER.debug("frame callback failed", exc_info=True)

        link_stats = self.driver.stats(stamp)
        self.state.set_status(
            driver_connected=link_stats.connected,
            driver_packets=link_stats.send_packets,
        )
        return frame

    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_time = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self.step(now)
            except Exception as exc:  # never let the fusion thread die
                LOGGER.exception("fusion step failed: %s", exc)
                self.state.set_status(last_error=f"fusion: {exc}")
            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_time = time.monotonic()
