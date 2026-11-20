"""CLI entry point for the OpenVR Cardboard desktop app."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import __version__
from .config import AppConfig, load_config, save_config, validate_config
from .runtime import Runtime

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openvrcardboard",
        description=(
            "Webcam headset/hand tracking, gamepad VR controllers and phone "
            "video streaming for SteamVR (Linux and Windows)."
        ),
    )
    parser.add_argument("--config", metavar="PATH", help="configuration file to use")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the Qt user interface",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="start with tracking stopped (driver/phone links still run)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="start phone video streaming immediately",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="do not start phone video streaming",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="headless status print interval (0 disables)",
    )
    parser.add_argument(
        "--status-json",
        metavar="PATH",
        help="write periodic JSON diagnostics (used by the end-to-end test)",
    )
    parser.add_argument(
        "--status-json-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="JSON diagnostics write interval",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the effective configuration and exit",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="write the effective configuration and exit",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="import optional dependencies and exit (diagnose a portable build)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.video:
        config.phone.video_enabled = True
    if args.no_video:
        config.phone.video_enabled = False
        config.phone.stream_on_launch = False
    if args.video and not args.no_video:
        config.phone.stream_on_launch = True
    if args.no_start:
        config.ui.start_tracking_on_launch = False
    validate_config(config)


def run_self_test() -> int:
    """Import and exercise the optional stacks, then report versions."""
    checks: dict[str, str] = {}
    failures: list[str] = []

    def check(name: str, action) -> None:
        try:
            checks[name] = str(action())
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    def opencv() -> str:
        import cv2

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco (opencv-contrib) is missing")
        return cv2.__version__

    def mediapipe() -> str:
        import mediapipe

        try:
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("the MediaPipe Tasks API is missing") from exc
        if not hasattr(vision, "HandLandmarker") or not hasattr(
            vision, "PoseLandmarker"
        ):
            raise RuntimeError("the hand/pose landmarkers are missing")
        return mediapipe.__version__

    def pygame_version() -> str:
        import pygame

        return pygame.version.ver

    def mss_version() -> str:
        import mss

        return str(getattr(mss, "__version__", "available"))

    def video_codec() -> str:
        from .config import PhoneConfig
        from .encoder import choose_stream_codec
        from .gpu import detect_gpu

        caps = detect_gpu()
        codec, reason = choose_stream_codec(PhoneConfig())
        suffix = f"ffmpeg={caps.ffmpeg}" if caps.ffmpeg else "no ffmpeg"
        return f"{codec} ({reason}; {suffix})"

    def qt_version() -> str:
        from PySide6 import QtCore
        from PySide6.QtWidgets import QApplication

        existing = QApplication.instance()
        created = existing is None
        if created:
            old_platform = os.environ.get("QT_QPA_PLATFORM")
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
            app = QApplication([])
        else:
            app = existing
            old_platform = None
        try:
            return str(QtCore.qVersion())
        finally:
            if created:
                app.quit()
                if old_platform is None:
                    os.environ.pop("QT_QPA_PLATFORM", None)
                else:
                    os.environ["QT_QPA_PLATFORM"] = old_platform

    check("opencv", opencv)
    check("mediapipe", mediapipe)
    check("pygame", pygame_version)
    check("mss", mss_version)
    check("video", video_codec)
    check("qt", qt_version)
    print(json.dumps({"version": __version__, "checks": checks, "failures": failures}, indent=2))
    return 0 if not failures else 1


def print_config(config: AppConfig, path: Optional[Path]) -> None:
    import dataclasses
    import json

    payload = dataclasses.asdict(config)
    print(json.dumps(payload, indent=2))
    if path is not None:
        print(f"# config path: {path}", file=sys.stderr)


def _install_signal_handlers(stop_event) -> None:
    def handler(signum, frame):
        LOGGER.info("received signal %s; shutting down", signum)
        stop_event.set()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
    if os.name == "nt":
        try:
            signal.signal(signal.SIGBREAK, handler)
        except (AttributeError, ValueError, OSError):
            pass


class StatusJsonWriter:
    """Periodically writes ``Runtime.diagnostics()`` for tests and tooling."""

    def __init__(self, runtime: Runtime, path: Path, interval: float = 0.5):
        self.runtime = runtime
        self.path = Path(path).expanduser()
        self.interval = max(0.1, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error = ""

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._loop, name="openvrcardboard-status-json", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._write()

    def _write(self) -> None:
        try:
            payload = self.runtime.diagnostics()
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self.path)
            self._error = ""
        except Exception as exc:  # never take the app down for diagnostics
            self._error = f"{type(exc).__name__}: {exc}"
            LOGGER.debug("status JSON write failed: %s", exc)


def run_headless(runtime: Runtime, status_interval: float = 5.0) -> int:
    import threading

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)
    try:
        problems = runtime.start()
        for problem in problems:
            print(f"warning: {problem}", file=sys.stderr)
        print(
            "OpenVR Cardboard running headless. Press Ctrl+C to stop.\n"
            f"driver UDP {runtime.config.driver.udp_host}:{runtime.config.driver.udp_port}, "
            f"phone IMU {runtime.config.phone.imu_port}, "
            f"video {'on' if runtime.video else 'off'}"
        )
        last_print = 0.0
        while not stop_event.wait(0.25):
            if status_interval <= 0:
                continue
            now = time.monotonic()
            if now - last_print >= status_interval:
                last_print = now
                status = runtime.status()
                print(
                    f"camera {status.camera_fps:5.1f} fps "
                    f"{status.camera_resolution[0]}x{status.camera_resolution[1]} | "
                    f"markers {status.marker_count} | "
                    f"hands {'yes' if status.hands_visible else 'no'} | "
                    f"body {'yes' if status.body_visible else 'no'} | "
                    f"phone {status.phone_address or '-'} {status.phone_rate_hz:4.1f} Hz | "
                    f"driver {'connected' if status.driver_connected else 'waiting'} "
                    f"({status.driver_packets} packets) | "
                    f"video {status.video_clients} clients, {status.video_fps:4.1f} fps"
                )
                if status.last_error:
                    print(f"  last error: {status.last_error}")
    finally:
        runtime.stop()
    return 0


def run_gui(runtime: Runtime, start_tracking: bool) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is not installed. Install the desktop extra with\n"
            "  pip install -e '.[ui]'\n"
            "or run with --headless.",
            file=sys.stderr,
        )
        return 2

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("OpenVR Cardboard")
    app.setApplicationVersion(__version__)
    window = MainWindow(runtime, start_tracking=start_tracking)
    window.show()
    try:
        return int(app.exec())
    finally:
        runtime.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.self_test:
        return run_self_test()

    config_path = Path(args.config).expanduser() if args.config else None
    config, warnings = load_config(config_path)
    apply_overrides(config, args)

    for warning in warnings:
        LOGGER.warning("%s", warning)

    if args.print_config or args.save_config:
        if args.save_config:
            target = save_config(config, config_path)
            print(target)
        if args.print_config:
            print_config(config, config_path or None)
        return 0

    runtime = Runtime(config, config_path)
    for warning in runtime.warnings:
        LOGGER.info("%s", warning)

    writer = None
    if args.status_json:
        writer = StatusJsonWriter(
            runtime, Path(args.status_json), args.status_json_interval
        )
        writer.start()
    try:
        if args.headless:
            return run_headless(runtime, args.status_interval)
        return run_gui(runtime, config.ui.start_tracking_on_launch)
    finally:
        if writer is not None:
            writer.stop()


if __name__ == "__main__":
    raise SystemExit(main())
