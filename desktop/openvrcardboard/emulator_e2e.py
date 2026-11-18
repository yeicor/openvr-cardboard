"""Android emulator end-to-end harness.

This module is test tooling rather than part of the desktop runtime. It boots
an Android emulator (or attaches to one started by CI), starts the real desktop
app against the synthetic webcam and synthetic Xbox pads, launches Monado's
``openxr-simple-example`` through the capture shim, launches the phone app
against the desktop video server, records both screens, and verifies that the
live OpenXR stereo view reaches the phone while markers, body tracking, 6DoF
motion, grabbing and haptics all happen in the real OpenXR application.

The harness is deliberately self-contained: every step, log and recording is
written to the artifact directory so a failed run can be diagnosed without
re-running it. It expects Linux, KVM (unless an emulator is already running),
Android SDK platform-tools, ffmpeg, Xvfb, xdotool and an OpenXR stack built by
``scripts/build-openxr.sh``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, save_config
from .openxr_stack import OpenXRStack, find_openxr_prefix

PACKAGE = "dev.openvrcardboard.phone"
ACTIVITY = f"{PACKAGE}/.MainActivity"
TAG = "OpenVRCardboard"
TOKEN = "ovrc-e2e-token"
VIDEO_PORT = 8765
IMU_PORT = 39101
HOST_FOR_EMULATOR = "10.0.2.2"
DESKTOP_WIDTH = 1100
DESKTOP_HEIGHT = 720
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 720
EMULATOR_WIDTH = 1920
EMULATOR_HEIGHT = 1080
SCREENRECORD_PATH = "/sdcard/ovrc-e2e.mp4"
OPENXR_WINDOW = "OpenXR Example"
DESKTOP_WINDOW = "OpenVR Cardboard"
# 2400x1080 total stereo is 1200x1080 per eye, the target for a 1080p phone
# panel at 90 Hz. The capture shim, Monado view configuration, phone stream and
# recording all use this single geometry.
OPENXR_VIEW_WIDTH = 2400
OPENXR_VIEW_HEIGHT = 1080
PHONE_EYE_WIDTH = OPENXR_VIEW_WIDTH // 2
PHONE_EYE_HEIGHT = OPENXR_VIEW_HEIGHT
PHONE_FPS = 90
OPENXR_CAPTURE_FPS = 90
OPENXR_CAPTURE_QUALITY = 65
# The phone stream deliberately runs the CPU libx264 backend at half per-eye
# resolution. That keeps real MediaCodec H.264 decoding in the hardware-free
# CI path without starving the OpenXR example or body tracking; the capture
# shim still receives full 2400x1080 stereo eye views.
E2E_PHONE_EYE_WIDTH = 600
E2E_PHONE_EYE_HEIGHT = 540
E2E_PHONE_FPS = 30
RECORD_FPS = 60

# The example renders a 600x540 desktop preview (half the 1200x1080 per-eye
# view configuration). It is moved beside the desktop app's 1100x720 window so
# the desktop recording shows both the app UI and the real OpenXR scene.
OPENXR_WINDOW_X = DESKTOP_WIDTH
OPENXR_WINDOW_Y = (SCREEN_HEIGHT - PHONE_EYE_HEIGHT // 2) // 2


class E2EError(RuntimeError):
    """Raised when a step of the emulator E2E cannot complete."""


@dataclass
class E2EConfig:
    artifacts: Path = field(default_factory=lambda: Path("artifacts/emulator-e2e"))
    duration: float = 18.0
    minimum_frames: int = 60
    attach: bool = False
    keep_emulator: bool = False
    apk: Optional[Path] = None
    serial: Optional[str] = None
    avd_name: str = "ovrc_e2e"
    system_image: str = "system-images;android-34;google_apis;x86_64"
    device_profile: str = "pixel_2"
    boot_timeout: float = 300.0
    openxr_prefix: Optional[Path] = None
    phone_codec: str = "h264"
    phone_encoder: str = "x264"
    phone_width: int = E2E_PHONE_EYE_WIDTH
    phone_height: int = E2E_PHONE_EYE_HEIGHT
    phone_fps: int = E2E_PHONE_FPS


@dataclass
class E2EResult:
    frames: int
    video_connected: bool
    phone_imu_seen: bool
    desktop_imu_seen: bool
    screenshot_mean: float
    screenshot_std: float
    screenshot_diff: float
    screenshot_changed_pixels: int
    stereo_diff: float
    openxr_connected: bool
    openxr_frames: int
    markers_seen: list[int]
    body_seen: bool
    max_grab_left: float
    max_grab_right: float
    haptic_events: int
    gamepad_haptics: int
    pose_motion: dict[str, float]
    hardware_decode_seen: bool
    desktop_video: Path
    emulator_video: Path
    combined_video: Path
    summary: Path


def _log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[emulator-e2e {stamp}] {message}", flush=True)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _display_ready(display: str) -> bool:
    try:
        result = subprocess.run(
            ["xdpyinfo", "-display", display],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class EmulatorE2E:
    """Drive one complete desktop + OpenXR + emulator session."""

    def __init__(self, config: E2EConfig):
        self.config = config
        self.repo_root = Path(__file__).resolve().parents[2]
        self.desktop_dir = self.repo_root / "desktop"
        artifacts = Path(config.artifacts)
        self.artifact_dir = artifacts if artifacts.is_absolute() else self.repo_root / artifacts
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.env = os.environ.copy()
        self.sdk: Optional[Path] = None
        self.adb: Optional[str] = None
        self.emulator: Optional[str] = None
        self.avdmanager: Optional[str] = None
        self.serial: Optional[str] = None
        self.display: Optional[str] = None
        self.apk: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.status_path: Optional[Path] = None

        self._xvfb: Optional[subprocess.Popen] = None
        self._desktop: Optional[subprocess.Popen] = None
        self._desktop_record: Optional[subprocess.Popen] = None
        self._screen_record: Optional[subprocess.Popen] = None
        self._emulator: Optional[subprocess.Popen] = None
        self.openxr: Optional[OpenXRStack] = None
        self.openxr_remote_port = 4242
        self.openxr_capture_port = 39102
        self._started_emulator = False
        self._device_ready = False
        self._screen_record_pulled = False
        self._desktop_record_stopped = False
        self._evidence_collected = False
        self._combined = False
        self._summary_written = False
        self._record_started = 0.0
        self._run_started = 0.0
        self._screen_record_started = False
        self._frames_ready_at = 0.0

        self.frames = 0
        self.decode_fps: list[float] = []
        self.display_fps: list[float] = []
        self.video_connected = False
        self.phone_imu_seen = False
        self.desktop_imu_seen = False
        self.hardware_decode_seen = False
        self.screenshot_mean = 0.0
        self.screenshot_std = 0.0
        self.screenshot_diff = 0.0
        self.screenshot_changed_pixels = 0
        self.stereo_diff = 0.0
        self.openxr_connected = False
        self.openxr_frames = 0
        self.remote_connected = False
        self.remote_packets = 0
        self.markers_seen: set[int] = set()
        # Primary marker IDs of the HMD, left and right boards.  The board
        # tracker publishes one detection per board under its central ID.
        self.expected_markers: set[int] = {0, 5, 10}
        self.body_seen = False
        self.max_grab_left = 0.0
        self.max_grab_right = 0.0
        self.haptic_events = 0
        self.gamepad_haptics = 0
        self.status_samples = 0
        self._positions: dict[str, list[tuple[float, float, float]]] = {
            "hmd": [],
            "left": [],
            "right": [],
        }
        self.desktop_info: dict = {}
        self.emulator_info: dict = {}
        self.combined_info: dict = {}
        self.summary: dict = {
            "status": "running",
            "started_at": _utc_now(),
            "package": PACKAGE,
            "host_from_emulator": HOST_FOR_EMULATOR,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> E2EResult:
        run_started = time.monotonic()
        self._run_started = run_started
        self._clean_previous_outputs()
        try:
            self._check_prerequisites()
            self._find_sdk()
            self._prepare_apk()
            if self.config.attach:
                self._attach_device()
            else:
                self._ensure_avd()
                self._start_emulator()
            self._prepare_device()
            self._write_config()
            self._start_display()
            self._start_desktop()
            self._wait_for_video_port()
            self._start_openxr()
            self._wait_for_openxr()
            self._position_windows()
            time.sleep(1.0)

            self._record_started = time.monotonic()
            self._start_desktop_recording()
            self._start_screen_recording()
            self._launch_phone()
            self._wait_for_frames()
            self._capture_screenshots()
            self._hold_for_duration()

            self._stop_screen_recording()
            self._stop_desktop_recording()
            self._collect_evidence()
            self._stop_phone()
            self._combine_videos()
            self._verify_outputs()
            self.summary["status"] = "passed"
        except Exception as exc:
            self.summary["status"] = "failed"
            self.summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._cleanup()
        return self._result()

    # ------------------------------------------------------------------
    # Prerequisites and SDK discovery
    # ------------------------------------------------------------------
    def _clean_previous_outputs(self) -> None:
        for name in (
            "desktop-app.mp4",
            "android-emulator.mp4",
            "combined-session.mp4",
            "android-screenshot-1.png",
            "android-screenshot-2.png",
            "desktop-app-frame.png",
            "android-emulator-frame.png",
            "desktop-app.log",
            "emulator.log",
            "screenrecord.log",
            "ffmpeg-desktop.log",
            "xvfb.log",
            "logcat.txt",
            "phone-log.txt",
            "status.json",
            "status.json.tmp",
            "summary.json",
        ):
            try:
                (self.artifact_dir / name).unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(self.artifact_dir / "openxr", ignore_errors=True)

    def _check_prerequisites(self) -> None:
        for tool in ("ffmpeg", "ffprobe", "Xvfb", "xdpyinfo", "xdotool"):
            if shutil.which(tool) is None:
                raise E2EError(f"required tool not found: {tool}")
        if not self.config.attach:
            kvm = Path("/dev/kvm")
            if not kvm.exists():
                raise E2EError(
                    "/dev/kvm is missing; the Android emulator needs KVM. "
                    "Run Docker with --device /dev/kvm or enable KVM on the host."
                )
            if not os.access(kvm, os.R_OK | os.W_OK):
                raise E2EError(f"{kvm} is not readable/writable by this user")
        prefix = self.config.openxr_prefix or find_openxr_prefix()
        if prefix is None:
            raise E2EError(
                "the OpenXR runtime was not found. Build it first with\n"
                "  scripts/build-openxr.sh\n"
                "or point OVRC_OPENXR_ROOT at an installed runtime."
            )
        _log(f"OpenXR runtime: {Path(prefix).resolve()}")
        if sys.version_info < (3, 10):
            raise E2EError("Python 3.10 or newer is required")

    def _find_sdk(self) -> None:
        candidates: list[Path] = []
        for key in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
            value = os.environ.get(key)
            if value:
                candidates.append(Path(value).expanduser())
        candidates.extend(
            [
                Path.home() / "Android" / "Sdk",
                Path("/opt/android-sdk"),
                Path("/usr/local/lib/android/sdk"),
            ]
        )
        for base in candidates:
            adb = base / "platform-tools" / "adb"
            if adb.exists():
                self.sdk = base
                self.adb = str(adb)
                break
        if self.adb is None:
            adb_which = shutil.which("adb")
            if adb_which is None:
                raise E2EError("adb not found; install Android SDK platform-tools")
            self.adb = adb_which
            self.sdk = Path(adb_which).resolve().parent.parent

        assert self.sdk is not None
        emulator = self.sdk / "emulator" / "emulator"
        self.emulator = str(emulator) if emulator.exists() else shutil.which("emulator")
        avdmanager = self.sdk / "cmdline-tools" / "latest" / "bin" / "avdmanager"
        if not avdmanager.exists():
            avdmanager = self.sdk / "tools" / "bin" / "avdmanager"
        self.avdmanager = str(avdmanager) if avdmanager.exists() else shutil.which("avdmanager")

        self.env["ANDROID_HOME"] = str(self.sdk)
        self.env["ANDROID_SDK_ROOT"] = str(self.sdk)
        path_parts = [
            str(self.sdk / "platform-tools"),
            str(self.sdk / "emulator"),
            str(self.sdk / "cmdline-tools" / "latest" / "bin"),
        ]
        self.env["PATH"] = os.pathsep.join(path_parts + [self.env.get("PATH", "")])
        _log(f"Android SDK: {self.sdk}")

    # ------------------------------------------------------------------
    # APK and AVD
    # ------------------------------------------------------------------
    def _prepare_apk(self) -> None:
        candidates: list[Path] = []
        if self.config.apk is not None:
            candidates.append(Path(self.config.apk))
        candidates.append(self.repo_root / "android" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk")
        debug_dir = self.repo_root / "android" / "app" / "build" / "outputs" / "apk" / "debug"
        if debug_dir.is_dir():
            candidates.extend(sorted(debug_dir.glob("*.apk")))
        for candidate in candidates:
            candidate = candidate if candidate.is_absolute() else self.repo_root / candidate
            if candidate.is_file():
                self.apk = candidate
                _log(f"APK: {candidate}")
                return
        raise E2EError(
            "Android debug APK not found. Build it first with\n"
            "  scripts/run-e2e.sh android assembleDebug"
        )

    def _ensure_avd(self) -> None:
        if self.avdmanager is None:
            raise E2EError(
                "avdmanager not found; install Android cmdline-tools and a system image"
            )
        listing = self._exec([self.avdmanager, "list", "avd"], quiet=True, check=False)
        if re.search(rf"^\s*Name:\s*{re.escape(self.config.avd_name)}\s*$", listing.stdout, re.M):
            _log(f"reusing AVD {self.config.avd_name}")
            return
        _log(f"creating AVD {self.config.avd_name} ({self.config.system_image})")
        result = self._exec(
            [
                self.avdmanager,
                "create",
                "avd",
                "--force",
                "-n",
                self.config.avd_name,
                "-k",
                self.config.system_image,
                "-d",
                self.config.device_profile,
            ],
            input_text="no\n",
            check=False,
        )
        if result.returncode != 0:
            raise E2EError(
                "could not create the Android AVD. Install the system image with\n"
                f"  sdkmanager '{self.config.system_image}'\n"
                f"{result.stdout}\n{result.stderr}"
            )

    def _start_emulator(self) -> None:
        if self.emulator is None:
            raise E2EError("Android emulator binary not found")
        accel = self._exec([self.emulator, "-accel-check"], quiet=True, check=False)
        accel_text = f"{accel.stdout}\n{accel.stderr}".strip()
        _log(f"emulator acceleration: {accel_text.splitlines()[0] if accel_text else 'unknown'}")
        if "usable" not in accel_text.lower():
            raise E2EError(
                "the Android emulator does not report usable KVM acceleration:\n"
                f"{accel_text}"
            )
        command = [
            self.emulator,
            "-avd",
            self.config.avd_name,
            "-no-window",
            "-no-audio",
            "-no-boot-anim",
            "-no-snapshot",
            "-gpu",
            "swiftshader_indirect",
            "-accel",
            "on",
            "-netdelay",
            "none",
            "-netspeed",
            "full",
            "-camera-back",
            "none",
            "-camera-front",
            "none",
        ]
        self._emulator = self._spawn(command, "emulator.log")
        self._started_emulator = True
        self._wait_for_device()
        self._wait_for_boot()

    def _attach_device(self) -> None:
        _log("attaching to an already running Android device")
        self._wait_for_device()
        self._wait_for_boot()

    # ------------------------------------------------------------------
    # Device control
    # ------------------------------------------------------------------
    def _adb_command(self, *args: str) -> list[str]:
        assert self.adb is not None
        command = [self.adb]
        if self.serial:
            command.extend(["-s", self.serial])
        command.extend(args)
        return command

    def _adb(self, *args: str, timeout: float = 30.0, check: bool = True) -> subprocess.CompletedProcess:
        return self._exec(self._adb_command(*args), timeout=timeout, check=check, quiet=True)

    def _wait_for_device(self) -> None:
        assert self.adb is not None
        self._exec([self.adb, "start-server"], quiet=True, check=False, timeout=30)
        requested = self.config.serial or os.environ.get("ANDROID_SERIAL")
        deadline = time.monotonic() + self.config.boot_timeout
        last_output = ""
        while time.monotonic() < deadline:
            result = self._exec([self.adb, "devices"], quiet=True, check=False)
            last_output = result.stdout
            devices: list[tuple[str, str]] = []
            for line in result.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    devices.append((parts[0], parts[1]))
            if requested:
                for serial, state in devices:
                    if serial == requested and state == "device":
                        self.serial = serial
                        _log(f"using device {serial}")
                        return
            ready = [serial for serial, state in devices if state == "device" and serial.startswith("emulator-")]
            if not ready:
                ready = [
                    serial
                    for serial, state in devices
                    if state == "device" and len(devices) == 1
                ]
            if ready:
                self.serial = ready[0]
                _log(f"using device {self.serial}")
                return
            if self._emulator is not None and self._emulator.poll() is not None:
                raise E2EError(
                    "the Android emulator exited during startup:\n"
                    f"{self._tail(self.artifact_dir / 'emulator.log')}"
                )
            time.sleep(1.0)
        raise E2EError(f"no Android device became available:\n{last_output}")

    def _wait_for_boot(self) -> None:
        assert self.serial is not None
        deadline = time.monotonic() + self.config.boot_timeout
        while time.monotonic() < deadline:
            result = self._adb("shell", "getprop", "sys.boot_completed", check=False, timeout=10)
            booted = result.stdout.strip() == "1"
            if booted:
                animation = self._adb("shell", "getprop", "init.svc.bootanim", check=False, timeout=10)
                if animation.stdout.strip() in ("stopped", ""):
                    _log(f"device {self.serial} booted")
                    time.sleep(1.0)
                    return
            time.sleep(1.0)
        raise E2EError(f"device {self.serial} did not finish booting in {self.config.boot_timeout:.0f} s")

    def _prepare_device(self) -> None:
        self._adb("shell", "settings", "put", "global", "window_animation_scale", "0", check=False)
        self._adb("shell", "settings", "put", "global", "transition_animation_scale", "0", check=False)
        self._adb("shell", "settings", "put", "global", "animator_duration_scale", "0", check=False)
        self._adb("shell", "settings", "put", "system", "screen_off_timeout", "2147483647", check=False)
        self._adb("shell", "settings", "put", "secure", "immersive_mode_confirmations", "confirmed", check=False)
        self._adb("shell", "svc", "power", "stayon", "true", check=False)
        self._adb("shell", "input", "keyevent", "82", check=False)
        self._adb("shell", "wm", "dismiss-keyguard", check=False)

        assert self.apk is not None
        _log(f"installing {self.apk.name}")
        install = self._adb("install", "-r", "-t", str(self.apk), timeout=180, check=False)
        if install.returncode != 0 or "Success" not in install.stdout:
            raise E2EError(f"adb install failed:\n{install.stdout}\n{install.stderr}")
        self._adb("shell", "pm", "clear", PACKAGE, check=False)
        self._device_ready = True

    def _launch_phone(self) -> None:
        self._adb("logcat", "-c", check=False)
        self._adb("shell", "am", "force-stop", PACKAGE, check=False)
        command = [
            "shell",
            "am",
            "start",
            "-W",
            "-S",
            "-n",
            ACTIVITY,
            "--es",
            "host",
            HOST_FOR_EMULATOR,
            "--ei",
            "imu_port",
            str(IMU_PORT),
            "--ei",
            "video_port",
            str(VIDEO_PORT),
            "--ei",
            "sensor_hz",
            "60",
            "--es",
            "token",
            TOKEN,
            "--ez",
            "autostart",
            "true",
        ]
        result = self._adb(*command, timeout=60, check=False)
        if result.returncode != 0:
            raise E2EError(f"could not launch the phone app:\n{result.stdout}\n{result.stderr}")
        _log("phone app launched")

    def _read_phone_log(self) -> str:
        # Tailing the recent buffer keeps the polling loop cheap on software
        # rendered CI emulators; the interesting status lines are frequent and
        # the latches below remember anything that scrolls out.
        result = self._adb(
            "logcat",
            "-d",
            "-t",
            "400",
            "-s",
            f"{TAG}:I",
            "*:S",
            timeout=30,
            check=False,
        )
        return result.stdout

    def _wait_for_frames(self) -> None:
        deadline = time.monotonic() + 120.0
        last = ""
        while time.monotonic() < deadline:
            last = self._read_phone_log()
            self.video_connected = self.video_connected or "Video connected" in last
            self.phone_imu_seen = self.phone_imu_seen or "imu:" in last
            self.hardware_decode_seen = (
                self.hardware_decode_seen
                or "hardware decoder started: h264" in last
            )
            counts = [int(value) for value in re.findall(r"displayed (\d+) video frames", last)]
            if counts:
                self.frames = max(self.frames, max(counts))
            self._sample_status()
            if self.frames >= self.config.minimum_frames:
                self._frames_ready_at = time.monotonic()
                _log(f"phone displayed {self.frames} OpenXR stereo frames")
                return
            time.sleep(1.0)
        tail = "\n".join(last.splitlines()[-40:])
        raise E2EError(
            f"the phone did not display {self.config.minimum_frames} frames "
            f"(last count {self.frames}). Logcat tail:\n{tail}"
        )

    def _wait_for_more_frames(self, count: int, timeout: float = 20.0) -> None:
        """Wait until the phone renders ``count`` more frames, or time out."""
        target = self.frames + max(1, int(count))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            counts = [
                int(value)
                for value in re.findall(
                    r"displayed (\d+) video frames", self._read_phone_log()
                )
            ]
            if counts:
                self.frames = max(self.frames, max(counts))
            # Keep sampling while the phone renders the extra frames so pose
            # motion and trigger pulses are captured even when the next
            # "displayed" log line takes several seconds to appear.
            self._sample_status()
            if self.frames >= target:
                return
            time.sleep(0.5)
        _log(
            f"phone reached {self.frames} frames while waiting for {target}; "
            "capturing anyway"
        )

    def _capture_screenshots(self) -> None:
        first = self.artifact_dir / "android-screenshot-1.png"
        second = self.artifact_dir / "android-screenshot-2.png"
        self._screencap(first)
        # Require fresh animation frames between the captures: software
        # emulators can render only a few frames per second, and a fixed sleep
        # can otherwise compare two identical frames.
        self._wait_for_more_frames(5)
        self._screencap(second)
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependencies are installed by the image
            raise E2EError(f"Pillow and NumPy are required for screenshots: {exc}") from exc
        left = np.asarray(Image.open(first).convert("RGB"), dtype=np.float32)
        right = np.asarray(Image.open(second).convert("RGB"), dtype=np.float32)
        difference = np.abs(left - right)
        self.screenshot_mean = float(left.mean())
        self.screenshot_std = float(left.std())
        self.screenshot_diff = float(difference.mean())
        # The example's scene is sparse (a few cubes on a flat background), so
        # the mean difference stays small even when the cubes visibly rotate.
        # Count strongly changed pixels instead.
        self.screenshot_changed_pixels = int(
            (difference.max(axis=2) > 16.0).sum()
        )
        middle = left.shape[1] // 2
        eye_left = left[:, :middle]
        eye_right = left[:, middle : middle * 2]
        self.stereo_diff = float(np.abs(eye_left - eye_right).mean())
        self.summary["screenshots"] = {
            "mean": round(self.screenshot_mean, 2),
            "std": round(self.screenshot_std, 2),
            "difference": round(self.screenshot_diff, 2),
            "changed_pixels": self.screenshot_changed_pixels,
            "stereo_difference": round(self.stereo_diff, 2),
            "width": int(left.shape[1]),
            "height": int(left.shape[0]),
        }
        _log(
            "screenshots: mean %.1f, std %.1f, frame difference %.2f "
            "(%d changed pixels), left/right eye difference %.2f"
            % (
                self.screenshot_mean,
                self.screenshot_std,
                self.screenshot_diff,
                self.screenshot_changed_pixels,
                self.stereo_diff,
            )
        )

    def _screencap(self, target: Path) -> None:
        assert self.adb is not None
        command = self._adb_command("exec-out", "screencap", "-p")
        data = subprocess.run(command, capture_output=True, timeout=60, env=self.env)
        if data.returncode != 0 or not data.stdout.startswith(b"\x89PNG"):
            raise E2EError(f"screencap failed: {data.stderr.decode(errors='replace')}")
        target.write_bytes(data.stdout)

    # ------------------------------------------------------------------
    # Desktop app, OpenXR stack and recording
    # ------------------------------------------------------------------
    def _write_config(self) -> None:
        self.openxr_remote_port = _free_port()
        self.openxr_capture_port = _free_port()
        while self.openxr_capture_port == self.openxr_remote_port:
            self.openxr_capture_port = _free_port()
        config = AppConfig()
        config.camera.source = "synthetic"
        config.camera.width = 1280
        config.camera.height = 720
        config.camera.fps = 24
        config.tracking.markers_enabled = True
        config.tracking.hands_enabled = False
        config.tracking.body_enabled = True
        config.tracking.hmd_source = "markers"
        config.tracking.left_source = "markers"
        config.tracking.right_source = "markers"
        self.expected_markers = {
            int(config.tracking.hmd_marker_id),
            int(config.tracking.left_marker_id),
            int(config.tracking.right_marker_id),
        }
        # Software-rendered CI emulators run the synthetic scene at ~5 fps
        # while MediaPipe and the encoder compete for the same two cores. Keep
        # the last marker pose valid across that cadence; the production
        # default (0.25 s) is tuned for real 30-60 fps webcams.
        config.tracking.pose_max_age_s = 1.5
        config.tracking.mediapipe_max_fps = 15
        config.gamepad.enabled = True
        config.gamepad.synthetic = True
        config.openxr.enabled = True
        config.openxr.manage_service = False
        config.openxr.remote_port = self.openxr_remote_port
        config.openxr.capture_port = self.openxr_capture_port
        config.openxr.send_rate_hz = 100
        config.phone.capture_source = "openxr"
        config.phone.layout = "sbs"
        # The OpenXR stack keeps its native 2400x1080 stereo geometry; the
        # phone stream uses CPU H.264 at the configured E2E size so MediaCodec
        # is exercised even on CI machines without a GPU.
        config.phone.width = self.config.phone_width
        config.phone.height = self.config.phone_height
        config.phone.fps = self.config.phone_fps
        config.phone.codec = self.config.phone_codec
        config.phone.encoder = self.config.phone_encoder
        config.phone.jpeg_quality = OPENXR_CAPTURE_QUALITY
        config.phone.token = TOKEN
        config.phone.video_port = VIDEO_PORT
        config.phone.imu_port = IMU_PORT
        config.phone.video_enabled = True
        config.phone.stream_on_launch = True
        config.ui.start_tracking_on_launch = True
        self.config_path = self.artifact_dir / "app-config.json"
        self.status_path = self.artifact_dir / "status.json"
        save_config(config, self.config_path)
        self.summary["config"] = "app-config.json"
        self.summary["openxr_ports"] = {
            "remote": self.openxr_remote_port,
            "capture": self.openxr_capture_port,
        }
        _log(
            "desktop config written to %s (OpenXR remote %d, capture %d)"
            % (self.config_path, self.openxr_remote_port, self.openxr_capture_port)
        )

    def _start_display(self) -> None:
        for number in range(99, 121):
            display = f":{number}"
            if Path(f"/tmp/.X11-unix/X{number}").exists():
                continue
            process = self._spawn(
                ["Xvfb", display, "-screen", "0", f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}x24", "-nolisten", "tcp", "-ac"],
                "xvfb.log",
            )
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if _display_ready(display):
                    self.display = display
                    self._xvfb = process
                    _log(f"Xvfb running on {display} ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
                    return
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            self._stop_process(process, "Xvfb")
        raise E2EError(f"could not start Xvfb:\n{self._tail(self.artifact_dir / 'xvfb.log')}")

    def _desktop_environment(self) -> dict[str, str]:
        env = self.env.copy()
        assert self.display is not None
        runtime_dir = self.artifact_dir / "xdg-runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        try:
            runtime_dir.chmod(0o700)
        except OSError:
            pass
        env.update(
            {
                "DISPLAY": self.display,
                "XDG_RUNTIME_DIR": str(runtime_dir),
                "QT_QPA_PLATFORM": "xcb",
                "QT_LOGGING_RULES": "qt.qpa.xcb=false",
                "SDL_AUDIODRIVER": "dummy",
                "SDL_VIDEODRIVER": "x11",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": os.pathsep.join(
                    part
                    for part in (
                        str(self.desktop_dir),
                        env.get("PYTHONPATH", ""),
                    )
                    if part
                ),
            }
        )
        return env

    def _start_desktop(self) -> None:
        assert self.config_path is not None
        assert self.status_path is not None
        env = self._desktop_environment()
        command = [
            sys.executable,
            "-m",
            "openvrcardboard",
            "--config",
            str(self.config_path),
            "--video",
            "--status-json",
            str(self.status_path),
            "--status-json-interval",
            "0.25",
            "--log-level",
            "debug",
        ]
        self._desktop = self._spawn(command, "desktop-app.log", env=env)
        _log("desktop app started")

    def _wait_for_video_port(self) -> None:
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            if self._desktop is not None and self._desktop.poll() is not None:
                raise E2EError(
                    f"the desktop app exited before streaming:\n"
                    f"{self._tail(self.artifact_dir / 'desktop-app.log')}"
                )
            if _port_open("127.0.0.1", VIDEO_PORT):
                _log(f"desktop video server listening on {VIDEO_PORT}")
                return
            time.sleep(0.25)
        raise E2EError(f"video port {VIDEO_PORT} did not open:\n{self._tail(self.artifact_dir / 'desktop-app.log')}")

    def _start_openxr(self) -> None:
        self.openxr = OpenXRStack(
            prefix=self.config.openxr_prefix,
            work_dir=self.artifact_dir / "openxr",
            remote_port=self.openxr_remote_port,
            capture_port=self.openxr_capture_port,
            view_width=OPENXR_VIEW_WIDTH,
            view_height=OPENXR_VIEW_HEIGHT,
            capture_fps=OPENXR_CAPTURE_FPS,
            capture_quality=OPENXR_CAPTURE_QUALITY,
            display=self.display,
            extra_env={
                "SDL_VIDEODRIVER": "x11",
                "SDL_AUDIODRIVER": "dummy",
            },
        )
        self.openxr.start()
        _log(
            "OpenXR stack started (monado-service pid %d, example pid %d)"
            % (self.openxr.stats.service_pid, self.openxr.stats.example_pid)
        )

    def _wait_for_openxr(self) -> None:
        assert self.openxr is not None
        self.openxr.wait_example_connected(timeout=40.0)
        _log("openxr-simple-example connected to the capture shim")
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self._desktop is not None and self._desktop.poll() is not None:
                raise E2EError(
                    "the desktop app exited while waiting for tracking:\n"
                    f"{self._tail(self.artifact_dir / 'desktop-app.log')}"
                )
            self._sample_status()
            if (
                self.openxr_connected
                and self.openxr_frames >= 5
                and self.expected_markers.issubset(self.markers_seen)
                and self.body_seen
            ):
                _log(
                    "markers %s and body tracking visible; "
                    "capture has %d stereo frames"
                    % (sorted(self.markers_seen), self.openxr_frames)
                )
                return
            time.sleep(0.5)
        raise E2EError(
            "the OpenXR session did not become ready "
            f"(capture connected={self.openxr_connected}, frames={self.openxr_frames}, "
            f"markers={sorted(self.markers_seen)}, body={self.body_seen})\n"
            "example log:\n"
            f"{self._tail(self.artifact_dir / 'openxr' / 'openxr-example.log')}"
        )

    def _position_windows(self) -> None:
        """Place the app and the OpenXR preview side by side for recording."""
        desktop = self._find_window(DESKTOP_WINDOW, timeout=20.0)
        if desktop is None:
            _log("warning: the desktop window was not found; recording the root window")
            return
        self._xdotool("windowsize", desktop, str(DESKTOP_WIDTH), str(DESKTOP_HEIGHT))
        self._xdotool("windowmove", desktop, "0", "0")
        example = self._find_window(OPENXR_WINDOW, timeout=20.0)
        if example is None:
            _log("warning: the OpenXR example window was not found")
            return
        self._xdotool("windowsize", example, str(PHONE_EYE_WIDTH // 2), str(PHONE_EYE_HEIGHT // 2))
        self._xdotool("windowmove", example, str(OPENXR_WINDOW_X), str(OPENXR_WINDOW_Y))
        time.sleep(0.5)
        _log(
            "windows positioned: desktop at 0,0, OpenXR preview at %d,%d"
            % (OPENXR_WINDOW_X, OPENXR_WINDOW_Y)
        )

    def _find_window(self, name: str, timeout: float) -> Optional[str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", name],
                env=self._desktop_environment(),
                capture_output=True,
                text=True,
                timeout=10,
            )
            ids = result.stdout.split()
            if ids:
                return ids[0]
            time.sleep(0.2)
        return None

    def _xdotool(self, *args: str) -> None:
        result = subprocess.run(
            ["xdotool", *args],
            env=self._desktop_environment(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            _log(f"xdotool {' '.join(args)} failed: {result.stderr.strip()}")

    def _start_desktop_recording(self) -> None:
        assert self.display is not None
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "x11grab",
            "-framerate",
            str(RECORD_FPS),
            "-draw_mouse",
            "0",
            "-video_size",
            f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            "-i",
            f"{self.display}.0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.artifact_dir / "desktop-app.mp4"),
        ]
        self._desktop_record = self._spawn(command, "ffmpeg-desktop.log")
        time.sleep(1.0)
        if self._desktop_record.poll() is not None:
            raise E2EError(f"desktop recording failed:\n{self._tail(self.artifact_dir / 'ffmpeg-desktop.log')}")
        _log("desktop recording started")

    def _start_screen_recording(self) -> None:
        self._adb("shell", "rm", "-f", SCREENRECORD_PATH, check=False)
        command = self._adb_command(
            "shell",
            "screenrecord",
            "--bit-rate",
            "4000000",
            "--time-limit",
            "180",
            "--size",
            f"{EMULATOR_WIDTH}x{EMULATOR_HEIGHT}",
            SCREENRECORD_PATH,
        )
        self._screen_record = self._spawn(command, "screenrecord.log")
        self._screen_record_started = True
        time.sleep(1.0)
        if self._screen_record.poll() is not None:
            raise E2EError(f"emulator screen recording failed:\n{self._tail(self.artifact_dir / 'screenrecord.log')}")
        _log("emulator screen recording started")

    def _hold_for_duration(self) -> None:
        target = self._record_started + self.config.duration
        if self._frames_ready_at > 0.0:
            # Software-rendered CI emulators can spend the whole recording
            # budget booting the phone app. Guarantee a few seconds of steady
            # state so animation, grabs and pose motion are sampled.
            target = max(
                target, self._frames_ready_at + min(self.config.duration, 8.0)
            )
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            self._sample_status()
            time.sleep(min(0.5, remaining))
        _log(f"recorded for {max(0.0, time.monotonic() - self._record_started):.1f} s")

    def _stop_desktop_recording(self) -> None:
        if self._desktop_record is None or self._desktop_record_stopped:
            return
        self._desktop_record_stopped = True
        self._stop_process(self._desktop_record, "desktop recording", sig=signal.SIGINT)
        self._desktop_record = None
        _log("desktop recording stopped")

    def _stop_screen_recording(self) -> None:
        if self._screen_record is None and (self._screen_record_pulled or not self._screen_record_started):
            return
        if self._screen_record is not None:
            self._adb("shell", "pkill", "-INT", "screenrecord", check=False, timeout=15)
            # screenrecord writes the MP4 index when it exits. Pulling too early
            # gives a file without a ``moov`` atom, so wait for the device-side
            # process to disappear before stopping the local adb client.
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                live = self._adb("shell", "pidof", "screenrecord", check=False, timeout=10)
                if not live.stdout.strip():
                    break
                time.sleep(0.5)
            else:
                _log("screenrecord did not stop after SIGINT; stopping the local client")
            self._stop_process(self._screen_record, "emulator screen recording", timeout=15)
            self._screen_record = None
            # Give the device a moment to flush the finalized file.
            time.sleep(2.0)
        if self._screen_record_pulled or not self._device_ready:
            return
        self._screen_record_pulled = True
        target = self.artifact_dir / "android-emulator.mp4"
        pulled = self._adb("pull", SCREENRECORD_PATH, str(target), check=False, timeout=120)
        if pulled.returncode != 0 or not target.exists():
            _log(f"could not pull the emulator recording:\n{pulled.stdout}\n{pulled.stderr}")
            self.summary["screenrecord_error"] = pulled.stderr.strip() or pulled.stdout.strip()
        else:
            _log(f"emulator recording pulled ({target.stat().st_size} bytes)")

    # ------------------------------------------------------------------
    # Status sampling and evidence
    # ------------------------------------------------------------------
    def _read_status(self) -> Optional[dict]:
        if self.status_path is None:
            return None
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _sample_status(self) -> None:
        payload = self._read_status()
        if payload is not None:
            self._record_status(payload)

    def _record_status(self, payload: dict) -> None:
        self.status_samples += 1
        openxr = payload.get("openxr") or {}
        capture = openxr.get("capture") or {}
        remote = openxr.get("remote") or {}
        self.openxr_connected = self.openxr_connected or bool(capture.get("connected"))
        self.openxr_frames = max(self.openxr_frames, int(capture.get("frames") or 0))
        self.remote_connected = self.remote_connected or bool(remote.get("connected"))
        self.remote_packets = max(self.remote_packets, int(remote.get("packets") or 0))
        self.haptic_events = max(self.haptic_events, int(capture.get("haptic_events") or 0))
        self.max_grab_left = max(self.max_grab_left, float(capture.get("grab_left") or 0.0))
        self.max_grab_right = max(self.max_grab_right, float(capture.get("grab_right") or 0.0))
        # The shim's current grab value is only a snapshot. On a slow runner a
        # status poll can easily land between the synthetic trigger pulses, so
        # scan the recent event history as well; it covers several seconds and
        # each poll sees everything that happened since the pulse.
        for event in payload.get("capture_events") or []:
            if not isinstance(event, dict) or event.get("type") != "grab":
                continue
            try:
                value = float(event.get("value") or 0.0)
            except (TypeError, ValueError):
                continue
            hand = event.get("hand")
            if hand == "left":
                self.max_grab_left = max(self.max_grab_left, value)
            elif hand == "right":
                self.max_grab_right = max(self.max_grab_right, value)

        markers = payload.get("markers") or {}
        for marker_id, pose in markers.items():
            if isinstance(pose, dict) and pose.get("valid"):
                try:
                    self.markers_seen.add(int(marker_id))
                except (TypeError, ValueError):
                    continue
        body = payload.get("body") or {}
        if any(isinstance(pose, dict) and pose.get("valid") for pose in body.values()):
            self.body_seen = True

        composed = payload.get("composed") or {}
        for name in self._positions:
            pose = composed.get(name)
            if not isinstance(pose, dict) or not pose.get("valid"):
                continue
            position = pose.get("position") or []
            if len(position) == 3:
                self._positions[name].append(
                    (float(position[0]), float(position[1]), float(position[2]))
                )

        tracking = payload.get("tracking") or {}
        gamepads = tracking.get("gamepads") or {}
        haptics = gamepads.get("haptics") or {}
        self.gamepad_haptics = max(
            self.gamepad_haptics, int(haptics.get("count") or 0)
        )

    def pose_motion(self) -> dict[str, float]:
        motion = {}
        for name, positions in self._positions.items():
            if len(positions) < 2:
                motion[name] = 0.0
                continue
            spread = 0.0
            for axis in range(3):
                values = [position[axis] for position in positions]
                spread = max(spread, max(values) - min(values))
            motion[name] = round(spread, 4)
        return motion

    def _collect_evidence(self) -> None:
        if self._evidence_collected:
            return
        self._evidence_collected = True
        self._sample_status()
        if self._device_ready:
            logcat = self._adb("logcat", "-d", "-v", "threadtime", check=False, timeout=60)
            (self.artifact_dir / "logcat.txt").write_text(logcat.stdout, encoding="utf-8")
            filtered = self._read_phone_log()
            (self.artifact_dir / "phone-log.txt").write_text(filtered, encoding="utf-8")
            # ``-t`` counts lines before the tag filter, so the connection line
            # can scroll out of the short tail on a busy emulator. The full
            # dump is the authoritative record, and displayed frames imply a
            # working video connection even if the status line was missed.
            self.video_connected = (
                self.video_connected
                or "Video connected" in filtered
                or "Video connected" in logcat.stdout
                or self.frames > 0
            )
            # The live poll only looks at the last 400 log lines, which a busy
            # software-rendered emulator can fill in well under a second. The
            # full dump is the authoritative record for one-shot startup lines
            # and for the periodic frame-rate summaries.
            log_text = filtered + "\n" + logcat.stdout
            self.phone_imu_seen = self.phone_imu_seen or "imu:" in log_text
            self.hardware_decode_seen = (
                self.hardware_decode_seen
                or "hardware decoder started: h264" in log_text
            )
            counts = [int(value) for value in re.findall(r"displayed (\d+) video frames", log_text)]
            if counts:
                self.frames = max(self.frames, max(counts))
            self.decode_fps = [
                float(value)
                for value in re.findall(r"video: Video \d+ frames .*? ([\d.]+) fps", log_text)
            ]
            self.display_fps = [
                float(value) for value in re.findall(r"display ([\d.]+) fps", log_text)
            ]
        desktop_log = self._read_text(self.artifact_dir / "desktop-app.log")
        self.desktop_imu_seen = "phone IMU samples from" in desktop_log
        self.summary["phone"] = {
            "frames": self.frames,
            "video_connected": self.video_connected,
            "imu_seen": self.phone_imu_seen,
            "hardware_decode_seen": self.hardware_decode_seen,
            "stereo_difference": round(self.stereo_diff, 2),
            "decode_fps_mean": _mean(self.decode_fps),
            "decode_fps_max": round(max(self.decode_fps), 1) if self.decode_fps else 0.0,
            "display_fps_mean": _mean(self.display_fps),
            "display_fps_max": round(max(self.display_fps), 1) if self.display_fps else 0.0,
        }
        self.summary["desktop_imu_seen"] = self.desktop_imu_seen
        self.summary["openxr"] = {
            "capture_connected": self.openxr_connected,
            "capture_frames": self.openxr_frames,
            "remote_connected": self.remote_connected,
            "remote_packets": self.remote_packets,
            "max_grab_left": round(self.max_grab_left, 3),
            "max_grab_right": round(self.max_grab_right, 3),
            "haptic_events": self.haptic_events,
            "gamepad_haptics": self.gamepad_haptics,
            "markers": sorted(self.markers_seen),
            "body": self.body_seen,
            "pose_motion_m": self.pose_motion(),
            "status_samples": self.status_samples,
        }
        _log(
            "phone video connected=%s, frames=%d, phone IMU=%s, desktop IMU=%s"
            % (self.video_connected, self.frames, self.phone_imu_seen, self.desktop_imu_seen)
        )
        _log(
            "OpenXR: capture=%s frames=%d, remote=%s packets=%d, "
            "grab L/R=%.2f/%.2f, haptics=%d (gamepad %d), markers=%s, body=%s"
            % (
                self.openxr_connected,
                self.openxr_frames,
                self.remote_connected,
                self.remote_packets,
                self.max_grab_left,
                self.max_grab_right,
                self.haptic_events,
                self.gamepad_haptics,
                sorted(self.markers_seen),
                self.body_seen,
            )
        )
        _log(
            "phone decode FPS mean %.1f max %.1f, display FPS mean %.1f max %.1f"
            % (
                _mean(self.decode_fps),
                max(self.decode_fps) if self.decode_fps else 0.0,
                _mean(self.display_fps),
                max(self.display_fps) if self.display_fps else 0.0,
            )
        )

    def _combine_videos(self) -> None:
        if self._combined:
            return
        desktop = self.artifact_dir / "desktop-app.mp4"
        emulator = self.artifact_dir / "android-emulator.mp4"
        if not desktop.exists() or not emulator.exists():
            return
        self._combined = True
        combined = self.artifact_dir / "combined-session.mp4"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(desktop),
            "-i",
            str(emulator),
            "-filter_complex",
            (
                f"[0:v]fps={RECORD_FPS},scale=1440:540:force_original_aspect_ratio=decrease,"
                f"pad=1440:540:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[a];"
                f"[1:v]fps={RECORD_FPS},scale=960:540:force_original_aspect_ratio=decrease,"
                f"pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[b];"
                f"[a][b]hstack=inputs=2[v]"
            ),
            "-map",
            "[v]",
            "-r",
            str(RECORD_FPS),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            str(combined),
        ]
        result = self._exec(command, check=False, quiet=True, timeout=600)
        if result.returncode != 0 or not combined.exists():
            self.summary["combine_error"] = (result.stderr or result.stdout).strip()
            _log(f"could not combine the recordings: {result.stderr.strip()}")
            return
        _log(f"combined recording written ({combined.stat().st_size} bytes)")

    def _verify_outputs(self) -> None:
        desktop = self.artifact_dir / "desktop-app.mp4"
        emulator = self.artifact_dir / "android-emulator.mp4"
        combined = self.artifact_dir / "combined-session.mp4"
        if self.frames < self.config.minimum_frames:
            raise E2EError(
                f"only {self.frames} phone frames were displayed "
                f"(expected at least {self.config.minimum_frames})"
            )
        if not self.video_connected:
            raise E2EError("the phone never logged a video connection")
        if self.screenshot_mean < 6.0 or self.screenshot_std < 6.0:
            raise E2EError(
                "the emulator screenshot looks blank "
                f"(mean {self.screenshot_mean:.1f}, std {self.screenshot_std:.1f})"
            )
        if self.screenshot_changed_pixels < 2000:
            raise E2EError(
                "the emulator screenshots are nearly identical "
                f"({self.screenshot_changed_pixels} changed pixels)"
            )
        if self.stereo_diff < 0.5:
            raise E2EError(
                "the phone is not showing two distinct stereo eye views "
                f"(left/right difference {self.stereo_diff:.2f})"
            )
        screenshot = self.summary.get("screenshots") or {}
        if int(screenshot.get("width") or 0) <= int(screenshot.get("height") or 0):
            raise E2EError("the emulator screenshot is not in landscape orientation")

        if not self.openxr_connected:
            raise E2EError("the OpenXR application never connected to the capture shim")
        if self.openxr_frames < 30:
            raise E2EError(f"only {self.openxr_frames} OpenXR stereo frames were captured")
        if not self.remote_connected or self.remote_packets <= 0:
            raise E2EError(
                "the desktop app did not feed poses to Monado "
                f"(connected={self.remote_connected}, packets={self.remote_packets})"
            )
        if not self.expected_markers.issubset(self.markers_seen):
            raise E2EError(f"synthetic markers were not all tracked: {sorted(self.markers_seen)}")
        if not self.body_seen:
            raise E2EError("the synthetic human skeleton was never tracked")
        if self.max_grab_left <= 0.75 or self.max_grab_right <= 0.75:
            raise E2EError(
                "the synthetic Xbox triggers did not trigger a grab in the OpenXR app "
                f"(left {self.max_grab_left:.2f}, right {self.max_grab_right:.2f})"
            )
        if self.haptic_events <= 0 or self.gamepad_haptics <= 0:
            raise E2EError(
                "the OpenXR app's haptics did not reach the synthetic Xbox pads "
                f"(shim {self.haptic_events}, gamepads {self.gamepad_haptics})"
            )
        motion = self.pose_motion()
        for name, minimum in (("hmd", 0.02), ("left", 0.02), ("right", 0.02)):
            if motion.get(name, 0.0) <= minimum:
                raise E2EError(
                    f"the {name} pose barely moved in the OpenXR session "
                    f"({motion.get(name, 0.0):.3f} m); 6DoF motion is not visible"
                )
        for path, minimum in ((desktop, 20_000), (emulator, 20_000), (combined, 20_000)):
            if not path.exists() or path.stat().st_size < minimum:
                raise E2EError(f"recording is missing or too small: {path.name}")

        self.desktop_info = self._probe_and_verify_video(desktop, "desktop-app-frame.png")
        self.emulator_info = self._probe_and_verify_video(emulator, "android-emulator-frame.png")
        self.combined_info = self._probe_video(combined)
        if self.emulator_info["width"] <= self.emulator_info["height"]:
            raise E2EError(
                "the emulator recording is portrait "
                f"({self.emulator_info['width']}x{self.emulator_info['height']}); "
                "screenrecord did not capture the landscape session"
            )
        self.summary["videos"] = {
            "desktop": self.desktop_info,
            "android": self.emulator_info,
            "combined": self.combined_info,
        }
        self.summary["frames"] = self.frames

    def _probe_and_verify_video(self, path: Path, frame_name: str) -> dict:
        info = self._probe_video(path)
        frame = self.artifact_dir / frame_name
        at = max(0.5, min(info["duration"] * 0.5, 5.0))
        result = self._exec(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{at:.2f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                str(frame),
            ],
            check=False,
            quiet=True,
            timeout=60,
        )
        if result.returncode == 0 and frame.exists():
            try:
                from PIL import Image
                import numpy as np

                pixels = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32)
                info["frame_mean"] = round(float(pixels.mean()), 2)
                info["frame_std"] = round(float(pixels.std()), 2)
                if info["frame_mean"] < 4.0 or info["frame_std"] < 4.0:
                    raise E2EError(f"{path.name} appears blank")
            except ImportError:
                pass
        return info

    def _probe_video(self, path: Path) -> dict:
        result = self._exec(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            quiet=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise E2EError(f"ffprobe could not read {path.name}:\n{result.stderr}")
        payload = json.loads(result.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        fmt = payload.get("format") or {}
        return {
            "file": path.name,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": round(float(fmt.get("duration", 0.0) or 0.0), 2),
            "bytes": int(fmt.get("size", path.stat().st_size) or path.stat().st_size),
        }

    # ------------------------------------------------------------------
    # Shutdown and summary
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        for step, action in (
            ("stop screen recording", self._stop_screen_recording),
            ("stop desktop recording", self._stop_desktop_recording),
            ("collect evidence", self._collect_evidence),
            ("stop phone app", self._stop_phone),
            ("stop OpenXR stack", self._stop_openxr),
            ("stop desktop app", self._stop_desktop),
            ("stop display", self._stop_display),
            ("stop emulator", self._stop_emulator),
            ("combine recordings", self._combine_videos),
            ("write summary", self._write_summary),
        ):
            try:
                action()
            except Exception as exc:  # cleanup must never mask the original failure
                _log(f"{step} failed: {type(exc).__name__}: {exc}")

    def _stop_phone(self) -> None:
        if not self._device_ready:
            return
        self._adb("shell", "am", "force-stop", PACKAGE, check=False, timeout=15)

    def _stop_openxr(self) -> None:
        stack = self.openxr
        self.openxr = None
        if stack is not None:
            stack.stop()
            _log("OpenXR stack stopped")

    def _stop_desktop(self) -> None:
        if self._desktop is None:
            return
        self._stop_process(self._desktop, "desktop app")
        self._desktop = None

    def _stop_display(self) -> None:
        if self._xvfb is None:
            return
        self._stop_process(self._xvfb, "Xvfb")
        self._xvfb = None

    def _stop_emulator(self) -> None:
        if self._emulator is None:
            return
        if not self._started_emulator:
            # Attached emulator (for example started by CI); leave its lifecycle
            # to whoever launched it.
            self._emulator = None
            return
        if self.config.keep_emulator:
            _log(f"leaving emulator {self.serial} running (--keep-emulator)")
            self._emulator = None
            return
        if self.serial:
            self._adb("emu", "kill", check=False, timeout=15)
        self._stop_process(self._emulator, "Android emulator", timeout=20)
        self._emulator = None

    def _write_summary(self) -> None:
        if self._summary_written:
            return
        self._summary_written = True
        self.summary["finished_at"] = _utc_now()
        if self._run_started:
            self.summary["duration"] = round(time.monotonic() - self._run_started, 2)
        if self.screenshot_mean:
            self.summary.setdefault("screenshots", {})["mean"] = round(self.screenshot_mean, 2)
        (self.artifact_dir / "summary.json").write_text(
            json.dumps(self.summary, indent=2) + "\n", encoding="utf-8"
        )

    def _result(self) -> E2EResult:
        return E2EResult(
            frames=self.frames,
            video_connected=self.video_connected,
            phone_imu_seen=self.phone_imu_seen,
            desktop_imu_seen=self.desktop_imu_seen,
            screenshot_mean=self.screenshot_mean,
            screenshot_std=self.screenshot_std,
            screenshot_diff=self.screenshot_diff,
            screenshot_changed_pixels=self.screenshot_changed_pixels,
            stereo_diff=self.stereo_diff,
            openxr_connected=self.openxr_connected,
            openxr_frames=self.openxr_frames,
            markers_seen=sorted(self.markers_seen),
            body_seen=self.body_seen,
            max_grab_left=self.max_grab_left,
            max_grab_right=self.max_grab_right,
            haptic_events=self.haptic_events,
            gamepad_haptics=self.gamepad_haptics,
            pose_motion=self.pose_motion(),
            hardware_decode_seen=self.hardware_decode_seen,
            desktop_video=self.artifact_dir / "desktop-app.mp4",
            emulator_video=self.artifact_dir / "android-emulator.mp4",
            combined_video=self.artifact_dir / "combined-session.mp4",
            summary=self.artifact_dir / "summary.json",
        )

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------
    def _exec(
        self,
        command: list[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
        timeout: float = 120.0,
        input_text: Optional[str] = None,
        check: bool = True,
        quiet: bool = False,
    ) -> subprocess.CompletedProcess:
        if not quiet:
            _log("run: " + shlex.join(command))
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                env=env or self.env,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise E2EError(f"command not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise E2EError(f"command timed out after {timeout:.0f} s: {shlex.join(command)}") from exc
        if check and result.returncode != 0:
            raise E2EError(
                f"command failed ({result.returncode}): {shlex.join(command)}\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return result

    def _spawn(
        self,
        command: list[str],
        log_name: str,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.Popen:
        target = self.artifact_dir / log_name
        _log("spawn: " + shlex.join(command))
        with target.open("wb") as handle:
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env or self.env,
                start_new_session=True,
            )
        return process

    def _stop_process(
        self,
        process: subprocess.Popen,
        name: str,
        sig: int = signal.SIGTERM,
        timeout: float = 10.0,
    ) -> None:
        if process.poll() is not None:
            return
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _log(f"{name} did not stop; killing it")
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _tail(path: Path, lines: int = 60) -> str:
        text = EmulatorE2E._read_text(path)
        return "\n".join(text.splitlines()[-lines:])
