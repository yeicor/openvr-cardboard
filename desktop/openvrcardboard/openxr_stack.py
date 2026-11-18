"""Launcher for the real OpenXR stack used by recorded end-to-end tests.

The stack is built by ``scripts/build-openxr.sh`` and contains Monado with the
remote driver and null compositor, the Khronos OpenXR loader, Monado's
``openxr-simple-example`` demo and the OpenVR Cardboard capture shim. This
module only orchestrates those pinned binaries: it writes a throw-away Monado
configuration, keeps the service's stdin pipe open (Monado exits when stdin
closes) and starts the example with ``LD_PRELOAD`` pointing at the shim.

It deliberately does not import the rest of the application, so both the
desktop test suite and the Android emulator harness can use it.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .monado_ipc import monado_socket_fits

LOGGER = logging.getLogger(__name__)

DEFAULT_REMOTE_PORT = 4242
DEFAULT_CAPTURE_PORT = 39102
DEFAULT_VIEW_WIDTH = 1280
DEFAULT_VIEW_HEIGHT = 720

# Monado's null compositor shares textures between the application's OpenGL
# context and its own Vulkan device. Both must come from the same GPU vendor:
# importing an AMD/Intel dma-buf into an NVIDIA Vulkan device (or the other way
# round) makes Mesa abort while the application creates its first swapchain.
_VULKAN_ICD_DIRS = (
    Path("/usr/share/vulkan/icd.d"),
    Path("/usr/local/share/vulkan/icd.d"),
    Path("/etc/vulkan/icd.d"),
)


class OpenXRStackError(RuntimeError):
    """Raised when the OpenXR runtime or the example cannot start."""


def _find_vulkan_icd(prefixes: tuple[str, ...]) -> Optional[Path]:
    for directory in _VULKAN_ICD_DIRS:
        try:
            entries = sorted(directory.glob("*.json"))
        except OSError:
            continue
        for entry in entries:
            name = entry.name.lower()
            if any(name.startswith(prefix) for prefix in prefixes):
                return entry
    return None


def _opengl_vendor() -> str:
    """Return the GLX renderer vendor (``amd``, ``nvidia``, ...), or ``""``."""
    try:
        result = subprocess.run(
            ["glxinfo", "-B"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "vendor":
            return value.strip().lower()
    return ""


def choose_vulkan_icd() -> Optional[Path]:
    """Pick a Vulkan implementation that can share textures with GLX.

    ``VK_ICD_FILENAMES`` is only set for the test stack when the caller has not
    chosen an ICD already. Matching the GL vendor avoids the cross-vendor
    external-memory crash described above; software GL (Xvfb/llvmpipe) prefers
    lavapipe, which is both correct and what CI uses.
    """
    vendor = _opengl_vendor()
    if "nvidia" in vendor:
        return _find_vulkan_icd(("nvidia_icd",))
    if any(token in vendor for token in ("amd", "ati", "radeon")):
        return _find_vulkan_icd(("radeon_icd",))
    if "intel" in vendor:
        return _find_vulkan_icd(("intel_icd",))
    # Mesa software rendering, or an unknown/absent GLX vendor.
    return _find_vulkan_icd(("lvp_icd", "swrast"))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_openxr_prefix() -> Optional[Path]:
    """Return the first installed OpenXR stack, or ``None`` when absent."""
    candidates = []
    configured = os.environ.get("OVRC_OPENXR_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(_repository_root() / "out" / "openxr")
    candidates.append(Path("/opt/openxr"))
    for candidate in candidates:
        if (candidate / "bin" / "openxr-example").is_file() and (
            candidate / "bin" / "monado-service"
        ).is_file():
            return candidate.resolve()
    return None


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _tail(path: Path, lines: int = 30) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


@dataclass(slots=True)
class OpenXRStackStats:
    running: bool = False
    service_pid: int = 0
    example_pid: int = 0
    remote_port: int = 0
    capture_port: int = 0
    prefix: str = ""
    error: str = ""


class OpenXRStack:
    """Own ``monado-service`` and the shimmed OpenXR example process."""

    def __init__(
        self,
        prefix: Optional[Path] = None,
        work_dir: Optional[Path] = None,
        remote_port: int = DEFAULT_REMOTE_PORT,
        capture_port: int = DEFAULT_CAPTURE_PORT,
        view_width: int = DEFAULT_VIEW_WIDTH,
        view_height: int = DEFAULT_VIEW_HEIGHT,
        display: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
        capture_fps: int = 30,
        capture_quality: int = 80,
    ):
        configured = prefix or find_openxr_prefix()
        if configured is None:
            raise OpenXRStackError(
                "OpenXR runtime not found; run scripts/build-openxr.sh or set "
                "OVRC_OPENXR_ROOT"
            )
        self.prefix = Path(configured).resolve()
        self.work_dir = Path(work_dir or (Path.home() / ".cache" / "openvrcardboard" / "openxr-e2e"))
        self.remote_port = int(remote_port)
        self.capture_port = int(capture_port)
        self.view_width = int(view_width)
        self.view_height = int(view_height)
        self.capture_fps = max(1, int(capture_fps))
        self.capture_quality = max(20, min(95, int(capture_quality)))
        self.display = display or os.environ.get("DISPLAY", "")
        self.extra_env = dict(extra_env or {})
        self._service: Optional[subprocess.Popen] = None
        self._example: Optional[subprocess.Popen] = None
        self._temporary_runtime_dir: Optional[Path] = None
        self._started_at = 0.0
        self._error = ""
        self.service_log = self.work_dir / "monado-service.log"
        self.example_log = self.work_dir / "openxr-example.log"

    # ------------------------------------------------------------------
    # Paths and configuration
    # ------------------------------------------------------------------
    @property
    def example(self) -> Path:
        return self.prefix / "bin" / "openxr-example"

    @property
    def service(self) -> Path:
        return self.prefix / "bin" / "monado-service"

    @property
    def shim(self) -> Path:
        return self.prefix / "lib" / "libovrc_xr_capture.so"

    @property
    def runtime_manifest(self) -> Path:
        return self.prefix / "share" / "openxr" / "1" / "openxr_monado.json"

    @property
    def running(self) -> bool:
        return self._service is not None or self._example is not None

    @property
    def stats(self) -> OpenXRStackStats:
        return OpenXRStackStats(
            running=self.running,
            service_pid=self._service.pid if self._service else 0,
            example_pid=self._example.pid if self._example else 0,
            remote_port=self.remote_port,
            capture_port=self.capture_port,
            prefix=str(self.prefix),
            error=self._error,
        )

    def _ensure_installed(self) -> None:
        missing = [
            path
            for path in (self.example, self.service, self.shim, self.runtime_manifest)
            if not path.exists()
        ]
        if missing:
            raise OpenXRStackError(
                "OpenXR stack is incomplete; missing: "
                + ", ".join(str(path) for path in missing)
            )

    def _runtime_directory(self) -> Path:
        """Return a runtime dir where Monado's IPC socket path fits.

        Monado appends its fixed socket filename to ``XDG_RUNTIME_DIR``; when
        the work directory is nested deeply enough that the result exceeds
        ``sockaddr_un.sun_path``, use a short temporary directory instead. The
        same path is passed to the service and to the OpenXR application.
        """
        preferred = self.work_dir / "xdg-runtime"
        if monado_socket_fits(preferred):
            return preferred
        if self._temporary_runtime_dir is None:
            self._temporary_runtime_dir = Path(
                tempfile.mkdtemp(prefix="openvrcardboard-openxr-")
            )
            LOGGER.info(
                "using %s for Monado's IPC socket because %s is too long",
                self._temporary_runtime_dir,
                preferred,
            )
        return self._temporary_runtime_dir

    def _environment(self) -> dict[str, str]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        config_home = self.work_dir / "xdg-config"
        runtime_dir = self._runtime_directory()
        cache_home = self.work_dir / "xdg-cache"
        monado_config = config_home / "monado"
        for directory in (config_home, runtime_dir, cache_home, monado_config):
            directory.mkdir(parents=True, exist_ok=True)
        try:
            runtime_dir.chmod(0o700)
        except OSError:
            pass
        (monado_config / "config_v0.json").write_text(
            "{\n"
            '  "active": "remote",\n'
            '  "remote": {\n'
            '    "version": 0,\n'
            f'    "port": {self.remote_port},\n'
            '    "view_count": 2,\n'
            f'    "w_pixels": {self.view_width},\n'
            f'    "h_pixels": {self.view_height},\n'
            '    "w_meters": 0.13,\n'
            '    "h_meters": 0.07\n'
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env.update(self.extra_env)
        env["XDG_CONFIG_HOME"] = str(config_home)
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
        env["XDG_CACHE_HOME"] = str(cache_home)
        env["XR_RUNTIME_JSON"] = str(self.runtime_manifest)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(self.prefix / "lib"), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
        env["XRT_COMPOSITOR_NULL"] = "1"
        env["XRT_COMPOSITOR_NULL_FPS"] = str(max(30, min(240, self.capture_fps)))
        env["P_OVERRIDE_ACTIVE_CONFIG"] = "remote"
        if self.display:
            env["DISPLAY"] = self.display
        if not env.get("VK_ICD_FILENAMES") and not env.get("VK_DRIVER_FILES"):
            icd = choose_vulkan_icd()
            if icd is not None:
                env["VK_ICD_FILENAMES"] = str(icd)
        return env

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, timeout: float = 25.0) -> None:
        if self.running:
            return
        self._ensure_installed()
        env = self._environment()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.service_log.write_text("", encoding="utf-8")
        self.example_log.write_text("", encoding="utf-8")

        service_out = self.service_log.open("ab")
        try:
            self._service = subprocess.Popen(
                [str(self.service)],
                stdin=subprocess.PIPE,
                stdout=service_out,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        finally:
            service_out.close()
        LOGGER.info("started monado-service (pid %d)", self._service.pid)

        deadline = time.monotonic() + timeout
        ready = False
        while time.monotonic() < deadline:
            if self._service.poll() is not None:
                raise OpenXRStackError(
                    "monado-service exited during startup:\n"
                    + _tail(self.service_log, 40)
                )
            content = self.service_log.read_text(encoding="utf-8", errors="replace")
            if "Config selected remote" in content and "null_compositor" in content:
                ready = True
                break
            time.sleep(0.1)
        if not ready:
            raise OpenXRStackError(
                "monado-service did not become ready:\n" + _tail(self.service_log, 40)
            )

        example_env = dict(env)
        example_env["LD_PRELOAD"] = str(self.shim)
        example_env["OVRC_CAPTURE_HOST"] = "127.0.0.1"
        example_env["OVRC_CAPTURE_PORT"] = str(self.capture_port)
        example_env["OVRC_CAPTURE_FPS"] = str(self.capture_fps)
        example_env["OVRC_CAPTURE_QUALITY"] = str(self.capture_quality)
        # Capture at the runtime's native per-eye size. The example's
        # recommended views are half of the configured stereo display, so the
        # resulting side-by-side frame has the same 16:9 aspect as the phone
        # panel and fills it without letterboxing.
        example_env["OVRC_CAPTURE_EYE_WIDTH"] = str(max(1, self.view_width // 2))
        example_env["OVRC_CAPTURE_EYE_HEIGHT"] = str(max(1, self.view_height))
        example_out = self.example_log.open("ab")
        try:
            self._example = subprocess.Popen(
                [str(self.example)],
                stdout=example_out,
                stderr=subprocess.STDOUT,
                env=example_env,
                start_new_session=True,
            )
        finally:
            example_out.close()
        self._started_at = time.monotonic()
        LOGGER.info("started openxr-simple-example (pid %d)", self._example.pid)

    def wait_example_connected(self, timeout: float = 20.0) -> None:
        """Wait until the shim reports that the desktop capture server accepted it."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._example is not None and self._example.poll() is not None:
                raise OpenXRStackError(
                    "openxr-example exited before connecting:\n"
                    + _tail(self.example_log, 60)
                )
            content = self.example_log.read_text(encoding="utf-8", errors="replace")
            if "connected to the desktop capture server" in content:
                return
            time.sleep(0.1)
        raise OpenXRStackError(
            "openxr-example did not connect to the capture server:\n"
            + _tail(self.example_log, 60)
        )

    def stop(self) -> None:
        example = self._example
        self._example = None
        if example is not None:
            if example.poll() is None:
                example.terminate()
                try:
                    example.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    example.kill()
                    example.wait(timeout=5.0)

        service = self._service
        self._service = None
        if service is not None:
            if service.stdin is not None:
                try:
                    service.stdin.close()
                except OSError:
                    pass
            if service.poll() is None:
                service.terminate()
                try:
                    service.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.wait(timeout=5.0)

        temporary = self._temporary_runtime_dir
        self._temporary_runtime_dir = None
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    def __enter__(self) -> "OpenXRStack":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
