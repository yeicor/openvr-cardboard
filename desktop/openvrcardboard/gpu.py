"""NVIDIA/ffmpeg capability detection.

The desktop app is usable on any machine with a webcam, but it is designed for
the common VR PC with an NVIDIA GPU. Everything GPU-specific is capability
gated: when ``nvidia-smi`` or ffmpeg's NVENC encoders are missing, callers fall
back to the CPU/JPEG paths used by CI and by machines without a dGPU.

Detection is intentionally cheap and cached; ``nvidia-smi`` is only queried
once per process unless ``refresh=True`` is passed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

_CACHE: dict[tuple[str, str], "GpuCapabilities"] = {}
_LOCK = threading.Lock()


@dataclass(slots=True)
class GpuCapabilities:
    available: bool = False
    name: str = ""
    driver: str = ""
    memory_mb: int = 0
    ffmpeg: str = ""
    encoders: set[str] = field(default_factory=set)
    reason: str = ""

    @property
    def nvenc(self) -> bool:
        return self.available and any(
            name.endswith("_nvenc") for name in self.encoders
        )

    def supports(self, codec: str) -> bool:
        codec = (codec or "").lower()
        return self.available and f"{codec}_nvenc" in self.encoders

    def best_codec(self, allow_av1: bool = True) -> str:
        """Return the preferred hardware codec, or ``jpeg``.

        HEVC is preferred over AV1 because every recent phone decodes it in
        hardware; AV1 is opt-in through ``phone.codec``.
        """
        if not self.nvenc:
            return "jpeg"
        if self.supports("hevc"):
            return "hevc"
        if allow_av1 and self.supports("av1"):
            return "av1"
        if self.supports("h264"):
            return "h264"
        return "jpeg"

    def summary(self) -> str:
        if self.available:
            memory = f", {self.memory_mb} MiB" if self.memory_mb else ""
            codecs = ", ".join(
                sorted(name for name in self.encoders if name.endswith("_nvenc"))
            )
            return f"{self.name}{memory} (NVENC: {codecs or 'none'})"
        return f"no NVIDIA GPU detected ({self.reason or 'nvidia-smi unavailable'})"


def _bundled_ffmpeg() -> Optional[str]:
    """Return an ffmpeg shipped beside the frozen app."""
    candidates: list[Path] = []
    override = os.environ.get("OPENVRCARDBOARD_FFMPEG")
    if override:
        candidates.append(Path(override).expanduser())
    bundle = getattr(sys, "_MEIPASS", "")
    if bundle:
        candidates.append(Path(bundle) / "ffmpeg")
        candidates.append(Path(bundle) / "bin" / "ffmpeg")
    executable = Path(sys.executable).resolve()
    candidates.append(executable.parent / "ffmpeg")
    candidates.append(executable.parent / "bin" / "ffmpeg")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    # ``imageio-ffmpeg`` ships a static ffmpeg with NVENC encoders. It is the
    # preferred fallback because it works in the frozen package, on machines
    # without ffmpeg on PATH, and in CI. Importing it is cheap and cached by
    # Python; a missing optional dependency just falls through.
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    return found


def find_ffmpeg(configured: str = "") -> str:
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return _bundled_ffmpeg() or ""


def _run(command: list[str], timeout: float = 4.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _query_nvidia() -> tuple[str, str, int, str]:
    executable = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    if not Path(executable).exists():
        return "", "", 0, "nvidia-smi not found"
    try:
        result = _run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", "", 0, str(exc)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip().splitlines()
        return "", "", 0, message[0] if message else f"nvidia-smi exit {result.returncode}"
    line = next((row for row in result.stdout.splitlines() if row.strip()), "")
    if not line:
        return "", "", 0, "nvidia-smi returned no GPU"
    parts = [part.strip() for part in line.split(",")]
    name = parts[0] if parts else "NVIDIA GPU"
    try:
        memory = int(float(parts[1])) if len(parts) > 1 else 0
    except ValueError:
        memory = 0
    driver = parts[2] if len(parts) > 2 else ""
    return name, driver, memory, ""


def _query_encoders(ffmpeg: str) -> tuple[set[str], str]:
    if not ffmpeg:
        return set(), "ffmpeg not found"
    try:
        result = _run([ffmpeg, "-hide_banner", "-encoders"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return set(), str(exc)
    if result.returncode != 0:
        return set(), "ffmpeg could not list encoders"
    encoders = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        flags = fields[0]
        # ffmpeg prints a legend line ("V..... = Video") before the list.
        if len(flags) != 6 or flags[0] not in "VAS" or flags[0] != "V":
            continue
        name = fields[1]
        if name in {"=", "-"} or not name:
            continue
        encoders.add(name)
    return encoders, ""


def detect_gpu(ffmpeg: str = "", refresh: bool = False) -> GpuCapabilities:
    """Detect an NVIDIA GPU and the NVENC encoders available to ffmpeg."""
    executable = find_ffmpeg(ffmpeg)
    disabled = os.environ.get("OPENVRCARDBOARD_DISABLE_GPU", "").strip().lower()
    if disabled in ("1", "true", "yes", "on"):
        # Still inspect ffmpeg so the CPU libx264 backend remains available;
        # only the NVIDIA/NVENC path is disabled.
        encoders, _ = _query_encoders(executable)
        return GpuCapabilities(
            available=False,
            ffmpeg=executable,
            encoders=encoders,
            reason="GPU acceleration disabled by OPENVRCARDBOARD_DISABLE_GPU",
        )
    key = (executable, os.environ.get("OPENVRCARDBOARD_GPU", ""))
    if not refresh:
        with _LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached

    name, driver, memory, gpu_error = _query_nvidia()
    encoders, ffmpeg_error = _query_encoders(executable)
    nvenc = {encoder for encoder in encoders if encoder.endswith("_nvenc")}
    available = bool(nvenc) and not gpu_error
    if available:
        reason = ""
    elif gpu_error and not nvenc:
        reason = gpu_error
    elif ffmpeg_error and not encoders:
        reason = ffmpeg_error
    elif not nvenc:
        reason = "ffmpeg has no NVIDIA encoders"
    else:
        reason = gpu_error

    capabilities = GpuCapabilities(
        available=available,
        name=name or ("NVIDIA GPU" if nvenc else ""),
        driver=driver,
        memory_mb=memory,
        ffmpeg=executable,
        encoders=encoders,
        reason=reason,
    )
    with _LOCK:
        _CACHE[key] = capabilities
    if available:
        LOGGER.info("GPU acceleration available: %s", capabilities.summary())
    else:
        LOGGER.info("GPU acceleration unavailable: %s", capabilities.summary())
    return capabilities


def reset_cache() -> None:
    """Clear the detection cache (used by tests and after settings change)."""
    with _LOCK:
        _CACHE.clear()
