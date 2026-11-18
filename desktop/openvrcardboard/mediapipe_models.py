"""MediaPipe Tasks model assets: bundled, cached and downloaded on demand.

The Tasks API loads a ``.task`` model file instead of the model that used to be
compiled into the legacy ``mediapipe.solutions`` packages. Keeping the files in
a cache makes the desktop app small and lets users run fully offline once a
model has been fetched. Portable builds ship the same files next to the
executable, and ``OPENVRCARDBOARD_MODEL_DIR`` overrides both locations.

Downloads are verified against a pinned SHA-256 before they become visible to
MediaPipe, so an interrupted transfer or a captive portal can never leave a
corrupt model behind.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

LOGGER = logging.getLogger(__name__)

MODEL_DIR_ENV = "OPENVRCARDBOARD_MODEL_DIR"
_MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models"


class ModelUnavailable(RuntimeError):
    """Raised when a MediaPipe model cannot be found or downloaded."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One pinned MediaPipe Tasks model."""

    key: str
    filename: str
    url: str
    sha256: str
    size: int

    @property
    def short_name(self) -> str:
        return self.filename.removesuffix(".task")


def _spec(key: str, folder: str, filename: str, sha256: str, size: int) -> ModelSpec:
    # The pose variants live under their own model folder (for example
    # pose_landmarker/pose_landmarker_full/...), not under a shared folder
    # named after the family twice. Deriving the URL from the filename keeps
    # all four assets on the official MediaPipe storage path.
    model_name = filename.removesuffix(".task")
    return ModelSpec(
        key=key,
        filename=filename,
        url=f"{_MODEL_BASE_URL}/{folder}/{model_name}/float16/1/{filename}",
        sha256=sha256,
        size=size,
    )


HAND_LANDMARKER = _spec(
    "hand_landmarker",
    "hand_landmarker",
    "hand_landmarker.task",
    "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    7_819_105,
)
POSE_LANDMARKER_LITE = _spec(
    "pose_landmarker_lite",
    "pose_landmarker",
    "pose_landmarker_lite.task",
    "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    5_777_746,
)
POSE_LANDMARKER_FULL = _spec(
    "pose_landmarker_full",
    "pose_landmarker",
    "pose_landmarker_full.task",
    "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
    9_398_198,
)
POSE_LANDMARKER_HEAVY = _spec(
    "pose_landmarker_heavy",
    "pose_landmarker",
    "pose_landmarker_heavy.task",
    "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
    30_664_242,
)

MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        HAND_LANDMARKER,
        POSE_LANDMARKER_LITE,
        POSE_LANDMARKER_FULL,
        POSE_LANDMARKER_HEAVY,
    )
}

POSE_MODELS: dict[str, ModelSpec] = {
    "lite": POSE_LANDMARKER_LITE,
    "full": POSE_LANDMARKER_FULL,
    "heavy": POSE_LANDMARKER_HEAVY,
}


def pose_model_spec(variant: str) -> ModelSpec:
    """Return the pose model for a user-facing ``lite``/``full``/``heavy``."""
    return POSE_MODELS.get(str(variant).strip().lower(), POSE_LANDMARKER_FULL)


def model_spec(name: str | ModelSpec) -> ModelSpec:
    if isinstance(name, ModelSpec):
        return name
    try:
        return MODELS[str(name)]
    except KeyError as exc:
        raise ModelUnavailable(f"unknown MediaPipe model {name!r}") from exc


def default_model_dir() -> Path:
    """Directory used for downloaded models.

    ``OPENVRCARDBOARD_MODEL_DIR`` wins so tests and portable installs can point
    at an existing cache. Otherwise models live beside the configuration, which
    is writable for both per-user installs and the frozen app.
    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    from .config import default_config_dir

    return default_config_dir() / "models"


def bundled_model_dirs() -> list[Path]:
    """Locations where a packaged build may ship models."""
    here = Path(__file__).resolve().parent
    candidates = [here / "models"]
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidates.append(Path(bundle) / "openvrcardboard" / "models")
    # PyInstaller onedir stores data files relative to the executable as well.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "openvrcardboard" / "models")
    return candidates


def search_dirs(cache_dir: str | Path | None = None) -> list[Path]:
    result: list[Path] = []
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        result.append(Path(override).expanduser())
    if cache_dir is not None:
        result.append(Path(cache_dir).expanduser())
    result.extend(bundled_model_dirs())
    if cache_dir is None and not override:
        result.append(default_model_dir())
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in result:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: str | Path, spec: ModelSpec) -> bool:
    """True when ``path`` is exactly the pinned model."""
    target = Path(path)
    try:
        if not target.is_file() or target.stat().st_size != spec.size:
            return False
        return sha256_file(target) == spec.sha256
    except OSError:
        return False


def _progress_printer(spec: ModelSpec) -> Callable[[int, int], None]:
    state = {"last": 0.0}

    def progress(received: int, total: int) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.5 and received < total:
            return
        state["last"] = now
        percent = 100.0 * received / total if total > 0 else 0.0
        LOGGER.info(
            "downloading %s: %.1f%% (%d/%d bytes)",
            spec.filename,
            percent,
            received,
            total,
        )

    return progress


ProgressCallback = Callable[[int, int], None]


def download_model(
    name: str | ModelSpec,
    target_dir: str | Path | None = None,
    *,
    timeout: float = 120.0,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download and verify one model, atomically publishing it on success."""
    spec = model_spec(name)
    directory = Path(target_dir).expanduser() if target_dir else default_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename
    if verify_model(target, spec):
        return target

    partial = directory / (spec.filename + ".part")
    callback = progress or _progress_printer(spec)
    LOGGER.info("fetching MediaPipe model %s", spec.url)
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "openvrcardboard model fetcher"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or spec.size)
            received = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    callback(received, total)
                    if received > spec.size:
                        raise ModelUnavailable(
                            f"{spec.filename} is larger than the pinned size"
                        )
    except (OSError, urllib.error.URLError) as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise ModelUnavailable(
            f"could not download {spec.filename}: {exc}. "
            f"Place it in {directory} or set {MODEL_DIR_ENV}."
        ) from exc

    if not verify_model(partial, spec):
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise ModelUnavailable(
            f"downloaded {spec.filename} failed its SHA-256/size check; "
            f"delete {directory / spec.filename} and retry"
        )
    os.replace(partial, target)
    LOGGER.info("MediaPipe model ready: %s", target)
    return target


_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _lock_for(spec: ModelSpec) -> threading.Lock:
    with _download_locks_guard:
        return _download_locks.setdefault(spec.key, threading.Lock())


def resolve_model(
    name: str | ModelSpec,
    cache_dir: str | Path | None = None,
    *,
    allow_download: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    """Find a valid model in the search path, downloading it when allowed.

    A corrupt file in a writable cache is removed before retrying so a single
    bad download cannot poison every later start.
    """
    spec = model_spec(name)
    candidates = search_dirs(cache_dir)
    for directory in candidates:
        candidate = directory / spec.filename
        if not candidate.is_file():
            continue
        if verify_model(candidate, spec):
            return candidate
        LOGGER.warning("ignoring corrupt MediaPipe model %s", candidate)
        if _is_writable_cache(directory):
            try:
                candidate.unlink()
            except OSError:
                pass

    if not allow_download:
        locations = ", ".join(str(path) for path in candidates)
        raise ModelUnavailable(
            f"{spec.filename} is missing; looked in {locations}. "
            f"Download it or set {MODEL_DIR_ENV}."
        )

    target_dir = Path(cache_dir).expanduser() if cache_dir else default_model_dir()
    with _lock_for(spec):
        candidate = target_dir / spec.filename
        if verify_model(candidate, spec):
            return candidate
        return download_model(spec, target_dir, progress=progress)


def _is_writable_cache(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def download_all(
    target_dir: str | Path,
    names: Optional[Iterable[str]] = None,
    *,
    progress: Optional[ProgressCallback] = None,
) -> list[Path]:
    """Fetch a set of models (used by packaging and container builds)."""
    wanted = list(names) if names is not None else list(MODELS)
    paths: list[Path] = []
    for name in wanted:
        paths.append(download_model(name, target_dir, progress=progress))
    return paths


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names",
        nargs="*",
        default=None,
        help=f"models to fetch (default: all). Choices: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help=f"destination directory (default: {default_model_dir()})",
    )
    parser.add_argument("--list", action="store_true", help="list known models")
    args = parser.parse_args(argv)
    if args.list:
        for spec in MODELS.values():
            print(f"{spec.key:28} {spec.size:>10} bytes  {spec.url}")
        return 0
    try:
        paths = download_all(args.dir or default_model_dir(), args.names or None)
    except ModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual/packaging entry point
    raise SystemExit(_main())
