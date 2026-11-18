"""Reversible desktop integration: SteamVR driver and OpenXR runtime.

The portable build keeps every file it needs inside its own folder, but two
things must live outside that folder for other applications to find them:

* the SteamVR driver under ``<SteamVR>/drivers/openvrcardboard``;
* the user-level OpenXR active runtime and Monado remote-driver configuration.

Everything this module changes is recorded in ``integration.json`` next to the
application configuration. Original files are backed up before they are
replaced, and :func:`uninstall` restores them and removes the generated driver
folder. No registry keys, environment variables or system-wide directories are
modified.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from .config import default_config_dir

DRIVER_NAME = "openvrcardboard"
BACKUP_DIRNAME = ".openvrcardboard-backups"
OPENXR_MANIFEST_NAME = "openvrcardboard_runtime.json"
DEFAULT_OPENXR_VIEW_WIDTH = 2400
DEFAULT_OPENXR_VIEW_HEIGHT = 1080
DEFAULT_OPENXR_REMOTE_PORT = 4242


class IntegrationError(RuntimeError):
    """Raised when an integration step cannot be completed safely."""


def platform_dir() -> str:
    if sys.platform.startswith("win"):
        return "win64"
    if sys.platform == "darwin":
        return "osx64"
    return "linux64"


def library_name() -> str:
    if sys.platform.startswith("win"):
        return "driver_openvrcardboard.dll"
    if sys.platform == "darwin":
        return "driver_openvrcardboard.dylib"
    return "driver_openvrcardboard.so"


def default_state_path() -> Path:
    override = os.environ.get("OPENVRCARDBOARD_INTEGRATION_STATE")
    if override:
        return Path(override).expanduser()
    return default_config_dir() / "integration.json"


def default_xdg_config_home() -> Path:
    return Path(
        os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    ).expanduser()


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _same_file_content(path: Path, text: str) -> bool:
    try:
        return path.read_text(encoding="utf-8") == text
    except OSError:
        return False


def _backup_file(path: Path, backup_root: Path, label: str) -> str:
    """Copy *path* into *backup_root* and return the backup path."""
    if not path.is_file():
        return ""
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / f"{label}-{_timestamp()}"
    counter = 1
    while destination.exists():
        destination = backup_root / f"{label}-{_timestamp()}-{counter}"
        counter += 1
    shutil.copy2(path, destination)
    return str(destination)


@dataclass
class InstallState:
    version: int = 1
    steamvr: str = ""
    driver: str = ""
    driver_backup: str = ""
    openxr_manifest: str = ""
    openxr_active_runtime: str = ""
    openxr_active_backup: str = ""
    monado_config: str = ""
    monado_config_backup: str = ""
    installed_at: str = ""

    @property
    def installed(self) -> bool:
        return bool(self.driver or self.openxr_active_runtime)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "InstallState":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(payload, dict):
            return cls()
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


# ---------------------------------------------------------------------------
# SteamVR discovery
# ---------------------------------------------------------------------------
def _steam_library_roots() -> list[Path]:
    """Read Steam library folders without depending on a full VDF parser."""
    candidates: list[Path] = []
    home = Path.home()
    if sys.platform.startswith("win"):
        roots = [
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
            / "Steam",
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Steam",
        ]
        try:
            import winreg
        except ImportError:  # pragma: no cover - only on Windows
            winreg = None
        if winreg is not None:
            for hive, subkey in (
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
            ):
                try:
                    with winreg.OpenKey(hive, subkey) as key:
                        value, _ = winreg.QueryValueEx(key, "SteamPath")
                        roots.append(Path(value))
                except OSError:
                    continue
        candidates.extend(roots)
    else:
        candidates.extend(
            [
                home / ".steam" / "steam",
                home / ".steam" / "root",
                home / ".local" / "share" / "Steam",
                home / ".var" / "app" / "com.valvesoftware.Steam" / "data" / "Steam",
            ]
        )

    libraries: list[Path] = []
    for root in candidates:
        if root not in libraries:
            libraries.append(root)
        for vdf in (
            root / "steamapps" / "libraryfolders.vdf",
            root / "config" / "libraryfolders.vdf",
        ):
            try:
                text = vdf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in re.finditer(r'"path"\s+"([^"]+)"', text):
                path = Path(match.group(1).replace("\\\\", "\\"))
                if path not in libraries:
                    libraries.append(path)
    return libraries


def find_steamvr(explicit: Optional[Path | str] = None) -> Optional[Path]:
    """Return a SteamVR root containing ``drivers`` and ``bin``."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for variable in ("OPENVRCARDBOARD_STEAMVR", "STEAMVR_HOME"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value).expanduser())
    for library in _steam_library_roots():
        candidates.append(library / "steamapps" / "common" / "SteamVR")

    checked: set[Path] = set()
    for candidate in candidates:
        for variant in (
            candidate,
            candidate / "SteamVR",
            candidate / "steamapps" / "common" / "SteamVR",
        ):
            resolved = variant.resolve() if variant.exists() else variant
            if resolved in checked:
                continue
            checked.add(resolved)
            if (variant / "drivers").is_dir() and (variant / "bin").is_dir():
                return variant
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def validate_driver_source(source: Path) -> None:
    manifest = source / "driver.vrdrivermanifest"
    binary = source / "bin" / platform_dir() / library_name()
    if not manifest.is_file():
        raise IntegrationError(f"missing {manifest}; build the driver first")
    if not binary.is_file():
        raise IntegrationError(f"missing {binary}; build the driver first")


def _is_our_driver(path: Path) -> bool:
    return (path / "driver.vrdrivermanifest").is_file()


def _install_driver(
    source: Path,
    steamvr: Path,
    state: InstallState,
    backup_root: Path,
    force: bool,
) -> tuple[Path, str]:
    validate_driver_source(source)
    drivers_dir = steamvr / "drivers"
    destination = drivers_dir / DRIVER_NAME
    drivers_dir.mkdir(parents=True, exist_ok=True)
    backup = state.driver_backup

    if destination.exists():
        if not _is_our_driver(destination) and not force:
            raise IntegrationError(
                f"{destination} exists but does not look like a driver; "
                "use --force to replace it"
            )
        if not backup:
            backup_root.mkdir(parents=True, exist_ok=True)
            candidate = backup_root / f"driver-{_timestamp()}"
            counter = 1
            while candidate.exists():
                candidate = backup_root / f"driver-{_timestamp()}-{counter}"
                counter += 1
            shutil.move(str(destination), str(candidate))
            backup = str(candidate)
        else:
            # The original driver is already safe in the backup. This is our
            # previous installation, so replace it in place.
            shutil.rmtree(destination)

    staging = drivers_dir / f".{DRIVER_NAME}-staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(source, staging)
        staging.rename(destination)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if backup and Path(backup).exists() and not destination.exists():
            shutil.move(backup, destination)
        raise
    return destination, backup


def _uninstall_driver(
    state: InstallState, steamvr: Optional[Path], force: bool
) -> None:
    destination = Path(state.driver) if state.driver else None
    if destination is None and steamvr is not None:
        destination = steamvr / "drivers" / DRIVER_NAME
    if destination is not None and destination.exists():
        if not _is_our_driver(destination) and not force:
            raise IntegrationError(
                f"{destination} no longer looks like an OpenVR Cardboard driver; "
                "use --force to remove it"
            )
        shutil.rmtree(destination)
    backup = Path(state.driver_backup) if state.driver_backup else None
    if backup is not None and backup.exists():
        if destination is None:
            raise IntegrationError("cannot restore the driver backup without a destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(backup), str(destination))
        backup_root = backup.parent
        try:
            backup_root.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OpenXR
# ---------------------------------------------------------------------------
def _pick_openxr_library(prefix: Path) -> Path:
    candidates = [
        prefix / "lib" / "libopenxr_monado.so",
        prefix / "bin" / "libopenxr_monado.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise IntegrationError(
        f"no Monado OpenXR runtime library under {prefix}; "
        "build it with scripts/build-openxr.sh"
    )


def _pick_monado_library(prefix: Path) -> Optional[Path]:
    for name in ("libmonado.so.25", "libmonado.so"):
        candidate = prefix / "lib" / name
        if candidate.is_file():
            return candidate
    matches = sorted((prefix / "lib").glob("libmonado.so.*"))
    return matches[-1] if matches else None


def _runtime_manifest(prefix: Path) -> str:
    runtime = {
        "name": "OpenVR Cardboard Monado",
        "library_path": str(_pick_openxr_library(prefix).resolve()),
    }
    monado = _pick_monado_library(prefix)
    if monado is not None:
        runtime["MND_libmonado_path"] = str(monado.resolve())
    return (
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "runtime": runtime,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _monado_config(
    view_width: int = DEFAULT_OPENXR_VIEW_WIDTH,
    view_height: int = DEFAULT_OPENXR_VIEW_HEIGHT,
    remote_port: int = DEFAULT_OPENXR_REMOTE_PORT,
    existing_text: str = "",
) -> str:
    data: dict = {}
    if existing_text.strip():
        try:
            loaded = json.loads(existing_text)
            if isinstance(loaded, dict):
                data = loaded
        except ValueError:
            data = {}
    data["active"] = "remote"
    remote = data.get("remote")
    if not isinstance(remote, dict):
        remote = {}
    remote.update(
        {
            "version": 0,
            "port": int(remote_port),
            "view_count": 2,
            "w_pixels": int(view_width),
            "h_pixels": int(view_height),
            "w_meters": 0.13,
            "h_meters": 0.07,
        }
    )
    data["remote"] = remote
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _install_openxr(
    prefix: Path,
    config_home: Path,
    state: InstallState,
    backup_root: Path,
    view_width: int,
    view_height: int,
    remote_port: int,
) -> tuple[str, str, str, str, str]:
    manifest_text = _runtime_manifest(prefix)
    manifest_path = default_config_dir() / "openxr" / OPENXR_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    active = config_home / "openxr" / "1" / "active_runtime.json"
    active.parent.mkdir(parents=True, exist_ok=True)
    active_backup = state.openxr_active_backup
    if active.is_file() and not _same_file_content(active, manifest_text):
        if not active_backup:
            active_backup = _backup_file(
                active, backup_root, "openxr-active-runtime"
            )
    active.write_text(manifest_text, encoding="utf-8")

    monado = config_home / "monado" / "config_v0.json"
    monado.parent.mkdir(parents=True, exist_ok=True)
    existing_text = ""
    if monado.is_file():
        try:
            existing_text = monado.read_text(encoding="utf-8")
        except OSError:
            existing_text = ""
    monado_backup = state.monado_config_backup
    monado_text = _monado_config(
        view_width=view_width,
        view_height=view_height,
        remote_port=remote_port,
        existing_text=existing_text,
    )
    if monado.is_file() and not _same_file_content(monado, monado_text):
        if not monado_backup:
            monado_backup = _backup_file(monado, backup_root, "monado-config")
    monado.write_text(monado_text, encoding="utf-8")
    return (
        str(manifest_path),
        str(active),
        active_backup,
        str(monado),
        monado_backup,
    )


def _restore_or_remove(path: Path, backup: str) -> None:
    backup_path = Path(backup) if backup else None
    if backup_path is not None and backup_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, path)
        backup_path.unlink()
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _uninstall_openxr(state: InstallState) -> None:
    if state.openxr_active_runtime:
        _restore_or_remove(
            Path(state.openxr_active_runtime), state.openxr_active_backup
        )
    if state.monado_config:
        _restore_or_remove(Path(state.monado_config), state.monado_config_backup)
    if state.openxr_manifest:
        try:
            Path(state.openxr_manifest).unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def install(
    driver_source: Optional[Path] = None,
    steamvr: Optional[Path | str] = None,
    openxr_prefix: Optional[Path] = None,
    register_openxr: bool = True,
    install_driver: bool = True,
    force: bool = False,
    state_path: Optional[Path] = None,
    config_home: Optional[Path] = None,
    view_width: int = DEFAULT_OPENXR_VIEW_WIDTH,
    view_height: int = DEFAULT_OPENXR_VIEW_HEIGHT,
    remote_port: int = DEFAULT_OPENXR_REMOTE_PORT,
) -> InstallState:
    """Install the driver and/or OpenXR runtime, recording every change."""
    state_file = Path(state_path).expanduser() if state_path else default_state_path()
    state = InstallState.load(state_file)
    backup_root = state_file.parent / "backups"

    steamvr_root: Optional[Path] = None
    if install_driver:
        if driver_source is None:
            raise IntegrationError("a driver source folder is required")
        steamvr_root = find_steamvr(steamvr)
        if steamvr_root is None:
            raise IntegrationError(
                "could not find SteamVR; pass --steamvr PATH or --no-driver"
            )
        destination, backup = _install_driver(
            Path(driver_source),
            steamvr_root,
            state,
            steamvr_root / BACKUP_DIRNAME,
            force,
        )
        state.steamvr = str(steamvr_root)
        state.driver = str(destination)
        state.driver_backup = backup

    if register_openxr:
        if openxr_prefix is None:
            raise IntegrationError("an OpenXR runtime prefix is required")
        prefix = Path(openxr_prefix)
        if not prefix.is_dir():
            raise IntegrationError(f"OpenXR runtime folder does not exist: {prefix}")
        (
            state.openxr_manifest,
            state.openxr_active_runtime,
            state.openxr_active_backup,
            state.monado_config,
            state.monado_config_backup,
        ) = _install_openxr(
            prefix,
            Path(config_home).expanduser() if config_home else default_xdg_config_home(),
            state,
            backup_root,
            view_width,
            view_height,
            remote_port,
        )

    state.installed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state.save(state_file)
    return state


def uninstall(
    state_path: Optional[Path] = None,
    force: bool = False,
) -> InstallState:
    """Undo everything :func:`install` changed and remove the state file."""
    state_file = Path(state_path).expanduser() if state_path else default_state_path()
    state = InstallState.load(state_file)
    if not state.installed:
        return state
    steamvr = Path(state.steamvr) if state.steamvr else None
    _uninstall_driver(state, steamvr, force)
    _uninstall_openxr(state)
    try:
        state_file.unlink()
    except FileNotFoundError:
        pass
    backup_root = state_file.parent / "backups"
    try:
        backup_root.rmdir()
    except OSError:
        pass
    return state


def status(state_path: Optional[Path] = None) -> InstallState:
    path = Path(state_path).expanduser() if state_path else default_state_path()
    return InstallState.load(path)


def bundled_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Return ``(driver_source, openxr_prefix)`` from a portable build."""
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parents[2]
    driver_candidates = [
        root / "driver",
        root / "driver" / "build" / "openvrcardboard",
    ]
    openxr_candidates = [
        root / "openxr",
        root / "out" / "openxr",
    ]
    driver = next(
        (path for path in driver_candidates if (path / "driver.vrdrivermanifest").is_file()),
        None,
    )
    openxr = next(
        (
            path
            for path in openxr_candidates
            if (path / "lib" / "libopenxr_monado.so").is_file()
        ),
        None,
    )
    return driver, openxr
