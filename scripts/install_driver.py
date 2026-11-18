#!/usr/bin/env python3
"""Build and install the OpenVR Cardboard SteamVR driver.

Typical use:

    python scripts/install_driver.py                 # build + auto-detect SteamVR
    python scripts/install_driver.py --steamvr PATH
    python scripts/install_driver.py --dest PATH --no-build

The script assembles the driver through CMake (the same folder a manual copy
would use), backs up an existing ``openvrcardboard`` driver, and installs the new
folder under ``<SteamVR>/drivers/openvrcardboard``. It never edits
``steamvr.vrsettings``; SteamVR discovers drivers by directory.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "driver" / "build"
ASSEMBLED = BUILD_DIR / "openvrcardboard"


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


def _steam_library_roots() -> list[Path]:
    """Read Steam library folders without depending on a VDF parser."""
    candidates: list[Path] = []
    home = Path.home()
    if sys.platform.startswith("win"):
        roots = [
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Steam",
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
        for root in roots:
            candidates.append(root)
    else:
        roots = [
            home / ".steam" / "steam",
            home / ".steam" / "root",
            home / ".local" / "share" / "Steam",
            home / ".var" / "app" / "com.valvesoftware.Steam" / "data" / "Steam",
        ]
        candidates.extend(roots)

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


def find_steamvr(explicit: str | None = None) -> Path | None:
    """Return a SteamVR root containing ``drivers`` and ``bin``."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("OPENVRCARDBOARD_STEAMVR") or os.environ.get("STEAMVR_HOME")
    if env:
        candidates.append(Path(env).expanduser())
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


def run_build() -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("cmake was not found on PATH; install CMake 3.16 or newer")
    configure = [
        cmake,
        "-S",
        str(ROOT / "driver"),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        if shutil.which("ninja") is not None:
            configure.extend(["-G", "Ninja"])
    subprocess.run(configure, check=True)
    subprocess.run(
        [cmake, "--build", str(BUILD_DIR), "--config", "Release", "--parallel"],
        check=True,
    )


def validate_source(source: Path) -> None:
    manifest = source / "driver.vrdrivermanifest"
    binary = source / "bin" / platform_dir() / library_name()
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}; build the driver first")
    if not binary.is_file():
        raise SystemExit(f"missing {binary}; build the driver first")


def install(source: Path, drivers_dir: Path, force: bool = False) -> Path:
    destination = drivers_dir / "openvrcardboard"
    drivers_dir.mkdir(parents=True, exist_ok=True)
    staging = drivers_dir / f".openvrcardboard-staging-{os.getpid()}"
    backup_root = drivers_dir.parent / ".openvrcardboard-backups"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)

    backup: Path | None = None
    if destination.exists():
        if not force and not (destination / "driver.vrdrivermanifest").is_file():
            shutil.rmtree(staging)
            raise SystemExit(
                f"{destination} exists but does not look like a SteamVR driver; "
                "use --force to replace it"
            )
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"openvrcardboard-{time.strftime('%Y%m%d-%H%M%S')}"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(destination), str(backup))

    try:
        staging.rename(destination)
    except OSError:
        if backup is not None and not destination.exists():
            shutil.move(str(backup), str(destination))
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if backup is not None:
        print(f"backed up the previous driver to {backup}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steamvr", metavar="PATH", help="SteamVR root directory")
    parser.add_argument(
        "--dest",
        metavar="PATH",
        help="install the driver folder here instead of <SteamVR>/drivers/openvrcardboard",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="use the existing driver/build/openvrcardboard folder",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing destination even if it is not a SteamVR driver",
    )
    parser.add_argument("--dry-run", action="store_true", help="print paths and exit")
    args = parser.parse_args(argv)

    if not args.no_build:
        run_build()
    elif not ASSEMBLED.is_dir():
        raise SystemExit(
            f"{ASSEMBLED} does not exist; run without --no-build to build it"
        )
    validate_source(ASSEMBLED)

    if args.dest:
        destination_parent = Path(args.dest).expanduser().resolve().parent
        destination = Path(args.dest).expanduser().resolve()
        if args.dry_run:
            print(f"would install {ASSEMBLED} -> {destination}")
            return 0
        drivers_dir = destination_parent
        # install() always appends openvrcardboard; use a temporary name-friendly path.
        if destination.name != "openvrcardboard":
            staging = destination_parent / f".openvrcardboard-copy-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(ASSEMBLED, staging)
            if destination.exists():
                if not args.force:
                    shutil.rmtree(staging)
                    raise SystemExit(f"{destination} exists; use --force to replace it")
                shutil.rmtree(destination)
            staging.rename(destination)
            print(f"installed driver to {destination}")
            return 0
        installed = install(ASSEMBLED, drivers_dir, force=args.force)
        print(f"installed driver to {installed}")
        return 0

    steamvr = find_steamvr(args.steamvr)
    if steamvr is None:
        raise SystemExit(
            "could not find SteamVR. Pass --steamvr /path/to/SteamVR or --dest PATH."
        )
    drivers_dir = steamvr / "drivers"
    if args.dry_run:
        print(f"SteamVR: {steamvr}")
        print(f"would install {ASSEMBLED} -> {drivers_dir / 'openvrcardboard'}")
        return 0
    installed = install(ASSEMBLED, drivers_dir, force=args.force)
    print(f"installed driver to {installed}")
    print("Restart SteamVR to load the driver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
