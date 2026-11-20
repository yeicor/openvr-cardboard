#!/usr/bin/env python3
"""Collect CI artifacts into flat, versioned release asset names.

The consolidated CI artifact keeps the original job layout. GitHub Releases
want unique basenames, so this script copies the user-facing files into a flat
``release-upload`` directory and packs the standalone SteamVR drivers.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional


class ReleaseError(RuntimeError):
    """Raised when an expected build artifact is missing."""


def _first(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _require(source: Optional[Path], description: str) -> Path:
    if source is None or not source.exists():
        raise ReleaseError(f"{description} not found")
    return source


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _archive_directory(source: Path, target: Path, arcname: str) -> Path:
    if target.suffix == ".zip":
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(arcname) / path.relative_to(source))
    else:
        with tarfile.open(target, "w:gz") as archive:
            archive.add(source, arcname=arcname)
    return target


def _driver_directory(build_assets: Path, platform: str) -> Path:
    root = build_assets / f"driver-{platform}"
    nested = root / "openvrcardboard"
    # upload-artifact strips a single uploaded directory, so a CI artifact
    # normally has the driver files at its root. Accept the nested layout too.
    source = nested if nested.is_dir() else root
    if not (source / "driver.vrdrivermanifest").is_file():
        raise ReleaseError(f"SteamVR driver folder not found under {root}")
    return source


def collect(
    build_assets: Path,
    android_apk: Optional[Path],
    output: Path,
    version: str,
) -> list[Path]:
    """Copy release files into ``output`` and return them."""
    build_assets = Path(build_assets)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    windows = build_assets / "package-windows"
    linux = build_assets / "package-linux"
    python = build_assets / "desktop-package"
    demo = build_assets / "demo-windows"

    windows_setup = _require(
        _first(windows, "OpenVR-Cardboard-*-Setup.exe"),
        f"Windows installer in {windows}",
    )
    outputs.append(
        _copy(
            windows_setup,
            output / f"OpenVR-Cardboard-{version}-windows-x64-Setup.exe",
        )
    )
    outputs.append(
        _copy(
            _require(windows / "openvrcardboard.zip", "portable Windows package"),
            output / f"OpenVR-Cardboard-{version}-windows-x64-portable.zip",
        )
    )

    outputs.append(
        _copy(
            _require(linux / "openvrcardboard.tar.gz", "portable Linux package"),
            output / f"OpenVR-Cardboard-{version}-linux-x86_64.tar.gz",
        )
    )
    appimage = _require(
        _first(linux, "*.AppImage"), f"Linux AppImage in {linux}"
    )
    outputs.append(
        _copy(
            appimage,
            output / f"OpenVR-Cardboard-{version}-linux-x86_64.AppImage",
        )
    )

    wheel = _require(_first(python, "*.whl"), "Python wheel")
    sdist = _require(_first(python, "*.tar.gz"), "Python source distribution")
    outputs.append(
        _copy(wheel, output / f"openvrcardboard-{version}-py3-none-any.whl")
    )
    outputs.append(
        _copy(sdist, output / f"openvrcardboard-{version}.tar.gz")
    )

    demo_zip = _require(_first(demo, "*.zip"), "OpenXR demo archive")
    outputs.append(
        _copy(demo_zip, output / f"openxr-simple-example-{version}-windows-x64.zip")
    )

    if android_apk is not None:
        outputs.append(
            _copy(
                _require(android_apk, "signed Android APK"),
                output / f"OpenVR-Cardboard-{version}-android.apk",
            )
        )

    for platform, suffix in (("linux-x64", ".tar.gz"), ("windows-x64", ".zip")):
        driver = _driver_directory(build_assets, platform)
        target = output / f"openvrcardboard-steamvr-driver-{version}-{platform}{suffix}"
        _archive_directory(driver, target, "openvrcardboard")
        outputs.append(target)

    return outputs


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-assets", type=Path, required=True)
    parser.add_argument("--android-apk", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        paths = collect(args.build_assets, args.android_apk, args.output, args.version)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
