#!/usr/bin/env python3
"""Build a self-contained OpenVR Cardboard desktop folder.

The output contains the PyInstaller application with Qt, OpenCV, MediaPipe,
gamepad and video dependencies, the compiled SteamVR driver, and (on Linux) the
bundled Monado OpenXR runtime. ``openvrcardboard-install`` in the same folder
registers the driver and runtime; ``uninstall`` restores the previous files.

Usage:
    python scripts/package.py --out dist
    python scripts/package.py --out dist --appimage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"
DEFAULT_NAME = "openvrcardboard"


def _sys_path() -> None:
    if str(DESKTOP) not in sys.path:
        sys.path.insert(0, str(DESKTOP))


def _version() -> str:
    _sys_path()
    from openvrcardboard import __version__

    return __version__


def platform_tag() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64"
    if sys.platform.startswith("win"):
        return f"windows-{machine}"
    if sys.platform == "darwin":
        return f"macos-{machine}"
    return f"linux-{machine}"


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def build_driver() -> Path:
    """Build the native driver when no prebuilt artifact is present."""
    assembled = ROOT / "driver" / "build" / "openvrcardboard"
    manifest = assembled / "driver.vrdrivermanifest"
    if manifest.is_file():
        return assembled
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("cmake is required to build the SteamVR driver")
    configure = [
        cmake,
        "-S",
        str(ROOT / "driver"),
        "-B",
        str(ROOT / "driver" / "build"),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if not (ROOT / "driver" / "build" / "CMakeCache.txt").exists():
        if shutil.which("ninja") is not None:
            configure.extend(["-G", "Ninja"])
    _run(configure)
    _run(
        [
            cmake,
            "--build",
            str(ROOT / "driver" / "build"),
            "--config",
            "Release",
            "--parallel",
        ]
    )
    if not manifest.is_file():
        raise SystemExit(f"driver build did not produce {manifest}")
    return assembled


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pyinstaller(out: Path, name: str, debug: bool) -> Path:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        name,
        "--paths",
        str(DESKTOP),
        "--distpath",
        str(out),
        "--workpath",
        str(ROOT / "packaging" / "build"),
        "--specpath",
        str(ROOT / "packaging" / "build"),
    ]
    if debug:
        command.append("--debug=imports")
    if sys.platform.startswith("win"):
        # Keep the console bootloader: the same executable doubles as the
        # installer/status CLI, and CI smoke tests read --version/--self-test
        # output. A separate windowed build can come later if needed.
        icon = ROOT / "packaging" / "windows" / "openvrcardboard.ico"
        if icon.is_file():
            command.extend(["--icon", str(icon)])
    # Optional stacks are imported lazily; collect their data and binaries
    # explicitly so the portable build behaves like a full `[all]` install.
    # PySide6 is deliberately left to PyInstaller's Qt hooks: collecting the
    # whole namespace would drag in every Qt add-on (and its system libraries)
    # instead of only QtCore, QtGui and QtWidgets.
    import importlib.util

    for module in ("mediapipe", "cv2", "pygame", "mss", "imageio_ffmpeg"):
        if importlib.util.find_spec(module) is not None:
            command.extend(["--collect-all", module])
    command.append(str(ROOT / "packaging" / "entry.py"))
    _run(command)
    folder = out / name
    if not folder.is_dir():
        raise SystemExit(f"PyInstaller did not create {folder}")
    return folder


def _copy_tree(src: Path, destination: Path, ignore=()) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        src,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns(*ignore),
    )


def prepare_folder(
    folder: Path,
    name: str,
    driver: Path | None,
    openxr: Path | None,
    ffmpeg: Path | None = None,
) -> Path:
    executable = folder / (name + (".exe" if sys.platform.startswith("win") else ""))
    if not executable.is_file():
        raise SystemExit(f"portable executable is missing: {executable}")

    if driver is not None:
        _copy_tree(driver, folder / "driver")
    if openxr is not None:
        _copy_tree(
            openxr,
            folder / "openxr",
            ignore=("src", "build", "include", ".git"),
        )
    if ffmpeg is not None:
        # NVENC HEVC/AV1 and CPU H.264 are optional: without a bundled ffmpeg
        # the app uses imageio-ffmpeg or a system ffmpeg and falls back to JPEG
        # when neither exists. Ship a static build here so the portable folder
        # is self-contained.
        ffmpeg_name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
        shutil.copy2(ffmpeg, folder / ffmpeg_name)
        try:
            (folder / ffmpeg_name).chmod(0o755)
        except OSError:
            pass

    installer = folder / (
        name + "-install" + (".exe" if sys.platform.startswith("win") else "")
    )
    shutil.copy2(executable, installer)
    try:
        installer.chmod(0o755)
    except OSError:
        pass

    # Launch other OpenXR applications with the capture shim preloaded. The
    # desktop app starts the service and receives the per-eye JPEG stream.
    if openxr is not None and sys.platform.startswith("linux"):
        launcher = folder / f"{name}-openxr"
        launcher.write_text(
            "#!/bin/sh\n"
            "# Run an OpenXR application with the OpenVR Cardboard capture shim.\n"
            'HERE="$(dirname "$(readlink -f "$0")")"\n'
            'SHIM="$HERE/openxr/lib/libovrc_xr_capture.so"\n'
            'if [ ! -f "$SHIM" ]; then\n'
            '  echo "error: the bundled OpenXR capture shim is missing" >&2\n'
            "  exit 2\n"
            "fi\n"
            'if [ "$#" -eq 0 ]; then\n'
            '  echo "usage: $(basename "$0") /path/to/openxr-application [args...]" >&2\n'
            "  exit 2\n"
            "fi\n"
            'export LD_PRELOAD="$SHIM${LD_PRELOAD:+:$LD_PRELOAD}"\n'
            ': "${XR_RUNTIME_JSON:=$HERE/openxr/share/openxr/1/openxr_monado.json}"\n'
            'export XR_RUNTIME_JSON\n'
            ': "${OVRC_CAPTURE_HOST:=127.0.0.1}"\n'
            ': "${OVRC_CAPTURE_PORT:=39102}"\n'
            ': "${OVRC_CAPTURE_FPS:=90}"\n'
            ': "${OVRC_CAPTURE_QUALITY:=65}"\n'
            ': "${OVRC_CAPTURE_EYE_WIDTH:=1200}"\n'
            ': "${OVRC_CAPTURE_EYE_HEIGHT:=1080}"\n'
            "export OVRC_CAPTURE_HOST OVRC_CAPTURE_PORT OVRC_CAPTURE_FPS\n"
            "export OVRC_CAPTURE_QUALITY OVRC_CAPTURE_EYE_WIDTH OVRC_CAPTURE_EYE_HEIGHT\n"
            'exec "$@"\n',
            encoding="utf-8",
        )
        try:
            launcher.chmod(0o755)
        except OSError:
            pass

        # The reference OpenXR demo is part of the bundled runtime. A tiny
        # wrapper makes it usable like a normal packaged application without
        # asking users to locate the runtime JSON or set LD_LIBRARY_PATH.
        demo = folder / f"{name}-demo"
        demo.write_text(
            "#!/bin/sh\n"
            "# Run the bundled openxr-simple-example on the bundled Monado runtime.\n"
            'HERE="$(dirname "$(readlink -f "$0")")"\n'
            'DEMO="$HERE/openxr/bin/openxr-example"\n'
            'if [ ! -x "$DEMO" ]; then\n'
            '  echo "error: the bundled OpenXR demo is missing" >&2\n'
            "  exit 2\n"
            "fi\n"
            'export LD_LIBRARY_PATH="$HERE/openxr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            ': "${XR_RUNTIME_JSON:=$HERE/openxr/share/openxr/1/openxr_monado.json}"\n'
            "export XR_RUNTIME_JSON\n"
            'exec "$DEMO" "$@"\n',
            encoding="utf-8",
        )
        try:
            demo.chmod(0o755)
        except OSError:
            pass

    for document in ("README.md", "LICENSE"):
        source = ROOT / document
        if source.is_file():
            shutil.copy2(source, folder / document)

    info = [
        f"OpenVR Cardboard {_version()}",
        f"platform: {platform_tag()}",
        f"python: {platform.python_version()}",
        f"built: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"driver: {driver if driver else 'not bundled'}",
        f"openxr: {openxr if openxr else 'not bundled'}",
        f"ffmpeg: {ffmpeg if ffmpeg else 'system PATH / JPEG fallback'}",
    ]
    if driver is not None:
        manifest = driver / "driver.vrdrivermanifest"
        if manifest.is_file():
            info.append(f"driver manifest sha256: {_sha256(manifest)}")
    if openxr is not None:
        runtime = openxr / "lib" / "libopenxr_monado.so"
        if runtime.is_file():
            info.append(f"openxr runtime sha256: {_sha256(runtime)}")
    (folder / "BUILD-INFO.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    return folder


def generate_markers(folder: Path) -> bool:
    """Ship a printable marker sheet so packaged builds are self-contained.

    The generator only needs OpenCV and Pillow, both installed by the packaging
    environments. A failure must not stop the release build: the sheet can also
    be generated from a source checkout.
    """
    output = folder / "markers"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_markers.py"),
        "--output",
        str(output),
    ]
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"warning: could not generate printable markers: {error}", file=sys.stderr)
        return False
    return (output / "boards.html").is_file()


def make_archive(folder: Path, out: Path, kind: str) -> Path | None:
    if kind == "none":
        return None
    if kind == "auto":
        kind = "zip" if sys.platform.startswith("win") else "tar"
    if kind == "tar":
        target = out / f"{folder.name}.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            archive.add(folder, arcname=folder.name)
        return target
    target = out / f"{folder.name}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            archive.write(path, path.relative_to(folder.parent))
    return target


def make_appimage(folder: Path, out: Path, name: str) -> Path | None:
    if not sys.platform.startswith("linux"):
        return None
    tool = os.environ.get("APPIMAGETOOL") or shutil.which("appimagetool")
    if not tool:
        print(
            "appimagetool was not found; skipping AppImage "
            "(set APPIMAGETOOL or install appimagetool)",
            file=sys.stderr,
        )
        return None
    appdir = out / f"{name}.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    payload = appdir / "usr" / "lib" / name
    payload.parent.mkdir(parents=True)
    shutil.copytree(folder, payload)
    (appdir / "usr" / "bin").mkdir(parents=True)
    (appdir / "usr" / "bin" / name).write_text(
        "#!/bin/sh\n"
        'HERE="$(dirname "$(readlink -f "$0")")"\n'
        f'exec "$HERE/../lib/{name}/{name}" "$@"\n',
        encoding="utf-8",
    )
    (appdir / "usr" / "bin" / name).chmod(0o755)
    (appdir / "AppRun").write_text(
        "#!/bin/sh\n"
        'HERE="$(dirname "$(readlink -f "$0")")"\n'
        "case \"${1:-}\" in\n"
        "  install|uninstall|status)\n"
        f'    exec "$HERE/usr/lib/{name}/{name}-install" "$@"\n'
        "    ;;\n"
        "  openxr)\n"
        "    shift\n"
        f'    exec "$HERE/usr/lib/{name}/{name}-openxr" "$@"\n'
        "    ;;\n"
        "  demo)\n"
        "    shift\n"
        f'    exec "$HERE/usr/lib/{name}/{name}-demo" "$@"\n'
        "    ;;\n"
        "esac\n"
        f'exec "$HERE/usr/lib/{name}/{name}" "$@"\n',
        encoding="utf-8",
    )
    (appdir / "AppRun").chmod(0o755)
    icon = ROOT / "driver" / "resources" / "icons" / "controller.svg"
    icon_target = appdir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"
    icon_target.mkdir(parents=True)
    if icon.is_file():
        shutil.copy2(icon, icon_target / f"{name}.svg")
    # AppImage tools require a top-level PNG named after Icon= in the desktop
    # entry; they generate .DirIcon from it. The scalable SVG alone produces a
    # warning and will be rejected by future appimagetool releases.
    png_icon = ROOT / "driver" / "resources" / "icons" / "controller.png"
    if png_icon.is_file():
        shutil.copy2(png_icon, appdir / f"{name}.png")
        png_target = (
            appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        )
        png_target.mkdir(parents=True)
        shutil.copy2(png_icon, png_target / f"{name}.png")
    desktop = appdir / f"{name}.desktop"
    desktop.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name=OpenVR Cardboard\n"
        f"Exec={name}\n"
        f"Icon={name}\n"
        "Categories=Utility;\n"
        "Terminal=false\n",
        encoding="utf-8",
    )
    target = out / f"{name}-{_version()}-{platform_tag()}.AppImage"
    command = [tool]
    if str(tool).endswith(".AppImage"):
        command.append("--appimage-extract-and-run")
    command.extend(["--no-appstream", str(appdir), str(target)])
    env = os.environ.copy()
    env.setdefault("ARCH", "x86_64" if platform.machine().lower() in ("x86_64", "amd64") else platform.machine())
    env.setdefault("VERSION", _version())
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=env)
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"AppImage creation failed: {error}", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(appdir, ignore_errors=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--driver", type=Path, help="prebuilt driver folder")
    parser.add_argument("--openxr", type=Path, help="OpenXR runtime prefix")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="static ffmpeg binary to bundle for NVENC HEVC/AV1 and CPU H.264 streaming",
    )
    parser.add_argument("--skip-driver", action="store_true")
    parser.add_argument("--skip-openxr", action="store_true")
    parser.add_argument("--no-build-driver", action="store_true")
    parser.add_argument("--debug", action="store_true", help="PyInstaller import trace")
    parser.add_argument(
        "--archive",
        choices=("auto", "tar", "zip", "none"),
        default="auto",
        help="archive format for the portable folder",
    )
    parser.add_argument(
        "--appimage",
        action="store_true",
        help="also build an AppImage when appimagetool is available",
    )
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    driver: Path | None = None
    if not args.skip_driver:
        if args.driver:
            driver = args.driver.expanduser().resolve()
        elif not args.no_build_driver:
            driver = build_driver()
        else:
            candidate = ROOT / "driver" / "build" / "openvrcardboard"
            driver = candidate if candidate.is_dir() else None
        if driver is None or not (driver / "driver.vrdrivermanifest").is_file():
            raise SystemExit("driver folder is missing its driver.vrdrivermanifest")

    openxr: Path | None = None
    if not args.skip_openxr and sys.platform.startswith("linux"):
        openxr = (
            args.openxr.expanduser().resolve()
            if args.openxr
            else ROOT / "out" / "openxr"
        )
        if not (openxr / "lib" / "libopenxr_monado.so").is_file():
            if args.openxr:
                raise SystemExit(f"OpenXR runtime is incomplete: {openxr}")
            print(
                "OpenXR runtime not found; packaging without it "
                "(run scripts/build-openxr.sh to include it)",
                file=sys.stderr,
            )
            openxr = None

    folder = run_pyinstaller(out, args.name, args.debug)
    ffmpeg = args.ffmpeg.expanduser().resolve() if args.ffmpeg else None
    if ffmpeg is not None and (not ffmpeg.is_file() or not os.access(ffmpeg, os.X_OK)):
        raise SystemExit(f"ffmpeg is not an executable file: {ffmpeg}")
    prepare_folder(folder, args.name, driver, openxr, ffmpeg)
    generate_markers(folder)

    archive = make_archive(folder, out, args.archive)
    appimage = make_appimage(folder, out, args.name) if args.appimage else None

    print(
        json.dumps(
            {
                "folder": str(folder),
                "archive": str(archive) if archive else "",
                "appimage": str(appimage) if appimage else "",
                "driver": str(driver) if driver else "",
                "openxr": str(openxr) if openxr else "",
                "ffmpeg": str(ffmpeg) if ffmpeg else "",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
