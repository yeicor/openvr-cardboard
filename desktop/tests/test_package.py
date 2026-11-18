"""Packaging helpers that do not need PyInstaller or appimagetool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _package_module():
    spec = importlib.util.spec_from_file_location(
        "openvrcardboard_package", ROOT / "scripts" / "package.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="AppImage creation is Linux-only"
)
def test_appimage_has_a_top_level_png_icon(tmp_path, monkeypatch):
    package = _package_module()
    folder = tmp_path / "openvrcardboard"
    folder.mkdir()
    executable = folder / "openvrcardboard"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    listing = tmp_path / "appdir-files.txt"
    tool = tmp_path / "appimagetool"
    tool.write_text(
        "#!/bin/sh\n"
        'find "$2" -maxdepth 1 -type f | sort > "$APPIMAGE_LISTING"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    monkeypatch.setenv("APPIMAGETOOL", str(tool))
    monkeypatch.setenv("APPIMAGE_LISTING", str(listing))

    target = package.make_appimage(folder, tmp_path / "out", "openvrcardboard")

    assert target is not None
    files = listing.read_text(encoding="utf-8").splitlines()
    assert any(path.endswith("/openvrcardboard.png") for path in files)
    assert any(path.endswith("/openvrcardboard.desktop") for path in files)
    assert any(path.endswith("/AppRun") for path in files)
