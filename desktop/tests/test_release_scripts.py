"""Unit coverage for the release asset and SBOM generators."""

from __future__ import annotations

import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_release_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_release = _load("prepare_release")
generate_sbom = _load("generate_sbom")


def _fixture_assets(tmp_path: Path) -> Path:
    assets = tmp_path / "build-assets"

    def write(relative: str, content: bytes = b"data") -> None:
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    write("package-windows/OpenVR-Cardboard-1.2.3-Setup.exe")
    write("package-windows/openvrcardboard.zip")
    write("package-linux/openvrcardboard.tar.gz")
    write("package-linux/openvrcardboard-1.2.3-linux-x86_64.AppImage")
    write("desktop-package/openvrcardboard-1.2.3-py3-none-any.whl")
    write("desktop-package/openvrcardboard-1.2.3.tar.gz")
    write("demo-windows/openxr-simple-example-windows-x64.zip")
    # upload-artifact strips the uploaded directory, so CI driver artifacts
    # contain these files at their root rather than under openvrcardboard/.
    write("driver-linux-x64/driver.vrdrivermanifest")
    write("driver-windows-x64/driver.vrdrivermanifest")
    return assets


def test_prepare_release_flattens_names_and_archives_drivers(tmp_path):
    assets = _fixture_assets(tmp_path)
    apk = tmp_path / "app-release.apk"
    apk.write_bytes(b"apk")
    output = tmp_path / "release-upload"

    paths = prepare_release.collect(assets, apk, output, "1.2.3")
    names = {path.name for path in paths}

    assert len(paths) == 10
    assert "OpenVR-Cardboard-1.2.3-windows-x64-Setup.exe" in names
    assert "OpenVR-Cardboard-1.2.3-linux-x86_64.AppImage" in names
    assert "OpenVR-Cardboard-1.2.3-android.apk" in names
    assert "openvrcardboard-1.2.3-py3-none-any.whl" in names
    assert "openvrcardboard-steamvr-driver-1.2.3-linux-x64.tar.gz" in names
    assert "openvrcardboard-steamvr-driver-1.2.3-windows-x64.zip" in names
    assert all(path.parent == output for path in paths)

    with tarfile.open(output / "openvrcardboard-steamvr-driver-1.2.3-linux-x64.tar.gz") as archive:
        assert "openvrcardboard/driver.vrdrivermanifest" in archive.getnames()
    with zipfile.ZipFile(output / "openvrcardboard-steamvr-driver-1.2.3-windows-x64.zip") as archive:
        assert "openvrcardboard/driver.vrdrivermanifest" in archive.namelist()


def test_prepare_release_reports_missing_windows_installer(tmp_path):
    assets = _fixture_assets(tmp_path)
    (assets / "package-windows" / "OpenVR-Cardboard-1.2.3-Setup.exe").unlink()
    with pytest.raises(prepare_release.ReleaseError, match="Windows installer"):
        prepare_release.collect(assets, None, tmp_path / "out", "1.2.3")


def test_prepare_release_accepts_nested_driver_artifact(tmp_path):
    assets = _fixture_assets(tmp_path)
    nested = assets / "driver-linux-x64" / "openvrcardboard"
    manifest = assets / "driver-linux-x64" / "driver.vrdrivermanifest"
    nested.mkdir()
    manifest.rename(nested / manifest.name)

    output = tmp_path / "out"
    prepare_release.collect(assets, None, output, "1.2.3")
    with tarfile.open(
        output / "openvrcardboard-steamvr-driver-1.2.3-linux-x64.tar.gz"
    ) as archive:
        assert "openvrcardboard/driver.vrdrivermanifest" in archive.getnames()


def test_prepare_release_reports_missing_driver(tmp_path):
    assets = _fixture_assets(tmp_path)
    (assets / "driver-linux-x64" / "driver.vrdrivermanifest").unlink()
    with pytest.raises(prepare_release.ReleaseError, match="SteamVR driver"):
        prepare_release.collect(assets, None, tmp_path / "out", "1.2.3")


def test_prepare_release_main_reports_errors(tmp_path, capsys):
    assert prepare_release.main(
        [
            "--build-assets",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
            "--version",
            "1.2.3",
        ]
    ) == 2
    assert "error:" in capsys.readouterr().err


def test_sbom_components_are_sorted_and_have_purls(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """
version = 1
[[package]]
name = "zeta"
version = "2.0"

[[package]]
name = "alpha_pkg"
version = "1.0"
"""
    )
    components = generate_sbom.components_from_lock(lock)
    assert [item["name"] for item in components] == ["alpha_pkg", "zeta"]
    assert components[0]["purl"] == "pkg:pypi/alpha-pkg@1.0"


def test_sbom_is_deterministic(tmp_path):
    components = [{"type": "library", "name": "a", "version": "1", "purl": "pkg:pypi/a@1"}]
    first = generate_sbom.build_sbom(components, "1.2.3")
    second = generate_sbom.build_sbom(components, "1.2.3")
    assert first == second
    assert first["serialNumber"].startswith("urn:uuid:")
    assert first["metadata"]["component"]["version"] == "1.2.3"


def test_sbom_main_writes_json_and_rejects_a_missing_lock(tmp_path, capsys):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """
version = 1
[[package]]
name = "numpy"
version = "2.0"
"""
    )
    output = tmp_path / "sbom.json"
    assert generate_sbom.main(
        ["--lock", str(lock), "--version", "1.2.3", "--output", str(output)]
    ) == 0
    payload = json.loads(output.read_text())
    assert payload["bomFormat"] == "CycloneDX"
    assert payload["components"][0]["name"] == "numpy"
    assert generate_sbom.main(
        [
            "--lock",
            str(tmp_path / "missing.lock"),
            "--version",
            "1.2.3",
            "--output",
            str(output),
        ]
    ) == 2
    assert "could not read" in capsys.readouterr().err
