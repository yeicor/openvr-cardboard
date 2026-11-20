import json
from pathlib import Path

import pytest

from openvrcardboard.integration import (
    InstallState,
    IntegrationError,
    install,
    library_name,
    platform_dir,
    status,
    uninstall,
)


def _make_driver(root: Path, version: str = "new") -> Path:
    driver = root / "driver"
    binary = driver / "bin" / platform_dir() / library_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(f"binary-{version}".encode())
    (driver / "driver.vrdrivermanifest").write_text(
        json.dumps({"name": "openvrcardboard", "version": version}),
        encoding="utf-8",
    )
    return driver


def _make_steamvr(root: Path) -> Path:
    steamvr = root / "SteamVR"
    (steamvr / "drivers").mkdir(parents=True)
    (steamvr / "bin").mkdir(parents=True)
    return steamvr


def _make_openxr(root: Path) -> Path:
    prefix = root / "openxr"
    (prefix / "lib").mkdir(parents=True, exist_ok=True)
    (prefix / "lib" / "libopenxr_monado.so").write_bytes(b"runtime")
    (prefix / "lib" / "libmonado.so.25").write_bytes(b"monado")
    return prefix


def test_install_and_uninstall_restore_every_file(tmp_path):
    steamvr = _make_steamvr(tmp_path)
    previous_driver = steamvr / "drivers" / "openvrcardboard"
    previous_driver.mkdir()
    (previous_driver / "driver.vrdrivermanifest").write_text(
        '{"name": "previous"}', encoding="utf-8"
    )

    config_home = tmp_path / "config"
    active = config_home / "openxr" / "1" / "active_runtime.json"
    active.parent.mkdir(parents=True)
    active.write_text('{"runtime": "previous"}', encoding="utf-8")
    monado = config_home / "monado" / "config_v0.json"
    monado.parent.mkdir(parents=True)
    monado.write_text('{"active": "other", "keep": true}', encoding="utf-8")

    state_path = tmp_path / "app" / "integration.json"
    state = install(
        driver_source=_make_driver(tmp_path),
        steamvr=steamvr,
        openxr_prefix=_make_openxr(tmp_path),
        state_path=state_path,
        config_home=config_home,
        remote_port=4444,
        view_width=2000,
        view_height=1000,
    )

    assert state.driver == str(steamvr / "drivers" / "openvrcardboard")
    assert Path(state.driver, "bin", platform_dir(), library_name()).read_bytes() == b"binary-new"
    assert Path(state.driver_backup).parent == steamvr / ".openvrcardboard-backups"
    assert Path(state.driver_backup).is_dir()
    assert status(state_path).installed
    runtime = json.loads(active.read_text(encoding="utf-8"))
    assert runtime["runtime"]["name"] == "OpenVR Cardboard Monado"
    assert runtime["runtime"]["library_path"].endswith("libopenxr_monado.so")
    monado_payload = json.loads(monado.read_text(encoding="utf-8"))
    assert monado_payload["active"] == "remote"
    assert monado_payload["keep"] is True
    assert monado_payload["remote"]["port"] == 4444
    assert monado_payload["remote"]["w_pixels"] == 2000

    # A second install must keep the original backups instead of treating our
    # own files as user data.
    state = install(
        driver_source=_make_driver(tmp_path, version="newer"),
        steamvr=steamvr,
        openxr_prefix=_make_openxr(tmp_path),
        state_path=state_path,
        config_home=config_home,
        remote_port=4444,
        view_width=2000,
        view_height=1000,
    )
    assert Path(state.driver_backup).is_dir()

    uninstall(state_path=state_path)
    assert not state_path.exists()
    assert previous_driver.is_dir()
    assert not (
        previous_driver / "bin" / platform_dir() / library_name()
    ).exists()
    assert (previous_driver / "driver.vrdrivermanifest").read_text() == '{"name": "previous"}'
    assert active.read_text(encoding="utf-8") == '{"runtime": "previous"}'
    assert monado.read_text(encoding="utf-8") == '{"active": "other", "keep": true}'
    assert not (state_path.parent / "backups").exists()
    assert not (steamvr / ".openvrcardboard-backups").exists()


def test_install_rejects_foreign_driver_without_force(tmp_path):
    steamvr = _make_steamvr(tmp_path)
    foreign = steamvr / "drivers" / "openvrcardboard"
    foreign.mkdir()
    (foreign / "important.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(IntegrationError):
        install(
            driver_source=_make_driver(tmp_path),
            steamvr=steamvr,
            register_openxr=False,
            state_path=tmp_path / "state.json",
        )
    assert (foreign / "important.txt").read_text(encoding="utf-8") == "keep me"


def test_state_roundtrip_ignores_unknown_keys(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"driver": "/tmp/driver", "future": 1}), encoding="utf-8"
    )
    state = InstallState.load(path)
    assert state.driver == "/tmp/driver"
    state.save(path)
    assert json.loads(path.read_text())["driver"] == "/tmp/driver"
