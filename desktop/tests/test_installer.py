"""Unit coverage for the command-line installer front end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openvrcardboard import installer
from openvrcardboard.integration import IntegrationError


@dataclass
class FakeState:
    installed: bool
    driver: str = ""
    driver_backup: str = ""
    openxr_active_runtime: str = ""


def test_parser_defaults_to_install():
    args = installer.build_parser().parse_args([])
    assert args.command == "install"
    assert not args.no_driver
    assert not args.no_openxr
    assert not args.force


def test_status_prints_a_short_message_when_not_installed(monkeypatch, capsys):
    monkeypatch.setattr(installer, "status", lambda path: FakeState(False))
    assert installer.main(["status"]) == 0
    assert "not installed" in capsys.readouterr().out


def test_status_prints_json_when_installed(monkeypatch, capsys):
    monkeypatch.setattr(
        installer, "status", lambda path: FakeState(True, driver="/steamvr/driver")
    )
    assert installer.main(["status", "--state", "/tmp/state.json"]) == 0
    output = capsys.readouterr().out
    assert '"installed": true' in output
    assert '"driver": "/steamvr/driver"' in output


def test_uninstall_is_a_noop_when_not_installed(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(installer, "status", lambda path: FakeState(False))
    monkeypatch.setattr(installer, "uninstall", lambda **kwargs: calls.append(kwargs))
    assert installer.main(["uninstall"]) == 0
    assert calls == []
    assert "not installed" in capsys.readouterr().out


def test_uninstall_removes_an_installed_integration(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(installer, "status", lambda path: FakeState(True))
    monkeypatch.setattr(installer, "uninstall", lambda **kwargs: calls.append(kwargs))
    assert installer.main(["uninstall", "--force", "--state", "/tmp/state.json"]) == 0
    assert calls == [
        {"state_path": Path("/tmp/state.json"), "force": True}
    ]
    assert "Removed" in capsys.readouterr().out


def test_install_skips_steamvr_when_it_is_not_installed(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_install(**kwargs):
        captured.update(kwargs)
        return FakeState(False, openxr_active_runtime="/home/me/.config/openxr/1")

    monkeypatch.setattr(installer, "_default_driver_source", lambda: Path("/bundle/driver"))
    monkeypatch.setattr(installer, "_default_openxr_prefix", lambda: Path("/bundle/openxr"))
    monkeypatch.setattr(installer, "find_steamvr", lambda path: None)
    monkeypatch.setattr(installer, "install", fake_install)

    assert installer.main(["install", "--config-home", str(tmp_path)]) == 0
    assert captured["install_driver"] is False
    assert captured["driver_source"] is None
    assert captured["register_openxr"] is True
    assert captured["config_home"] == tmp_path
    stderr = capsys.readouterr().err
    assert "SteamVR was not found" in stderr


def test_explicit_missing_steamvr_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr(installer, "_default_driver_source", lambda: Path("/bundle/driver"))
    monkeypatch.setattr(installer, "find_steamvr", lambda path: None)
    assert installer.main(["install", "--steamvr", "/missing"]) == 2
    assert "SteamVR was not found at /missing" in capsys.readouterr().err


def test_install_without_a_bundled_driver_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr(installer, "_default_driver_source", lambda: None)
    assert installer.main(["install", "--no-openxr"]) == 2
    assert "no bundled driver" in capsys.readouterr().err


def test_install_can_skip_both_integrations(monkeypatch, capsys):
    captured = {}

    def fake_install(**kwargs):
        captured.update(kwargs)
        return FakeState(False)

    monkeypatch.setattr(installer, "install", fake_install)
    assert installer.main(["install", "--no-driver", "--no-openxr"]) == 0
    assert captured["install_driver"] is False
    assert captured["register_openxr"] is False
    assert "No integration changes were made" in capsys.readouterr().out


def test_integration_errors_return_exit_code_two(monkeypatch, capsys):
    def failing_install(**kwargs):
        raise IntegrationError("disk is on fire")

    monkeypatch.setattr(installer, "install", failing_install)
    assert installer.main(["install", "--no-driver", "--no-openxr"]) == 2
    assert "error: disk is on fire" in capsys.readouterr().err


def test_missing_default_openxr_prefix_is_reported(monkeypatch, capsys):
    captured = {}

    def fake_install(**kwargs):
        captured.update(kwargs)
        return FakeState(False)

    monkeypatch.setattr(installer, "_default_openxr_prefix", lambda: None)
    monkeypatch.setattr(installer, "install", fake_install)
    assert installer.main(["install", "--no-driver"]) == 0
    assert captured["register_openxr"] is False
    assert "No bundled OpenXR runtime" in capsys.readouterr().err
