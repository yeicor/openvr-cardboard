"""Unit coverage for the Android emulator harness' evidence extraction."""

import json
import os

import pytest

from e2e.emulator import E2EConfig, EmulatorE2E


def _status(step: int) -> dict:
    return {
        "openxr": {
            "enabled": True,
            "capture": {
                "connected": True,
                "frames": 100 + step,
                "grab_left": 0.2 + step * 0.3,
                "grab_right": 0.4 + step * 0.3,
                "haptic_events": 3 + step,
            },
            "remote": {"connected": True, "packets": 500 + step},
        },
        "markers": {
            "0": {"valid": True, "position": [0.0, 1.6, 0.0]},
            "5": {"valid": True, "position": [-0.3, 1.2, -0.4]},
            "10": {"valid": True, "position": [0.3, 1.2, -0.4]},
            "100": {"valid": False, "position": [0.0, 1.0, 0.0]},
        },
        "body": {
            "waist": {"valid": True, "position": [0.0, 1.0, 0.0]},
        },
        "composed": {
            "hmd": {"valid": True, "position": [0.0 + step * 0.02, 1.6, 0.0]},
            "left": {"valid": True, "position": [-0.3, 1.2 - step * 0.01, -0.4]},
            "right": {"valid": True, "position": [0.3, 1.2, -0.4 + step * 0.03]},
        },
        "tracking": {
            "gamepads": {
                "haptics": {"count": 2 + step, "last": {"amplitude": 0.5}},
            }
        },
    }


def test_harness_extracts_openxr_evidence(tmp_path):
    harness = EmulatorE2E(E2EConfig(artifacts=tmp_path))
    for step in range(4):
        harness._record_status(_status(step))

    assert harness.openxr_connected
    assert harness.openxr_frames == 103
    assert harness.remote_connected
    assert harness.remote_packets == 503
    assert harness.markers_seen == {0, 5, 10}
    assert harness.body_seen
    assert harness.max_grab_left > 0.75
    assert harness.max_grab_right > 0.75
    assert harness.haptic_events == 6
    assert harness.gamepad_haptics == 5
    motion = harness.pose_motion()
    assert motion["hmd"] > 0.05
    assert motion["left"] > 0.02
    assert motion["right"] > 0.08


def test_harness_uses_recent_grab_events_when_snapshots_miss_a_pulse(tmp_path):
    harness = EmulatorE2E(E2EConfig(artifacts=tmp_path))
    payload = _status(0)
    payload["openxr"]["capture"]["grab_left"] = 0.0
    payload["openxr"]["capture"]["grab_right"] = 0.0
    payload["capture_events"] = [
        {"type": "grab", "hand": "left", "value": 1.0, "active": True},
        {"type": "grab", "hand": "right", "value": 0.8, "active": True},
        "malformed",
        {"type": "haptic", "hand": "left", "amplitude": 0.5},
    ]
    harness._record_status(payload)

    assert harness.max_grab_left == 1.0
    assert harness.max_grab_right == 0.8


def test_harness_config_uses_synthetic_openxr_stack(tmp_path):
    harness = EmulatorE2E(E2EConfig(artifacts=tmp_path))
    harness._write_config()
    payload = json.loads((tmp_path / "app-config.json").read_text())

    assert payload["camera"]["source"] == "synthetic"
    assert payload["tracking"]["hmd_source"] == "markers"
    assert payload["tracking"]["body_enabled"] is True
    assert payload["tracking"]["pose_max_age_s"] == 1.5
    assert payload["gamepad"]["synthetic"] is True
    assert payload["openxr"]["enabled"] is True
    assert payload["phone"]["capture_source"] == "openxr"
    assert payload["phone"]["layout"] == "sbs"
    assert (payload["phone"]["width"], payload["phone"]["height"]) == (600, 540)
    assert payload["phone"]["fps"] == 30
    assert payload["phone"]["codec"] == "h264"
    assert payload["phone"]["encoder"] == "x264"
    assert payload["openxr"]["manage_service"] is False
    assert payload["ui"]["start_tracking_on_launch"] is True
    assert harness.openxr_remote_port != harness.openxr_capture_port
    assert payload["openxr"]["remote_port"] == harness.openxr_remote_port
    assert payload["openxr"]["capture_port"] == harness.openxr_capture_port


def test_harness_helpers_format_and_clip_text(tmp_path):
    from e2e.emulator import EmulatorE2E, _mean, _utc_now

    assert _mean([]) == 0.0
    assert _mean([1.0, 2.0]) == 1.5
    assert _utc_now().endswith("+00:00")

    path = tmp_path / "log.txt"
    path.write_text("one\ntwo\nthree\n")
    assert EmulatorE2E._read_text(path) == "one\ntwo\nthree\n"
    assert EmulatorE2E._tail(path, lines=2) == "two\nthree"
    assert EmulatorE2E._read_text(tmp_path / "missing") == ""


def test_harness_detects_display_and_ports(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from e2e.emulator import _display_ready, _free_port, _port_open

    monkeypatch.setattr(
        "e2e.emulator.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert _display_ready(":99")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("xdpyinfo", 5)

    monkeypatch.setattr("e2e.emulator.subprocess.run", timeout)
    assert not _display_ready(":99")

    port = _free_port()
    assert 0 < port < 65536
    assert not _port_open("127.0.0.1", port)


@pytest.mark.skipif(os.name == "nt", reason="the harness expects a Unix adb layout")
def test_harness_finds_an_android_sdk(monkeypatch, tmp_path):
    adb = tmp_path / "platform-tools" / "adb"
    adb.parent.mkdir(parents=True)
    adb.write_text("#!/bin/sh\n")
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    harness = EmulatorE2E(E2EConfig(artifacts=tmp_path / "artifacts"))

    harness._find_sdk()
    assert harness.sdk == tmp_path
    assert harness.adb == str(adb)
    assert harness.env["ANDROID_HOME"] == str(tmp_path)
    assert str(adb.parent) in harness.env["PATH"].split(os.pathsep)


def test_harness_uses_an_explicit_apk(monkeypatch, tmp_path):
    apk = tmp_path / "app-debug.apk"
    apk.write_bytes(b"apk")
    harness = EmulatorE2E(E2EConfig(artifacts=tmp_path / "artifacts", apk=apk))

    harness._prepare_apk()
    assert harness.apk == apk


def test_harness_writes_summary_and_result(tmp_path):
    harness = EmulatorE2E(E2EConfig(artifacts=tmp_path))
    harness.markers_seen = {10, 0, 5}
    harness.screenshot_mean = 12.345
    harness._write_summary()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "running"
    assert summary["package"] == "dev.openvrcardboard.phone"
    assert summary["screenshots"]["mean"] == 12.35
    assert harness._result().markers_seen == [0, 5, 10]
    assert harness._result().summary == tmp_path / "summary.json"
