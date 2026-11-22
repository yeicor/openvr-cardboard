"""Cross-language camera fallback ladder and ordering cases."""

from pathlib import Path

from openvrcardboard.camera_modes import (
    DEFAULT_CAMERA_MODES,
    CameraMode,
    camera_mode_attempts,
)


def _protocol_file(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "protocol" / name


def _parse_mode(values: list[str]) -> CameraMode:
    return CameraMode(int(values[0]), int(values[1]), int(values[2]))


def test_default_ladder_matches_shared_spec():
    expected = [
        _parse_mode(line.split())
        for line in _protocol_file("camera_modes.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert expected == list(DEFAULT_CAMERA_MODES)


def test_fallback_attempts_match_shared_cases():
    cases = [
        line
        for line in _protocol_file("camera_mode_attempts.txt")
        .read_text()
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(cases) == 15
    for line in cases:
        target_text, attempts_text = line.split("|", 1)
        target = _parse_mode(target_text.split())
        expected = [
            _parse_mode(part.split())
            for part in attempts_text.split(",")
        ]
        assert camera_mode_attempts(target) == expected


def test_manual_target_is_not_upscaled_by_the_fallback():
    attempts = camera_mode_attempts(CameraMode(1280, 720, 60))
    assert all(mode.width <= 1280 and mode.height <= 720 for mode in attempts)
    assert all(mode.fps <= 60 for mode in attempts)
