"""Android emulator end-to-end test.

The test is skipped during normal unit runs. Enable it with
``OVRC_EMULATOR_E2E=1`` after an emulator is running (``OVRC_EMULATOR_ATTACH=1``)
or let the harness boot one from an installed AVD and system image.

The recorded session runs the real desktop app and Monado's
``openxr-simple-example``: synthetic webcam markers and a human skeleton drive
head and controller poses, synthetic Xbox pads grab and receive haptics, and
the phone displays the example's live side-by-side stereo eye views. The phone
stream uses CPU libx264 H.264 at 600x540 per eye / 30 FPS by default so the
MediaCodec decode path is covered without a GPU; the OpenXR capture itself
stays at 2400x1080. Override the geometry/codec with ``OVRC_E2E_PHONE_*``
environment variables.

See ``docs/TESTING.md`` and ``scripts/run-e2e.sh emulator``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OVRC_EMULATOR_E2E") != "1",
    reason="set OVRC_EMULATOR_E2E=1 to run the Android emulator end-to-end test",
)


def test_android_emulator_streams_the_openxr_application():
    from openvrcardboard.emulator_e2e import E2EConfig, EmulatorE2E

    artifacts = Path(os.environ.get("OVRC_E2E_ARTIFACTS", "artifacts/emulator-e2e"))
    minimum_frames = int(os.environ.get("OVRC_E2E_MIN_FRAMES", "60"))
    harness = EmulatorE2E(
        E2EConfig(
            artifacts=artifacts,
            duration=float(os.environ.get("OVRC_E2E_DURATION", "18")),
            minimum_frames=minimum_frames,
            attach=os.environ.get("OVRC_EMULATOR_ATTACH") == "1",
            keep_emulator=os.environ.get("OVRC_KEEP_EMULATOR") == "1",
            phone_codec=os.environ.get("OVRC_E2E_PHONE_CODEC", "h264"),
            phone_encoder=os.environ.get("OVRC_E2E_PHONE_ENCODER", "x264"),
            phone_width=int(os.environ.get("OVRC_E2E_PHONE_WIDTH", "600")),
            phone_height=int(os.environ.get("OVRC_E2E_PHONE_HEIGHT", "540")),
            phone_fps=int(os.environ.get("OVRC_E2E_PHONE_FPS", "30")),
        )
    )
    result = harness.run()

    assert result.video_connected, "phone never connected to the video WebSocket"
    assert result.frames >= minimum_frames, f"phone displayed only {result.frames} frames"
    assert result.phone_imu_seen, "phone did not start its IMU stream"
    assert result.desktop_imu_seen, "desktop did not receive phone IMU packets"
    assert result.hardware_decode_seen, "phone did not start MediaCodec H.264 decoding"
    assert result.screenshot_mean > 6.0, "emulator screenshot is blank"
    assert result.screenshot_std > 6.0, "emulator screenshot has no contrast"
    assert result.screenshot_changed_pixels > 2000, "the scene is not animating on the emulator"
    assert result.stereo_diff > 0.5, "the phone is not showing distinct eye views"

    assert result.openxr_connected, "the OpenXR application never connected"
    assert result.openxr_frames >= 30, "too few stereo frames came from OpenXR"
    # 0, 5 and 10 are the central markers of the HMD, left and right boards.
    assert {0, 5, 10}.issubset(set(result.markers_seen)), "tracked markers are missing"
    assert result.body_seen, "the synthetic human skeleton was not tracked"
    assert result.max_grab_left > 0.75, "the left synthetic trigger never grabbed"
    assert result.max_grab_right > 0.75, "the right synthetic trigger never grabbed"
    assert result.haptic_events > 0, "the OpenXR app did not request haptics"
    assert result.gamepad_haptics > 0, "haptics did not reach the synthetic pads"
    for name, minimum in (("hmd", 0.02), ("left", 0.02), ("right", 0.02)):
        assert result.pose_motion.get(name, 0.0) > minimum, (
            f"the {name} pose barely moved ({result.pose_motion.get(name, 0.0):.3f} m)"
        )

    for video in (result.desktop_video, result.emulator_video, result.combined_video):
        assert video.is_file() and video.stat().st_size > 20_000, f"missing recording {video}"
    assert result.summary.is_file()
