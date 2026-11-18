#!/usr/bin/env python3
"""Run the Android emulator OpenXR end-to-end harness without pytest.

Examples:
    python3 scripts/emulator_e2e.py --artifacts artifacts/emulator-e2e
    python3 scripts/emulator_e2e.py --attach --duration 12

The harness needs Linux, KVM (unless --attach), Android SDK platform-tools,
ffmpeg and Xvfb. The repository's Docker target packages all of them:

    scripts/run-e2e.sh emulator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "desktop"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        default="artifacts/emulator-e2e",
        help="directory for logs, screenshots and recordings",
    )
    parser.add_argument("--duration", type=float, default=18.0, help="recorded seconds")
    parser.add_argument("--minimum-frames", type=int, default=60)
    parser.add_argument(
        "--attach",
        action="store_true",
        help="use an emulator that is already running instead of booting one",
    )
    parser.add_argument(
        "--keep-emulator",
        action="store_true",
        help="leave a harness-started emulator running after the test",
    )
    parser.add_argument("--apk", help="path to the debug APK")
    parser.add_argument("--serial", help="adb serial of the emulator to use")
    parser.add_argument("--avd", default="ovrc_e2e", help="AVD name to create or reuse")
    parser.add_argument(
        "--system-image",
        default="system-images;android-34;google_apis;x86_64",
    )
    parser.add_argument("--device-profile", default="pixel_2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from openvrcardboard.emulator_e2e import E2EConfig, EmulatorE2E

    result = EmulatorE2E(
        E2EConfig(
            artifacts=Path(args.artifacts),
            duration=args.duration,
            minimum_frames=args.minimum_frames,
            attach=args.attach,
            keep_emulator=args.keep_emulator,
            apk=Path(args.apk) if args.apk else None,
            serial=args.serial,
            avd_name=args.avd,
            system_image=args.system_image,
            device_profile=args.device_profile,
        )
    ).run()
    print(
        f"emulator E2E passed: {result.frames} phone frames, "
        f"{result.openxr_frames} OpenXR stereo frames, "
        f"eye difference {result.stereo_diff:.2f}, "
        f"grabs {result.max_grab_left:.2f}/{result.max_grab_right:.2f}, "
        f"haptics {result.haptic_events}, "
        f"recording {result.combined_video}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
