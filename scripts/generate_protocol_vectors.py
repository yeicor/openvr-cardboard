#!/usr/bin/env python3
"""Generate the cross-language golden protocol vectors.

The output is a plain-text file with one ``name hex`` line per packet. Python,
C++ and Kotlin tests read the same file so a wire-layout change cannot pass in
one language while silently breaking another.

Run with ``--check`` in CI to fail when the committed file is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))

from openvrcardboard.protocol import (  # noqa: E402
    Buttons,
    ControllerInput,
    DeviceFlags,
    DeviceId,
    DevicePose,
    pack_haptic,
    pack_imu,
    pack_state,
    pack_status,
    pack_time_ping,
)

OUTPUT = ROOT / "protocol" / "golden_vectors.txt"
TOKEN = "test-token"


def vectors() -> dict[str, bytes]:
    return {
        "state": pack_state(
            42,
            12.5,
            [
                DevicePose(
                    device_id=int(DeviceId.HMD),
                    position=(0.1, 1.7, -0.2),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    flags=DeviceFlags.POSE_VALID | DeviceFlags.CONNECTED,
                ),
                DevicePose(
                    device_id=int(DeviceId.LEFT),
                    position=(-0.3, 1.2, -0.4),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    flags=DeviceFlags.POSE_VALID | DeviceFlags.CONNECTED,
                ),
            ],
            [
                ControllerInput(
                    device_id=int(DeviceId.LEFT),
                    buttons=Buttons.TRIGGER_CLICK | Buttons.A,
                    trigger=0.75,
                    grip=0.25,
                    joystick_x=-0.5,
                    joystick_y=0.5,
                )
            ],
            prediction_ms=35.0,
        ),
        "haptic": pack_haptic(7, int(DeviceId.LEFT), 0.8, 120.0, 0.05),
        "status": pack_status(8, device_count=5, uptime_ms=1000, state_packets=321, parse_errors=2),
        "imu": pack_imu(
            0x01020304,
            1234.5,
            (0.1, 0.2, 0.3, 0.9),
            (1.5, -2.5, 3.5),
            (0.0, -9.81, 0.25),
            accuracy=3,
            screen_rotation=2,
        ),
        "imu_auth": pack_imu(
            0x01020304,
            1234.5,
            (0.1, 0.2, 0.3, 0.9),
            (1.5, -2.5, 3.5),
            (0.0, -9.81, 0.25),
            accuracy=3,
            screen_rotation=2,
            token=TOKEN,
        ),
        "time_ping": pack_time_ping(9, 55.125),
        "time_ping_auth": pack_time_ping(9, 55.125, token=TOKEN),
    }


def render() -> str:
    lines = [
        "# Golden protocol vectors shared by the Python, C++ and Kotlin tests.",
        "# Regenerate with scripts/generate_protocol_vectors.py",
        "# state: sequence 42, timestamp 12.5, prediction 35 ms, HMD + left pose,",
        "#        left controller input (trigger 0.75, A); haptic: 7/left/0.8/120/0.05;",
        "#        status: 8, 5 devices, 1000 ms, 321 packets, 2 parse errors;",
        "#        imu: sequence 0x01020304, timestamp 1234.5, accuracy 3, rotation 2;",
        "#        time_ping: sequence 9, client time 55.125.",
        "# _auth variants append SHA-256(test-token)[:16].",
    ]
    for name, packet in vectors().items():
        lines.append(f"{name} {packet.hex()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            print(f"{OUTPUT.relative_to(ROOT)} is stale; run scripts/generate_protocol_vectors.py")
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
