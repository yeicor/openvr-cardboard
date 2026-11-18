#!/usr/bin/env python3
"""Generate printable marker sheets and a calibration chessboard.

Each tracked object uses one rigid board: a large central ArUco marker plus
four smaller peripheral markers that share a single PnP solve.  Print the HTML
at 100% / actual size on white paper, check the 100 mm ruler with a real ruler,
cut along the dashed outline, and mount the board flat on the device.

Examples:

    python scripts/generate_markers.py
    python scripts/generate_markers.py --config ~/.config/openvrcardboard/config.json
    python scripts/generate_markers.py --output markers --dpi 600
    python scripts/generate_markers.py --calibration --output markers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = REPO_ROOT / "desktop"
if str(DESKTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(DESKTOP_ROOT))

from openvrcardboard.boards import boards_from_config  # noqa: E402
from openvrcardboard.config import AppConfig, load_config  # noqa: E402
from openvrcardboard.printables import (  # noqa: E402
    PrintableError,
    generate_boards,
    generate_calibration_board,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default=None,
        help="load board IDs/geometry from a config.json (default: built-in layout)",
    )
    parser.add_argument(
        "--output", default="markers", help="output directory (default: markers)"
    )
    parser.add_argument(
        "--dictionary",
        default=None,
        help="override the ArUco dictionary (default: config/camera dictionary)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="source PNG resolution (default: 300)",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=8.0,
        help="white border around the cut line (default: 8)",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="also write the printable calibration chessboard",
    )
    args = parser.parse_args(argv)

    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    if args.margin_mm < 0:
        raise SystemExit("--margin-mm must be zero or positive")
    if args.config:
        config, warnings = load_config(args.config)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    else:
        config = AppConfig()
    dictionary_name = str(args.dictionary or config.camera.dictionary).strip().upper()
    boards = boards_from_config(config.tracking)
    output = Path(args.output).expanduser()
    try:
        sheet = generate_boards(
            output, boards, dictionary_name, args.dpi, args.margin_mm
        )
        print(f"wrote {sheet}")
        print(f"wrote {output / 'README.txt'}")
        if args.calibration:
            calibration = generate_calibration_board(
                output,
                config.camera.calibration_board_cols,
                config.camera.calibration_board_rows,
                config.camera.calibration_square_mm,
            )
            print(f"wrote {calibration}")
    except PrintableError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
