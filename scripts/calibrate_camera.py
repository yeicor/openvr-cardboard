#!/usr/bin/env python3
"""Calibrate a webcam for marker tracking.

The app falls back to a 60° horizontal field-of-view estimate when no
calibration exists. A real chessboard calibration improves marker depth and
off-centre accuracy, especially for wide-angle webcams.

Usage:

    python scripts/calibrate_camera.py --camera 0 --width 1280 --height 720 \
        --cols 9 --rows 6 --square-mm 25

Print a chessboard with 9x6 **inner** corners (10x7 squares) or pass the right
counts for your board. Hold it in front of the camera, press SPACE for each
pose (at least 8, ideally 15–20), varying distance, tilt and image position.
Press q or Esc to finish. The resulting JSON can be copied to the app config
directory as ``camera_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_cv2():
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "camera calibration needs opencv-contrib-python:\n"
            "  pip install opencv-contrib-python"
        )
    return cv2


def find_corners(cv2, gray, pattern: tuple[int, int]):
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        return found, corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def write_calibration(path: Path, width: int, height: int, matrix, distortion) -> Path:
    payload = {
        "width": int(width),
        "height": int(height),
        "camera_matrix": np.asarray(matrix, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(distortion, dtype=float).reshape(-1).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cols", type=int, default=9, help="inner corners along width")
    parser.add_argument("--rows", type=int, default=6, help="inner corners along height")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--samples", type=int, default=30, help="maximum accepted views")
    parser.add_argument(
        "--output",
        default="camera_calibration.json",
        help="output JSON path (default: ./camera_calibration.json)",
    )
    args = parser.parse_args(argv)

    if args.cols < 2 or args.rows < 2:
        raise SystemExit("--cols and --rows must be at least 2")
    if args.square_mm <= 0:
        raise SystemExit("--square-mm must be positive")
    if args.samples < 8:
        raise SystemExit("--samples must be at least 8")

    cv2 = load_cv2()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)

    pattern = (args.cols, args.rows)
    square_m = args.square_mm / 1000.0
    object_template = np.zeros((args.cols * args.rows, 3), dtype=np.float32)
    grid = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    object_template[:, :2] = grid * square_m

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    frame_size = (0, 0)
    window = "OpenVR Cardboard camera calibration"
    print("SPACE: capture a view | q/Esc: finish | need at least 8 views")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("camera returned no frame", file=sys.stderr)
                break
            height, width = frame.shape[:2]
            frame_size = (width, height)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(cv2, gray, pattern)
            preview = frame.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(preview, pattern, corners, found)
            cv2.putText(
                preview,
                f"views {len(object_points)}/{args.samples}  SPACE capture  q finish",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 120) if found else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and found and len(object_points) < args.samples:
                object_points.append(object_template.copy())
                image_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 2))
                print(f"captured view {len(object_points)} at {width}x{height}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if len(object_points) < 8:
        raise SystemExit(
            f"only {len(object_points)} views captured; at least 8 are needed "
            "(15–20 well-spread views are better)"
        )

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, frame_size, None, None
    )
    path = write_calibration(Path(args.output).expanduser(), frame_size[0], frame_size[1], matrix, distortion)
    print(f"wrote {path}")
    print(f"reprojection RMS error: {rms:.3f} px")
    if rms > 1.0:
        print("warning: RMS above 1 px; recapture with less blur and more varied views")
    print(
        "copy it to the app config directory as camera_calibration.json, or set "
        "camera.calibration_file to this absolute path"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
