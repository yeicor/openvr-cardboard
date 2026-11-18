"""Printable marker boards and calibration chessboards.

Both the desktop app and ``scripts/generate_markers.py`` use this module so the
HTML that users print is always generated from the same board geometry the
tracker expects.  Sheets are written in millimetres and must be printed at
100% / actual size.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

from .aruco import get_dictionary, marker_image
from .boards import BoardSpec


class PrintableError(RuntimeError):
    """Raised when board generation needs a missing dependency or ID."""


def load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise PrintableError(
            "board generation needs opencv-contrib-python:\n"
            "  pip install opencv-contrib-python"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise PrintableError(
            "this OpenCV build has no aruco module; install opencv-contrib-python"
        )
    return cv2


def _layout(
    board: BoardSpec, quiet_margin_mm: float
) -> tuple[float, float, dict[int, tuple[float, float, float]]]:
    """Return cut size and per-marker ``(left, top, size)`` in millimetres."""
    markers = list(board.markers)
    extents = []
    for marker in markers:
        half = float(marker.size_m) * 500.0
        extents.append(
            (
                float(marker.center_x_m) * 1000.0 - half,
                float(marker.center_x_m) * 1000.0 + half,
                float(marker.center_y_m) * 1000.0 - half,
                float(marker.center_y_m) * 1000.0 + half,
            )
        )
    min_x = min(item[0] for item in extents)
    max_x = max(item[1] for item in extents)
    min_y = min(item[2] for item in extents)
    max_y = max(item[3] for item in extents)
    content_width = max_x - min_x
    content_height = max_y - min_y
    if len(markers) == 1:
        module_mm = max(float(markers[0].size_m) * 1000.0, 1.0) / 7.0
        pad = max(6.0, module_mm)
        cut_width = content_width + 2.0 * pad
        cut_height = content_height + 2.0 * pad
    else:
        cut_width = float(board.width_m) * 1000.0
        cut_height = float(board.height_m) * 1000.0
    left_origin = (cut_width - content_width) * 0.5
    top_origin = (cut_height - content_height) * 0.5
    positions = {}
    for marker, extent in zip(markers, extents):
        size = float(marker.size_m) * 1000.0
        positions[int(marker.marker_id)] = (
            quiet_margin_mm + left_origin + (extent[0] - min_x),
            quiet_margin_mm + top_origin + (max_y - extent[3]),
            size,
        )
    return cut_width, cut_height, positions


def _board_sheet(
    board: BoardSpec,
    dictionary_name: str,
    files: dict[int, str],
    margin_mm: float = 0.0,
) -> str:
    cut_width_mm, cut_height_mm, positions = _layout(board, margin_mm)
    width_mm = cut_width_mm + 2.0 * margin_mm
    height_mm = cut_height_mm + 2.0 * margin_mm
    markers = "".join(
        f'<img class="marker" '
        f'style="left:{positions[int(marker.marker_id)][0]:.3f}mm;'
        f'top:{positions[int(marker.marker_id)][1]:.3f}mm;'
        f'width:{positions[int(marker.marker_id)][2]:.3f}mm;'
        f'height:{positions[int(marker.marker_id)][2]:.3f}mm" '
        f'src="{html.escape(files[marker.marker_id])}" '
        f'alt="marker {marker.marker_id}">'
        for marker in board.markers
    )
    mapping = ", ".join(
        f"{marker.role}{'' if marker.role == 'central' else f' {index}'}: "
        f"ID {marker.marker_id} ({float(marker.size_m) * 1000.0:.0f} mm)"
        for index, marker in enumerate(board.markers)
        if marker.role != "central"
    )
    central = board.central
    return f"""
<section class="sheet">
  <header>
    <h1>{html.escape(board.name.upper())} board</h1>
    <p class="meta">{html.escape(dictionary_name)} · central ID {central.marker_id}
      · cut size {cut_width_mm:.0f} × {cut_height_mm:.0f} mm
      · sheet {width_mm:.0f} × {height_mm:.0f} mm</p>
  </header>
  <div class="top-label">TOP ↑</div>
  <div class="board" style="width:{width_mm:.3f}mm;height:{height_mm:.3f}mm">
    <div class="cut" style="inset:{margin_mm:.3f}mm"></div>
    {markers}
  </div>
  <div class="ruler" style="width:100mm"><span>100 mm check</span></div>
  <p class="ids">
    Cut along the dashed outline. Marker IDs:
    centre ID {central.marker_id} ({float(central.size_m) * 1000.0:.0f} mm);
    {mapping}.
  </p>
  <p class="ids">Mount this board flat and rigid on the
    {html.escape(board.name)} device. Do not fold or trim inside the outline.</p>
</section>
"""


def generate_boards(
    output: Path,
    boards: dict[str, BoardSpec],
    dictionary_name: str,
    dpi: int = 300,
    quiet_margin_mm: float = 8.0,
) -> Path:
    """Write one PNG per marker plus a print-ready ``boards.html``."""
    if int(dpi) < 72:
        raise PrintableError("DPI must be at least 72")
    cv2 = load_cv2()
    try:
        dictionary = get_dictionary(cv2, dictionary_name)
    except ValueError as exc:
        raise PrintableError(str(exc)) from exc
    capacity_bytes = getattr(dictionary, "bytesList", None)
    capacity = int(capacity_bytes.shape[0]) if capacity_bytes is not None else 0

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files: dict[int, str] = {}
    for board in boards.values():
        for marker in board.markers:
            marker_id = int(marker.marker_id)
            if marker_id < 0 or (capacity and marker_id >= capacity):
                raise PrintableError(
                    f"marker ID {marker_id} is outside {dictionary_name} "
                    f"(valid range 0-{capacity - 1})"
                )
            size_mm = float(marker.size_m) * 1000.0
            pixels = max(64, int(math.ceil(size_mm / 25.4 * dpi)))
            image = marker_image(cv2, dictionary, marker_id, pixels)
            filename = f"{board.name}_id_{marker_id:03d}.png"
            if not cv2.imwrite(str(output / filename), image):
                raise PrintableError(f"could not write {output / filename}")
            files[marker_id] = filename

    sheets = "".join(
        _board_sheet(board, dictionary_name, files, quiet_margin_mm)
        for board in boards.values()
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OpenVR Cardboard board sheets</title>
<style>
  @page {{ size: A4 landscape; margin: 10mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #fff; color: #111;
    font-family: system-ui, sans-serif;
  }}
  .sheet {{
    page-break-after: always; break-after: page;
    padding-top: 2mm;
  }}
  .sheet:last-child {{ page-break-after: auto; break-after: auto; }}
  header {{ margin: 0 0 7mm; }}
  h1 {{ font-size: 17pt; margin: 0 0 2mm; letter-spacing: 0.02em; }}
  .meta {{ font-size: 10pt; margin: 0; color: #333; }}
  .top-label {{
    width: 200mm; max-width: 100%; text-align: center;
    font-size: 9pt; letter-spacing: 0.2em; margin: 0 0 2mm;
  }}
  .board {{
    position: relative; margin: 0 auto; background: #fff;
  }}
  .cut {{
    position: absolute; border: 0.3mm dashed #777;
    pointer-events: none;
  }}
  .marker {{
    position: absolute; display: block;
    image-rendering: pixelated; background: #fff;
  }}
  .ruler {{
    height: 4mm; margin: 8mm auto 3mm;
    border-bottom: 0.35mm solid #111;
    background-image: repeating-linear-gradient(
      to right, #111 0 0.25mm, transparent 0.25mm 10mm
    );
    background-size: 100% 2.5mm; background-repeat: no-repeat;
    background-position: left bottom;
  }}
  .ruler span {{ font-size: 7pt; color: #444; }}
  .ids {{ font-size: 9pt; line-height: 1.4; margin: 2mm 0 0; }}
</style>
</head>
<body>
{sheets}
</body>
</html>
"""
    sheet = output / "boards.html"
    sheet.write_text(page, encoding="utf-8")

    lines = [
        "OpenVR Cardboard marker boards",
        f"Dictionary: {dictionary_name}",
        "",
        "Print boards.html at 100% / actual size on white paper (A4/Letter",
        "landscape; the 200 mm board does not fit portrait with margins).",
        "Disable 'fit to page', 'shrink to fit' and borderless scaling.",
        "Measure the printed 100 mm ruler with a real ruler before cutting.",
        "Cut along the dashed line and keep the white border outside the board.",
        "",
    ]
    for board in boards.values():
        lines.append(
            f"{board.name.upper()}: {float(board.width_m) * 1000:.0f} x "
            f"{float(board.height_m) * 1000:.0f} mm, IDs "
            + ", ".join(str(marker.marker_id) for marker in board.markers)
        )
        for marker in board.markers:
            lines.append(
                f"  ID {marker.marker_id:3d}  {float(marker.size_m) * 1000:5.0f} mm  "
                f"centre ({float(marker.center_x_m) * 1000:+.0f}, "
                f"{float(marker.center_y_m) * 1000:+.0f}) mm  {marker.role}"
            )
    lines += [
        "",
        "The board origin is the centre marker; +X is right and +Y is up on",
        "the printed face. The board mounting offset and rotation are set in",
        "the desktop application under Settings -> Tracking -> Boards.",
    ]
    (output / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sheet


def generate_calibration_board(
    output: Path,
    cols: int = 9,
    rows: int = 6,
    square_mm: float = 25.0,
    margin_mm: float = 0.0,
) -> Path:
    """Write a print-ready chessboard with ``cols × rows`` inner corners.

    Print at 100% / actual size, then measure one square with a ruler.  The
    calibration wizard expects inner corners, so the printed board has one
    extra square in each direction.
    """
    cols = max(3, int(cols))
    rows = max(3, int(rows))
    square = max(5.0, float(square_mm))
    margin = max(0.0, float(margin_mm))
    squares_x = cols + 1
    squares_y = rows + 1
    board_w = squares_x * square
    board_h = squares_y * square
    sheet_w = board_w + 2.0 * margin
    sheet_h = board_h + 2.0 * margin

    cells = []
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                continue
            cells.append(
                f'<rect x="{margin + col * square:.3f}" '
                f'y="{margin + row * square:.3f}" '
                f'width="{square:.3f}" height="{square:.3f}" fill="#000"/>'
            )
    cells = "\n    ".join(cells)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OpenVR Cardboard calibration chessboard</title>
<style>
  @page {{ size: A4 landscape; margin: 5mm; }}
  body {{ margin: 0; background: #fff; color: #111; font-family: system-ui, sans-serif; }}
  h1 {{ font-size: 16pt; margin: 0 0 2mm; }}
  p {{ font-size: 10pt; margin: 0 0 4mm; }}
  svg {{ display: block; }}
</style>
</head>
<body>
<h1>OpenVR Cardboard calibration board</h1>
<p>{cols} × {rows} inner corners · {square:.1f} mm squares ·
   board {board_w:.0f} × {board_h:.0f} mm. Print at 100% / actual size on
   A4/Letter landscape and measure one square before using it.</p>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{sheet_w:.3f}mm" height="{sheet_h:.3f}mm"
     viewBox="0 0 {sheet_w:.3f} {sheet_h:.3f}">
    <rect x="0" y="0" width="{sheet_w:.3f}" height="{sheet_h:.3f}" fill="#fff"/>
    {cells}
</svg>
</body>
</html>
"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sheet = output / "calibration-board.html"
    sheet.write_text(page, encoding="utf-8")
    (output / "CALIBRATION.txt").write_text(
        "OpenVR Cardboard calibration board\n"
        f"Inner corners: {cols} x {rows}\n"
        f"Square size: {square:.1f} mm\n"
        f"Printed board: {board_w:.1f} x {board_h:.1f} mm\n\n"
        "Print calibration-board.html at 100% / actual size on A4/Letter "
        "landscape. Measure one square with a ruler; if it is not the size above, "
        "adjust the printer scale before calibrating. Use the same square size in "
        "the calibration dialog.\n",
        encoding="utf-8",
    )
    return sheet
