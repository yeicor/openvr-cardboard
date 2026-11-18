from pathlib import Path

import pytest

from openvrcardboard.config import AppConfig
from openvrcardboard.printables import generate_boards, generate_calibration_board

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV aruco module is required for printable tests", allow_module_level=True)


def test_generate_boards_writes_printable_sheets(tmp_path: Path):
    from openvrcardboard.boards import boards_from_config

    config = AppConfig()
    boards = boards_from_config(config.tracking)
    sheet = generate_boards(
        tmp_path / "markers", boards, config.camera.dictionary, dpi=150
    )
    page = sheet.read_text(encoding="utf-8")
    assert "DICT_3X3_15" in page
    assert "200 × 112 mm" in page
    assert "central ID 0" in page
    assert "ID 14" in page
    pngs = sorted(path.name for path in (tmp_path / "markers").glob("*.png"))
    assert len(pngs) == 15
    assert "hmd_id_000.png" in pngs
    assert "right_id_014.png" in pngs
    readme = (tmp_path / "markers" / "README.txt").read_text(encoding="utf-8")
    assert "A4/Letter" in readme
    assert "DICT_3X3_15" in readme


def test_generate_calibration_board_uses_configured_geometry(tmp_path: Path):
    sheet = generate_calibration_board(tmp_path / "calibration")
    page = sheet.read_text(encoding="utf-8")
    assert "9 × 6 inner corners" in page
    assert 'width="250.000mm" height="175.000mm"' in page
    notes = (tmp_path / "calibration" / "CALIBRATION.txt").read_text(encoding="utf-8")
    assert "9 x 6" in notes
    assert "25.0 mm" in notes
    assert "A4/Letter landscape" in notes
