import sys
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[1]
if str(DESKTOP) not in sys.path:
    sys.path.insert(0, str(DESKTOP))


@pytest.fixture(autouse=True)
def _disable_gpu_by_default(monkeypatch):
    """Keep the CPU/JPEG path deterministic on GPU-equipped test machines."""
    monkeypatch.setenv("OPENVRCARDBOARD_DISABLE_GPU", "1")
    from openvrcardboard.gpu import reset_cache

    reset_cache()
    yield
    reset_cache()
