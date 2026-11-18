"""Shared PyInstaller entry point for the app and the installer.

The portable folder contains two executable names (``openvrcardboard`` and
``openvrcardboard-install``). PyInstaller stores the same entry point in both,
so the program selects its mode from ``sys.argv[0]``. This keeps the GUI and
the command-line installer in one self-contained bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    name = Path(sys.argv[0]).stem.lower()
    if "install" in name or "uninstall" in name:
        from openvrcardboard.installer import main as installer_main

        return int(installer_main())
    from openvrcardboard.app import main as app_main

    return int(app_main())


if __name__ == "__main__":
    raise SystemExit(main())
