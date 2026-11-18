"""Path limits for Monado's Unix-domain IPC socket.

Monado places its service socket at ``$XDG_RUNTIME_DIR/monado_comp_ipc``. On
Linux the path must fit in ``sockaddr_un.sun_path``, which is 108 bytes
including the terminating NUL. A deep checkout or home directory otherwise
makes ``monado-service`` exit during startup with ``Total IPC path too long``.
Both launchers use :func:`monado_socket_fits` to detect that case and fall back
to a short temporary runtime directory.
"""

from __future__ import annotations

import os
from pathlib import Path

MONADO_SOCKET_FILENAME = "monado_comp_ipc"

# sizeof(sockaddr_un.sun_path) on Linux: 108 including the trailing NUL.
UNIX_SOCKET_PATH_LIMIT = 108


def monado_socket_fits(runtime_dir: str | Path) -> bool:
    """Return whether Monado's IPC socket fits in ``sockaddr_un.sun_path``.

    Monado uses named pipes on Windows, where the limit does not apply.
    """
    if os.name == "nt":
        return True
    socket_path = Path(runtime_dir) / MONADO_SOCKET_FILENAME
    return len(os.fsencode(str(socket_path))) < UNIX_SOCKET_PATH_LIMIT
