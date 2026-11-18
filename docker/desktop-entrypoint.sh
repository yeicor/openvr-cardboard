#!/usr/bin/env bash
# Entry point for the desktop E2E container: start a headless X server (mss,
# pygame and Qt can use it) and run the test suite against /workspace/desktop.
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    Xvfb "${DISPLAY}" -screen 0 1280x1024x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    for _ in $(seq 1 100); do
        if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
fi

if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "error: Xvfb failed to start on ${DISPLAY}" >&2
    cat /tmp/xvfb.log >&2 || true
    exit 1
fi

cd /workspace/desktop
exec python3 -m pytest "$@"
