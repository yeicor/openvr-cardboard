# Desktop end-to-end test image.
#
# Builds the Python app with every optional dependency (OpenCV contrib,
# MediaPipe, PySide6, pygame, mss) and a virtual X server so the hardware-free
# E2E suite can run reproducibly on any Docker host.
#
# Build and run through scripts/run-e2e.sh.
#
# The base matches the Ubuntu 24.04 GitHub runner used by CI so the OpenXR
# runtime built there has the same C library and shared-library SONAMEs as the
# container that runs the desktop E2E suite.

FROM ubuntu:24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Runtime libraries for Qt/PySide6, OpenCV, pygame, mss and Xvfb. The Python
# packages ship their own bundled wheels inside the virtual environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3.12 \
        python3.12-venv \
        libpython3.12 \
        binutils \
        file \
        libgssapi-krb5-2 \
        xvfb \
        xauth \
        x11-utils \
        xdotool \
        libgl1 \
        libopengl0 \
        libgles2 \
        libegl-mesa0 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libglib2.0-0 \
        libegl1 \
        libvulkan1 \
        mesa-vulkan-drivers \
        libbsd0 \
        libudev1 \
        libusb-1.0-0 \
        libgbm1 \
        libdrm2 \
        libjpeg-turbo8 \
        libpng16-16 \
        libsamplerate0 \
        libasound2t64 \
        libpulse0 \
        libxrandr2 \
        libxfixes3 \
        libxi6 \
        libxcursor1 \
        libxss1 \
        libdecor-0-0 \
        libwayland-client0 \
        libwayland-egl1 \
        libwayland-cursor0 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libfontconfig1 \
        libfreetype6 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xkb1 \
        libxext6 \
        libxrender1 \
        libxtst6 \
        libsm6 \
        libice6 \
        libsdl2-2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# The project is installed editable from the path where the repository gets
# bind-mounted at run time, so the tests always exercise the mounted sources.
RUN python3 -m venv "${VIRTUAL_ENV}"
COPY desktop /workspace/desktop
RUN pip install -e "/workspace/desktop[all]" pytest

COPY docker/desktop-entrypoint.sh /usr/local/bin/desktop-entrypoint.sh
RUN chmod +x /usr/local/bin/desktop-entrypoint.sh

WORKDIR /workspace/desktop

ENTRYPOINT ["/usr/local/bin/desktop-entrypoint.sh"]
CMD ["-o", "addopts=", "-q"]
