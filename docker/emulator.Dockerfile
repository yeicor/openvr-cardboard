# Android emulator end-to-end image.
#
# Extends the desktop test image (Python, Qt, OpenCV, Xvfb, ffmpeg) with the
# Android SDK, emulator and an API 34 x86_64 system image. Run it with
# `--device /dev/kvm --shm-size=2g` through scripts/run-e2e.sh emulator.
#
# Build arguments:
#   DESKTOP_IMAGE  base image built from docker/desktop.Dockerfile

ARG DESKTOP_IMAGE=openvrcardboard-e2e-desktop
FROM ${DESKTOP_IMAGE}

USER root

ENV ANDROID_HOME=/opt/android-sdk \
    ANDROID_SDK_ROOT=/opt/android-sdk \
    ANDROID_EMULATOR_HOME=/root/.android \
    DEBIAN_FRONTEND=noninteractive

# The emulator is a Qt application that links against the host's GL, audio and
# XCB libraries even when it runs headless. Java is needed by sdkmanager.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        wget \
        unzip \
        ca-certificates \
        ffmpeg \
        libpulse0 \
        libasound2t64 \
        libgbm1 \
        libgles2 \
        libglu1-mesa \
        libxcb-dri3-0 \
        libxcb-present0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libxtst6 \
        libnspr4 \
        libnss3 \
        libatk1.0-0t64 \
        libatk-bridge2.0-0t64 \
        libcups2t64 \
        libdrm2 \
        libxfixes3 \
        libxshmfence1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${ANDROID_HOME}/cmdline-tools" \
    && wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O /tmp/cmdline-tools.zip \
    && unzip -q /tmp/cmdline-tools.zip -d /tmp/cmdline-tools \
    && mv /tmp/cmdline-tools/cmdline-tools "${ANDROID_HOME}/cmdline-tools/latest" \
    && rm -rf /tmp/cmdline-tools /tmp/cmdline-tools.zip

# Licenses are accepted once at build time so the container can create AVDs at
# run time without interactive prompts.
RUN yes | "${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager" --licenses >/dev/null \
    && yes | "${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager" \
        "platform-tools" \
        "emulator" \
        "system-images;android-34;google_apis;x86_64" >/dev/null \
    && rm -rf /root/.android/cache

COPY docker/emulator-entrypoint.sh /usr/local/bin/emulator-entrypoint.sh
RUN chmod +x /usr/local/bin/emulator-entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/emulator-entrypoint.sh"]
CMD []
