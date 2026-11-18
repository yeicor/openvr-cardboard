# Android build/test image.
#
# Contains JDK 17, Gradle 9.9.1 and the Android SDK (platform 37.0,
# build-tools 37.0.0) so `testDebugUnitTest` and `assembleDebug` run without a
# local Android Studio installation.
#
# Build and run through scripts/run-e2e.sh.

FROM gradle:9.9.1-jdk17

USER root

# Match the invoking user so Gradle can write into the bind-mounted checkout.
# GitHub Actions runners do not use UID 1000, and `scripts/run-e2e.sh` passes
# the caller's IDs instead of chmodding the host tree or running builds as root.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupmod -o -g "${GROUP_ID}" gradle \
    && usermod -o -u "${USER_ID}" gradle \
    && chown -R gradle:gradle /home/gradle /opt/gradle

ENV ANDROID_HOME=/opt/android-sdk \
    ANDROID_SDK_ROOT=/opt/android-sdk \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${ANDROID_HOME}/cmdline-tools" \
    && wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O /tmp/cmdline-tools.zip \
    && unzip -q /tmp/cmdline-tools.zip -d /tmp/cmdline-tools \
    && mv /tmp/cmdline-tools/cmdline-tools "${ANDROID_HOME}/cmdline-tools/latest" \
    && rm -rf /tmp/cmdline-tools /tmp/cmdline-tools.zip

RUN yes | "${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager" --licenses >/dev/null \
    && "${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager" \
        "platform-tools" \
        "platforms;android-37.0" \
        "build-tools;37.0.0" >/dev/null

COPY android /workspace/android
COPY docker/android-entrypoint.sh /usr/local/bin/android-entrypoint.sh
RUN chmod +x /usr/local/bin/android-entrypoint.sh \
    && chown -R gradle:gradle /workspace /opt/android-sdk

USER gradle
WORKDIR /workspace/android

ENTRYPOINT ["/usr/local/bin/android-entrypoint.sh"]
