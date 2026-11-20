#!/usr/bin/env bash
# Reproducible end-to-end tests in Docker.
#
#   scripts/run-e2e.sh desktop            # full Python suite + hardware-free E2E
#   scripts/run-e2e.sh desktop -k video   # extra args go to pytest
#   scripts/run-e2e.sh android            # JVM tests + assembleDebug
#   scripts/run-e2e.sh android testDebugUnitTest
#   scripts/run-e2e.sh emulator           # recorded Android emulator OpenXR E2E
#   scripts/run-e2e.sh all                # desktop + Android + emulator (needs KVM)
#
# The repository is bind-mounted at /workspace, so the containers always test
# the current working tree. The OpenXR stack built by scripts/build-openxr.sh
# (out/openxr by default) is mounted read-only at /opt/openxr. Set
# OVRC_DOCKER_BUILD_ARGS (e.g. "--no-cache") to pass extra flags to docker build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
if [ "$#" -gt 0 ]; then
    shift
fi

DESKTOP_IMAGE="${OVRC_DESKTOP_IMAGE:-openvrcardboard-e2e-desktop}"
ANDROID_IMAGE="${OVRC_ANDROID_IMAGE:-openvrcardboard-e2e-android}"
EMULATOR_IMAGE="${OVRC_EMULATOR_IMAGE:-openvrcardboard-e2e-emulator}"
GRADLE_CACHE_VOLUME="${OVRC_GRADLE_CACHE_VOLUME:-openvrcardboard-gradle-cache}"
OPENXR_DIR="${OVRC_OPENXR_ROOT:-${ROOT}/out/openxr}"
OPENXR_MOUNTS=()
OPENXR_ENV=()
if [ -x "${OPENXR_DIR}/bin/openxr-example" ]; then
    OPENXR_MOUNTS=(-v "${OPENXR_DIR}:/opt/openxr:ro")
    OPENXR_ENV=(-e OVRC_OPENXR_ROOT=/opt/openxr)
fi
BUILD_ARGS=()
if [ -n "${OVRC_DOCKER_BUILD_ARGS:-}" ]; then
    # shellcheck disable=SC2206  # intentional word splitting
    BUILD_ARGS=(${OVRC_DOCKER_BUILD_ARGS})
fi

usage() {
    cat <<'EOF'
Usage: scripts/run-e2e.sh <desktop|android|emulator|all> [extra args...]

  desktop   Build the Python image, run pytest with OVRC_E2E=1.
  android   Build the Android image, run Gradle unit tests and assembleDebug.
  emulator  Build the APK and emulator image, run the recorded Android E2E.
  all       All three targets (the emulator needs /dev/kvm).

Extra arguments are forwarded to pytest or Gradle for the selected target.
Environment:
  OVRC_DOCKER_BUILD_ARGS   Extra options for `docker build` (e.g. --no-cache)
  OVRC_DESKTOP_IMAGE       Override the desktop image tag
  OVRC_ANDROID_IMAGE       Override the Android image tag
  OVRC_EMULATOR_IMAGE      Override the emulator image tag
  OVRC_GRADLE_CACHE_VOLUME Override the Gradle cache volume name
  OVRC_OPENXR_ROOT         OpenXR install tree to mount (default out/openxr)
  OVRC_E2E_DURATION        Recorded seconds in the emulator session (default 18)
  OVRC_ANDROID_KEYSTORE    Keystore path (container path) for signed APKs
  OVRC_ANDROID_KEYSTORE_PASSWORD, OVRC_ANDROID_KEY_ALIAS,
  OVRC_ANDROID_KEY_PASSWORD  Remaining release-signing values; all four are
                           forwarded to the Android container when set.
EOF
}

build_desktop() {
    docker build \
        "${BUILD_ARGS[@]}" \
        -f "${ROOT}/docker/desktop.Dockerfile" \
        -t "${DESKTOP_IMAGE}" \
        "${ROOT}"
}

build_android() {
    docker build \
        "${BUILD_ARGS[@]}" \
        --build-arg "USER_ID=$(id -u)" \
        --build-arg "GROUP_ID=$(id -g)" \
        -f "${ROOT}/docker/android.Dockerfile" \
        -t "${ANDROID_IMAGE}" \
        "${ROOT}"
}

build_emulator() {
    build_desktop
    docker build \
        "${BUILD_ARGS[@]}" \
        --build-arg "DESKTOP_IMAGE=${DESKTOP_IMAGE}" \
        -f "${ROOT}/docker/emulator.Dockerfile" \
        -t "${EMULATOR_IMAGE}" \
        "${ROOT}"
}

run_desktop() {
    build_desktop
    mkdir -p "${ROOT}/artifacts/desktop"
    docker run --rm \
        -e OVRC_E2E=1 \
        "${OPENXR_ENV[@]}" \
        -v "${ROOT}:/workspace" \
        "${OPENXR_MOUNTS[@]}" \
        "${DESKTOP_IMAGE}" "$@"
}

run_android() {
    build_android
    local signing_env=()
    local name
    for name in \
        OVRC_ANDROID_KEYSTORE \
        OVRC_ANDROID_KEYSTORE_PASSWORD \
        OVRC_ANDROID_KEY_ALIAS \
        OVRC_ANDROID_KEY_PASSWORD; do
        if [ -n "${!name:-}" ]; then
            signing_env+=(-e "${name}")
        fi
    done
    docker run --rm \
        "${signing_env[@]}" \
        -v "${ROOT}:/workspace" \
        -v "${GRADLE_CACHE_VOLUME}:/home/gradle/.gradle" \
        "${ANDROID_IMAGE}" "$@"
}

run_emulator() {
    if [ ! -e /dev/kvm ]; then
        echo "error: /dev/kvm is required for the Android emulator E2E" >&2
        echo "       enable KVM, or run the desktop/android targets separately" >&2
        exit 2
    fi
    if [ "${#OPENXR_MOUNTS[@]}" -eq 0 ]; then
        echo "error: the OpenXR runtime was not found in ${OPENXR_DIR}" >&2
        echo "       build it first with scripts/build-openxr.sh" >&2
        exit 2
    fi
    # The harness installs this APK into the emulator.
    run_android assembleDebug
    build_emulator
    mkdir -p "${ROOT}/artifacts/emulator-e2e"
    docker run --rm \
        --device /dev/kvm \
        --shm-size=2g \
        -e OVRC_EMULATOR_E2E=1 \
        -e "OVRC_E2E_DURATION=${OVRC_E2E_DURATION:-18}" \
        "${OPENXR_ENV[@]}" \
        -v "${ROOT}:/workspace" \
        "${OPENXR_MOUNTS[@]}" \
        "${EMULATOR_IMAGE}" "$@"
}

case "${TARGET}" in
    desktop)
        run_desktop "$@"
        ;;
    android)
        run_android "$@"
        ;;
    emulator)
        run_emulator "$@"
        ;;
    all)
        if [ "$#" -gt 0 ]; then
            echo "error: extra arguments are only supported for a single target" >&2
            exit 2
        fi
        run_desktop
        run_android
        run_emulator
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "error: unknown target '${TARGET}'" >&2
        usage >&2
        exit 2
        ;;
esac
