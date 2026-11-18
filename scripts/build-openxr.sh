#!/usr/bin/env bash
# Build the real OpenXR stack used by the recorded emulator E2E:
#
#   * Monado with the remote driver and the null compositor
#   * the Khronos OpenXR loader (a recent loader is needed by the example)
#   * Monado's openxr-simple-example application
#   * the OpenVR Cardboard OpenXR capture shim
#
# Everything is installed under out/openxr by default so the test harness can
# find it without touching the system. Sources are cloned into out/openxr/src.
#
# Build dependencies on Debian/Ubuntu:
#   ca-certificates build-essential cmake ninja-build pkg-config git \
#   libeigen3-dev libvulkan-dev vulkan-tools glslang-tools \
#   libgl1-mesa-dev libglu1-mesa-dev libx11-dev libx11-xcb-dev \
#   libxrandr-dev libxinerama-dev \
#   libxcb1-dev libxcb-randr0-dev libxcb-xfixes0-dev \
#   libxcb-xinerama0-dev libxcb-present-dev libxcb-sync-dev \
#   libxcb-dri3-dev libxcb-glx0-dev libudev-dev libusb-1.0-0-dev \
#   zlib1g-dev libbsd-dev libsdl2-dev libjpeg-dev libpng-dev
#
# At run time the example also needs libSDL2, libvulkan1, Mesa's lavapipe
# (mesa-vulkan-drivers) and Xvfb.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${OVRC_OPENXR_ROOT:-${ROOT}/out/openxr}"
JOBS="${OVRC_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
FORCE=0

MONADO_URL="https://gitlab.freedesktop.org/monado/monado.git"
MONADO_REV="9dc151cd2efae21d1308fcc4e974b6c9a97a7513"
OPENXR_SDK_URL="https://github.com/KhronosGroup/OpenXR-SDK.git"
OPENXR_SDK_REV="f90488c4fb1537f4256d09d4a4d3ad5543ebaf24"
EXAMPLE_URL="https://gitlab.freedesktop.org/monado/demos/openxr-simple-example.git"
EXAMPLE_REV="bf920800ff156424e2710e3ca63f84be121e62cb"

usage() {
    cat <<EOF
Usage: scripts/build-openxr.sh [options]

Options:
  --prefix DIR   install prefix (default: out/openxr)
  --jobs N       parallel build jobs (default: nproc)
  --force        rebuild even when the outputs already exist
  -h, --help     show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PREFIX="$(mkdir -p "${PREFIX}" && cd "${PREFIX}" && pwd)"
SRC="${PREFIX}/src"
mkdir -p "${SRC}" "${PREFIX}/include" "${PREFIX}/lib" "${PREFIX}/bin"

need() {
    for tool in "$@"; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            echo "error: required build tool not found: ${tool}" >&2
            exit 1
        fi
    done
}

need git cmake ninja cc pkg-config

clone_at() {
    local url="$1" revision="$2" target="$3"
    if [ -d "${target}/.git" ]; then
        if [ "$(git -C "${target}" rev-parse HEAD)" = "${revision}" ]; then
            return
        fi
        git -C "${target}" fetch --depth 1 origin "${revision}"
    else
        rm -rf "${target}"
        git clone --filter=blob:none --no-checkout "${url}" "${target}"
        git -C "${target}" fetch --depth 1 origin "${revision}"
    fi
    git -C "${target}" checkout --detach "${revision}"
}

install_doc() {
    local source="$1" name="$2"
    [ -f "${source}/LICENSE" ] || return 0
    mkdir -p "${PREFIX}/share/doc/${name}"
    install -m 644 "${source}/LICENSE" "${PREFIX}/share/doc/${name}/LICENSE"
}

install_example_docs() {
    local source="$1"
    install_doc "${source}" openxr-simple-example
    if [ -f "${source}/Readme.md" ]; then
        mkdir -p "${PREFIX}/share/doc/openxr-simple-example"
        install -m 644 "${source}/Readme.md" \
            "${PREFIX}/share/doc/openxr-simple-example/Readme.md"
    fi
}

build_monado() {
    local source="${SRC}/monado"
    local build="${PREFIX}/build/monado"
    local stamp="${PREFIX}/lib/libopenxr_monado.so"
    if [ "${FORCE}" -eq 0 ] && [ -f "${stamp}" ] && [ -x "${PREFIX}/bin/monado-service" ]; then
        echo "-- Monado already built"
        install_doc "${source}" monado
        return
    fi
    echo "-- Building Monado"
    clone_at "${MONADO_URL}" "${MONADO_REV}" "${source}"
    rm -rf "${build}"
    cmake -S "${source}" -B "${build}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DXRT_MODULE_COMPOSITOR_MAIN=OFF \
        -DXRT_MODULE_COMPOSITOR_NULL=ON \
        -DXRT_BUILD_DRIVER_REMOTE=ON \
        -DXRT_BUILD_DRIVER_SIMULATED=ON \
        -DXRT_FEATURE_SERVICE=ON \
        -DXRT_FEATURE_DEBUG_GUI=OFF \
        -DXRT_FEATURE_WINDOW_PEEK=OFF \
        -DXRT_FEATURE_OPENVR=OFF \
        -DXRT_FEATURE_STEAMVR_PLUGIN=OFF \
        -DXRT_BUILD_SAMPLES=OFF \
        -DXRT_MODULE_AUX_VIVE=ON \
        -DXRT_MODULE_AUX_HTC=ON \
        -DXRT_FEATURE_SLAM=OFF \
        -DXRT_FEATURE_IPC_CLIENT=OFF
    cmake --build "${build}" --parallel "${JOBS}"
    cmake --install "${build}"
    install_doc "${source}" monado
}

build_loader() {
    local source="${SRC}/OpenXR-SDK"
    local build="${PREFIX}/build/openxr-sdk"
    local stamp="${PREFIX}/lib/libopenxr_loader.so"
    if [ "${FORCE}" -eq 0 ] && [ -e "${stamp}" ]; then
        echo "-- OpenXR loader already built"
        return
    fi
    echo "-- Building the Khronos OpenXR loader"
    clone_at "${OPENXR_SDK_URL}" "${OPENXR_SDK_REV}" "${source}"
    rm -rf "${build}"
    cmake -S "${source}" -B "${build}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DBUILD_TESTS=OFF \
        -DBUILD_API_LAYERS=OFF \
        -DBUILD_LOADER=ON
    cmake --build "${build}" --parallel "${JOBS}"
    cmake --install "${build}"
}

build_example() {
    local source="${SRC}/openxr-simple-example"
    local build="${PREFIX}/build/openxr-example"
    local stamp="${PREFIX}/bin/openxr-example"
    if [ "${FORCE}" -eq 0 ] && [ -x "${stamp}" ]; then
        echo "-- openxr-simple-example already built"
        install_example_docs "${source}"
        return
    fi
    echo "-- Building openxr-simple-example"
    clone_at "${EXAMPLE_URL}" "${EXAMPLE_REV}" "${source}"
    rm -rf "${build}"
    cmake -S "${source}" -B "${build}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}"
    cmake --build "${build}" --parallel "${JOBS}"
    install -m 755 "${build}/openxr-example" "${stamp}"
    # The example is redistributed in Linux packages, so ship its BSL-1.0
    # license with the installed runtime tree. The Windows demo script copies
    # the same files next to the executable.
    install_example_docs "${source}"
}

build_shim() {
    local source="${ROOT}/shim/openxr_capture_shim.c"
    local target="${PREFIX}/lib/libovrc_xr_capture.so"
    if [ "${FORCE}" -eq 0 ] && [ -f "${target}" ] &&
        [ "${target}" -nt "${source}" ]; then
        echo "-- capture shim already built"
        return
    fi
    echo "-- Building the OpenXR capture shim"
    mkdir -p "${PREFIX}/include"
    if [ ! -f "${PREFIX}/include/openxr/openxr.h" ]; then
        echo "error: OpenXR headers are missing from ${PREFIX}/include" >&2
        exit 1
    fi
    cc -shared -fPIC -O2 -Wall -Wextra \
        -I"${PREFIX}/include" \
        -o "${target}" "${source}" \
        -ljpeg -ldl -lpthread -lm
}

build_monado
build_loader
build_example
build_shim

cat <<EOF

OpenXR runtime ready in ${PREFIX}

  runtime manifest : ${PREFIX}/share/openxr/1/openxr_monado.json
  service          : ${PREFIX}/bin/monado-service
  example          : ${PREFIX}/bin/openxr-example
  capture shim     : ${PREFIX}/lib/libovrc_xr_capture.so

Run the recorded emulator E2E with:

  OVRC_OPENXR_ROOT=${PREFIX} scripts/run-e2e.sh emulator
EOF
