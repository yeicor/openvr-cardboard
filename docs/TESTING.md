# Testing

The project has five test layers:

| Layer | Command | Needs |
| --- | --- | --- |
| Lint, types and coverage | `cd desktop && uv run ruff check . && uv run mypy && uv run pytest -q --cov` | `uv` (or install the `dev` dependency group) |
| Python unit tests | `cd desktop && python3 -m pytest -q` | Python 3.12+ and the base dependencies |
| C++ driver test | `ctest --test-dir driver/build --output-on-failure` | A built driver |
| Container suites | `scripts/run-e2e.sh desktop` / `scripts/run-e2e.sh android` | Docker |
| Android emulator OpenXR E2E | `scripts/run-e2e.sh emulator` | Docker, `/dev/kvm`, ~6 GB disk, OpenXR runtime |

## Unit tests

```bash
cmake -S driver -B driver/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build driver/build -j4
ctest --test-dir driver/build --output-on-failure

cd desktop
uv sync --locked --all-extras
uv run ruff check .
uv run mypy
uv run pytest -q --cov
```

`uv.lock` pins every dependency. Ruff checks correctness and import order,
mypy covers the fully annotated core modules listed in `pyproject.toml`, and
`pytest-cov` enforces the floor in `[tool.coverage.report]`. The same commands
run in the CI `quality` job. To wire them into a local git hook, run
`uv run --directory desktop pre-commit install --config ../.pre-commit-config.yaml`
from the repository root. The hooks invoke the same locked Ruff and mypy that
CI uses. Test-only emulator tooling lives in `desktop/e2e/` rather than the
installed runtime package.

The Python suite covers the wire protocol, pose composition, board geometry,
board-level PnP, the error-state Kalman filter and tracking state machine,
ArUco detection, chessboard calibration, phone IMU handling, driver links,
gamepad mapping, the built-in test scene, the OpenXR remote packet and capture
protocol, the synthetic webcam/gamepad fixtures, the video server, the settings
dialog, the encoder/demuxers, MediaPipe Tasks model selection and delegate
fallback, and the Qt lifecycle. Tests that need optional packages or a container
skip themselves on a bare workstation. NVENC detection is disabled with
`OPENVRCARDBOARD_DISABLE_GPU=1` by default, so the video suite exercises the
same JPEG/CPU fallback that CI and non-NVIDIA machines use; GPU-specific
encoder checks are capability-gated and only run when an NVIDIA device and
NVENC ffmpeg are actually present.

`test_boards.py` checks exact synthetic projections, robust PnP with a
corrupted marker, object mounting transforms, covariance behaviour and
central-only solves. `test_tracking_state.py` checks the filter, NIS gating,
state transitions, exact-time prediction, acceleration estimation,
reacquisition blending, a 10,000-update positive-definiteness run and a
stationary tracker-to-composer jitter check. `test_fusion.py` includes the
quaternion glitch-hold policy: a single >140° jump is held and only accepted
after three consistent samples. `test_synthetic.py` renders complete
five-marker boards with the custom 3×3 dictionary and asserts detection, pose
accuracy and frame-to-frame orientation continuity. `test_phone_link.py` covers
the clock-offset estimate, one-sample reorder buffer and stale/duplicate
rejection. `test_installer.py` covers the CLI install/uninstall/status paths,
`test_mediapipe_models.py` covers model lookup, cache verification and atomic
downloads, `test_netutil.py` covers the per-source UDP rate limiter, and
`test_release_scripts.py` covers release asset collection and SBOM generation.

## Desktop container suite

`docker/desktop.Dockerfile` starts from Ubuntu 24.04 (matching the GitHub
runner and the OpenXR runtime build environment), creates a Python 3.12
virtual environment and installs every optional dependency
(`opencv-contrib-python`, `mediapipe` Tasks, `PySide6`, `pygame-ce`, `mss`)
plus Xvfb. The entry point starts a virtual display and runs pytest from
`/workspace/desktop`. The repository is bind-mounted at `/workspace`, so the
container always tests the current working tree.

```bash
scripts/run-e2e.sh desktop            # full suite with OVRC_E2E=1
scripts/run-e2e.sh desktop -k video   # extra arguments go to pytest
```

`OVRC_E2E=1` enables the checks that are skipped on a bare workstation:

- `test_container_optional_stack` — imports and smoke-tests OpenCV contrib,
  MediaPipe Tasks landmarkers, pygame and Qt;
- `test_container_marker_detection` — renders complete five-marker boards and
  checks board-level PnP through `BoardTracker`;
- `test_container_screen_capture_source` — captures one frame through `mss`;
- `test_container_qt_ui_lifecycle` — creates the real `MainWindow` and checks
  Start/Stop/close behaviour.

The hardware-free end-to-end tests always run and cover:

- driver STATE/STATUS/HAPTIC round trips against a fake UDP driver endpoint;
- phone IMU ingestion, screen-rotation bit packing, time sync and HMD
  orientation updates;
- the WebSocket video server, welcome/recenter messages, token rejection,
  codec/keyframe/config flags, phone-requested JPEG fallback and client
  accounting;
- the encoder helpers: MPEG-TS and IVF demuxing, parameter-set extraction,
  keyframe detection, NVENC/CPU codec selection with and without a GPU, and a
  real `libx264` encode whose Annex-B access units are demuxed and validated;
- camera FourCC-first mode negotiation, fallback ordering and manual controls;
- the settings dialog round-trip for camera controls, board geometry and object
  offsets, calibration defaults, tracking inference size and MediaPipe
  delegate/model choices, streaming codec/bitrate/keyframes, lens geometry and
  OpenXR options;
- a complete side-by-side stereo game stream, including a check that the two
  eye views differ and that consecutive frames animate;
- `test_openxr_runtime` — the real desktop runtime, synthetic webcam and Xbox
  pads, Monado's `openxr-simple-example` and the capture shim in one process
  tree, asserting boards, tracked body, moving 6DoF poses, grabs, haptics
  routed to the synthetic pads and distinct stereo eye textures.

The OpenXR test needs the pinned runtime. Build it on the host; CI builds it
once and passes the artifact to the desktop and emulator jobs.
`scripts/run-e2e.sh` mounts `out/openxr` (or `$OVRC_OPENXR_ROOT`) at
`/opt/openxr` when it exists, otherwise the test skips with a clear reason.

## Android container suite

`docker/android.Dockerfile` starts from Gradle 9.9.1/JDK 17 and installs
Android SDK Platform 37.0 and Build-Tools 37.0.0. The default entry point runs
`testDebugUnitTest assembleDebug`; the JVM tests validate the Kotlin wire
encoder against the layouts in [`PROTOCOL.md`](PROTOCOL.md). Gradle caches live
in the named volume `openvrcardboard-gradle-cache`, so repeated runs are faster.

```bash
scripts/run-e2e.sh android
scripts/run-e2e.sh android testDebugUnitTest
```

The debug APK is left in `android/app/build/outputs/apk/debug/`.

## Android emulator OpenXR E2E

This is the recorded end-to-end run: a real Android emulator displays the live
stereo output of Monado's `openxr-simple-example`, driven by the real desktop
app. It is not a mock and does not use a prerecorded video.

First build the pinned OpenXR stack (Monado with the remote driver and null
compositor, the Khronos loader, the example and the capture shim):

```bash
scripts/build-openxr.sh                 # installs under out/openxr
scripts/run-e2e.sh emulator
OVRC_E2E_DURATION=25 scripts/run-e2e.sh emulator
```

`scripts/run-e2e.sh` mounts `out/openxr` at `/opt/openxr` and sets
`OVRC_OPENXR_ROOT`; set that variable to use a different install tree. The
emulator target exits with a clear error when the runtime is missing.

`docker/emulator.Dockerfile` extends the desktop image with JDK 17, Android
SDK platform-tools, the emulator, ffmpeg and the `android-34;google_apis;x86_64`
system image. The runner passes `/dev/kvm` into the container. Without KVM the
target exits with a clear error; use `desktop` and `android` separately instead.

The harness (`desktop/e2e/emulator.py`) performs these steps:

1. creates or reuses an AVD, boots the emulator headless with software GL and
   waits for `sys.boot_completed`;
2. installs the debug APK and clears its preferences;
3. writes a temporary desktop configuration with the synthetic webcam
   (animated human + five-marker boards 0/5/10), board-driven 6DoF head and
   controllers, body tracking, synthetic Xbox pads and
   `phone.capture_source = openxr`. The OpenXR capture shim receives native
   1200×1080 per eye at 90 fps; the phone stream uses CPU `libx264` H.264 at
   600×540 per eye / 30 fps by default so the real MediaCodec decode path and
   the no-GPU encoder fallback are both covered in CI, without starving the
   OpenXR example or body tracking;
4. starts Xvfb and the real Qt desktop app with `--video --status-json`, then
   waits for the WebSocket server and the OpenXR capture listener;
5. starts Monado and the shimmed `openxr-simple-example`, waits until the app
   receives stereo frames and the desktop reports boards plus a tracked body,
   then places the app and the example preview side by side;
6. starts 60 FPS `ffmpeg` recording of the desktop display and
   `adb shell screenrecord` on the emulator;
7. launches the phone app through an intent with the emulator's host address
   (`10.0.2.2`), the ports, the token and `autostart=true`;
8. waits until logcat reports at least 60 displayed video frames and a started
   MediaCodec H.264 decoder while sampling `--status-json` for capture frames,
   remote packets, board IDs, body poses, grab values, haptics and composed
   head/controller motion;
9. takes two screenshots 2.5 s apart, checks that the scene animates and that
   the left/right halves of the phone image show distinct eye views;
10. pulls the emulator recording, stops the desktop recording, combines both
    into `combined-session.mp4` and verifies the recordings with `ffprobe`.

The test fails unless both synthetic triggers cross the example's grab
threshold, the example's haptic requests reach the synthetic Xbox pads, all
three boards and the body skeleton are tracked, every composed pose moves by
at least 2 cm during the recorded session, and the phone reports hardware
H.264 decoding.

The emulator stream runs through the real Android video client and
`HardwareVideoDecoder`, so CI exercises the same Annex-B/MediaCodec path that a
phone uses with NVENC. `summary.json` records `phone.hardware_decode_seen` plus
the measured decode/display means and maxima under `phone`. The AVD panel is
60 Hz and decodes H.264 on emulated x86 cores, so those numbers measure the
pipeline rather than phone hardware. The client requests the highest refresh
rate the panel reports and drops stale frames instead of queueing them;
sustained 90 FPS at the full 2400×1080 geometry still needs to be confirmed on
a real 90 Hz phone over 5 GHz Wi-Fi. Set `OVRC_E2E_PHONE_CODEC=jpeg`,
`OVRC_E2E_PHONE_WIDTH=1200` and `OVRC_E2E_PHONE_HEIGHT=1080` to run the JPEG
path instead.

Artifacts are written to `artifacts/emulator-e2e/`:

| File | Contents |
| --- | --- |
| `combined-session.mp4` | Desktop app + OpenXR preview and emulator, side by side |
| `desktop-app.mp4` | Desktop display recording (app UI, synthetic camera, OpenXR preview) |
| `android-emulator.mp4` | Emulator screen recording of the stereo OpenXR view |
| `android-screenshot-1/2.png` | Two live phone screenshots |
| `desktop-app-frame.png`, `android-emulator-frame.png` | Extracted verification frames |
| `status.json` | Last `Runtime.diagnostics()` snapshot sampled by the harness |
| `openxr/monado-service.log`, `openxr/openxr-example.log` | OpenXR stack logs |
| `logcat.txt`, `phone-log.txt` | Full and app-filtered Android logs |
| `desktop-app.log`, `emulator.log`, `screenrecord.log` | Process logs |
| `summary.json` | Frame counts, screenshot metrics, OpenXR evidence, video probes, pass/fail |
| `junit.xml` | Pytest JUnit report (Docker/CI runs) |

Environment variables:

| Variable | Meaning |
| --- | --- |
| `OVRC_E2E_DURATION` | Recorded seconds (default `18`) |
| `OVRC_E2E_MIN_FRAMES` | Displayed frame threshold (default `60`) |
| `OVRC_E2E_PHONE_CODEC` | Phone stream codec (default `h264`) |
| `OVRC_E2E_PHONE_ENCODER` | Phone encoder backend (default `x264`) |
| `OVRC_E2E_PHONE_WIDTH` / `HEIGHT` / `FPS` | Per-eye stream geometry (default `600×540`, 30 fps) |
| `OVRC_E2E_ARTIFACTS` | Output directory (default `artifacts/emulator-e2e`) |
| `OVRC_EMULATOR_ATTACH=1` | Use an already running emulator instead of booting one |
| `OVRC_KEEP_EMULATOR=1` | Leave a harness-started emulator running |
| `OVRC_OPENXR_ROOT` | OpenXR install tree (default `out/openxr`) |

The standalone harness can also be run directly:

```bash
python3 scripts/emulator_e2e.py --artifacts artifacts/emulator-e2e
```

## CI and published artifacts

`.github/workflows/ci.yml` runs every layer on pushes and pull requests:

| Job | What it runs | Published artifact |
| --- | --- | --- |
| `openxr-runtime` | Pinned Monado + loader + example + shim build | `openxr-runtime` |
| `driver-linux` | CMake build + CTest | `driver-linux-x64` |
| `driver-windows` | Visual Studio build + CTest | `driver-windows-x64` |
| `desktop` | Full container Python suite with JUnit and OpenXR integration | `desktop-test-results` |
| `desktop-windows` | Qt/OpenCV/MediaPipe/pygame suite on Windows with an offscreen platform | `desktop-windows-test-results` |
| `android` | JVM tests + debug APK | `android-debug-apk`, `android-test-results` |
| `emulator-e2e` | Recorded Android OpenXR E2E on a KVM runner | `emulator-e2e-video` |
| `desktop-package` | Wheel and source distribution | `desktop-package` |
| `package-linux` | Frozen onedir app + driver + Monado runtime, tar.gz and AppImage | `package-linux` |
| `package-windows` | Frozen onedir app + driver, portable zip and Inno Setup `.exe` | `package-windows` |
| `demo-windows` | `openxr-simple-example` built with vcpkg and VS 2022 | `demo-windows` |

The OpenXR runtime is built on an Ubuntu 24.04 runner. The desktop container
and the emulator runner use the same release, so the runtime binaries see the
same C library and shared-library SONAMEs in every job.

The emulator job uses a GitHub-hosted Linux runner with KVM, the API 34
`google_apis` x86_64 image and the same harness as the Docker target
(`OVRC_EMULATOR_ATTACH=1`). It downloads the pinned OpenXR runtime artifact and
uploads the combined video, both individual recordings, screenshots, status
JSON, OpenXR logs, logcat and JUnit results even when the test fails.

The portable package jobs download the driver and OpenXR artifacts, build a
PyInstaller onedir folder, add `openvrcardboard-install` and (on Linux)
`openvrcardboard-openxr`, then smoke-test the frozen executables. The Linux job
also builds an AppImage when `appimagetool` is available. The Windows job wraps
the same folder in an Inno Setup installer and verifies that
`OpenVR-Cardboard-<version>-Setup.exe` was produced. These packages are
self-contained: they need no Python installation, and the Linux bundle includes
the pinned Monado runtime.

Every successful push to `main` (and every manual run) additionally runs the
`build-artifacts` job. It downloads every job artifact and publishes one
consolidated `openvrcardboard-build-<commit>` workflow artifact containing the
driver builds, wheel/sdist, portable tar/AppImage/zip, Windows setup executable,
APK, demo archive, test reports and videos. GitHub releases and tags are not
used; download the artifact from the **Actions** tab instead.

## Not covered by automation

These still need a real machine and are tracked as manual validation:

- loading the driver in SteamVR and checking that the HMD/controllers appear;
- webcam capture, board printing/lighting and tracking quality;
- phone IMU/video over Wi-Fi, including firewall and AP-isolation issues;
- Xbox gamepad mapping and rumble;
- hibernation, device replug and long-running stability;
- real Windows behaviour: the headless Windows job checks imports, the Qt
  lifecycle and packaging, but SteamVR, webcams, gamepads, the installer wizard
  and Android tooling still need a real Windows machine;
- sustained 90 FPS on real phone hardware and Wi-Fi (the emulator measures the
  pipeline, but its software renderer and host networking are not a phone).
