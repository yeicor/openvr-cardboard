# OpenVR Cardboard

> **Vibe-coded project.** Most of this codebase was written by an AI coding
> agent from high-level prompts, then reviewed and exercised through automated
> tests. It is a hobby project, not a commercial product. Expect rough edges,
> read the code before running it, and treat the MIT license as the only
> warranty.

OpenVR Cardboard turns a webcam, a phone and ordinary gamepads into a PC VR
setup for SteamVR and OpenXR:

- **6DoF webcam tracking** with printed five-marker boards for the headset and
  both hands. Each board is solved as one rigid object with a predictive
  error-state Kalman filter. Body tracking is MediaPipe-only and never claims
  marker IDs.
- **MediaPipe hands and body tracking** using the current Tasks API, with GPU
  inference when available and a CPU fallback.
- **Gamepad controllers** — one Xbox-style pad per hand drives a virtual OpenVR
  controller with sticks, triggers, grip, face buttons, D-pad and rumble.
- **Cardboard phone HMD** — an Android phone streams IMU orientation at sensor
  rate and displays stereo video from the PC. The desktop fuses phone rotation
  with board position for a 6DoF headset.
- **Phone camera mode** — the same app can upload its rear camera as 1080p120
  H.264, falling back through 720p120, 1080p90, 720p90 and down to whatever the
  camera accepts. The desktop uses it as the tracking camera while connected,
  releases the local webcam, and reopens it automatically after the phone
  stops. Each phone camera gets its own persisted chessboard profile. USB
  tethering is recommended for the cleanest link.
- **Native SteamVR driver and Monado OpenXR runtime** — the C++17 driver
  creates a virtual HMD, controllers and generic trackers on Linux and Windows.
  On Linux the app also registers a Monado-based OpenXR runtime, so OpenXR
  applications run without SteamVR while Steam games keep using the native
  driver.
- **Guided chessboard calibration** available from the main window and the
  settings dialog whenever the camera moves or the lens is refocused, with
  separate per-camera profiles for phone camera mode.

The project is independent and unaffiliated with Valve, Google, or any headset
vendor. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces
fit together.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Windows 10/11 x64, or Linux x86_64 with X11 |
| VR runtime | SteamVR for Steam games; Monado is bundled for OpenXR on Linux |
| Webcam | 720p/30 up to 1080p120; the shared ladder falls back to the best supported mode |
| Python (source install) | 3.12 or newer |
| Gamepads (optional) | Two Xbox-style pads for two-handed controller emulation |
| Phone (optional) | Android 8.0+, rotation-vector sensor, same Wi-Fi or USB tethering |

The desktop app is tested with Python 3.12–3.14, OpenCV 5, MediaPipe 1.2 Tasks,
PySide6 6.10+ and pygame-ce 2.5.8. The Android app targets
API 37 with AGP 9.5.1 and Gradle 9.9.1.

## Components

| Path | What it is |
| --- | --- |
| `driver/` | Native OpenVR driver (C++17, CMake, vendored OpenVR headers) |
| `desktop/` | Python desktop app (PySide6 UI, OpenCV/MediaPipe tracking, gamepads, phone link) |
| `android/` | Kotlin companion app for phone IMU, stereo video and phone camera |
| `shim/` | OpenXR capture shim used to stream another application's eye textures |
| `docs/PROTOCOL.md` | Wire protocol shared by all components |
| `docs/ARCHITECTURE.md` | Coordinate frames, threads, fusion and lifecycle |
| `docs/DESIGN.md` | Board geometry, marker family choice, estimator and benchmark design |
| `docs/BENCHMARKS.md` | Measured tracking accuracy, latency and CPU numbers |
| `docs/SETUP.md` | Detailed installation, calibration, board and firewall guide |
| `docs/ANDROID.md` | Phone app build and configuration |
| `docs/TESTING.md` | Unit, container E2E, OpenXR emulator E2E and manual hardware testing |

## Downloads

Tagged builds are published under
[**GitHub Releases**](https://github.com/yeicor/openvr-cardboard/releases).
Each release includes the Windows setup executable, portable Windows and Linux
packages, the signed Android APK, the Python wheel/source distribution, the
standalone SteamVR drivers, a CycloneDX SBOM and `SHA256SUMS`. No Python
installation is required for the packaged builds.

Every push to `main` also produces an untagged CI artifact. Open the
**Actions** tab, select the latest successful **CI** run and download
`openvrcardboard-build-<commit>`:

| Release asset | Contents |
| --- | --- |
| `OpenVR-Cardboard-<version>-windows-x64-Setup.exe` | Windows installer (Inno Setup), app, frozen PyInstaller runtime, SteamVR driver |
| `OpenVR-Cardboard-<version>-windows-x64-portable.zip` | Portable Windows folder + installer executable |
| `OpenVR-Cardboard-<version>-linux-x86_64.tar.gz` | Portable Linux folder + installer executable + bundled demo |
| `OpenVR-Cardboard-<version>-linux-x86_64.AppImage` | Linux single-file app |
| `OpenVR-Cardboard-<version>-android.apk` | Signed Android phone app |
| `openvrcardboard-<version>-py3-none-any.whl` | Python wheel for source installs |
| `openxr-simple-example-<version>-windows-x64.zip` | Free/open-source OpenXR demo for Windows |
| `openvrcardboard-steamvr-driver-<version>-linux-x64.tar.gz` | Standalone SteamVR driver for Linux |
| `openvrcardboard-steamvr-driver-<version>-windows-x64.zip` | Standalone SteamVR driver for Windows |
| `sbom.cdx.json`, `SHA256SUMS` | Dependency inventory and SHA-256 checksums |

The Windows `.exe` is produced by GitHub Actions on `windows-latest`, not by a
local Linux build. The portable packages include the SteamVR driver compiled by
the matching CI job. Linux packages also include the pinned Monado OpenXR
runtime, the OpenXR loader, the capture shim and `openxr-simple-example`.
The Android APK is signed with the project release key so updates install over
each other; debug and release builds are not interchangeable.

The installer registers the driver in the detected SteamVR installation and
records every change so it can be undone:

```text
Windows:  OpenVR-Cardboard-<version>-Setup.exe
          (uninstall through Windows Settings > Apps)
Linux:    ./openvrcardboard-install
          ./openvrcardboard-install uninstall
          ./openvrcardboard-install status
          ./openvrcardboard-demo
AppImage: ./openvrcardboard-*.AppImage install
          ./openvrcardboard-*.AppImage demo
```

The portable installer backs up any previous `openvrcardboard` driver folder
and restores it on uninstall. It only touches the SteamVR `drivers` folder and,
on Linux, the user's OpenXR/Monado configuration.

## Quick start with a free OpenXR demo

The first application in this walkthrough is
[`openxr-simple-example`](https://gitlab.freedesktop.org/monado/demos/openxr-simple-example),
Monado's Boost-licensed OpenXR reference app. It renders a stereo field of
colourful cubes and tracked controller blocks, prints the OpenXR runtime, and
exercises grabbing and haptics. It is small enough to read and modify, runs on
Linux and Windows, and behaves like a normal standalone VR application.

Linux packages bundle the demo and the Monado runtime. Windows packages install
the SteamVR driver; take `demo-windows/openxr-simple-example-windows-x64.zip`
from the same CI build artifact and use SteamVR as the OpenXR runtime.

### 1. Prepare the VR runtime

- **Linux:** no SteamVR install is needed. The portable package carries the
  pinned Monado runtime, the OpenXR loader and the demo.
- **Windows:** install **SteamVR** (free from Steam) and start it once so it
  creates its configuration. In **SteamVR Settings → Developer**, set **SteamVR
  as the OpenXR runtime** if another runtime is selected.

### 2. Install OpenVR Cardboard

- **Windows:** run `OpenVR-Cardboard-<version>-Setup.exe`, keep **Install the
  SteamVR driver** selected, and finish the wizard. Start **OpenVR Cardboard**
  from the Start menu.
- **Linux portable:** extract `openvrcardboard.tar.gz`, run
  `./openvrcardboard-install`, then start `./openvrcardboard`.
- **AppImage:** make it executable, run `./openvrcardboard-*.AppImage install`,
  then run the AppImage.

The app opens a window with a camera preview and status badges. The driver
badge turns green when SteamVR answers on `127.0.0.1:39100`.

### 3. Print and mount the tracking boards

Each tracked device carries one rigid five-marker board: an 80 mm central
marker and four 40 mm peripheral markers at ±70 mm horizontally and ±26 mm
vertically, printed on a 200 × 112 mm plate. The fifteen markers use the
project's custom 3×3 ArUco dictionary (`DICT_3X3_15`): a 3×3 payload has much
larger modules than a 4×4 or 5×5 tag at the same printed size, which is what
keeps the 40 mm peripherals decodable at 2 m and under motion blur. All five
markers are solved together as one board, so the peripheral corners provide a
180 mm baseline while the centre keeps the board trackable when an edge is
occluded.

There are exactly fifteen codes: five for the HMD and five for each hand. Body
tracking uses MediaPipe and does not use marker IDs, so there are no body
markers to print.

Packaged builds ship a printable sheet at `markers/boards.html` inside the
installed folder (Windows: `C:\Program Files\OpenVR Cardboard\markers`). In the
app, use **Tools → Generate printable marker boards…** to write the sheet and
the per-marker PNGs to a folder. From a source checkout you can also run:

```bash
python scripts/generate_markers.py
```

Print the HTML at **100% / actual size on A4/Letter landscape**, not “fit to
page”, and check the printed 100 mm ruler with a real ruler. Cut along the
dashed line and mount the board flat on the device. The generator also writes
one PNG per marker and a `README.txt` with the exact geometry.

| IDs | Device |
| --- | --- |
| 0–4 | HMD / phone holder (central marker 0) |
| 5–9 | left controller (central marker 5) |
| 10–14 | right controller (central marker 10) |

Mount the board rigidly and keep its white border visible. Put the HMD board
above or below the lenses so it does not cover the phone screen. If a board is
not at the device origin, enter its offset and rotation in **Settings →
Tracking → Boards**; the app composes the object pose from the board pose.

### 4. Calibrate the camera (recommended)

1. Print a chessboard with **9×6 inner corners** and 25 mm squares. **Tools →
   Generate calibration board…** writes a printable HTML sheet on A4/Letter
   landscape at the configured size if you do not already have one.
2. In the app choose **Tools → Calibrate camera…** (or **Settings → Camera &
   markers → Calibrate camera…**).
3. Show the board at the centre, corners, close up and at an angle. The wizard
   captures automatically when the board is still and has moved to a new pose.
4. Collect at least 12 views, then choose **Calibrate and save…**.

The result is written to `camera_calibration.json` in the app configuration
directory and used immediately. Recalibrate whenever the camera moves, the
focus changes or the resolution changes. With phone camera mode connected, use
**Tools → Calibrate phone camera…** to store a separate profile keyed by the
reported phone and Camera2 ID in `phone_calibrations.json`. A command-line
workflow is also available:

```bash
python scripts/calibrate_camera.py --camera 0 --width 1920 --height 1080 \
  --cols 9 --rows 6 --square-mm 25 --output camera_calibration.json
```

### 5. Start tracking and the OpenXR bridge

1. Place the camera near eye height, 1.5–3 m from the play area, with the whole
   area in view. Avoid direct sunlight and strong backlighting.
2. In the app, press **Start tracking**.
3. Confirm the preview shows marker IDs and distances and the header badges
   report a camera frame rate.
4. **Linux:** open the **OpenXR** tab, enable **Run the bundled OpenXR runtime
   (Monado)** and **Start the Monado service with the desktop app**.
   **Windows:** leave SteamVR running; the native driver supplies the OpenXR
   devices through SteamVR.
5. Connect one Xbox-style gamepad per hand (optional). Windows usually detects
   them automatically; on Linux check `jstest /dev/input/js0`.

### 6. Run the demo

- **Linux portable:**
  ```bash
  ./openvrcardboard-demo
  ```
- **AppImage:**
  ```bash
  ./openvrcardboard-*.AppImage demo
  ```
- **Windows:** extract `openxr-simple-example-windows-x64.zip` from the CI
  build artifact, start SteamVR and the desktop app, then run
  `openxr-example.exe`.
- **Source build:** see [OpenXR demo](#openxr-demo) below.

The demo opens a desktop window with stereo output. Move your head and the
phone (or the HMD board) and the virtual view follows. The two controller
blocks move with the left and right boards or the gamepads. Squeeze the grip to
grab a cube; supported controllers receive the demo's haptic pulses.

If the demo reports no runtime or no headset, check that tracking is running,
the driver/OpenXR badge is green, and the active OpenXR runtime is Monado
(Linux) or SteamVR (Windows).

### 7. Add the phone (optional)

Build or install the Android APK from [`docs/ANDROID.md`](docs/ANDROID.md), put
the phone in a Cardboard-style holder with the HMD board (IDs 0–4) facing the
camera, and enter the desktop's LAN IP in the app. See **Phone headset** below.
To use the phone as the tracking camera instead of the webcam, choose **Camera**
mode in the app and set the desktop's camera port (default `8766`).

### 8. Use other applications

Any SteamVR title works on Windows and Linux once the driver badge is green.
On Linux, any OpenXR application also works through the bundled Monado runtime.
Start the application normally; use `openvrcardboard-openxr` (or
`./openvrcardboard-*.AppImage openxr`) only when you want its rendered eyes
streamed to the phone.

## Markers and tracking

### Boards and tracking settings

- **Dictionary** — `DICT_3X3_15` by default: a custom 3×3 family with exactly
  fifteen rotation-unique codes. The three boards consume all fifteen IDs.
  Predefined OpenCV dictionaries remain selectable for custom printed tags, but
  the selected dictionary must contain every configured ID.
- **Board geometry** — width, height, central and peripheral marker sizes, and
  peripheral offsets are editable in **Settings → Tracking → Boards**. The
  default board is 200 × 112 mm with 80 mm and 40 mm markers at ±70/±26 mm. The
  printed board must match these values, otherwise depth and orientation are
  wrong.
- **Object offset** — the board pose is converted to the device pose with
  `object_position` and `object_rotation`. The board origin is the centre of the
  large marker; it does not need to coincide with the device origin.
- **Detection resolution** — ArUco detection runs on the full capture frame.
  At 2 m and a 60° horizontal FOV the 40 mm peripheral markers are only about
  33 px wide, so downscaling costs those markers and the long baseline they
  provide.
- **Pose solve** — all detected corners from the board's markers go into one
  `solvePnP` / IPPE problem with robust outlier rejection. The app never solves
  five marker poses independently and averages them.
- **Estimator** — a 12-state error-state Kalman filter tracks position, linear
  velocity, orientation quaternion and angular velocity; linear and angular
  acceleration are derived from the corrections and smoothed rather than
  carried as noisy filter states. Measurement noise depends on reprojection
  error, board span, viewing angle and marker count; NIS gating rejects
  inconsistent frames.
- **Prediction** — the driver is the single prediction owner. The desktop
  publishes each pose at its measured timestamp and includes
  `tracking.pose_prediction_ms` (default 35 ms) as a horizon in the STATE
  packet; the driver applies that horizon through `poseTimeOffset`. No tracker
  or composer adds a hidden second lead.
- **Comfort presets** — **Settings → Tracking → Board tracking preset** offers
  `Comfort`, `Balanced` and `Responsive`, plus `Custom`. Comfort lowers the
  board filter noise (0.5 linear, 1.0 angular) and applies 6 Hz output
  smoothing; Responsive raises the noise to 2.0/4.0 for fast games. Prediction
  is the separate driver horizon above and is not changed by a preset.
- **Filter tuning** — `marker_process_noise` and `marker_angular_noise` control
  how quickly the board filter follows measurements. With
  `marker_adaptive_noise` enabled, the linear noise is scaled down to
  `marker_static_noise_scale` (default 0.2) while the board is still, which
  removes most resting jitter without adding lag during real motion.
- **Output smoothing** — `marker_output_smoothing_hz` (default 0, off) applies
  an optional One-Euro stage to the final board pose. It makes slow motion
  softer but adds latency; tune the filter itself first.
- **States** — each board independently reports `TRACKING`, `DEGRADED`,
  `PREDICTING`, `REACQUIRING` or `LOST`. Short occlusions are bridged; after
  `marker_hold_s` the pose is predicted, and after `marker_lost_s` the board is
  reported lost. Reacquisition blends over `marker_reacquire_frames` frames.
- **Occlusion** — a board remains usable with three of its five markers. The
  central marker alone gives a pose, but the app reports lower confidence and
  larger uncertainty.
- **Debug view** — enable the tracking overlay to see inlier corners (green),
  rejected corners (red), board axes, raw (yellow), filtered (cyan) and
  predicted (magenta) poses, velocity arrows, confidence, reprojection error and
  the per-board state.

### Comfort and motion sickness

Start with the `Comfort` preset. It trades a little responsiveness for a calmer
image: lower process noise removes resting jitter, and a small amount of output
smoothing softens residual motion. Prediction is the separate driver horizon
above, so it is not changed by the preset; lower it if the view overshoots
during direction changes. If the virtual world feels laggy instead, move to
`Balanced`, then `Responsive`. Every preset value can be edited individually,
so the preset becomes `Custom` as soon as a field changes.

Other settings that matter more than smoothing:

- Run the camera at 60 FPS when possible and keep exposure short; motion blur
  at 30 FPS is much harder to filter than corner noise. Marker-aware automatic
  exposure and gain are enabled by default and shorten exposure until adding
  more light would start dropping markers; the **Camera** settings expose the
  aggressiveness and target brightness, and disabling automation leaves the
  manual controls in charge.
- Fix the camera and the boards rigidly. A wobbling mount looks like head
  motion and cannot be corrected in software.
- Calibrate the camera after changing focus, resolution or mounting.
- Prefer a stable frame rate over a higher resolution. A steady 720p60 feed is
  usually more comfortable than an uneven 1080p30 one.
- If prediction overshoots during direction changes, lower
  `tracking.pose_prediction_ms` in 5–10 ms steps; `0` is only for debugging
  because it increases perceived latency.

### Hands and body

Enable **Settings → Tracking → Hands** or **Body**. The app downloads the
MediaPipe Tasks `.task` models into a per-user cache (or uses
`tracking.model_dir`). Inference runs on a downscaled copy
(`tracking.inference_width`, default 960) while markers keep the full frame.
`tracking.mediapipe_delegate` defaults to `auto`: GPU first, then CPU. Hands
work without markers; body tracking needs the player's full body in view.

### Camera placement

- Camera near eye height, 0.8–2 m from the play area, whole area visible.
- 1920×1080 is the reference tracking mode. The capture path starts at
  1080p120 and walks down the shared 720p120, 1080p90, 720p90, 1080p60,
  720p60, 1080p30 and 720p30 ladder; it requests the pixel format before the
  resolution because many USB webcams only expose high rates in MJPG.
- Manual focus, exposure and white balance are more stable than automatic
  control once the scene is lit.
- Mirroring is preview-only; tracking always uses the original frame.

## Phone headset

The Android app streams IMU orientation at sensor rate and displays the PC's
stereo video. For a phone-only HMD:

1. Mount the phone in a Cardboard-style holder and put the HMD board (IDs 0–4)
   on the holder facing the camera.
2. Start the desktop app and the phone video server.
3. Enter the desktop's LAN IP in the phone app and press **Start**.
4. Recenter after putting the headset on.

Defaults are 1200×1080 per eye (2400×1080 side-by-side) at 90 FPS. On a machine
with an NVIDIA GPU and ffmpeg, the desktop encodes HEVC (or AV1) with NVENC and
the phone decodes it in hardware; otherwise the always-available JPEG path
(quality 65) is used. Non-NVIDIA machines can use `phone.codec=h264` with
`phone.encoder=x264` for Android hardware decoding over the CPU encoder. Lower
the size or quality if the phone or Wi-Fi link cannot keep up.

Settings → Phone video selects the codec, encoder backend, bitrate and keyframe
interval. Settings → Lens & display carries the Cardboard lens separation,
eye-screen distance, FOV, distortion and overscan sent to the phone.

The phone is optional. A normal monitor, a TV or another SteamVR headset can be
used while the app supplies tracking and controllers.

### Phone camera mode

Choose **Camera** in the Android app instead of **Headset**. The phone uploads
its rear camera to `phone.camera_port` (default `8766`), targeting 1920×1080 at
120 FPS by default. Both cameras use the same fallback ladder:
1080p120, 720p120, 1080p90, 720p90, 1080p60, 720p60, 1080p30, 720p30, then
whatever the camera accepts. A manually configured target is tried first and
the ladder resumes at the closest supported entry. Only one camera is kept
open: the desktop releases the local camera while fresh phone frames arrive
and reopens it in the background after the phone stops, so expect a short gap
on fallback. Both directions have independent quality settings:
`phone.camera_bitrate_mbps` (default 40 Mbps) for the phone upload and
`phone.bitrate_mbps` (default 40 Mbps) for the headset video. USB tethering
with internet sharing is the recommended link; raise the camera bitrate to
60–80 Mbps on USB. The app reports its device and Camera2 ID so
**Tools → Calibrate phone camera…** can keep a separate intrinsics profile per
phone lens.

## Controllers and input

By default pad index 0 drives the left controller and pad index 1 the right.
Swap them in **Settings → Gamepad** if needed.

| Gamepad | Controller |
| --- | --- |
| Left stick | joystick / movement |
| Right stick | trackpad |
| Left/right trigger | analog trigger + click at threshold |
| Bumper | grip + grip click |
| A/B/X/Y | face buttons |
| D-pad | D-pad |
| Start/Back/Guide | system/menu/select |
| Stick clicks | joystick/trackpad click |

SteamVR rumble events are routed back to the matching pad when
`gamepad.rumble` is enabled. Hand pinches can drive trigger/grip when MediaPipe
hands are enabled.

## OpenXR

On Linux, the package registers the bundled Monado runtime as the user's active
OpenXR runtime and writes a Monado `remote` configuration. When
`openxr.enabled` is true — it is enabled automatically when
`phone.capture_source` is `openxr` — the desktop app starts `monado-service` and
stops it on exit. Other OpenXR applications then see the same tracked head and
controllers. `./openvrcardboard-demo` (or
`./openvrcardboard-*.AppImage demo`) runs the bundled `openxr-simple-example`
against that runtime.

On Windows there is no bundled runtime: install SteamVR, set it as the active
OpenXR runtime, and the native driver supplies the devices to both OpenXR and
SteamVR applications. Run `openxr-example.exe` from the packaged
`openxr-simple-example-windows-x64.zip`.

To stream another OpenXR application's rendered eyes to the phone, start the
app with `phone.capture_source` set to `openxr` and launch the application
through the wrapper:

```bash
# Linux portable
./openvrcardboard-openxr /path/to/openxr-application
# AppImage
./openvrcardboard-*.AppImage openxr /path/to/openxr-application
```

Capturing another application's eye textures is currently a Linux feature,
because it relies on the bundled runtime manifest. On Windows, phone video uses
the test pattern, a screen region, the webcam or a game capture window instead.

## Configuration

The configuration file lives at:

- Linux: `~/.config/openvrcardboard/config.json`
- Windows: `%APPDATA%\OpenVR Cardboard\config.json`

Use `openvrcardboard --print-config` to inspect the effective values and
`openvrcardboard --save-config` to write them. Most fields are editable in the
UI. Useful command-line options:

```bash
openvrcardboard                 # Qt window
openvrcardboard --headless      # terminal only
openvrcardboard --no-start      # window + driver/phone links, camera stopped
openvrcardboard --video         # start phone video immediately
openvrcardboard --self-test     # verify the frozen Qt/OpenCV/MediaPipe stack
openvrcardboard --log-level debug
```

The full configuration reference is in [`docs/SETUP.md`](docs/SETUP.md).

## Build from source

### Driver

```bash
cmake -S driver -B driver/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build driver/build -j4
ctest --test-dir driver/build --output-on-failure
```

On Windows use `-G "Visual Studio 17 2022" -A x64` and add `--config Release`
to the build and test commands.

The driver folder is assembled in `driver/build/openvrcardboard/`. Install it
with:

```bash
python3 scripts/install_driver.py
# or: python3 scripts/install_driver.py --steamvr "/path/to/SteamVR"
```

### Desktop

```bash
cd desktop
python3 -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[all]'
openvrcardboard
```

The extras are `ui` (PySide6), `tracking` (OpenCV contrib + MediaPipe),
`gamepad` (pygame-ce), `screen` (mss) and `video` (imageio-ffmpeg). `all`
installs everything. Without MediaPipe the board and gamepad paths still work;
without an NVIDIA GPU for encoding, the app falls back to JPEG or the CPU
encoder.

### OpenXR demo

`openxr-simple-example` is pinned in `scripts/build-openxr.sh` and built with
the rest of the OpenXR stack:

```bash
scripts/build-openxr.sh
```

Linux packages install the demo as `openvrcardboard-demo`. On Windows CI builds
it with CMake, SDL2 and the OpenXR loader from vcpkg:

```powershell
scripts\build_demo_windows.ps1
```

The resulting `out/openxr-simple-example-windows-x64.zip` contains
`openxr-example.exe`, SDL2, the OpenXR loader and the upstream Boost license.
It runs against SteamVR as the OpenXR runtime.

### Windows setup executable

CI builds the `.exe` only. To reproduce it locally on Windows:

```powershell
python -m pip install -e "desktop[all]" pyinstaller
python scripts\package.py --out dist --driver package-driver --skip-openxr `
  --no-build-driver --archive zip
choco install innosetup -y
scripts\build_windows_installer.ps1 -SourceDir dist\openvrcardboard
```

### Android

See [`docs/ANDROID.md`](docs/ANDROID.md). The short version:

```bash
scripts/run-e2e.sh android testDebugUnitTest assembleDebug
```

## Tests and CI

```bash
cd desktop
uv sync --locked --all-extras           # exact versions from uv.lock
uv run ruff check .                     # lint
uv run mypy                             # typed core modules
uv run pytest -q --cov                  # unit suite + coverage floor
```
```bash
# from the repository root, once:
uv run --directory desktop pre-commit install --config ../.pre-commit-config.yaml
```

```bash
ctest --test-dir driver/build --output-on-failure
scripts/build-openxr.sh                 # Monado + OpenXR loader + example + shim
scripts/run-e2e.sh desktop              # full Python suite in Docker
scripts/run-e2e.sh android              # Kotlin tests + debug APK
scripts/run-e2e.sh emulator             # recorded OpenXR emulator E2E (needs KVM)
```

CI runs lint, type checking, the coverage floor, the C++ driver tests, the
Linux and Windows desktop suites, Kotlin tests and the recorded emulator E2E.
Tag pushes additionally build a signed Android release and publish the release
assets listed above.

The desktop suite covers the wire protocol, pose fusion, board PnP and the
error-state filter, ArUco maths, calibration, MediaPipe wrappers, gamepads, the
Qt lifecycle, phone IMU/video, the OpenXR remote protocol and the hardware-free
end-to-end scenarios. The emulator job boots Android, installs the phone app,
runs the desktop app with a synthetic webcam and synthetic Xbox pads, drives
Monado's real `openxr-simple-example` through the capture shim, and checks board
tracking, body tracking, 6DoF head/controller motion, grabs and haptics. It
records the desktop and emulator screens and writes `combined-session.mp4` under
`artifacts/emulator-e2e/`.

CI builds the pinned OpenXR stack, runs the driver tests on Linux and Windows,
runs the desktop suite on Linux and Windows, runs the Android JVM tests and APK
build, runs the OpenXR emulator E2E, and produces the Linux tar/AppImage, the
Windows portable zip, the Windows setup executable and the Windows OpenXR demo
archive. Each successful run also publishes one consolidated
`openvrcardboard-build-<commit>` artifact. See
[`docs/TESTING.md`](docs/TESTING.md) for the matrix and artifact list.

## Troubleshooting

| Problem | Check |
| --- | --- |
| Driver never connects | Is the driver folder directly under `SteamVR/drivers/`? Does `bin/<platform>/driver_openvrcardboard.so/.dll` exist? Restart SteamVR. |
| `driver: waiting` forever | SteamVR is not running, the driver failed to load, or UDP `39100` is blocked by a local firewall. |
| No camera | Camera index, permissions, another app using it, `camera.backend`, MJPG support. |
| Boards detected but jittery | Calibrate intrinsics, improve lighting, use a rigid flat mount, verify the printed board geometry, then inspect the debug overlay for outlier corners. |
| Board distance is wrong | The printed board dimensions and marker sizes must match **Settings → Tracking → Boards**; measure the central marker and the 180 mm peripheral spacing. |
| Controllers point backwards | Set the board's `object_rotation` in **Settings → Tracking → Boards** to a half-turn about Y. |
| Hands/body unavailable | MediaPipe needs Python 3.12+ and the `tracking` extra; check `openvrcardboard --self-test`. |
| Gamepad not detected | pygame-ce installed? Correct index? `/dev/input` permissions? Reconnect the pad and wait 2 s. |
| Phone IMU not seen | PC IP, UDP `39101`, firewall, AP isolation, phone on the same subnet. |
| Phone video not seen | TCP `8765`, token mismatch, server started, capture source/test pattern. |
| Phone camera not used | TCP `8766`, token mismatch, **Camera** mode selected, camera permission granted, firewall allows the link. |
| Phone camera calibration unavailable | update the Android app and reconnect in **Camera** mode after frames start; older apps do not report a camera ID. |
| Video lag | Use 5 GHz or USB tethering, lower `phone.fps`, `width`/`height` and JPEG quality. |
| Virtual HMD conflicts with a real headset | Disable `driver.enable_virtual_hmd`. |
| SteamVR crashes after install | Remove `drivers/openvrcardboard`, restore the backup, and check that the driver architecture matches SteamVR (64-bit). |
| Portable app fails at startup | Run `openvrcardboard --self-test`; it reports which bundled stack failed to import. |
| OpenXR app sees no tracking | Is the desktop app running with `openxr.enabled`? Check `openvrcardboard-install status` and `monado-service.log`. |
| OpenXR app plays but the phone stays black | Launch it through `openvrcardboard-openxr`, set `phone.capture_source` to `openxr`, and check port `39102`. |

Logs:

- Desktop: the UI log panel, or `openvrcardboard --log-level debug`.
- Driver/SteamVR: `vrserver.txt` under the Steam installation or Steam logs.
- Android: `adb logcat | grep -i openvrcardboard`.

## Safety and privacy

- The driver link is loopback-only by default and carries no authentication.
  Keep `driver.udp_host` at `127.0.0.1` unless you understand the risk.
- Phone IMU, clock sync, video and camera frames are plaintext on your LAN or
  USB tether. Set `phone.token` to authenticate the IMU/clock packets, the
  video WebSocket and the camera WebSocket, and use a trusted network. The
  Android app requests `INTERNET`, `ACCESS_NETWORK_STATE`, `ACCESS_WIFI_STATE`,
  `CAMERA`, `ACCESS_LOCAL_NETWORK` and `WAKE_LOCK`; camera permission is
  requested only when you start Camera mode, and Android 17+ asks for the
  **Nearby devices** permission before the first LAN stream. It does not request
  microphone, location or storage permissions.
- Webcam frames stay on the machine unless you explicitly capture them or use
  the phone camera source. Phone camera frames are decoded in memory for
  tracking and are not recorded unless you start a recording.
- Test recordings contain synthetic scenes, not real camera or desktop data.

## License

MIT, except the vendored OpenVR header (BSD 3-Clause) and the bundled
`openxr-simple-example` demo (Boost Software License 1.0). OpenVR and SteamVR
are trademarks of Valve Corporation. Cardboard is a trademark of Google LLC.
This project is independent and unaffiliated with Valve, Google, or any headset
vendor.
