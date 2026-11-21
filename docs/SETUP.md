# Setup guide

This guide assumes a Linux or Windows PC with SteamVR and a webcam. The Android
phone app is optional; it needs a separate build step documented in
[`ANDROID.md`](ANDROID.md).

> The code and unit tests pass in this repository, but the full chain still
> needs validation on real hardware. Work through the checklist in order and
> keep the SteamVR log open when something does not appear.

## Portable packages

The CI-built Windows setup executable, Windows zip, Linux tar and AppImage
already contain the desktop application, every optional Python dependency, the
SteamVR driver and (on Linux) the pinned Monado OpenXR runtime. This avoids the
Python and CMake steps below.

- **Windows:** run `OpenVR-Cardboard-<version>-Setup.exe`. It installs under
  `Program Files\OpenVR Cardboard`, adds Start-menu and optional desktop
  shortcuts, and registers the SteamVR driver unless you clear the task.
  Uninstall through **Settings → Apps**; the bundled uninstaller removes the
  driver it registered.
- **Windows portable:** extract `openvrcardboard.zip`, run
  `openvrcardboard-install.exe`, then `openvrcardboard.exe`.
- **Linux portable:** extract `openvrcardboard.tar.gz`, run
  `./openvrcardboard-install`, then `./openvrcardboard`.
- **AppImage:** make it executable, run `./openvrcardboard-*.AppImage install`,
  then run the AppImage.

The portable installer auto-detects SteamVR, copies the bundled driver, backs
up any previous `openvrcardboard` driver folder, registers the bundled OpenXR
runtime and writes the Monado `remote` configuration.
`openvrcardboard-install status` shows what is installed.
`openvrcardboard-install uninstall` restores the previous driver and OpenXR
configuration and removes every file the installer added.

The installer only writes inside the SteamVR `drivers` folder and the user's
OpenXR/Monado configuration. It does not modify registry keys, environment
variables or system-wide directories.

## Checklist

If you use a portable package, skip steps 1–3 and continue with the markers.

1. Build the native driver.
2. Install it under SteamVR's `drivers/` directory and restart SteamVR.
3. Install the desktop app in a Python virtual environment.
4. Print markers and mount them on the headset/controllers.
5. Calibrate the camera (recommended) and optionally build the Android app.
6. Start the desktop app and confirm the driver/phone status.
7. Start SteamVR and check that the virtual devices appear.

## 1. Prerequisites

| Component | Requirement |
| --- | --- |
| SteamVR | Installed and able to start |
| Python | 3.12 or newer (tested through 3.14) |
| Build tools | CMake 3.16+, a C++17 compiler, Ninja or Visual Studio 2022 |
| Webcam | 720p/60 or 1080p/60, manual exposure recommended |
| Gamepads | Two Xbox-style pads for two-handed controller emulation |
| Phone | Android 8.0+, rotation-vector sensor, same Wi-Fi or USB tethering |

Linux packages (Debian/Ubuntu names):

```bash
sudo apt install build-essential cmake ninja-build python3-venv python3-pip \
  libgl1 libgles2 libegl1 libegl-mesa0 libgl1-mesa-dri libglib2.0-0 \
  libvulkan1 mesa-vulkan-drivers libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0
```

Add your user to the `video` and `input` groups if the camera or gamepads are
not accessible, then log out and back in:

```bash
sudo usermod -aG video,input "$USER"
```

## 2. Build the driver

Linux/macOS:

```bash
cmake -S driver -B driver/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build driver/build -j4
ctest --test-dir driver/build --output-on-failure
```

Windows (PowerShell):

```powershell
cmake -S driver -B driver/build -G "Visual Studio 17 2022" -A x64
cmake --build driver/build --config Release
```

Output layout:

```text
driver/build/openvrcardboard/
├── driver.vrdrivermanifest
├── resources/
└── bin/
    ├── linux64/driver_openvrcardboard.so
    └── win64/driver_openvrcardboard.dll      (on Windows)
```

## 3. Install the driver into SteamVR

Use the helper:

```bash
python3 scripts/install_driver.py
# or choose the SteamVR root explicitly
python3 scripts/install_driver.py --steamvr "$HOME/.steam/steam/steamapps/common/SteamVR"
```

It looks in the usual Steam library locations, copies the assembled driver to
`<SteamVR>/drivers/openvrcardboard`, and moves any previous driver folder to
`<SteamVR>/.openvrcardboard-backups/` before replacing it. Add `--no-build` to reuse
an existing build and `--dry-run` to see paths without changing anything.

Manual install is just a folder copy:

| OS | Destination |
| --- | --- |
| Linux | `~/.steam/steam/steamapps/common/SteamVR/drivers/openvrcardboard` (or the `~/.local/share/Steam` equivalent) |
| Windows | `C:\Program Files (x86)\Steam\steamapps\common\SteamVR\drivers\openvrcardboard` |

The folder must contain `driver.vrdrivermanifest` directly. Restart SteamVR
after installing. The driver announces itself as `openvrcardboard`; no entry in
`steamvr.vrsettings` is required.

### Using a real headset

The default configuration creates a virtual HMD, which is what you want for a
phone-only headset. If you already have a real HMD and only want tracked
controllers/body trackers, disable **Create a virtual HMD** in
**Settings → Driver** (`driver.enable_virtual_hmd = false`) and set
`driver.tracker_count` to the number of generic trackers you need.

## 4. Install the desktop app

```bash
cd desktop
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[all]'
```

Windows:

```powershell
cd desktop
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[all]"
```

The extras are:

| Extra | Packages | Needed for |
| --- | --- | --- |
| `ui` | PySide6 6.10+ | Desktop window (headless works without it) |
| `tracking` | opencv-contrib-python 5, mediapipe 1.2+ Tasks | Boards, hands, body |
| `gamepad` | pygame-ce | Xbox controller input |
| `screen` | mss | Capturing a desktop monitor for phone video |
| `video` | imageio-ffmpeg | Static ffmpeg with NVENC and libx264 encoding |
| `all` | everything above | Normal desktop install |

The desktop app requires OpenCV 4.7 or newer at runtime for
`cv2.aruco.ArucoDetector`; the declared dependency is OpenCV 5 contrib, which
also provides the calibration helpers. Hand and body tracking use the MediaPipe
Tasks `HandLandmarker` and `PoseLandmarker`, not the removed
`mediapipe.solutions` API. The `.task` models are downloaded on first use into a
per-user cache; set `tracking.model_dir` to use pre-downloaded files, or set
`OPENVRCARDBOARD_MODEL_DIR` for a shared cache.

Board tracking uses OpenCV's CPU ArUco detector at full capture resolution.
It needs no GPU runtime: a 4×4-marker 1080p frame takes about 2.5 ms on a
modern desktop CPU, leaving ample headroom for MediaPipe.

Hardware video streaming uses an `ffmpeg` executable with NVIDIA NVENC support.
`imageio-ffmpeg` ships a static one, so `[all]` and the portable packages work
without a system install; `OPENVRCARDBOARD_FFMPEG` overrides the path. Without
an NVIDIA GPU or ffmpeg the app streams JPEG and everything else keeps working.
Set `phone.codec=h264` and `phone.encoder=x264` to use the CPU H.264 encoder
instead (useful on non-NVIDIA machines and to exercise Android hardware
decoding in CI); `auto` still keeps JPEG for non-GPU machines.
`openvrcardboard --self-test` reports the selected codec and ffmpeg path.

If MediaPipe has no wheel for your Python version, install the `ui`, `gamepad`,
`screen` and `video` extras without `tracking`, or use a supported Python
version. Marker tracking still needs OpenCV; hand/body tracking will report that
MediaPipe is unavailable.

Run:

```bash
openvrcardboard                 # Qt window
openvrcardboard --headless      # terminal only
openvrcardboard --no-start      # window + driver/phone links, camera stopped
openvrcardboard --video         # start phone video immediately
```

The configuration file is created at:

- Linux: `~/.config/openvrcardboard/config.json`
- Windows: `%APPDATA%\OpenVR Cardboard\config.json`

Use `openvrcardboard --print-config` to inspect the effective values and
`openvrcardboard --save-config` to write them.

### OpenXR integration

The portable installer registers the bundled Monado runtime as the user's
active OpenXR runtime and writes a Monado `remote` configuration. When
`openxr.enabled` is true — it is enabled automatically when
`phone.capture_source` is `openxr` — the desktop app starts the bundled
`monado-service` and stops it on exit. Other OpenXR applications launched while
the desktop app is running therefore see the same tracked head and controllers.
Set `openxr.manage_service` to `false` if another service already owns the
runtime.

To send another OpenXR application's rendered eyes to the phone, launch it
through the packaged wrapper:

```bash
./openvrcardboard-openxr /path/to/openxr-application
# AppImage:
./openvrcardboard-*.AppImage openxr /path/to/openxr-application
```

The wrapper preloads the capture shim and points the OpenXR loader at the
bundled runtime. Set `phone.capture_source` to `openxr` on the desktop first.

The package also bundles Monado's Boost-licensed `openxr-simple-example` as a
ready-to-run smoke test:

```bash
./openvrcardboard-demo
# AppImage:
./openvrcardboard-*.AppImage demo
```

On Windows the demo is a separate CI-built ZIP and uses SteamVR as the OpenXR
runtime.

## 5. Boards and camera

### Generate and print a board

Each tracked device uses one rigid five-marker board: an 80 mm central marker
and four 40 mm markers at ±70 mm horizontally and ±26 mm vertically, giving a
200 × 112 mm plate with a 180 mm baseline. The fifteen codes come from the
project's custom 3×3 dictionary, whose larger modules survive distance and
motion blur better than a 4×4 or 5×5 tag at the same physical size. Packaged
builds include the same sheet in their `markers` folder. In the app, choose
**Tools → Generate printable marker boards…**; for a source checkout or custom
geometry:

```bash
python3 scripts/generate_markers.py
```

This writes `markers/boards.html`, one PNG per marker, and a `README.txt` with
the exact geometry. Print the HTML at **100% / actual size on A4/Letter
landscape** (disable “fit to page”), measure the printed 100 mm ruler, and cut
along the dashed line. The default dictionary is `DICT_3X3_15`.

Default IDs:

| IDs | Device |
| --- | --- |
| 0–4 | HMD / phone holder (central 0; 1–4 are top-left, bottom-left, top-right, bottom-right) |
| 5–9 | left controller (central 5) |
| 10–14 | right controller (central 10) |

Body tracking is MediaPipe-only and does not use marker IDs, so there are no
body markers to print.

`tracking.hmd_marker_id`, `left_marker_id` and `right_marker_id` set the first
ID of each board; the next four IDs belong to that board. IDs must be disjoint
and inside the dictionary's range. The role order in the printed board is
central, top-left, bottom-left, top-right, bottom-right.

If you print a different physical board, update **Settings → Tracking →
Boards** with the measured width, height, central and peripheral marker sizes
and the peripheral offsets. Measurement errors appear directly as depth and
orientation errors.

### Mounting

- Mount each board **flat and rigid**. A board that bends or wobbles produces a
  wobbling pose even with perfect corner detection.
- Matte paper or laser print; glossy tape can create reflections.
- Keep at least three markers visible, including the large central one. The
  board remains usable when one or two peripheral markers are occluded, but
  confidence drops and the reported uncertainty grows.
- Put the HMD board above or below the lenses so it does not cover the phone
  screen, and mount the controller boards on the top/front face.
- A board facing the camera has its printed face toward the camera. The default
  object rotation already accounts for this. If a device points the wrong way,
  change that board's `object_rotation` in Settings.
- If the board is not at the device origin, set `object_position` and
  `object_rotation`; the app composes the device pose from the calibrated
  board-to-object transform.

### Camera placement

- Put the camera near eye height, 0.8–2 m from the play area, with the whole
  area in view. The default pose is `(0, 1.5, −1.5)` looking at the origin.
- Avoid strong backlighting and direct sunlight on boards.
- Use 1920×1080 as the reference tracking mode. The app requests the pixel
  format before the resolution because many USB webcams only expose 1080p60 in
  MJPG. If the camera cannot deliver it, `camera.allow_mode_fallback` accepts
  the highest mode it reports and the status panel shows the negotiated size.
- On Linux, `MJPG` avoids many USB bandwidth limits. Set `camera.fourcc` to
  `MJPG`; use `camera.backend = "v4l2"` if auto-detection picks the wrong API.
- Set `camera.autofocus`, `auto_exposure`, `gain` and `auto_white_balance` in
  Settings → Camera & markers. Manual focus and exposure are more stable once
  the scene is lit.
- ArUco detection always runs at capture resolution by default
  (`camera.marker_detection_width = 0`). At 2 m and a 60° horizontal FOV the
  40 mm peripheral markers are only about 33 px wide; a reduced-width copy
  loses them and the long baseline they provide.
- MediaPipe inference runs on a downscaled copy (`tracking.inference_width`,
  default 960 px) while boards still use the full-resolution frame.
- List cameras with `v4l2-ctl --list-devices` (Linux) or the Camera app
  (Windows). Change `camera.index` if the wrong one opens.

If the camera looks mirrored, enable `camera.mirror_preview`. Mirroring is
preview-only; tracking still uses the original frame. If the preview is mirrored
but controller movement is reversed, the camera is physically facing the player
from behind; move it in front or adjust `camera.rotation`.

### Calibrate intrinsics

The fallback assumes a 60° horizontal FOV. A real calibration makes marker
distance, controller scale and hand depth much more accurate, and only takes a
minute.

1. Print a chessboard with **9×6 inner corners** (10×7 squares) and 25 mm
   squares on A4/Letter landscape. Any board works if you enter its real
   geometry.
2. In the app choose **Tools → Calibrate camera…** or **Settings → Camera &
   markers → Calibrate camera…**. The wizard opens the configured camera and
   shows a live preview.
3. Show the whole board at the centre, near the corners, close up and tilted.
   The wizard captures automatically when the board is still and has moved to a
   new pose. The progress bar shows accepted views.
4. Collect at least 12 views, then press **Calibrate and save…**. The result is
   written to the path in `camera.calibration_file` (by default
   `camera_calibration.json` in the app config directory) and is used
   immediately, without restarting tracking.

Recalibrate whenever the camera moves, the focus or zoom changes, or you change
the capture resolution. Calibration is available at any time; it does not
require restarting the app.

The same maths is available without the UI:

```bash
python3 scripts/calibrate_camera.py --camera 0 --width 1280 --height 720 \
  --cols 9 --rows 6 --square-mm 25 --output camera_calibration.json
```

Press SPACE for 15–20 clearly different views; q finishes. Copy the JSON to the
app config directory as `camera_calibration.json`, or set
`camera.calibration_file` to its absolute path. The app scales intrinsics to
the actual capture resolution automatically.

#### Calibrate the phone camera

Phone-camera intrinsics are stored per camera instead of in the local webcam
file, so switching phones never overwrites another device's profile. Start the
desktop with phone camera mode enabled, connect the Android app in **Camera**
mode, then choose **Tools → Calibrate phone camera…** while the phone is
streaming. The wizard uses the running phone receiver, reports the detected
camera name, and stores the result under `phone:<device>:<camera_id>` in
`camera.phone_calibration_file` (default `phone_calibrations.json` in the app
config directory). Tracking restarts automatically after the dialog closes and
picks up the new calibration without rebuilding the runtime.

A stored phone calibration takes precedence over the approximate horizontal
FOV reported by the app. Calibrate each phone and each rear lens separately;
the Android app reports the Camera2 ID so profiles remain distinct. Recalibrate
after changing the phone's focus, zoom or streamed resolution.

### Calibrate the board-to-object transform

Printing gives the board frame, but the tracked device origin is usually
elsewhere. Measure the board origin in the device frame (for example, the
controller grip origin or the HMD centre between the lenses) and enter the
translation in **Settings → Tracking → Boards → Object offset**. Enter the
rotation as a quaternion if the board is not parallel to the device axes. The
default `[0, 1, 0, 0]` is a half-turn about Y, which maps the board's printed
`+Z` face onto the OpenVR device's forward `−Z` axis.

The transform does not need a special calibration target: mount the board
rigidly, measure its position with a ruler from the device origin, and verify
by moving the device while watching the debug axes. A wrong offset appears as a
constant translation or rotation of the virtual controller, not as jitter.

### Hands and body

Enable **Settings → Tracking → Hands** or **Body**. Hand and body tracking use
the MediaPipe Tasks landmarkers and download their `.task` models on first use.
The hand model is fixed; the pose model can be `lite`, `full` (default) or
`heavy`. `tracking.mediapipe_delegate` defaults to `auto`: the GPU delegate is
tried first and inference falls back to CPU if the GPU delegate cannot start.

Inference runs on a downscaled copy (`tracking.inference_width`, default 960)
while markers keep the full-resolution frame. Hands work without markers.
Body tracking needs the player's full body in view, and the skeletons are
exposed as generic trackers for the roles in `tracking.tracker_roles`.

Pre-download models with `OPENVRCARDBOARD_MODEL_DIR` pointing at a directory of
`.task` files, or set `tracking.model_dir` in the configuration. Webcam frames
are processed locally and are not uploaded anywhere.

## 6. Gamepads

Connect one Xbox-style pad per hand. By default, pad index 0 drives the left
controller and index 1 the right. If the indices are reversed, swap
`gamepad.left_index` / `right_index` in Settings.

Linux notes:

- The kernel `xpad` driver is usually enough; `jstest /dev/input/js0` checks it.
- If pygame sees no pads, ensure `pygame` is installed in the same virtual
  environment and that the user can read `/dev/input/event*`.
- Bluetooth pads pair through the desktop's Bluetooth settings; reconnect after
  pairing.

Binding summary:

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

Rumble events from SteamVR are routed back to the matching pad when
`gamepad.rumble` is true. Set the trigger threshold (default 0.5) and stick
deadzone (default 0.12) in Settings.

## 7. Phone headset

Build and install the APK, then enter the desktop's LAN IP, IMU port (`39101`),
video port (`8765`) and optional token. See [`ANDROID.md`](ANDROID.md) for the
build and firewall details. Start the desktop **phone video** server first, or
enable `phone.stream_on_launch`.

For a phone-only HMD, mount the phone in a cardboard-style holder, put the HMD
board (IDs 0–4) on the holder facing the camera, and recenter after putting it
on. The desktop uses the board for position and the phone IMU for fast
rotation.

### Phone as the tracking camera

Camera mode in the Android app sends the rear camera to the desktop, so the
phone can replace the webcam while it is connected. In the app choose
**Camera** instead of **Headset**, set the same host, token and camera port
(default `8766`), and press **Start**. Only the active camera stays open: while
fresh phone frames arrive (within `phone.camera_timeout_s`, default 0.5 s) the
tracking loop uses them and releases the local camera, and when the phone stops
the desktop reopens its local camera in the background. There can be a short
tracking gap while that device opens; no desktop restart or extra pairing step
is needed.

The phone targets 1280×720 at up to 120 FPS, using Camera2 constrained
high-speed capture when the device exposes it, and retries with a standard
capture session if the HAL rejects the high-speed mode. On Android 17+ allow
the **Nearby devices** prompt on first start, otherwise LAN access is blocked.
Quality is the priority:
`phone.camera_bitrate_mbps` (default 40 Mbps) controls the phone upload, and
`phone.bitrate_mbps` (default 40 Mbps) controls the desktop-to-phone headset
stream. Raise either on USB tethering or a clean 5 GHz link. The phone reports
its horizontal FOV and camera ID during the handshake; a **Tools → Calibrate
phone camera…** profile takes precedence over the FOV estimate. Mount the phone
close to where the webcam was, because the configured camera position and
rotation still define the play-space origin.
The desktop decodes the upload with ffmpeg (ffmpeg 5.1 or newer); the `[all]`
install and the portable packages include it, and camera mode falls back to the
local webcam if ffmpeg is missing.

USB tethering is the recommended link for camera mode and for 720p120. Connect
the phone by USB, enable **USB tethering** and **USB tethering internet
sharing** on the phone, and use the tether interface's desktop address in the
app. It avoids Wi-Fi jitter and has ample bandwidth for the 40–80 Mbps camera
upload plus the headset stream. On Linux, `ip addr` shows the tether interface
(usually `usb0` or `enp0s20u1`); on Windows it appears as a new Ethernet
adapter. Keep the phone on the same subnet as that adapter, or use the address
the DHCP server hands to the desktop.

## 8. Configuration reference

Most fields are editable in the UI. The JSON file uses the same names grouped
by section. This release uses configuration version 7, which introduced the
custom 3×3 board dictionary and marker-aware camera control; an older
`config.json` is ignored and the app starts from defaults rather than migrating
older single-marker or body-marker settings.

| Section | Key defaults | Notes |
| --- | --- | --- |
| `driver` | `udp_host 127.0.0.1`, `udp_port 39100`, `state_rate_hz 250` | Keep loopback unless you understand the risk |
| | `enable_virtual_hmd true`, `controller_profile vive_controller` | Profiles: `vive_controller`, `knuckles`, `oculus_touch` |
| | `tracker_count 6`, `pose_timeout_ms 1000` | Generic trackers and tracking-loss timeout |
| `camera` | `index 0`, `1920×1080`, `60 fps`, `fourcc MJPG` | Capture and marker settings |
| | `allow_mode_fallback true`, `verify_mode true` | Accept the camera's best supported mode |
| | `autofocus true`, `auto_exposure true`, `auto_white_balance true` | Manual control when disabled |
| | `auto_marker_control true`, `camera_aggressiveness 0.5`, `camera_target_brightness 0.52`, `camera_max_gain 64` | Marker-aware exposure/gain loop; disable for fully manual control |
| | `camera_timestamp_source auto`, `camera_timestamp_offset_ms 0` | `POS_MSEC`/read timestamp source and exposure-start correction |
| | `horizontal_fov_deg 60`, `calibration_file camera_calibration.json` | Local-camera intrinsics fallback/override |
| | `phone_calibration_file phone_calibrations.json` | Per-phone-camera profiles keyed by device and Camera2 ID |
| | `position [0,1.5,-1.5]`, `rotation [0,1,0,0]` | Camera pose in play space |
| | `dictionary DICT_3X3_15`, `marker_detection_width 0` | Board dictionary and full-resolution detection |
| | `marker_refine_pose true`, `max_reprojection_error 4.0` | LM refinement and outlier rejection threshold |
| | `calibration_board_cols 9`, `calibration_board_rows 6`, `calibration_square_mm 25` | Chessboard inner corners and square size |
| | `calibration_min_views 12`, `calibration_file camera_calibration.json` | Wizard threshold and intrinsics path |
| `tracking` | `markers_enabled true`, `hands_enabled false`, `body_enabled true` | Tracking sources |
| | `hmd_source auto`, `left_source auto`, `right_source auto` | Source priority |
| | `hmd_marker_id 0`, `left_marker_id 5`, `right_marker_id 10` | First (central) ID of each board |
| | `boards.hmd/left/right.width_m 0.200`, `height_m 0.112` | Printed board outline |
| | `boards.*.central_marker_m 0.080`, `peripheral_marker_m 0.040` | Marker edge lengths |
| | `boards.*.peripheral_x_m 0.070`, `peripheral_y_m 0.026` | Peripheral marker offsets from the board origin |
| | `boards.*.object_position [0,0,0]`, `object_rotation [0,1,0,0]` | Board-to-object transform |
| | `tracker_roles [waist, chest, left_foot, right_foot, left_shoulder, right_shoulder]` | MediaPipe body joints exposed as trackers; no marker IDs |
| | `pose_prediction_ms 35`, `marker_hold_s 0.22`, `marker_lost_s 1.0` | Driver-owned prediction horizon and state timeouts |
| | `marker_reacquire_frames 3`, `marker_min_markers 3`, `marker_min_confidence 0.25` | Reacquisition and quality gates |
| | `marker_process_noise 1.0`, `marker_angular_noise 2.0`, `marker_exposure_s 0.005` | Filter tuning and rolling-shutter model |
| | `marker_adaptive_noise true`, `marker_static_noise_scale 0.2` | Quiet the filter while the board is still |
| | `marker_output_smoothing_hz 0` | Optional final One-Euro stage; 0 = off, adds latency when enabled |
| | `marker_lost_position_sigma_m 0.5`, `marker_lost_orientation_sigma_deg 30` | Uncertainty while lost |
| | `body_model full`, `mediapipe_delegate auto` | Pose model (`lite`, `full`, `heavy`) and GPU/CPU choice |
| | `model_dir ""`, `mediapipe_max_fps 30` | Task model cache and inference budget |
| | `inference_width 960`, `inference_height 0` | MediaPipe downscale; markers stay full-res |
| | `smoothing_min_cutoff 2.0`, `smoothing_beta 0.25` | One-Euro pose smoothing |
| | `orientation_cutoff 4.0` | Rotation filtering for MediaPipe poses |
| | `body_min_visibility 0.45`, `landmark_cutoff 6.0` | Body/hand landmark filtering |
| | `hand_inputs true`, `hand_trigger_pinch 0.35` | Pinch/grip controller input |
| | `pose_max_age_s 0.25`, `body_scale 1.0` | Freshness and body scaling |
| | `manual_hmd_position [0,1.65,0]`, `controller_offset_*` | Fallback poses |
| `phone` | `imu_port 39101`, `video_port 8765`, `token ""` | Phone link |
| | `camera_enabled true`, `camera_port 8766`, `camera_bitrate_mbps 40`, `camera_timeout_s 0.5` | Phone-as-webcam receiver, upload budget and local-camera fallback |
| | `layout sbs`, `width 1200`, `height 1080` | Per-eye size; 2400×1080 side-by-side |
| | `fps 90`, `codec auto`, `encoder auto`, `bitrate_mbps 40`, `keyframe_interval_s 1.0` | Video budget and NVENC/CPU encoder choice |
| | `jpeg_quality 65` | JPEG fallback and non-GPU machines |
| | `lens.lens_separation_mm 63`, `lens.eye_screen_distance_mm 45` | Cardboard lens alignment sent to the phone |
| | `lens.fov_deg 90`, `distortion_k1/k2 0`, `overscan 0` | Distortion/overscan sent to the phone |
| | `capture_source test`, `test_pattern sbs` | `test`, `screen`, `camera`, `game`, `openxr` |
| | `capture_monitor 1`, `camera_index 1` | Screen/camera sources |
| `openxr` | `enabled true`, `remote_port 4242`, `capture_port 39102` | Loopback Monado bridge |
| | `manage_service true`, `view_width 2400`, `view_height 1080`, `ipd_m 0.063` | Start/stop the bundled `monado-service` with the app |
| `gamepad` | `enabled true`, `left_index 0`, `right_index 1` | Pad assignment |
| | `deadzone 0.12`, `trigger_threshold 0.5`, `rumble true` | Input shaping |
| `ui` | `start_tracking_on_launch true`, `preview_fps 30`, `preview_width 560` | Window behaviour |

The board-to-object transform is configured per board, for example:

```json
{
  "tracking": {
    "boards": {
      "left": {
        "object_position": [0.0, -0.04, 0.0],
        "object_rotation": [0.0, 0.7071, 0.0, 0.7071]
      }
    }
  }
}
```

`object_position` is in the tracked object frame and `object_rotation` maps the
board frame into the object frame. The board origin is the centre of the large
marker; `+X` is right, `+Y` is up and `+Z` points out of the printed face.

### Tracking comfort

The **Board tracking preset** in **Settings → Tracking** sets three fields at
once. Selecting any value afterwards switches the preset to `Custom`.

| Preset | Process noise | Rotation noise | Output smoothing |
| --- | ---: | ---: | ---: |
| Comfort | 0.5 | 1.0 | 6 Hz |
| Balanced | 1.0 | 2.0 | off |
| Responsive | 2.0 | 4.0 | off |

Comfort is the recommended starting point for users prone to motion sickness:
it quiets the resting filter and adds a small amount of final smoothing.
Responsive is for fast games where a little overshoot is preferable to lag.
Prediction is separate: `tracking.pose_prediction_ms` (default 35 ms) is the
single horizon sent to the driver and is not changed by a preset. The board
filter itself should be tuned before adding output smoothing;
`marker_output_smoothing_hz` is deliberately off by default because it adds
latency.

`marker_adaptive_noise` keeps the configured process noise while the board
moves and scales the linear noise down to `marker_static_noise_scale` (default
0.2) when it is still. Set the scale to 1.0 to disable the reduction, or turn
the checkbox off when comparing fixed filter settings.

If the image is jittery, fix the mechanical setup first: rigid camera and board
mounts, short exposure, even lighting, a fresh calibration and a stable frame
rate matter more than any filter value. If the image instead feels laggy or
overshoots, lower `tracking.pose_prediction_ms` in 5–10 ms steps.

## 9. First run

1. Start SteamVR (it may show “headset not found”; the desktop has not sent a
   pose yet).
2. Start `openvrcardboard`.
3. Confirm the header badge says **driver: connected** in the UI, or the
   headless log shows packets.
4. The camera preview should show detected markers with IDs and distances.
5. Open SteamVR's **Devices → Manage Trackers** or the dashboard to verify the
   HMD/controllers appear and move.
6. If you use a phone, start the phone app and check the phone IMU rate and
   video client count.

To check the phone video pipeline before SteamVR, a webcam or markers are
working, set `phone.capture_source` to `game`, press **Start phone video** and
start the phone app. The phone shows a head-tracked stereo flight scene driven
by its orientation; tilting the phone changes the view.

Expected status examples:

```text
camera 59.7 fps 1920x1080 | boards 3 (0 TRACKING, 5 TRACKING, 10 TRACKING) |
hands no | body no | phone 192.168.1.20 118.4 Hz |
driver connected (15234 packets) | video 1 clients, 89.6 fps
```

## 10. Development tests

Unit tests run directly once the app and driver are built:

```bash
cd desktop && python3 -m pytest -q
ctest --test-dir driver/build --output-on-failure
```

End-to-end tests run in Docker and need no SteamVR, camera or phone:

```bash
scripts/build-openxr.sh              # Monado + loader + example + capture shim
scripts/run-e2e.sh desktop            # full Python suite + hardware-free E2E
scripts/run-e2e.sh android            # Kotlin tests + debug APK
scripts/run-e2e.sh emulator           # recorded OpenXR emulator E2E (needs KVM)
scripts/run-e2e.sh all
```

The desktop image bundles every optional dependency and a virtual X server, so
OpenCV/MediaPipe/pygame/mss and the Qt window can be exercised headlessly. The
emulator target boots Android, runs the real desktop app plus Monado's
`openxr-simple-example`, streams the live stereo eye views to the phone app and
writes both screen recordings to `artifacts/emulator-e2e/`. See
[`TESTING.md`](TESTING.md) for the test layers, artifact list and what remains
manual.

## 11. Troubleshooting

| Problem | Check |
| --- | --- |
| Driver never connects | Is the folder directly under `SteamVR/drivers/`? Does `bin/<platform>/driver_openvrcardboard.so/.dll` exist? Restart SteamVR. Check the SteamVR log for `openvrcardboard`. |
| `driver: waiting` forever | SteamVR not running, driver failed to load, or UDP `39100` blocked by a local firewall. |
| No camera | Index, permissions, another app using it, `camera.backend`, `MJPG` support. |
| Boards detected but jittery | Calibrate intrinsics, improve lighting, use a rigid flat mount, verify the printed board geometry, and check the debug overlay for outlier corners. |
| Board distance is wrong | The printed dimensions and marker sizes must match `tracking.boards.*`; measure the central marker and the 180 mm peripheral spacing. |
| Controllers point backwards | Change the board's `object_rotation` to a half-turn about Y. |
| Hands/body unavailable | MediaPipe needs Python 3.12+ and the `tracking` extra; check `openvrcardboard --self-test` and the model download log. |
| Gamepad not detected | pygame installed? Correct index? `/dev/input` permissions? Reconnect pad and wait 2 s. |
| Phone IMU not seen | PC IP, UDP `39101`, firewall, AP isolation, phone on same subnet. |
| Phone video not seen | TCP `8765`, token mismatch, server started, capture source/test pattern. |
| Phone camera not used | TCP `8766`, token mismatch, **Camera** mode selected, phone camera permission granted, firewall allows the tether/Wi-Fi interface. |
| Video lag | Use 5 GHz, lower `phone.fps`, `width/height` and JPEG quality. |
| Virtual HMD conflicts with a real headset | Disable `driver.enable_virtual_hmd`. |
| SteamVR crashes after install | Remove `drivers/openvrcardboard`, restore the backup, and check that the driver architecture matches SteamVR (64-bit). |
| Portable app fails at startup | Run `openvrcardboard --self-test`; it reports which bundled stack failed to import. |
| OpenXR app sees no tracking | Is the desktop app running with `openxr.enabled`? Check `openvrcardboard-install status` and the `monado-service.log` next to the app config. |
| OpenXR app plays but the phone stays black | Launch it through `openvrcardboard-openxr`, set `phone.capture_source` to `openxr`, and check that port `39102` is free. |

Logs:

- Desktop: the UI log panel, or `openvrcardboard --log-level debug`.
- Driver/SteamVR: the SteamVR log (`vrserver.txt` under the Steam installation
  or `~/.steam/steam/logs`).
- Android: `adb logcat | grep -i openvrcardboard`.

## 12. Uninstall

Portable package:

```bash
./openvrcardboard-install uninstall
./openvrcardboard-*.AppImage uninstall     # AppImage
openvrcardboard-install.exe uninstall        # Windows
```

This removes `<SteamVR>/drivers/openvrcardboard`, restores a previous driver
from `<SteamVR>/.openvrcardboard-backups/`, restores or deletes the active
OpenXR runtime and Monado configuration, and removes the integration state.
Restart SteamVR afterwards.

Source install:

1. Remove `<SteamVR>/drivers/openvrcardboard` (or restore a backup from
   `<SteamVR>/.openvrcardboard-backups/`).
2. Remove `~/.config/openxr/1/active_runtime.json` if it points at the bundled
   Monado runtime and restore your previous runtime.
3. Delete the Python virtual environment and, if desired, the config directory
   (`~/.config/openvrcardboard` or `%APPDATA%\OpenVR Cardboard`).
4. Uninstall the phone APK normally.
