# Architecture

This document describes how the three pieces fit together, which coordinate
frames are used, and where each pose comes from. Wire formats are specified in
[`PROTOCOL.md`](PROTOCOL.md); installation is in [`SETUP.md`](SETUP.md).

## 1. Process model

```text
┌────────────────────────────── Desktop machine ──────────────────────────────┐
│                                                                             │
│  SteamVR (vrserver)                          OpenVR Cardboard desktop app          │
│  ┌────────────────────────────┐              ┌────────────────────────────┐ │
│  │ driver_openvrcardboard           │   UDP 39100  │ DriverLink (send/recv)     │ │
│  │  ├─ virtual HMD + display  │◀────────────▶│ FusionWorker @ 250 Hz      │ │
│  │  ├─ left/right controllers │   state /    │  ├─ PoseComposer           │ │
│  │  ├─ generic trackers       │   haptics /  │  └─ PhonePoseEstimator     │ │
│  │  └─ pose timeout           │   status     │                            │ │
│  └────────────────────────────┘              │ TrackingWorker             │ │
│                                              │  ├─ CameraCapture          │ │
│  ┌────────────────────────────┐              │  ├─ BoardTracker (ArUco)  │ │
│  │ PySide6 window / CLI       │              │  └─ MediaPipe hands/body   │ │
│  └────────────────────────────┘              │ GamepadManager @ 120 Hz    │ │
│                                              │ PhoneImuReceiver :39101    │ │
│                                              │ VideoStreamServer :8765    │ │
│                                              └──────────────┬─────────────┘ │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │ Wi-Fi
                                              ┌───────────────▼─────────────┐
                                              │ Android phone               │
                                              │  IMU → UDP :39101           │
                                              │  WebSocket client :8765     │
                                              └─────────────────────────────┘
```

The driver is loaded **inside** `vrserver`. It is not a separate process and
does not open a window. The desktop app can run headless; the Qt window is just
another consumer of `Runtime.status()` and the preview image.

When `openxr.enabled` is set, the desktop app also runs an OpenXR bridge. If
`openxr.manage_service` is true, it starts the bundled `monado-service` for the
lifetime of the app (and leaves an already-running service untouched), so other
OpenXR applications see the registered runtime and the tracked devices. The
bridge connects to Monado's remote driver over loopback TCP, sends the composed
head/controller poses as 376-byte `r_remote_data` packets, and listens for the
capture shim inside the OpenXR application. The shim intercepts the application
at `xrEndFrame`, reads both eye textures, encodes a side-by-side JPEG and
reports grab/haptic events. The `openxr` phone video source forwards those
frames, so the phone displays the real OpenXR application rather than a
duplicated mono image.

## 2. Threads and ownership

| Thread | Owner | Responsibility |
| --- | --- | --- |
| Qt main | `MainWindow` | 4 Hz status refresh, preview repaint, settings |
| `openvrcardboard-camera` | `CameraCapture` | Read newest webcam frame, retry on failure |
| `openvrcardboard-tracking` | `TrackingWorker` | Run markers/hands/body on the newest frame, draw overlay |
| `openvrcardboard-gamepad` | `GamepadManager` | Poll pygame pads, scan for hot-plug every 2 s |
| `openvrcardboard-phone-imu` | `PhoneImuReceiver` | Receive IMU, order packets, answer clock pings |
| `openvrcardboard-phone-camera` | `PhoneCameraReceiver` | Authenticate the phone camera socket, accept H.264 frames |
| `openvrcardboard-camera-h264-*` | `_H264Decoder` | Decode Annex-B H.264 with ffmpeg, publish the newest BGR frame |
| `openvrcardboard-fusion` | `FusionWorker` | Compose and publish frames at `driver.state_rate_hz` |
| `openvrcardboard-send` | `DriverLink` | Send the newest frame over UDP |
| `openvrcardboard-recv` | `DriverLink` | Receive haptics/status from the driver |
| `openvrcardboard-video-server` | `VideoStreamServer` | WebSocket accept loop |
| `openvrcardboard-video-capture` | `VideoStreamServer` | Capture + NVENC/CPU/JPEG encode + broadcast |
| `openvrcardboard-video-client` | `VideoStreamServer` | One per phone, drops stale frames |

Producers never block the fusion loop. `SharedState` keeps the newest marker,
hand, body, phone and gamepad snapshots under an `RLock`; the fusion worker
takes one consistent snapshot per frame. Camera and video each use a
latest-frame buffer rather than a queue, so a slow consumer drops frames instead
of building latency.

## 3. Coordinates

### OpenVR play space

Right-handed, metres: **+X right, +Y up, −Z forward**. HMD and controller
forward is local −Z; up is local +Y. All poses sent to the driver are in this
frame.

### Camera optical frame

OpenCV/ArUco uses **+X right, +Y down, +Z into the scene**. The conversion to an
OpenVR-style view frame is a 180° rotation about X:

```text
optical (x, y, z) ──Rx(180°)──▶ camera view (x, −y, −z)
```

The camera pose (`camera.position`, `camera.rotation`) maps the view frame into
play space. Defaults put the camera at `(0, 1.5, −1.5)` with a 180° Y rotation,
so it sits in front of the player and looks back at the origin. The mapping is:

```text
p_play = p_camera + R_camera · R_x(180°) · p_optical
q_play = q_camera · q_x(180°) · q_optical
```

A pinhole model with the configured `horizontal_fov_deg` is used when no
`camera_calibration.json` exists. A real calibration (intrinsics + distortion)
substantially improves marker depth and off-centre accuracy.

Phone frames use a separate, keyed calibration store. The Android handshake
reports `device` and the Camera2 `camera_id`; `CameraCalibrationStore` persists
one model per `phone:<device>:<camera_id>` in
`camera.phone_calibration_file`, merging entries atomically and moving a
corrupt file aside before rewriting it. `TrackingWorker` resolves intrinsics in
this order: stored phone calibration scaled to the streamed resolution,
reported `hfov_deg`, then the local camera model. The calibration wizard writes
through the runtime's shared store, so a new profile is picked up by the
tracking worker (which also uses the store revision in its model cache key)
without rebuilding the runtime.

### Board frame

A five-marker board is solved with one PnP problem over all its detected
corners. The board frame has its origin at the centre of the large marker,
**+X right, +Y up, +Z out of the printed face**. The peripheral markers have
known coordinates in this frame, so their corners add a 180 mm baseline without
requiring a separate solve. A board mounted on a device facing the camera has
its +Z pointing at the camera, while OpenVR device forward is −Z, so the default
object rotation is a half-turn about Y. Override it per board with
`tracking.boards.<name>.object_position` and `object_rotation`.

### Phone frame

Android device axes are already OpenVR-like: **+X right, +Y up the screen, +Z
out of the screen**. The Android world frame is ENU (east/north/up), while play
space is east/up/south (because −Z is forward), so the world frame is rotated
by −90° about X:

```text
q_play = R_x(−90°) · q_android
```

The current display rotation is applied around the screen normal (+Z):

```text
q_screen = q_play · R_z(screen_rotation)
q_relative = conj(q_reference) · q_screen
```

`screen_rotation` is the raw `Surface.ROTATION_*` value: 0, 1 (+90°), 2 (180°),
3 (−90°). This is why the Android app must send the **live** display rotation
rather than a constant: rotating the phone 180° in a headset holder changes the
value between 1 and 3.

## 4. Tracking pipeline

```text
webcam frame
   ├─ BoardTracker ─── full-res ArUco detection (OpenCV CPU)
   │                                → group corners by board
   │                                → one robust PnP solve per board
   │                                → object transform → optical→play
   │                                → 12-state ES-EKF + state machine
   ├─ HandTracker ───── MediaPipe HandLandmarker → 21 landmarks
   │                                → palm width → depth
   │                                → hand basis → play pose + pinch/grip
   └─ BodyTracker ───── MediaPipe PoseLandmarker → 33 landmarks
                                    → shoulder/hip width → depth
                                    → joints in play space + yaw rotations
                                             │
                                   SharedState snapshot
                                             │
                                   PoseComposer (fusion)
```

### Boards

`BoardTracker.detect()` converts the frame to grayscale and runs
`cv2.aruco.ArucoDetector` at capture resolution. Detected quads are grouped by
their known marker IDs into the configured boards. Each board contributes all
of its detected corners to one `solvePnPGeneric` problem; the planar IPPE
solutions are scored by reprojection RMS and per-marker consistency, and the
best solution is refined with Levenberg–Marquardt when
`camera.marker_refine_pose` is true. A marker whose corners disagree with the
rest of the board is rejected and the board is re-solved without it, so one
corrupt detection cannot drag the pose. A board needs at least
`marker_min_markers` markers (default 3) and a reprojection error below
`camera.max_reprojection_error` to produce a measurement.

The raw board pose is transformed into the tracked object frame, then into play
space, and passed to the per-board error-state Kalman filter. The filter keeps
position, linear velocity, orientation as a quaternion and angular velocity;
linear and angular acceleration are derived from the corrections and smoothed
rather than carried as noisy filter states. Between detections it predicts
forward with process noise, but no stage adds a hidden output lead: poses are
published at their measured timestamps and the driver applies the single
`tracking.pose_prediction_ms` horizon from the STATE packet. Measurement
covariance is based on the observed reprojection error, board span in pixels,
incidence angle and marker count, with conservative floors so a barely visible
board cannot become overconfident. Innovations are NIS-gated. When
`marker_adaptive_noise` is enabled, the linear process noise is reduced toward
`marker_static_noise_scale` while the board is still, which removes most
resting jitter without adding motion lag. Covariance updates are symmetrised
and eigenvalue-floored so a long run cannot produce a negative variance.

Each board runs its own state machine (`TRACKING`, `DEGRADED`, `PREDICTING`,
`REACQUIRING`, `LOST`). A single bad frame does not reset tracking; missing
markers reduce confidence and grow covariance. After `marker_hold_s` without a
measurement the board reports `PREDICTING`; after `marker_lost_s` it reports
`LOST` and the filter keeps a large but bounded uncertainty. Reacquisition
blends the new measurement over `marker_reacquire_frames` frames to avoid a
visible snap. The planar PnP ambiguity (two almost equally good faces) is
resolved by temporal continuity with the last accepted raw orientation; angular
velocity is not extrapolated through the ambiguity.

### Hands

MediaPipe Tasks returns normalised landmarks. Palm width in pixels and an
assumed palm width (8.5 cm) give depth. The wrist→middle-MCP vector defines
finger direction; the index→pinky vector defines the hand's across axis. These
are orthonormalised into a device frame, converted to play space, and mirrored
for the left hand. Pinch comes from thumb-tip/index-tip distance; grip is the
mean finger curl. Both are 0..1 and can drive trigger/grip.

### Body

Pose landmarks are projected to a common depth estimated from shoulder width
(40 cm) and hip width (32 cm). Joint positions are converted to play space and
scaled around the hips by `body_scale`. Yaw comes from the hip line (falling
back to the shoulder line) and from heel→toe for feet. Tracker roles map to
joints: `waist → hips`, `chest → chest`, `left_foot → left_foot`, and so on.

## 5. Source selection and fusion

`FusionWorker.step()` runs at 250 Hz, regardless of camera rate. That decouples
SteamVR's pose cadence from 30–60 fps inference and keeps the last pose alive
between camera frames.

HMD source (`tracking.hmd_source`):

| Mode | Behaviour |
| --- | --- |
| `auto` | board position + phone orientation → body+phone → phone → manual |
| `markers` | HMD board only |
| `phone` | phone IMU only at the manual/neutral position |
| `body` | MediaPipe head/neck |
| `manual` | configured fixed pose |

Controller source (`left_source` / `right_source`):

| Mode | Behaviour |
| --- | --- |
| `auto` | board → hand → body → synthetic (if never tracked) |
| `markers` | configured device board |
| `hands` | MediaPipe hand |
| `body` | body wrist |
| `synthetic` | fixed offset from `controller_offset_left/right` |

Freshness is bounded by `tracking.pose_max_age_s` (default 250 ms). When a
source disappears after being tracked, the composer sends the last pose with
`valid=false` rather than teleporting to the neutral point; the driver then
applies its own `pose_timeout_ms` and reports the device as searching. This
avoids visible snapping and lets SteamVR show a tracking-loss state.

Board-tracked devices arrive already filtered by their per-board error-state
Kalman filter, so `PoseComposer` does not run One-Euro smoothing on top of
them: the composer emits the measured-time pose and the driver applies the
single `tracking.pose_prediction_ms` horizon. Setting
`marker_output_smoothing_hz` above zero explicitly enables a second comfort
stage. Hand and body poses use One-Euro
(`smoothing_min_cutoff` default 2.0 Hz, `smoothing_beta` default 0.25 and
`orientation_cutoff` default 4.0 Hz) and their landmarks are filtered before
composition. Smoothing is cached by source timestamp, so the 250 Hz loop does
not repeatedly filter the same camera sample; that would add lag without
benefit. Phone orientation is used as measured; it also carries no private
prediction horizon.

Gamepad inputs are merged last and win over hand-derived inputs for the same
device: a physical trigger press is never overridden by a stale pinch estimate.

## 6. Phone pose estimator

`PhonePoseEstimator` receives `PhoneSample`s from `PhoneImuReceiver`. It:

1. normalises the quaternion and applies the world-frame and screen-rotation
   transforms above;
2. captures a reference orientation on the first sample or when recentered;
3. computes `q_relative = conj(q_reference) · q_screen`;
4. rotates the device-frame gyroscope by the inverse screen rotation so angular
   velocity is expressed in the phone's screen frame;
5. extrapolates the newest sample only to the requested consumer timestamp;
   the configured prediction horizon is the driver's job;
6. reports a fixed position (`manual_hmd_position`) and `orientation_only=true`.

When the HMD is also tracked by a board, `auto` keeps the board position and
uses the phone for fast rotation. `align_to_marker_orientation()` folds the
slow `q_marker · conj(q_phone)` yaw residual into the estimator's reference
with a low-pass, so the camera supplies absolute heading while the gyro keeps
the low-latency motion. If the board is lost, the last position is held and
the pose is marked orientation-only.

`PhoneImuReceiver` converts phone sensor timestamps into desktop monotonic time
using a robust minimum-delay clock offset, with `TIME_PING` samples as extra
candidates. A one-sample reorder buffer (20 ms flush deadline) absorbs Wi-Fi
reordering, and stale, duplicate or out-of-order samples are dropped. The
phone-side sender pairs gyroscope and accelerometer readings by
`SensorEvent.timestamp` and sends the rotation-vector event timestamp, so all
three streams share one sensor clock.

## 7. Driver

`DeviceProvider::RunFrame()` checks the UDP link, updates devices, and calls
`DeviceBase::CheckPoseTimeout()`. State packets carry up to 16 devices and 8
input states; the driver matches device IDs:

| ID | Device |
| --- | --- |
| 0 | HMD |
| 1 | left controller |
| 2 | right controller |
| 3+n | generic tracker n |

The HMD exposes `IVRDisplayComponent` with configurable render/window geometry
and refresh rate. Controllers select a profile from settings
(`vive_controller`, `knuckles`, `oculus_touch`) and expose the usual input
components; generic trackers expose tracker roles and input profiles from
`driver/resources/input/`. Haptic events are sent back to the desktop and
routed to the matching gamepad.

Every STATE packet carries a monotonic sequence and a `prediction_ms` horizon.
`DriverLink` rejects duplicate or out-of-order sequences from the same sender
using a signed 32-bit difference, so wraparound does not look like a stale
packet and a restarted desktop is accepted immediately. `DeviceBase` maps the
packet horizon to `poseTimeOffset` and never applies a second extrapolation;
the driver is the only component that turns measured poses into future ones.

If SteamVR is not running, the desktop keeps sending into the void; the
`driver: waiting` badge simply means no status packet has arrived. This is
harmless and makes startup order independent.

## 8. Video path

The desktop renders `phone.width × phone.height` **per eye** and composes:

- `mono`: `width × height`
- `sbs`: `(2·width) × height`
- `top_bottom`: `width × (2·height)`

The capture thread fits the source to one eye and duplicates it for stereo.
Screen and camera sources are only fitted and duplicated. The animated test
pattern draws an eye label and a moving sweep line for latency inspection. The
built-in `game` source is different: it renders a true stereo view for each eye
(a head-tracked flight scene with rings, stars and a HUD) using the phone
orientation and left-gamepad steering, so the two halves are not identical. The
`openxr` source does not render anything itself: it receives the left and right
eye textures captured from a running OpenXR application by
`shim/openxr_capture_shim.c`, which already emits a side-by-side JPEG.

The composed frame is passed to `encoder.FfmpegVideoEncoder` when
`phone.codec` resolves to H.264/HEVC/AV1 and ffmpeg has the matching encoder.
NVENC is preferred when an NVIDIA GPU is present; H.264 can also use the CPU
`libx264` encoder (selected explicitly with `phone.encoder=x264`, or with
`codec=h264` when NVENC H.264 is unavailable). ffmpeg runs as one long-lived
child process; raw frames go in on stdin and MPEG-TS (H.264/HEVC) or IVF (AV1)
comes back on stdout, where a small demuxer splits complete access units and
caches the latest keyframe. Without a usable encoder the capture thread uses
the always-available JPEG path, and a phone may request a mid-stream switch
with `video_fallback`. Frames are broadcast as:

```text
kind(1) layout(1) width(u16) height(u16) flags(u16)
sequence(u32) timestamp(f64) payload…
```

Flags carry the codec in bits 0–2, a keyframe bit and a configuration-packet
bit. JPEG, H.264/HEVC Annex-B and AV1 temporal units all share the same header,
so the Android `VideoClient` can route each frame to `BitmapFactory` or
`MediaCodec` from the header alone.

Each desktop client has a one-slot queue: if the phone is slower than the
desktop, the old frame is discarded and the new one is sent, which keeps
latency bounded at the cost of frame rate. The Android side adds a small
bounded hardware-decoder queue (three frames of slack) and drops late
non-keyframes; it submits MediaCodec presentation timestamps from the sender
timestamp rather than arrival time, and a keyframe may reset a full queue. The
JPEG path decodes to a `Bitmap` and draws it scaled to the panel; hardware
codecs render straight onto a `Surface`. A cardboard-style viewer splits the
left and right halves optically.

## 9. Failure and recovery

| Failure | Behaviour |
| --- | --- |
| Camera unplugged | Capture thread retries every second; fusion keeps last pose until timeout |
| OpenCV/MediaPipe missing | Tracker is disabled with a note; other sources keep working |
| Marker lost | Last pose marked invalid after `pose_max_age_s`; driver times out |
| Phone IMU gap | Estimator returns its last pose until `timeout_s` (1 s) |
| Phone video slow | Newest frame replaces the queued one; hardware decoder skips to the next keyframe |
| Encoder failure / no GPU | the encoder closes and the capture thread continues on the JPEG path |
| Phone lacks the codec decoder | Phone sends `video_fallback`; the desktop switches the session to JPEG |
| Phone disconnects | WebSocket reconnects every 2 s on the phone; desktop drops the client |
| Driver offline | Desktop keeps sending; haptics stop |
| Fusion exception | Logged, status note set, thread stays alive |
| Settings changed | Runtime is stopped, rebuilt from the new config and restarted if it was running; the tracking and manual video states are preserved |

## 10. Testing

The pure modules are deliberately free of Qt/OpenCV/socket dependencies where
possible. Tests cover quaternion math, camera projection/unprojection, marker
geometry, ArUco detection and pose refinement, chessboard calibration,
hand/body mapping, fusion source selection, estimator recentering, UDP framing,
gamepad mapping, video header composition, phone camera framing/auth/router
fallback, synthetic scene looping and runtime lifecycle. Hardware-specific paths
(SteamVR API, real camera, MediaPipe inference, pygame) are thin wrappers around
those tested helpers.

## 11. Extension points

- Add a tracking source: publish a `Pose` into `TrackingSnapshot` (or extend
  `SharedState`) and map a role in `PoseComposer`.
- Add a controller profile: add JSON resources under
  `driver/resources/input/` and select it in `driver.controller_profile`.
- Add a video codec: keep the 20-byte header, assign a codec number and add a
  `MediaCodec` MIME mapping plus a demuxer if the container differs. The
  JPEG/CPU path stays as the universal fallback.
- Add positional phone tracking: fuse the phone camera or a marker on the
  phone; the estimator's `orientation_only` flag and marker composition already
  support a position source.
