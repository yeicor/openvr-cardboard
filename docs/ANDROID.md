# Android phone app

The Android app turns a phone into a wireless SteamVR headset or a phone
camera for the desktop:

- streams the rotation-vector, gyroscope and linear-acceleration sensors over
  UDP (default port `39101`),
- receives and displays the desktop's stereo video over WebSocket (default port
  `8765`), targeting 1200×1080 per eye (2400×1080 side-by-side) at 90 FPS.
  H.264/HEVC/AV1 are decoded by MediaCodec straight onto a display surface;
  JPEG remains the universal fallback with quality 65,
- reports the current display rotation so the desktop can compensate when the
  phone is turned 180° in its holder,
- in **Camera** mode, uploads the rear camera as H.264 to the desktop's
  tracking-camera socket (default port `8766`), targeting 1280×720 at up to
  120 FPS.

It is intentionally small: no analytics, no background service, no Google
account. The phone must stay on the same network as the desktop while in use;
USB tethering with internet sharing is the recommended link.

## Requirements

- Android 8.0 (API 26) or newer.
- A phone with `TYPE_ROTATION_VECTOR` (almost every phone made since 2015).
- Android Studio (recommended) or a command-line Android SDK with:
  - Android SDK Platform 37.0 and Build-Tools 37.0.0,
  - JDK 17.
- Wi-Fi with the phone and desktop on the same subnet, or USB tethering with
  internet sharing. 5 GHz or USB is strongly recommended because the video
  stream is bandwidth-heavy.

The project uses Android Gradle Plugin 9.5.1, Gradle 9.9.1 and AGP 9's built-in
Kotlin support, so there is no separate `org.jetbrains.kotlin.android` plugin to
install or version. `compileSdk`/`targetSdk` are 37, `minSdk` is 26, and the app
version is `1.1.0` (`versionCode 3`).

## Build

### Android Studio

1. Open the `android/` directory as a project.
2. Let Gradle sync and download the Android Gradle Plugin.
3. Build → Build Bundle(s) / APK(s) → Build APK(s).
4. The debug APK is `android/app/build/outputs/apk/debug/app-debug.apk`.

### Command line

The Gradle wrapper is committed for Gradle 9.9.1:

```bash
cd android
./gradlew assembleDebug
```

### Release signing

Android only installs an update when it is signed by the same key as the
installed app. Release APKs therefore use one fixed keystore that is stored
only in GitHub Actions secrets:

| Secret | Contents |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | base64 of the `.jks` keystore |
| `ANDROID_KEYSTORE_PASSWORD` | keystore password |
| `ANDROID_KEY_ALIAS` | release key alias |
| `ANDROID_KEY_PASSWORD` | key password |

Create the keystore once and back it up; losing it means existing installations
must be uninstalled before they can install a new signed build:

```bash
keytool -genkeypair -v -keystore release.jks -keyalg RSA -keysize 4096 \
  -validity 10000 -alias openvrcardboard
base64 -w0 release.jks   # paste into ANDROID_KEYSTORE_BASE64
```

The **Android signed release** workflow runs for tags and manual dispatches and
uploads `android-release-apk`. For a local signed build, point the four
`OVRC_ANDROID_*` environment variables at the keystore before running
`./gradlew assembleRelease`; without them the release APK is unsigned.

### Docker (no local Android SDK)

If Docker is available, the repository builds and tests the app in a pinned
image with JDK 17, Gradle 9.9.1 and Android SDK Platform 37.0 / Build-Tools
37.0.0:

```bash
../scripts/run-e2e.sh android
# or run specific Gradle tasks
../scripts/run-e2e.sh android testDebugUnitTest
```

The workspace is bind-mounted, so the debug APK appears at
`android/app/build/outputs/apk/debug/app-debug.apk` and the JVM protocol tests
run as part of the default `testDebugUnitTest assembleDebug` task list.

The same runner can execute the recorded Android emulator OpenXR E2E, which
installs the APK, starts the real desktop app and Monado's
`openxr-simple-example`, and records both screens:

```bash
../scripts/build-openxr.sh
../scripts/run-e2e.sh emulator
```

That target needs Docker, `/dev/kvm` and the OpenXR runtime; see
[`TESTING.md`](TESTING.md).

### Install it on a connected phone

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

If you build on Windows, use `gradlew.bat assembleDebug` instead.

## Configure

Start the desktop app before the phone app. The status panel shows both
`0.0.0.0`-bound services; use the desktop's LAN address, not `127.0.0.1`.

| Field | Default | Notes |
| --- | --- | --- |
| Mode | Headset | **Headset** displays stereo video; **Camera** uploads the rear camera |
| Desktop IP | empty | e.g. `192.168.1.20`; a port may be included |
| IMU port | `39101` | must match `phone.imu_port` on the desktop |
| Video port | `8765` | must match `phone.video_port` |
| Camera port | `8766` | shown in Camera mode; must match `phone.camera_port` |
| Sensor Hz | `120` | requested sensor period, 10–400 |
| Pairing token | empty | must match `phone.token` if one is set; authenticates IMU, clock pings, the video socket and the camera socket |

Tap **Start**. The app:

1. opens a UDP socket and starts the sensor listeners on a dedicated thread,
2. connects to `ws://<desktop>:<video-port>/video?token=…`,
3. sends a `hello` message with the phone model and screen size,
4. reads the desktop's `welcome` and configures either MediaCodec or the JPEG
   decoder for the announced codec,
5. displays the stereo video full-screen.

For H.264, HEVC and AV1 the phone prefers a hardware-accelerated `MediaCodec`
decoder and configures it for realtime priority, a 90 Hz operating rate and the
API 30+ low-latency mode. The decoder renders into the display `Surface`, so
decoded frames never touch a `Bitmap` and are composed directly by the display
pipeline. It is driven from one worker thread, keeps at most three queued
frames, drops late non-keyframes when it falls behind, and lets a keyframe reset
a full queue. MediaCodec presentation timestamps come from the desktop's sender
timestamp rather than local arrival time, so decode scheduling cannot reorder
frames. It extracts SPS/PPS/VPS from the keyframe when the desktop did not send
`csd`. If the phone has no decoder for the negotiated codec, or the decoder
rejects the stream, the app sends `video_fallback` and switches to JPEG without
reconnecting. The recorded CI emulator test uses CPU `libx264` H.264, so this
MediaCodec path is exercised without an NVIDIA GPU; on a desktop with NVENC the
same path receives HEVC/AV1 instead.

The JPEG path uses one decode worker with a latest-frame queue: if decoding
falls behind, older frames are dropped instead of building latency. Bitmaps are
reused through a small pool, decoded as `RGB_565`, and released back to the
renderer after drawing. The activity asks for the highest refresh rate the
display reports and logs decode and display FPS separately, so a 90 FPS stream
can be checked against the panel's actual refresh rate. Smaller per-eye sizes,
a lower JPEG quality or a hardware codec are the first controls if the phone
cannot keep up.

## Camera mode

Choose **Camera** in the mode selector, confirm the desktop IP, camera port and
token, then press **Start**. The app requests the camera permission on first
use, opens the rear camera and connects to
`ws://<desktop>:<camera-port>/camera?token=…`. It targets 1280×720, picks the
highest rate up to 120 FPS that both the camera and the H.264 encoder support,
and uses a Camera2 constrained high-speed session when the device only exposes
120 FPS there. The desktop's `camera_welcome` bitrate overrides the local
default, so quality can be tuned from the desktop settings without editing the
phone.

The encoder runs in CBR (falling back to VBR if the encoder rejects CBR), keeps
one keyframe per second and sends Annex-B access units. Only the newest frames
are queued; when the socket falls behind, the queue is cleared and the app asks
the encoder for a keyframe so the desktop decoder never receives a broken
reference chain. The status line shows the selected resolution, frame rate,
bitrate and sensor rotation.

Camera mode reuses the same host, token and preferences as headset mode. The
desktop keeps its local camera open and switches to the phone automatically
while fresh frames arrive. Stop the stream on the phone, or switch back to
**Headset**, to make the desktop fall back to its local camera.

The phone reports an approximate horizontal FOV from its focal length and
sensor size. Keep the phone near the webcam's position and orientation; the
configured `camera.position` and `camera.rotation` still define where the
tracking camera sits in play space.

For 720p120, use USB tethering with **USB tethering internet sharing** enabled
and enter the tether adapter's desktop address. Raise
`phone.camera_bitrate_mbps` (desktop **Settings → Phone**) to 60–80 Mbps when
the wired link allows it; 40 Mbps is the default. The headset stream has its
own `phone.bitrate_mbps` setting.

The settings panel hides 1.5 seconds after a clean start. Tap the video to show
it again; tap once more to hide it. The hardware back button stops streaming
and returns to the settings panel. A second back press leaves the app.

The display stays awake while the activity is in front. The requested
orientation is sensor landscape, so flipping the phone 180° changes
`Surface.ROTATION_*` between `1` and `3`; the desktop uses that value to keep
the view upright.

## Automated launch

The activity accepts intent extras so it can be driven by the emulator E2E or
by a phone automation script:

| Extra | Type | Effect |
| --- | --- | --- |
| `host` | string | Desktop address or WebSocket base URL |
| `mode` | string | `headset` (default) or `webcam` |
| `imu_port` | int | IMU UDP port |
| `video_port` | int | Video WebSocket port |
| `camera_port` | int | Phone camera WebSocket port |
| `sensor_hz` | int | Requested sensor rate, 10–400 |
| `token` | string | Pairing token |
| `autostart` | boolean | Start streaming 600 ms after the activity opens |

Values are saved to shared preferences before autostart. Example:

```bash
adb shell am start -S -n dev.openvrcardboard.phone/.MainActivity \
  --es host 192.168.1.20 --ei imu_port 39101 --ei video_port 8765 \
  --ei sensor_hz 60 --es token my-token --ez autostart true
```

Camera mode from an automation script:

```bash
adb shell am start -S -n dev.openvrcardboard.phone/.MainActivity \
  --es mode webcam --es host 192.168.42.1 --ei camera_port 8766 \
  --es token my-token --ez autostart true
```

Logcat uses the tag `OpenVRCardboard`. A healthy session logs the IMU setup,
`Video connected`, the stream dimensions, and periodic decode/display FPS and
`displayed N video frames` lines. The emulator harness uses those lines,
together with screenshots, to prove that the game is actually being displayed.

## Desktop-side checklist

1. `phone.video_enabled` is true and `phone.stream_on_launch` is true, or press
   **Start phone video** / use `--video`.
2. Firewall rules allow inbound UDP `39101` and TCP `8765` on the private
   network.
3. The router does not isolate wireless clients (AP isolation).
4. `phone.capture_source` is set to `game`, `screen`, `camera`, `test` or
   `openxr` as desired. `game` streams the built-in head-tracked stereo flight
   scene and is the easiest way to check the whole chain without SteamVR or a
   webcam. `openxr` forwards the eye textures of an OpenXR application launched
   through the packaged `openvrcardboard-openxr` wrapper. The test pattern is
   useful for diagnosing the link itself.
5. `phone.width`/`height` describe one eye for stereo, not the full panel. The
   defaults produce a 2400×1080 side-by-side frame at 90 FPS.

The desktop logs the phone's first `hello` and the video client count. If the
status stays at *waiting for phone*, check the IP and UDP port first; if video
connects but no frame appears, check TCP `8765` and the selected capture source.

## Notes and limitations

- Traffic is plaintext WebSocket and UDP. Use a trusted LAN, and set
  `phone.token` if other devices share it. TLS is not implemented because
  self-signed certificates on a phone require extra trust-store work; add a
  reverse proxy with a real certificate if the stream crosses an untrusted
  network.
- The phone contributes orientation only. Position comes from markers, hands
  or the fixed manual HMD position on the desktop. This is why a marker on the
  headset gives a much better experience than phone IMU alone.
- The app sends sensor timestamps from `SystemClock.elapsedRealtimeNanos()`.
  Gyroscope and accelerometer samples are paired by `SensorEvent.timestamp`,
  and the rotation-vector event timestamp is used for the packet, so all three
  streams share one sensor clock. The desktop estimates the phone-to-desktop
  offset and reorders one sample before rejecting stale or duplicate data; the
  `TIME_PING` exchange remains available for diagnostics and offset candidates.
- The APK requests `INTERNET`, `ACCESS_NETWORK_STATE`, `ACCESS_WIFI_STATE`,
  `CAMERA` and `WAKE_LOCK`. The camera permission is requested only when you
  start **Camera** mode; the app does not request microphone, location or
  storage permissions. Camera frames are uploaded only after you press Start
  and are not recorded by the app.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `No rotation-vector sensor on this phone` | phone lacks the sensor or it is disabled in a custom ROM |
| `UDP setup failed` | malformed host/port or no network route |
| `Video link failed` | wrong IP/port, desktop firewall, AP isolation |
| Video shows but is very dark | headset brightness / capture source; check the test pattern first |
| Tracking drifts sideways | phone IMU is orientation-only; add a marker or recenter |
| View is rotated | current display rotation is sent live; make sure the phone is not locked in a fixed orientation |
| Camera mode says permission denied | grant the camera permission in Android app settings, then Start again |
| Camera stream connects but the desktop still uses the local camera | check TCP `8766`, token, Camera mode selected; the desktop falls back after `phone.camera_timeout_s` without fresh frames |
| Camera mode drops frames | use USB tethering, lower `phone.camera_bitrate_mbps`, or accept a lower camera frame rate; check `frames_skipped` in the desktop status |
| Decode/display FPS below 90 | lower `phone.width`/`height`, lower `jpeg_quality`, use 5 GHz Wi-Fi, check that the panel actually offers a 90 Hz mode, and let `phone.codec=auto` use the hardware decoder |
| `Switching to JPEG` message | the phone has no decoder for the negotiated codec or its decoder rejected the stream; this is expected on older devices and reduces bandwidth/quality rather than breaking video |
