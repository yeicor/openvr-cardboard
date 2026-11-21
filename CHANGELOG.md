# Changelog

All notable changes to OpenVR Cardboard are documented here. The project
follows semantic versioning for the desktop package and the release tags.

## [1.2.1] - 2026-11-21

### Added

- Android 17 local-network permission handling. The app declares and requests
  `ACCESS_LOCAL_NETWORK` before headset or camera streaming, shows a denial in
  the status panel, and resumes the pending start after approval. Android 17
  blocks LAN TCP and UDP for apps targeting API 37 until this permission is
  granted, which made Wi-Fi and USB-tethering connections time out.

### Fixed

- Camera mode prefers a fixed high-speed FPS range such as `[120, 120]` over a
  variable range such as `[30, 120]`. Some Camera2 HALs advertise the variable
  range but reject it with `ERROR_CAMERA_DEVICE`.
- A rejected constrained high-speed camera session now falls back once to a
  standard capture session at the best supported rate instead of reconnecting
  to the same failing mode every two seconds.
- The camera log now records the selected mode, FPS range and whether it is a
  high-speed session, making device-specific failures easier to diagnose.

### Verified

- An Android 17 phone: 1280×720 H.264 camera upload at
  120 FPS and 40 Mbps over Wi-Fi with zero dropped frames, plus a 720p30
  fallback when the variable high-speed range was forced.
- The same phone streamed HEVC 2400×1080 at 90 FPS to the desktop with zero
  dropped frames while sending IMU samples and completing UDP clock sync over
  Wi-Fi, with and without a pairing token.

## [1.2.0] - 2026-11-21

### Added

- Android **Camera** mode: the phone uploads its rear camera as H.264 to the
  desktop's `/camera` WebSocket, targeting 1280×720 at up to 120 FPS with
  Camera2 constrained high-speed capture when available.
- Desktop phone-camera receiver and ffmpeg decoder with token authentication,
  keyframe recovery, replacement handling and drop statistics.
- Per-phone-camera chessboard calibration. The Android handshake reports the
  device and Camera2 camera ID; **Tools → Calibrate phone camera…** stores each
  lens under `phone:<device>:<camera_id>` in `camera.phone_calibration_file`
  (default `phone_calibrations.json`). A stored profile takes precedence over
  the reported FOV.
- `CameraRouter` uses fresh phone frames while they arrive and releases the
  local camera, reopening it on a background thread after
  `phone.camera_timeout_s`. Only the active camera is kept open.
- Separate quality controls for both directions: `phone.camera_bitrate_mbps`
  for the phone upload and `phone.bitrate_mbps` for the headset stream.
- The phone reports its approximate horizontal FOV so marker PnP and body
  tracking use the phone's intrinsics at the streamed resolution.
- Synthetic camera scene now loops cleanly over six seconds with more joint and
  board motion for tracking tests, plus a cached `SyntheticClip`.

### Changed

- USB tethering with internet sharing is documented as the recommended
  phone-to-desktop link for camera mode and 720p120.
- Camera-mode bitrate defaults to 40 Mbps, with 60–80 Mbps recommended on USB.
- Phone camera switching is no longer instantaneous: the local camera is
  released while the phone is active and reopened in the background on
  fallback, trading a short tracking gap for exclusive device access.
- Phone calibration entries are merged atomically, so calibrating one camera
  never overwrites another device's profile; corrupt stores are moved aside
  before a new entry is written.

### Fixed

- A replaced phone camera connection can no longer close the new decoder or
  clear the new session state.
- Camera streaming ignores duplicate disconnect callbacks, cancels stale
  WebSocket clients and avoids joining its own camera thread.
- The phone camera ffmpeg decoder now emits live frames instead of waiting for
  the input stream to close; the `nobuffer` probe flag silently discarded a
  piped H.264 stream.
- Camera mode validates the camera port before saving settings and clears a
  pending permission start when the mode changes.
- Phone camera decoder shutdown now kills ffmpeg before closing its input
  pipe, preventing a Windows deadlock when the writer is still blocked in a
  pipe write.
- Camera mode reports the selected Camera2 ID in its status line so the lens
  used for a calibration is unambiguous.

## [1.1.0] - 2026-11-19

### Added

- Five-marker 3x3 ArUco boards for the headset and both controllers, solved as
  rigid bodies with a predictive error-state Kalman filter.
- Phone IMU and clock-sync authentication using a fixed 16-byte trailer derived
  from `phone.token`; video WebSocket authentication uses the same token.
- Cross-language protocol golden vectors shared by the Python, C++ and Kotlin
  test suites.
- Signed Android release builds in CI, with signing keys supplied only through
  GitHub secrets.
- Coarse per-source rate limiting for the UDP driver and phone receivers.
- `uv.lock`, Ruff, mypy, pre-commit, a coverage floor and a `quality` CI job.
- Tag-driven GitHub Releases with Windows, Linux, Android, Python and driver
  assets, `SHA256SUMS` and a CycloneDX SBOM.
- `CHANGELOG.md`, `SECURITY.md`, `.editorconfig` and `.gitattributes`.

### Changed

- Camera negotiation now tries `1080p60`, `720p60`, `1080p30` and `720p30` in
  that order and never upscales.
- Runtime start, tracking start/stop and phone video toggling run off the Qt
  GUI thread, so the window stays responsive while the camera and MediaPipe
  initialise.
- The gamepad hotplug scan is rate-limited instead of reinitialising SDL on
  every poll when no pads are present.
- The pairing token is masked in the settings dialog unless explicitly shown.
- The package classifier is `Development Status :: 4 - Beta`, matching the
  project's vibe-coded disclaimer.
- Large modules were split: `ui/settings_dialog.py`, `config_validation.py`,
  `video_server.py`, `pose_geometry.py`, and test-only emulator tooling moved
  to `desktop/e2e/`.

### Fixed

- Malformed STATE input IDs no longer kill the driver-link receiver.
- Optional-dependency tests skip cleanly instead of failing when `cv2` or
  MediaPipe is missing.
- README and Android documentation now match the manifest's network and
  wake-lock permissions.
- Desktop startup smoke tests and packaged builds run against the locked
  dependency set.

## [1.0.0] - 2026-11-18

- First published board-tracking release: webcam marker/hand/body tracking,
  gamepad controllers, phone HMD streaming, native OpenVR driver, bundled
  Monado OpenXR runtime and portable Linux/Windows packages.

[1.2.0]: https://github.com/yeicor/openvr-cardboard/releases/tag/v1.2.0
[1.1.0]: https://github.com/yeicor/openvr-cardboard/releases/tag/v1.1.0
[1.0.0]: https://github.com/yeicor/openvr-cardboard/commit/d3bbdf0
