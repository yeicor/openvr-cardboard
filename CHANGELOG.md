# Changelog

All notable changes to OpenVR Cardboard are documented here. The project
follows semantic versioning for the desktop package and the release tags.

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

[1.1.0]: https://github.com/yeicor/openvr-cardboard/releases/tag/v1.1.0
[1.0.0]: https://github.com/yeicor/openvr-cardboard/commit/d3bbdf0
