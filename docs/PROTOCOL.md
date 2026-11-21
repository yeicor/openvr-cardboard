# OpenVR Cardboard protocol v1

This document defines every wire format used by the project. The same
definitions exist in:

- `desktop/openvrcardboard/protocol.py`
- `driver/src/protocol.h`
- `android/app/src/main/java/dev/openvrcardboard/phone/Protocol.kt`

All multi-byte fields are **little-endian**. Floats are IEEE-754 binary32 and
doubles are IEEE-754 binary64. Quaternions are stored as `(x, y, z, w)`.
Positions are metres, velocities metres/second, angular velocities radians/second
and accelerations metres/second².

## 1. UDP ports

| Port | Direction | Purpose |
| --- | --- | --- |
| 39100 | desktop → driver, driver → desktop | tracked-device state, haptics, driver status |
| 39101 | phone → desktop, desktop → phone | phone IMU, clock synchronisation |
| 8765 | phone ← desktop (TCP/WebSocket) | stereo video frames and control messages |
| 8766 | phone → desktop (TCP/WebSocket) | phone camera frames and control messages |
| 4242 | desktop → Monado remote driver | OpenXR device poses |
| 39102 | OpenXR app → desktop | captured eye textures, grabs and haptics |

The driver binds `39100` on loopback only. The desktop app sends state to
`127.0.0.1:39100`; the driver answers haptics to the source address of the most
recent state packet. The phone IMU receiver binds `39101` on all interfaces so
a phone on the same LAN can reach it. The OpenXR links are loopback-only: the
desktop sends poses to Monado's `remote` driver and the capture shim sends
per-eye images back to the desktop app.

Ports are configurable in the desktop app and in the driver's
`default.vrsettings` (`openvrcardboard/udp_port`). The Android app has its own UDP
port setting and must match the desktop app.

## 2. UDP header

Every UDP datagram begins with a 12-byte header:

| Offset | Size | Type | Field | Value |
| --- | --- | --- | --- | --- |
| 0 | 4 | bytes | magic | ASCII `OVRC` |
| 4 | 1 | uint8 | version | `1` |
| 5 | 1 | uint8 | type | packet type |
| 6 | 2 | uint16 | flags | reserved, `0` |
| 8 | 4 | uint32 | sequence | wraps at 2³² |

Packet types:

| Type | Name | Direction |
| --- | --- | --- |
| `0x01` | `STATE` | desktop → driver |
| `0x02` | `HAPTIC` | driver → desktop |
| `0x03` | `STATUS` | driver → desktop |
| `0x20` | `IMU` | phone → desktop |
| `0x21` | `TIME_PING` | phone → desktop |
| `0x22` | `TIME_PONG` | desktop → phone |

Datagrams with a wrong magic, an unknown version or a truncated payload are
dropped without a reply. Drivers must ignore packets from non-loopback
addresses; the desktop app must ignore phone packets whose size does not match.

## 3. STATE (`0x01`)

Sent by the desktop app at a configurable rate (default 250 Hz). Devices that
are absent from the packet keep their last pose until the driver's pose timeout
(default 1000 ms), after which they are reported as tracking-lost.

```
header                          12 bytes
double timestamp                8
uint8  device_count             1
uint8  reserved[3]              3
float  prediction_ms            4
device_count × DevicePose      56 each
uint8  input_count              1
uint8  reserved[3]              3
input_count × ControllerInput  36 each
```

`prediction_ms` is the horizon the desktop already wants the driver to apply;
it is clamped to 0…1000 ms. The driver maps it to `poseTimeOffset` and never
predicts a second time. State packets are also sequence-numbered: a driver
must reject duplicate or older sequences from the same sender, using signed
32-bit comparison so wraparound is handled, and may reset the comparison when
the sender address changes.

### 3.1 DevicePose (56 bytes)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | uint8 | device id |
| 1 | uint8 | flags |
| 2 | uint16 | reserved |
| 4 | float[3] | position x, y, z |
| 16 | float[4] | rotation x, y, z, w |
| 32 | float[3] | linear velocity |
| 44 | float[3] | angular velocity |

Device ids:

| Id | Device |
| --- | --- |
| `0` | HMD |
| `1` | left controller |
| `2` | right controller |
| `100 + n` | generic tracker *n* (0 ≤ n < 16) |

Flags:

| Bit | Name | Meaning |
| --- | --- | --- |
| 0 | `POSE_VALID` | pose may be used |
| 1 | `CONNECTED` | device is present |
| 2 | `ORIENTATION_ONLY` | position is synthesised / not tracked |

The driver accepts up to 16 device entries and 8 input entries. Additional
entries are ignored. Device ids may be repeated in successive packets but not
inside one packet.

### 3.2 ControllerInput (36 bytes)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | uint8 | device id (`1` or `2`) |
| 1 | uint8[3] | reserved |
| 4 | uint32 | buttons bitmask |
| 8 | float | trigger 0…1 |
| 12 | float | grip 0…1 |
| 16 | float | joystick x −1…1 |
| 20 | float | joystick y −1…1 |
| 24 | float | trackpad x −1…1 |
| 28 | float | trackpad y −1…1 |
| 32 | float | reserved (0) |

Button bits:

| Bit | Button |
| --- | --- |
| 0 | trigger click |
| 1 | grip click |
| 2 | application menu |
| 3 | system |
| 4 | A |
| 5 | B |
| 6 | trackpad click |
| 7 | joystick click |
| 8 | trackpad touch |
| 9 | trigger touch |
| 10 | joystick touch |
| 11 | grip touch |
| 12 | X |
| 13 | Y |
| 14 | D-pad up |
| 15 | D-pad down |
| 16 | D-pad left |
| 17 | D-pad right |
| 18 | select |
| 19 | start |

## 4. HAPTIC (`0x02`)

The driver forwards SteamVR haptic events to the source of the last STATE packet.

```
header                  12 bytes
uint8  device_id         1
uint8  reserved[3]       3
float  amplitude         4
float  frequency_hz      4
float  duration_seconds  4
```

Total: 28 bytes. `device_id` is `1` or `2`. The desktop app may use this to
drive gamepad rumble. Events with a zero amplitude are ignored.

## 5. STATUS (`0x03`)

Sent by the driver once per second after it has received at least one STATE
packet, so the desktop app can display driver health.

```
header                  12 bytes
uint32 protocol_version  4
uint32 device_count      4
uint32 uptime_ms         4
uint32 state_packets     4
uint32 parse_errors      4
```

Total: 32 bytes.

## 6. IMU (`0x20`)

Sent by the Android app at sensor rate (default 120 Hz, configurable).

```
header                        12 bytes
double timestamp              8   phone elapsedRealtimeNanos / 1e9
float  quaternion[4]          16  x, y, z, w (TYPE_ROTATION_VECTOR)
float  gyroscope[3]           12  rad/s, device frame
float  linear_acceleration[3] 12  m/s², device frame, gravity removed
uint8  tail                   1   accuracy (bits 0–1) | screen rotation (bits 2–3)
uint8  reserved[3]            3
```

Total: 64 bytes, or 80 bytes with a pairing token.

When `phone.token` is set, the phone appends a 16-byte authenticator after
the reserved bytes: the first 16 bytes of `SHA-256(token UTF-8)`. The desktop
rejects IMU packets whose authenticator is missing or wrong. Without a token
no authenticator is sent and the desktop accepts the 64-byte form.

The quaternion describes the orientation of the phone in the Android world
frame. The desktop app removes the orientation captured at the last recenter
and maps the Android world frame (X east, Y north, Z up) to OpenVR play space
(X right, Y up, -Z forward) with a −90° rotation around X. The device axes stay
+X right, +Y up the screen and +Z out of the screen, matching OpenVR HMD axes.

Tail byte fields:

| Bits | Field | Meaning |
| --- | --- | --- |
| 0–1 | `accuracy` | `0` unreliable, `1` low, `2` medium, `3` high |
| 2–3 | `screen_rotation` | Android `Surface.ROTATION_*`: 0, 1 (+90° Z), 2 (180°), 3 (−90°) |
| 4–7 | reserved | `0` |

The screen rotation is applied around the phone's screen normal before the
recenter reference is removed. The phone must send the rotation of its current
display, not a fixed value. See `docs/ARCHITECTURE.md` for the recenter math.

## 7. Clock synchronisation

### TIME_PING (`0x21`), phone → desktop

```
header              12
double client_time   8    phone elapsedRealtimeNanos / 1e9
auth[16]            16    only when phone.token is set
```

The authenticator is the same `SHA-256(token)` prefix as for IMU packets. The
desktop drops unauthenticated pings without replying when `phone.token` is set.

### TIME_PONG (`0x22`), desktop → phone

```
header              12
double client_time   8    copied from the ping
double server_time   8    desktop monotonic seconds at reception
```

The phone computes `offset = server_time - (client_time + rtt / 2)` from a
number of samples and can use the median offset to stamp IMU packets in
desktop time. The desktop does not depend on that: `PhoneImuReceiver` estimates
its own robust phone-to-desktop offset from the minimum observed one-way delay
and the `TIME_PING` candidates, converts every sensor timestamp to desktop
monotonic time, and buffers one sample for up to 20 ms to undo network
reordering. Samples that are stale, duplicated or older than the newest
accepted timestamp are discarded. A sender change resets the offset and
reorder state.

## 8. Phone video WebSocket

The desktop app hosts a WebSocket server (default `ws://0.0.0.0:8765/video`).
Only binary frames carry video. Text frames carry JSON control messages.

- The phone connects to `/video`, optionally with `?token=<pairing token>`.
- The desktop immediately sends a `welcome` message describing the stream.
- Depending on `phone.codec`, the desktop sends JPEG frames, or H.264/HEVC/AV1
  access units encoded by NVIDIA NVENC. H.264 can instead run on the CPU
  through `libx264` when `phone.encoder` is `x264` (or when H.264 is requested
  and NVENC H.264 is unavailable). Without a usable encoder or ffmpeg it always
  uses JPEG, so CI and non-GPU machines keep working.
- Frames are dropped per client when the socket is backed up; the newest frame
  wins rather than queueing latency.
- The phone sends `hello` after the socket opens and may send `ping` or
  `video_fallback`.
- The desktop may request a sensor recenter with `recenter`, and announces a
  mid-stream codec change with `video_config`.

### 8.1 Binary video frame

```
offset 0   uint8  kind = 0x01
offset 1   uint8  layout (0 mono, 1 side-by-side, 2 top-bottom)
offset 2   uint16 width  (pixels, complete frame)
offset 4   uint16 height (pixels, complete frame)
offset 6   uint16 flags
offset 8   uint32 frame sequence
offset 12  double capture time (desktop monotonic seconds)
offset 20  payload
```

Total header: 20 bytes. `width` and `height` always describe the complete
encoded frame. A mono layout means the phone duplicates the image to both eyes.
Side-by-side and top-bottom mean the full frame is already a stereo pair and
the phone displays it across the full panel (one eye per half). The desktop
configuration stores per-eye dimensions for stereo layouts and multiplies them
by two when it encodes the frame.

Flags:

| Bits | Field | Meaning |
| --- | --- | --- |
| 0–2 | codec | `0` JPEG, `1` HEVC, `2` AV1, `3` H.264 |
| 3 | `KEYFRAME` | payload starts an independent (key) frame |
| 4 | `CONFIG` | payload carries codec configuration, not a display frame |
| 5–15 | reserved | `0` |

JPEG payloads are ordinary baseline/progressive JPEG. H.264 and HEVC payloads
are Annex-B access units (start codes, SPS/PPS/VPS included in keyframes). AV1
uses IVF temporal units; the phone reconstructs the OBUs and the sequence
header. The desktop caches the most recent keyframe and sends it immediately
after `welcome`, so a phone that connects mid-stream can start decoding without
waiting for the next periodic keyframe.

### 8.2 Control JSON

Phone → desktop:

```json
{"type":"hello","protocol":2,"name":"Example Phone","width":2400,"height":1080,"sensor_hz":200}
{"type":"ping","t":123.456}
{"type":"stats","fps":58.2,"decode_ms":7.1}
{"type":"video_fallback","reason":"no av1 decoder on this phone"}
```

`video_fallback` asks the desktop to switch the whole session to JPEG. The
phone sends it when it has no MediaCodec decoder for the negotiated codec or
when its hardware decoder rejects the stream; the desktop then closes the
encoder, re-broadcasts a JPEG `video_config` and continues streaming. This keeps
old or low-end phones usable without reconnecting.

Desktop → phone:

```json
{"type":"welcome","protocol":2,"session":"a4f2","video":{"format":"hevc","codec":"hevc","layout":"sbs","width":2400,"height":1080,"per_eye_width":1200,"per_eye_height":1080,"fps":90,"quality":65,"bitrate_mbps":40.0,"csd":"AAAA...","lens":{"lens_separation_mm":63.0,"eye_screen_distance_mm":45.0,"screen_width_mm":0.0,"screen_height_mm":0.0,"fov_deg":90.0,"distortion_k1":0.0,"distortion_k2":0.0,"chromatic_aberration":0.0,"image_scale":1.0,"eye_offset_x":0.0,"eye_offset_y":0.0,"overscan":0.0}}}
{"type":"video_config","codec":"jpeg","csd":""}
{"type":"pong","t":123.456}
{"type":"recenter"}
{"type":"error","message":"invalid pairing token"}
```

`format` is kept for older clients; `codec` is authoritative. `width` and
`height` are the complete stereo frame, while `per_eye_width` and
`per_eye_height` are the configured per-eye values. `csd` is base64
codec-specific data (H.264 SPS/PPS, HEVC VPS/SPS/PPS, or the AV1 sequence
header) and may be empty; `HardwareVideoDecoder` extracts parameter sets from
the first keyframe when needed. `lens` carries the Cardboard lens and panel
geometry so the phone can align the two eye images without rebuilding the app.

Unknown JSON fields and message types must be ignored for forward compatibility.

## 9. Phone camera WebSocket

The desktop app hosts a second WebSocket server (default
`ws://0.0.0.0:8766/camera`) for the Android app's **Camera** mode. The phone
captures its rear camera, encodes H.264 with MediaCodec and uploads Annex-B
access units. The desktop decodes them with its bundled ffmpeg and exposes the
newest frame to the tracking loop. Only one phone is accepted at a time; a new
authenticated connection replaces the old one.

- The phone connects to `/camera` (the desktop also accepts `/webcam` and `/`),
  optionally with `?token=<pairing token>`. The token is `phone.token`, the
  same secret used for IMU and video.
- The desktop immediately sends `camera_welcome` with the preferred stream
  parameters and the configured bitrate.
- The phone replies with `camera_hello`, then starts sending binary frames.
- The desktop may send `camera_request_keyframe` when its decoder needs a new
  reference frame.

`camera_welcome` (desktop → phone):

```json
{"type":"camera_welcome","protocol":1,"session":"a4f2","codec":"h264","bitrate_mbps":40.0,"prefer_width":1280,"prefer_height":720,"prefer_fps":120}
```

`camera_hello` (phone → desktop):

```json
{"type":"camera_hello","protocol":1,"name":"Example Phone","device":"Example Phone","camera_id":"0","camera_name":"Example Phone rear camera","width":1280,"height":720,"fps":120,"codec":"h264","rotation":90,"hfov_deg":66.5,"bitrate_mbps":40.0}
```

`bitrate_mbps` on the phone is advisory. `device` and `camera_id` identify the
Camera2 device and lens; together they form the stable calibration key
`phone:<device>:<camera_id>`. `camera_name` is a human-readable label used in
the calibration dialog. Older apps that omit `camera_id` keep using the FOV
estimate and cannot be calibrated per camera. `hfov_deg` is optional; when
present and between 10° and 150° the desktop uses it to build the pinhole
camera model for marker and body tracking at the streamed resolution instead
of assuming the local webcam's calibration. A stored calibration for the
camera key takes precedence over `hfov_deg`. `camera_stats` messages are
accepted for logging, and `ping` is answered with `pong`.

### 9.1 Binary camera frame

```
offset 0   uint8   magic "O"
offset 1   uint8   magic "V"
offset 2   uint8   magic "R"
offset 3   uint8   magic "C"
offset 4   uint8   version = 1
offset 5   uint8   type = 0x30
offset 6   uint16  flags (bit 0 = keyframe)
offset 8   uint32  frame sequence
offset 12  double  capture time (Android elapsedRealtime seconds)
offset 20  uint32  width
offset 24  uint32  height
offset 28  bytes   Annex-B H.264 access unit
```

Total header: 28 bytes. The payload is one encoded access unit starting with
`00 00 00 01` (or `00 00 01`) start codes. Keyframes include SPS/PPS. The
desktop drops malformed headers, rejects implausible dimensions and rebuilds
the decoder when the resolution changes. It requests a keyframe after a decode
error, so a lost reference frame does not permanently corrupt the picture.

### 9.2 Tracking integration

Only the active camera is kept open. `CameraRouter` serves a phone frame while
one arrived within `phone.camera_timeout_s` (default 0.5 s); while fresh phone
frames arrive it stops the local camera, releasing the USB/device resources it
holds. When the phone goes stale, the router reopens the local camera on a
background thread so the tracking loop is not blocked by device startup. There
is a short tracking gap while the local camera reopens; switching back to the
phone releases the local camera again. The router's statistics expose the
active source, phone connection state and dropped frame counts.

Phone intrinsics are resolved in this order: a stored calibration for the
reported `phone:<device>:<camera_id>` key (scaled to the streamed resolution),
then the reported `hfov_deg`, then the local camera model. Stored calibrations
live in `camera.phone_calibration_file` (default `phone_calibrations.json`) and
are written by **Tools → Calibrate phone camera…**.

## 10. Configuration files

The desktop app writes `config.json` (see `desktop/openvrcardboard/config.py`). The
driver reads `openvrcardboard` keys from SteamVR's merged settings. The Android app
stores its settings in `SharedPreferences`.

Defaults shared by all components:

| Setting | Default |
| --- | --- |
| driver UDP port | 39100 |
| phone IMU UDP port | 39101 |
| video WebSocket port | 8765 |
| phone camera WebSocket port | 8766 |
| phone camera bitrate | 40 Mbps |
| phone camera fallback timeout | 0.5 s |
| OpenXR remote-driver port | 4242 |
| OpenXR capture port | 39102 |
| state rate | 250 Hz |
| pose timeout | 1000 ms |
| video layout | `sbs` |
| video per-eye size | 1200 × 1080 |
| video FPS target | 90 |
| video codec | `auto` (HEVC NVENC, then AV1, H.264, JPEG) |
| video encoder | `auto` (NVENC, or CPU `x264` for explicit H.264) |
| video bitrate | 40 Mbps |
| keyframe interval | 1 s |
| JPEG quality (fallback) | 65 |
| Android sensor rate | 120 Hz |
