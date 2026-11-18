# Six-degree-of-freedom board tracking design

This document records the design of the rigid five-marker tracking system: why
the boards look the way they do, which marker family was chosen, how a pose is
estimated and predicted, and how the implementation is measured against the old
single-marker tracker. The measured numbers live in
[`BENCHMARKS.md`](BENCHMARKS.md); this document explains the reasoning and the
budgets behind them.

## 1. Goals and constraints

- Track three rigid objects simultaneously: HMD, left hand and right hand.
- Target 1920 × 1080 capture at 30 FPS, working range up to 2 m.
- Reject corner-detection noise at the source instead of hiding it with output
  smoothing.
- Keep end-to-end latency low enough for a 35 ms prediction horizon to be
  useful.
- Survive short occlusions and reacquire smoothly without teleporting.
- Run on OpenCV's CPU detector. Custom GPU marker work was removed after
  measurement showed the CPU detector leaves ample headroom.
- No backward compatibility with the single-marker configuration in this
  bleeding-edge release.

## 2. Board geometry

Each object carries one flat, rigid board. All coordinates below are in the
board frame: origin at the centre of the large marker, `+X` right, `+Y` up and
`+Z` out of the printed face.

| Element | Size | Centre |
| --- | --- | --- |
| Board outline | 200 × 112 mm | (0, 0) |
| Central marker (ID `n`) | 80 × 80 mm | (0, 0) |
| Peripheral marker (ID `n+1`) | 40 × 40 mm | (−70, +26) mm |
| Peripheral marker (ID `n+2`) | 40 × 40 mm | (−70, −26) mm |
| Peripheral marker (ID `n+3`) | 40 × 40 mm | (+70, +26) mm |
| Peripheral marker (ID `n+4`) | 40 × 40 mm | (+70, −26) mm |

The peripheral marker edges reach `x = ±90 mm` and `y = ±46 mm`, leaving 10 mm
of white board to every cut line and 12 mm between the upper and lower marker
pairs. A 3×3 ArUco marker occupies five modules including its black border, so
a 40 mm marker has 8 mm modules; the 10 mm and 12 mm gaps therefore preserve at
least one full quiet-zone module. The layout is deliberately asymmetric in
marker size and symmetric in position: the large centre provides a strong
close-range signal and keeps the board solvable when an edge is occluded, while
the four small markers provide a 180 mm horizontal and 92 mm vertical
corner-to-corner span for orientation. The resulting outline is 1.79:1, not a
literal 3:1.

The extra board height is functional, not cosmetic. It leaves room for the
3×3 quiet zone while keeping the 40 mm peripherals large enough to decode at
the design range. A tighter layout would either shrink the markers below the
detector's reliable pixel count or crowd their borders, which loses markers
before the pose solve ever sees them.

### Projected sizes

For a 1920 × 1080 frame and a 60° horizontal field of view, the pinhole focal
length is

```text
fx = 960 / tan(30°) = 1662.8 px
```

and the projected edge of an object of width `w` at distance `d` is
`fx · w / d`:

| Feature | At 0.5 m | At 1 m | At 2 m |
| --- | --- | --- | --- |
| Board width (200 mm) | 665 px | 333 px | 166 px |
| Board height (112 mm) | 372 px | 186 px | 93 px |
| Central marker (80 mm) | 266 px | 133 px | 66.5 px |
| Peripheral marker (40 mm) | 133 px | 66.5 px | 33.3 px |
| Peripheral 3×3 module (8 mm) | 26.6 px | 13.3 px | 6.7 px |

At 720p the same 2 m view gives a 110.8 px board width, a 44.3 px central
marker and a 22.2 px peripheral marker whose data modules are 4.4 px wide. The
3×3 payload keeps those modules about 40% larger than a 5×5 tag printed at the
same physical size, which is the main reason it survives 2 m and motion blur.
The vertical field of view for 16:9 at 60° horizontal is 36.0°, which projects
the 112 mm board height to 93 px at 2 m.

## 3. Marker family

The project ships a custom 3×3 ArUco dictionary, `DICT_3X3_15`, rather than
OpenCV's smallest predefined family. A 3×3 payload occupies five modules
including the black border, so a 40 mm marker has 8 mm modules; the 4×4 and
5×5 families get 6.7 mm and 5.7 mm at the same printed size. Fifteen
rotation-unique codes cover the HMD and both hands, and body tracking consumes
no marker IDs. The minimum rotation-invariant Hamming distance across the
fifteen codes is two, so the dictionary is constructed with
`maxCorrectionBits = 0`: a single flipped module must never be "corrected" into
a different valid code. The bit patterns are embedded in
`desktop/openvrcardboard/aruco.py` so every build and printed sheet agrees
independently of the installed OpenCV version.

The candidates were run through the same OpenCV detector with the shipped
parameters: adaptive threshold 3–23, `minMarkerPerimeterRate` 0.012, subpixel
corner refinement and a 5 px refinement window.

| Candidate | 15-marker 1080p mean | p99 | 720p, 2 m/60° | 720p + 6 px blur |
| --- | ---: | ---: | ---: | ---: |
| ArUco `DICT_3X3_15` (custom) | 1.93 ms | 3.29 ms | 5/5 | 4/5 |
| ArUco `DICT_4X4_50` | 1.46 ms | 2.06 ms | 4/5 | 1/5 |
| ArUco `DICT_5X5_100` | 1.53 ms | 1.91 ms | 3/5 | 1/5 |
| AprilTag `DICT_APRILTAG_36h11` | 1.87 ms | 2.70 ms | 3/5 | 1/5 |

The 15-marker timing is a 30-frame detector-only run on a 1080p frame containing
all three device boards. The two visibility columns render one board at 2 m and
60° yaw at 720p, where the 40 mm peripherals are only about 22 px wide, and
count the decoded markers out of five; the last column adds a 6-tap horizontal
box blur as an aggressive stress case. At full 1080p all four dictionaries
decode 5/5 at 2 m and 60°; the 3×3 advantage appears once the markers get
small or blurred, because its modules are 20% larger than 4×4 and 40% larger
than 5×5. The custom family costs about 0.3–0.5 ms more detector time than the
predefined families, which is small against the 7.5 ms total tracking stage.
The board still solves from fewer markers when only the centre or one pair is
visible, but reports the `DEGRADED` state.

BGR-to-grayscale conversion costs about 0.043 ms. Subpixel corner refinement and
board PnP are shared by every dictionary, so corner accuracy is the same
whenever a marker is decoded; the choice is about how often the payload
survives. Board PnP, robust rejection and filtering add about 5.5 ms on top of
the 2 ms detector at 1080p (see the runtime benchmark). Predefined OpenCV
dictionaries remain configurable for users with existing printed tags.

The synthetic renderer blends marker edges with a fractional alpha mask so that
sub-pixel edges are represented honestly. Residual corner bias in the renderer
is about 0.25–0.4 px inward radially; it is a property of the synthetic raster,
not of the detector, and is called out in the benchmark notes. Real cameras
add optical blur, rolling shutter and sensor noise that the synthetic scene
does not model.

## 4. Board-level pose estimation

Detection returns a set of decoded markers, each with four corners in image
coordinates. The tracker groups them by the configured board IDs and puts
**all inlier corners from all visible markers into one PnP problem**:

1. Object points are the known 3D corners of every marker in the board frame.
2. Image points are the detected corners, ordered to match the marker's
   top-left, top-right, bottom-right, bottom-left convention.
3. `solvePnPGeneric` with `SOLVEPNP_IPPE` returns the planar pose candidates.
4. Each candidate is scored by RMS reprojection error, inlier ratio and
   per-marker consistency.
5. The best candidate is refined with Levenberg–Marquardt when
   `camera.marker_refine_pose` is enabled.
6. A marker whose corners disagree with the consensus is marked as an outlier
   and the board is re-solved without it if enough markers remain.

Solving and averaging five independent marker poses is explicitly not done.
Each independent solve throws away the rigid geometry and has its own planar
ambiguity; averaging the results does not remove either problem. One board
solve also gives the peripheral markers a vote in the centre's pose and makes
the long baseline useful.

A board needs at least `tracking.marker_min_markers` markers (default 3) and a
reprojection error below `camera.max_reprojection_error` (default 4 px) to
produce a measurement. Fewer markers or a weaker fit produce a measurement with
larger covariance, not an immediate loss.

### Planar ambiguity

Two planar solutions can reproject almost identically while differing by
20–40° in orientation. Choosing the lower RMS alone is unstable across frames.
The tracker keeps the last **accepted raw orientation** and picks the candidate
closer to it. It deliberately does not extrapolate that prior with angular
velocity: during a fast rotation the velocity-extrapolated prior can be farther
from the truth than the other candidate. Once a frame is accepted it becomes
the new prior. This removed the 10°+ frame-to-frame orientation jumps that the
earlier "lowest reprojection error" rule produced.

## 5. State representation

Each board has an independent 12-state error-state Kalman filter. The nominal
state is

```text
position        p       (3)
linear velocity v       (3)
orientation     q       (4, unit quaternion)
angular velocity ω      (3, world frame, rad/s)
```

and the error state is

```text
δx = [δp, δv, δθ, δω]   (12)
```

where `δθ` is a small rotation vector applied on the left of the nominal
quaternion:

```text
q ← exp(δθ) ⊗ q
```

Linear acceleration is derived from the velocity corrections and the elapsed
time, then smoothed with a short time constant; it is part of the reported
6DoF state but is not a filter state. Keeping acceleration out of the state
avoids an unobservable, noisy third derivative while still giving consumers a
useful value. Angular acceleration is derived from angular-velocity
corrections in the same way.

A left-multiplicative error state is used so the covariance always has 12
dimensions with no quaternion-norm singularity. Euler-angle filtering is not
used anywhere: it is singular at pitch ±90° and its noise is anisotropic in a
way that does not match a rotating rigid body.

The filter's process model is constant-velocity with process noise on velocity
and angular velocity. `marker_process_noise` (default 1.0) and
`marker_angular_noise` (default 2.0) scale the translation and rotation
process noise. With `marker_adaptive_noise` enabled, the linear noise is
reduced toward `marker_static_noise_scale` (default 0.2) while the board is
nearly stationary and returns to the configured value once it moves. This is a
source-level fix for resting jitter: it does not add output latency during
real motion, and it leaves the angular model unchanged so fast hand rotations
stay responsive.

## 6. Measurement model and correction

The measurement is the board pose converted into the object frame, then into
play space. The measurement covariance is built from observable evidence rather
than a single fixed number:

- **Reprojection error** sets the base image noise, with `base_sigma_px = 0.8`.
- **Board span in pixels** scales translation uncertainty: a board that covers
  more pixels localizes the centre better.
- **Incidence angle** increases depth uncertainty at oblique views.
- **Marker count and inlier ratio** increase uncertainty when the periphery is
  missing or an outlier was rejected.
- **Distance** raises depth uncertainty faster than lateral uncertainty.

Conservative floors are applied: 3 mm lateral, 8 mm depth and 1.7° orientation.
These floors were added after the filter became overconfident on a large,
close board and then rejected valid measurements. The floors keep the filter
honest without making it sluggish.

Corrections use the standard Kalman gain. Innovations are gated by normalized
innovation squared (NIS): a measurement that disagrees with the prediction by
more than the gate is rejected for that frame, but the board keeps predicting.
This prevents a single bad solve from throwing the pose across the room. If
rejections persist, the filter covariance grows until the gate opens
naturally, so a genuine fast motion is not locked out forever.

Confidence is a separate bounded score used by the state machine and the UI. It
combines reprojection error, marker count, inlier ratio, viewing angle and
multi-frame consistency, and is reduced for motion blur derived from the
estimated angular velocity and configured exposure time. Confidence is not used
as a Kalman gain; it is evidence for choosing between `TRACKING`, `DEGRADED`
and `PREDICTING`.

## 7. Prediction, latency and timestamps

Every stage carries its own timestamp:

- **capture** — when the camera driver delivered the frame;
- **detection** — when detection and board PnP completed;
- **pose** — the effective timestamp of the raw board pose;
- **update** — when the filter incorporated the measurement;
- **output** — when a composed pose was published.

The filter never assumes that capture time equals processing time. A frame that
takes 12 ms to process is integrated at its capture time, and the filter's
interval is computed from that timestamp, not from wall-clock arrival.

`pose_at(t)` returns the estimate at exactly `t`; it never adds a hidden
horizon. The driver is the single prediction owner: the desktop publishes
measured-time poses and puts `tracking.pose_prediction_ms` (default 35 ms) in
the STATE packet, and the driver maps it to `poseTimeOffset`. This compensates
capture, USB, detection and composition latency exactly once. The composer,
the phone estimator and the board filter must not add their own lead, or fast
motion would overshoot.

The filter's own short-horizon extrapolation only bridges the gap between a
measurement and the current timestamp; it is bounded by the state machine.
After `marker_hold_s` (default 0.22 s) the board is `PREDICTING`, and after
`marker_lost_s` (default 1.0 s) it is `LOST` with a large but finite
uncertainty (default 0.5 m and 30°). The tracker does not extrapolate a lost
board indefinitely, and it does not apply aggressive overshoot correction: the
velocity estimate is already damped by process noise and measurement updates.

## 8. Smoothing and correction policy

There is one main filter, at the board tracker. `PoseComposer` skips One-Euro
smoothing for board-tracked devices by default; otherwise the same sample
would be filtered twice and the second filter would add lag without removing
detector noise. MediaPipe hands and body still use One-Euro because they have
no state estimator of their own.

`marker_output_smoothing_hz` adds an optional, explicit second stage for users
who prefer a softer image and accept the latency. It is off by default and is
documented as a comfort control, not as a substitute for detector or filter
tuning.

The correction policy is:

1. Detect all markers at full resolution.
2. Solve the board once with robust outlier rejection.
3. Gate the resulting measurement against the filter prediction.
4. Apply the Kalman correction with an evidence-based covariance.
5. Publish the exact-time estimate and let the driver apply the single horizon.

The detector is fixed first. Smoothing is not used to hide corner jitter,
because filtering bad measurements produces a smooth but wrong pose.

## 9. Tracking states and occlusion

Each board runs its own state machine; one hand being occluded never changes
the HMD state.

| State | Meaning | Entry |
| --- | --- | --- |
| `TRACKING` | Fresh, confident measurement | Measurement accepted with good confidence |
| `DEGRADED` | Measurement accepted but weak | Low marker count, high reprojection error or oblique view |
| `PREDICTING` | No recent measurement, filter extrapolating | `marker_hold_s` elapsed since last accepted measurement |
| `REACQUIRING` | New measurements returning after a gap | First accepted measurement after `PREDICTING`/`LOST` |
| `LOST` | No usable estimate | `marker_lost_s` elapsed |

Occlusion strategy:

- Three or more visible markers, including the centre, produce a normal
  measurement; confidence falls as markers disappear.
- One or two peripheral markers missing still leave a solvable board and a
  larger covariance.
- Centre-only is solvable but has weak orientation; it is reported as
  `DEGRADED` and its orientation variance is enlarged.
- A brief full occlusion is bridged by prediction for `marker_hold_s`.
- Reacquisition blends the new measurement over `marker_reacquire_frames`
  (default 3) frames. If the gap exceeds `marker_reinitialize_gap_s` (default
  0.30 s), the filter reinitializes instead of blending across stale state.

## 10. Calibration requirements

1. **Camera intrinsics and distortion.** Use the built-in chessboard wizard or
   `scripts/calibrate_camera.py`. At least 12 varied views; recalibrate after
   moving the camera, changing focus/zoom or changing resolution. Without a
   calibration the app falls back to a 60° pinhole model, which is usable but
   less accurate in depth and at the frame edges.
2. **Printed board geometry.** The configured width, height, marker sizes and
   peripheral offsets must match the printed board. Print at 100% and verify
   the generated 100 mm ruler. A scale error maps directly into distance error.
3. **Board-to-object transform.** The board origin is the centre of the large
   marker. Set `object_position` and `object_rotation` per board to the pose of
   the board in the device frame. The default rotation is a half-turn about Y,
   mapping the printed face onto the device's forward axis.
4. **Camera pose.** `camera.position` and `camera.rotation` place the camera in
   play space. An incorrect camera pose offsets every tracked object by the
   same rigid transform.
5. **Timestamps.** Capture timestamps must be monotonic. The filter uses frame
   capture time, not processing time; a source that supplies arrival time
   instead of capture time will show up as additional latency, not as pose
   jitter.

## 11. Benchmark methodology

The benchmark script compares the board tracker with the retired single-marker
tracker on the same synthetic scenes and the same camera model. It reports:

- **Static jitter and drift** — mean, p95 and max position/orientation error,
  plus frame-to-frame noise, over a stationary object.
- **Dynamic error** — position/orientation error during the scripted sway,
  depth and wrist motion, including peak error and overshoot relative to the
  ground-truth path.
- **Prediction** — error at the configured 35 ms lead versus the zero-lead
  estimate, and whether prediction overshoots during direction changes.
- **Composed static** — resting jitter after the full `BoardTracker` →
  `PoseComposer` path, which is what actually reaches the driver.
- **Step latency** — time to 63% and 90% of an injected rigid board step, plus
  overshoot, separating filter lag from detection latency.
- **Reacquisition** — time and error after a forced occlusion, and whether the
  pose blends or snaps.
- **Detection** — marker/board detection rate at 0.5 m, 1 m and 2 m, at yaw
  0°/45°/60°, with and without peripheral occlusion.
- **Runtime** — detection/PnP/filter timings, end-to-end FPS, latency
  percentiles, CPU time and memory, and dropped frames.

Synthetic benchmarks isolate estimator behaviour, but they do not replace real
camera validation. The report calls out the synthetic corner bias and the
missing sensor noise/rolling-shutter effects.

## 12. Deliberate non-goals

- No per-marker pose averaging.
- No Euler-angle filtering.
- No custom GPU marker path unless a measured CPU bottleneck appears.
- No unbounded prediction or "hold last pose forever".
- No backward-compatible single-marker configuration in this release.
