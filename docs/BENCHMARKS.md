# Tracking benchmarks

This document records the measured behaviour of the five-marker board tracker
against a single-marker baseline. The design and the reasoning behind the board
geometry are in [`DESIGN.md`](DESIGN.md); this page is the measurement report.

The short version:

- Detection is 100% at 0.5 m, 1 m and 2 m, at 0°, 45° and 60° yaw, with all
  five markers visible.
- The board stays solvable with two peripheral markers hidden. Centre-only is
  still solvable but its position error grows to about 37 mm, which is why it
  is reported as `DEGRADED`.
- At the 2 m design range the board's static position error is 3.4 mm versus
  8.0 mm for the central marker alone.
- Resting frame-to-frame jitter is 1.2–7.5× lower in position and 1.1–5.2×
  lower in orientation than the raw single-marker solve. The HMD position and
  the left-hand rotation improve the most.
- The custom 3×3 dictionary keeps all five markers decodable at 720p/2 m where
  the 4×4, 5×5 and 36h11 alternatives lose one or two, and it loses only one
  under a 6 px horizontal blur where the others lose four.
- The tracking stage costs 7.5 ms mean and 9.8 ms p99 at 1080p on the test
  machine, with no frame over the 33 ms budget in 300 frames.

## Environment

Absolute timings depend on the CPU, OpenCV build, enabled SIMD paths and the
number of worker threads. The numbers below are a reference point from a
desktop x86_64 machine running Python 3.14, OpenCV 5.1 and NumPy 2.5; rerun the
benchmark locally for your own hardware.

The measurement setup is a 1920 × 1080, 30 FPS synthetic camera using
`DICT_3X3_15` and the shipped 200 × 112 mm board with an 80 mm centre marker
and four 40 mm peripherals at ±70/±26 mm.

The accuracy scenarios render a synthetic scene at 4× resolution and downsample
with `INTER_AREA`. This removes most of the marker rasterizer's scale-dependent
corner bias, which otherwise shows up as a false board tilt. The runtime
scenario renders at the target resolution so that tracker timing is not mixed
with supersampling cost.

The synthetic camera is placed at `(0, 1.20, −2.00)` m with a 60° horizontal
field of view. The HMD board is about 2.0 m away; the hand boards are about
1.4 m away. This is the 2 m design range, not the 2.6 m framing used by the
in-app synthetic webcam (that framing exists to keep MediaPipe's full body in
view).

## Reproducing

From the repository root:

```bash
cd desktop
python -m pip install -e '.[tracking]'
python ../scripts/benchmark_tracking.py \
  --width 1920 --height 1080 --frames 180 --supersample 4 \
  --runtime-frames 300
```

The script writes `artifacts/tracking-benchmark.json` and prints the same JSON
to stdout. `--supersample 1` disables supersampling; `--skip-reacquire` omits
the occlusion cases. The runtime section below is the `--runtime-frames 300`
portion of the same run.

## Marker family

Two measurements are reported. The timing row is a detector-only microbenchmark
on full 1080p frames containing all three device boards, using the shipped
detector parameters and 30 frames. The two right-hand columns render one board
at 2 m and 60° yaw at 720p, where a 40 mm peripheral is only about 22 px wide,
and count how many of the five markers decode. The last column applies a
6-tap horizontal box blur, an aggressive stress case well beyond a normal
1/200 s exposure at walking speed.

| Candidate | 15-marker 1080p mean | p99 | 720p 2 m/60° | 720p + 6 px blur |
| --- | ---: | ---: | ---: | ---: |
| ArUco `DICT_3X3_15` (custom) | 1.93 ms | 3.29 ms | **5/5** | **4/5** |
| ArUco `DICT_4X4_50` | 1.46 ms | 2.06 ms | 4/5 | 1/5 |
| ArUco `DICT_5X5_100` | 1.53 ms | 1.91 ms | 3/5 | 1/5 |
| AprilTag 3 `DICT_APRILTAG_36h11` | 1.87 ms | 2.70 ms | 3/5 | 1/5 |

At full 1080p all four dictionaries decode 5/5 at 2 m and 60°; the choice only
shows up when the markers get small or blurred. The 3×3 payload has 8 mm
modules on a 40 mm marker, versus 6.7 mm for 4×4 and 5.7 mm for 5×5 once the
black border is included. That 40% module-size advantage is what keeps the
peripheral markers decodable at 720p and under blur, and it is the reason the
project ships a custom family instead of OpenCV's smallest predefined
dictionary.

`DICT_3X3_15` contains exactly fifteen rotation-unique codes, enough for the HMD
and both hands; body tracking is MediaPipe-only and consumes no marker IDs. The
minimum rotation-invariant Hamming distance across the set is two, so the
dictionary is built with `maxCorrectionBits = 0`: one flipped module must never
be "corrected" into a different valid code. The detector still refines corners
to sub-pixel accuracy. The timing cost is about 0.3–0.5 ms over 4×4/5×5, which
is small against a 7.5 ms total tracking stage.

Subpixel corner refinement and the board PnP are shared by every dictionary, so
corner accuracy is the same whenever a marker is decoded; the choice is about
how often the payload survives. The timings above are the complete per-frame
detector latency. Board PnP, robust rejection and filtering add about 5.5 ms on
top of the 2 ms detector at 1080p (see the runtime benchmark). Predefined
OpenCV dictionaries remain configurable for users with existing printed tags.

The synthetic renderer blends marker edges with a fractional alpha mask so that
sub-pixel edges are represented honestly. Residual corner bias in the renderer
is about 0.25–0.4 px inward radially; it is a property of the synthetic raster,
not of the detector, and is called out in the benchmark notes. Real cameras
add optical blur, rolling shutter and sensor noise that the synthetic scene
does not model.

## Static accuracy and jitter

A stationary board is rendered for 180 frames with 2.0 gray-level Gaussian
sensor noise and all three boards in view. Errors are against the exact
synthetic ground truth. `baseline` is one IPPE square solve of the 80 mm central
marker; `raw` is the board PnP measurement; `filtered` is the error-state filter
at the capture time; `+35 ms` evaluates the same filter 35 ms ahead, matching
the driver's `poseTimeOffset` horizon. No tracker or composer adds its own
prediction. The filter uses the default Balanced preset: adaptive process noise
on, `marker_static_noise_scale` 0.2, and output smoothing off.

### Position error (mm)

| Board | Estimator | Mean | p95 | Max |
| --- | --- | ---: | ---: | ---: |
| HMD | baseline | 8.01 | 8.38 | 8.75 |
| HMD | board raw | 3.37 | 3.59 | 4.81 |
| HMD | board filtered | 3.37 | 3.48 | 3.73 |
| HMD | board +35 ms | 3.37 | 3.49 | 3.78 |
| Left | baseline | 1.19 | 1.28 | 1.38 |
| Left | board raw | 2.10 | 2.19 | 2.25 |
| Left | board filtered | 2.10 | 2.15 | 2.19 |
| Left | board +35 ms | 2.10 | 2.15 | 2.22 |
| Right | baseline | 2.91 | 3.04 | 3.09 |
| Right | board raw | 3.78 | 4.22 | 4.28 |
| Right | board filtered | 3.77 | 4.32 | 4.38 |
| Right | board +35 ms | 3.76 | 4.37 | 4.43 |

### Orientation error (degrees)

| Board | Estimator | Mean | p95 | Max |
| --- | --- | ---: | ---: | ---: |
| HMD | baseline | 0.163 | 0.244 | 0.332 |
| HMD | board raw | 1.036 | 1.095 | 1.169 |
| HMD | board filtered | 1.035 | 1.070 | 1.100 |
| HMD | board +35 ms | 1.035 | 1.075 | 1.111 |
| Left | baseline | 0.353 | 0.385 | 0.400 |
| Left | board raw | 0.165 | 0.180 | 0.196 |
| Left | board filtered | 0.165 | 0.172 | 0.179 |
| Left | board +35 ms | 0.165 | 0.174 | 0.183 |
| Right | baseline | 0.130 | 0.140 | 0.144 |
| Right | board raw | 0.192 | 0.249 | 0.258 |
| Right | board filtered | 0.192 | 0.261 | 0.269 |
| Right | board +35 ms | 0.193 | 0.271 | 0.278 |

### Jitter and drift

Jitter is the RMS frame-to-frame change; drift is the magnitude of a linear fit
to the whole stationary sequence. Lower is better.

| Board | Position jitter (mm) | Rotation jitter (°) | Position drift (mm) |
| --- | ---: | ---: | ---: |
| HMD baseline | 0.331 | 0.086 | 0.010 |
| HMD board filtered | **0.044** | **0.023** | 0.037 |
| Left baseline | 0.095 | 0.031 | 0.011 |
| Left board filtered | **0.016** | **0.006** | 0.012 |
| Right baseline | 0.095 | 0.016 | 0.005 |
| Right board filtered | **0.079** | **0.015** | 0.031 |

Static jitter improves because the filter scales linear process noise down while
the board is at rest (`marker_static_noise_scale` 0.2 by default) instead of
relying on output smoothing. Angular process noise is unchanged.

The HMD board runs at 2 m where the central-marker baseline is weakest in depth;
its position error drops from 8.0 mm to 3.4 mm and its jitter drops 7.5×. The
hand boards run at about 1.4 m where the central marker is already large (about
92 px), and the single-marker depth error is only 1.2–2.9 mm. At that range the
board's position error is dominated by the same synthetic peripheral corner bias
that affects orientation, so it is roughly 1–2 mm worse rather than better. The
important result is that the board stays within 2.1–3.8 mm across the whole
1.4–2.0 m range instead of degrading with distance.

### Why the synthetic rotation error is pessimistic

Board-level rotation is the one metric where the synthetic scene does not show
an improvement at the HMD range. The cause is the renderer, not the estimator:

- The 80 mm centre is 66 px wide at 2 m and IPPE_SQUARE fits its four outer
  corners with a very strong square prior. Its measured orientation error in the
  same frames is 0.1–0.3°.
- The 40 mm peripherals are only 33 px wide. At that size the 4× rasterizer and
  `INTER_AREA` downsample leave a 0.3–0.7 px systematic corner offset. The
  offset is coherent across a marker and tilts the combined planar solve by
  about 0.7–1.0°.
- The bias is visible with zero sensor noise, changes sign when the subpixel
  phase or the supersampling factor changes, and is much larger than the
  detector's real subpixel refinement error. It is not stable enough to model
  as a calibration term.
- The filter cannot remove a static bias without also lagging real motion. It
  does remove the frame-to-frame component: HMD rotation jitter is 3.7× lower
  than the raw single-marker solve.

Real-camera validation is required before making a claim about absolute board
rotation accuracy. The measured jitter, position consistency, detection and
occlusion behaviour are the results that the synthetic scene supports.

## Dynamic accuracy and prediction

The scripted scene sways the root, bobs the head, and waves both wrists while
the three boards rotate. The same 180-frame, 30 FPS, 2.0-noise run is used.
`+35 ms` is the estimate 35 ms ahead compared with the ground truth at that
future time; the baseline cannot predict and is shown at its current frame.

### Position error (mm), 180 frames

| Board | Estimator | Mean | p95 | Max |
| --- | --- | ---: | ---: | ---: |
| HMD | baseline | 7.17 | 10.03 | 12.64 |
| HMD | board raw | 5.75 | 10.26 | 12.71 |
| HMD | board filtered | 5.85 | 8.04 | 9.18 |
| HMD | board +35 ms | 5.92 | 8.44 | 9.68 |
| Left | baseline | 3.34 | 4.52 | 5.15 |
| Left | board filtered | 3.61 | 5.11 | 5.62 |
| Left | board +35 ms | 3.83 | 5.39 | 6.11 |
| Right | baseline | 3.70 | 4.92 | 6.77 |
| Right | board filtered | 4.01 | 5.26 | 6.06 |
| Right | board +35 ms | 4.14 | 5.46 | 6.54 |

### Orientation error (degrees), 180 frames

| Board | Estimator | Mean | p95 | Max |
| --- | --- | ---: | ---: | ---: |
| HMD | baseline | 0.405 | 1.118 | 1.937 |
| HMD | board raw | 0.729 | 1.493 | 2.077 |
| HMD | board filtered | 0.603 | 1.068 | 1.387 |
| HMD | board +35 ms | 0.601 | 1.097 | 1.454 |
| Left | baseline | 0.297 | 0.765 | 2.596 |
| Left | board filtered | 0.777 | 1.490 | 1.826 |
| Left | board +35 ms | 1.049 | 1.854 | 2.150 |
| Right | baseline | 0.279 | 0.660 | 2.608 |
| Right | board filtered | 0.707 | 1.282 | 1.698 |
| Right | board +35 ms | 0.948 | 1.578 | 2.064 |

The board lowers the HMD position tail (p95 10.0 → 8.0 mm, max 12.6 → 9.2 mm)
and caps the worst single-frame orientation excursion. Filtered, the hand
maxima are 1.70–1.83° versus 2.60–2.61° for the baseline; applying the driver's
35 ms horizon raises the board maxima to 2.06–2.15°, still below the baseline
while removing the added latency. Dynamic orientation is otherwise the
synthetic weak spot: the peripheral corner bias raises the board's mean and p95
on the hands. The baseline has no temporal model and no prediction; its mean is
therefore an optimistic lower bound during motion.

Overshoot is the distance by which an estimate leads the current ground truth
along the direction of motion. The default filter keeps the HMD below 3.7 mm:

| Board | Baseline (mm) | Board filtered (mm) | Board +35 ms (mm) |
| --- | ---: | ---: | ---: |
| HMD | 4.76 | 3.36 | 3.67 |
| Left | 1.45 | 1.57 | 2.10 |
| Right | 3.36 | 4.29 | 4.42 |

The 35 ms lead adds less than 0.25 mm of mean position error on the hands and
less than 0.1 mm on the HMD. It is a real extrapolation, but a conservative one.

## Composed output and step response

`composed_static` runs a stationary board through `BoardTracker` and
`PoseComposer`, which is the path that actually reaches the driver. Over 170
samples after warm-up the composed output has 0.043 mm RMS radial scatter,
0.027 mm frame-to-frame jitter and 0.012° rotation jitter. The 8.6 mm mean
offset from the commanded board origin includes the scene's normal planar-solve
bias at that pose; the spread, not that constant, is the meaningful result.

`step_latency` injects a rigid 120 mm step in the HMD board at 0.35 s and
measures the composed position:

| Metric | Value |
| --- | ---: |
| Time to 63% | 316.7 ms |
| Time to 90% | 350.0 ms |
| Overshoot | 0.11% |
| Final progress | 100.1% |

This is not detection latency: the same five markers are visible before and
after the step. It is the default Balanced policy rejecting the first frames as
NIS outliers, holding through `marker_hold_s`, then blending the reacquired
measurement. A real head does not jump 120 mm between frames; the metric is a
stress test of the outlier/blend path. It also shows why the preset offers
`Responsive` and why `marker_reinitialize_gap_s` and the process noise are
exposed for tuning.

## Detection and occlusion

A single HMD board is rendered at each distance and yaw for 40 frames. All five
markers are visible.

| Distance | Yaw | Mean markers | Central marker | Baseline solve | Board solve |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 m | 0° | 5.0 | 100% | 100% | 100% |
| 0.5 m | 45° | 5.0 | 100% | 100% | 100% |
| 0.5 m | 60° | 5.0 | 100% | 100% | 100% |
| 1.0 m | 0° | 5.0 | 100% | 100% | 100% |
| 1.0 m | 45° | 5.0 | 100% | 100% | 100% |
| 1.0 m | 60° | 5.0 | 100% | 100% | 100% |
| 2.0 m | 0° | 5.0 | 100% | 100% | 100% |
| 2.0 m | 45° | 5.0 | 100% | 100% | 100% |
| 2.0 m | 60° | 5.0 | 100% | 100% | 100% |

Occlusion groups at 1 m:

| Visible markers | Detection rate | Position error mean | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| All five | 100% | 1.34 mm | 1.37 mm | 1.40 mm |
| Left pair hidden (3/5) | 100% | 4.50 mm | 4.62 mm | 4.67 mm |
| Centre only (1/5) | 100% | 37.01 mm | 37.10 mm | 37.14 mm |

Hiding one pair removes the horizontal baseline on that side and roughly
triples the position error while remaining fully usable. Centre-only keeps the
board detected but its planar depth and orientation are weak, so the tracker
reports reduced confidence and the `DEGRADED` state rather than pretending the
pose is as good as a five-marker solve.

## Reacquisition

A moving left-hand board is visible, fully occluded, and then visible again.
The tables report the error on the first accepted visible frame after the
occlusion, the number of frames until the tracker is back in `TRACKING` with
error under 50 mm, and the largest output step. The baseline is the raw
single-marker solve; the board tracker starts its blend from the predicted pose.

| Occlusion | Error at restore | Recovery | Baseline first step | Board max step |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 s | 0.51 mm | 0 frames | 30.3 mm | 4.9 mm |
| 0.5 s | 25.92 mm | 2 frames (67 ms) | 52.6 mm | 7.5 mm |
| 1.0 s | 118.17 mm | 3 frames (100 ms) | 42.4 mm | 40.9 mm |

At 0.2 s the board is still inside the prediction hold and reacquires without a
blend. A 0.5 s gap exceeds `marker_reinitialize_gap_s`, so the filter restarts,
but the first outputs still blend from the last prediction: the restore error
is 26 mm and the largest step is 7.5 mm instead of the baseline's 52.6 mm. After
a full second the hand has moved about 118 mm, so the first blended output is
still that far from the truth. It converges in 3 frames, and its 40.9 mm
maximum step is close to the baseline snap because there is no continuity left
to preserve. Very long occlusions remain a visible reacquisition, not a
teleport-free continuation.

## Runtime

A 300-frame target-resolution run at 1920 × 1080 and 30 FPS, all three boards
in view. `detect` measures `BoardTracker.detect` only; `frame` additionally
contains the CPU synthetic renderer and is not a camera-pipeline measurement.

| Metric | Value |
| --- | ---: |
| Frames | 300 |
| Detect mean | 7.53 ms |
| Detect p95 | 8.93 ms |
| Detect p99 | 9.76 ms |
| Detect max | 11.09 ms |
| Marker detection mean | 2.01 ms |
| Marker detection p95 | 2.46 ms |
| Frames over 33 ms | 0 / 300 |
| Process CPU time | 26.2 ms per frame, summed across threads |
| Wall time (detect) | 7.5 ms per frame |
| Peak RSS | 275.4 MB |
| Detection width | 1920 (full resolution) |

The tracking stage uses about 23% of the 33 ms frame budget on average and 30%
at p99. OpenCV parallelises the detector across multiple worker threads, so the
26 ms of process CPU is spread over several cores rather than one. The marker
detector itself is only 2.0 ms of the total; the rest is board PnP, outlier
checks, filtering and the debug metadata for three boards. On a machine with
fewer cores the wall time may increase, but there is more than 3× headroom
before the 30 FPS budget is exhausted, and no GPU marker path is needed.

The synthetic renderer costs about 77 ms per frame at 1080p on the test machine,
which is why `frame_*` is not a usable pipeline FPS. Camera capture and MediaPipe
inference are not part of this benchmark.

## Limitations

- These are synthetic results. The scene models perspective, marker edges,
  sensor noise and motion, but not lens blur, rolling shutter, compression
  artifacts, exposure changes or real webcam auto-exposure.
- The rasterizer still has the 0.3–0.7 px peripheral corner bias described
  above. It affects absolute board orientation most; jitter, detection,
  occlusion and reacquisition are measured with the same bias on both the board
  and the baseline.
- The baseline is a raw per-frame single-marker IPPE solve. The retired app also
  applied temporal smoothing and continuity gating, so the baseline here is
  noisier than a real application's published output and has no prediction.
- Accuracy scenarios place the camera at the 2 m design range. The in-app
  synthetic webcam uses a 2.6 m framing so MediaPipe can see the full body.
- Runtime numbers are from one desktop x86_64 machine. They are a headroom
  check, not a guarantee for every machine.
- Real webcam, phone, SteamVR, Monado and Windows behaviour still require manual
  hardware validation; see [`TESTING.md`](TESTING.md).
