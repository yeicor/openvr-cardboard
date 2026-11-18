"""Small, dependency-free temporal filters used by tracking and fusion.

The One-Euro filters (Casiez et al., CHI 2012) smooth slow motion heavily while
staying responsive when the signal moves quickly. They are shared by marker,
landmark and device-pose smoothing so every tracking stage has the same
behaviour and can be tuned from one place.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .mathutil import (
    IDENTITY,
    angle_between,
    clamp,
    normalize,
    slerp,
)


def _finite_vector(value, length: int, fallback=None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(length)
    except (TypeError, ValueError):
        result = None
    if result is None or not np.all(np.isfinite(result)):
        if fallback is None:
            return np.zeros(length, dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64).reshape(length).copy()
    return result


def _smoothing_factor(cutoff_hz: float, dt: float) -> float:
    """One-Euro low-pass alpha. A zero cutoff disables smoothing."""
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0 or dt <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return clamp(dt / (dt + tau), 0.0, 1.0)


class OneEuroFilter:
    """Scalar One-Euro filter (Casiez et al., CHI 2012)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._last_time = 0.0
        self._last_raw = 0.0
        self._last_value = 0.0
        self._derivative = 0.0

    @property
    def initialized(self) -> bool:
        return self._initialized

    def filter(self, value: float, timestamp: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            return self._last_value if self._initialized else 0.0
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            return value

        if not self._initialized:
            self._initialized = True
            self._last_time = timestamp
            self._last_raw = value
            self._last_value = value
            self._derivative = 0.0
            return value

        dt = timestamp - self._last_time
        if dt <= 0.0:
            # Duplicate or out-of-order sample: do not let time run backwards.
            self._last_raw = value
            return self._last_value
        if dt > 1.0:
            # Reacquisition after a long gap should not drag old state along.
            self.reset()
            return self.filter(value, timestamp)

        raw_derivative = (value - self._last_raw) / dt
        alpha_d = _smoothing_factor(self.derivative_cutoff, dt)
        self._derivative = alpha_d * raw_derivative + (1.0 - alpha_d) * self._derivative

        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        alpha = _smoothing_factor(cutoff, dt)
        self._last_value = alpha * value + (1.0 - alpha) * self._last_value
        self._last_raw = value
        self._last_time = timestamp
        return self._last_value

    @property
    def value(self) -> float:
        return self._last_value


class OneEuroVec3:
    """Component-wise One-Euro filter for 3-vectors."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ):
        self.filters = [
            OneEuroFilter(min_cutoff, beta, derivative_cutoff) for _ in range(3)
        ]

    @property
    def initialized(self) -> bool:
        return all(item.initialized for item in self.filters)

    def reset(self) -> None:
        for item in self.filters:
            item.reset()

    def filter(self, value, timestamp: float) -> np.ndarray:
        vector = _finite_vector(value, 3)
        return np.array(
            [item.filter(vector[index], timestamp) for index, item in enumerate(self.filters)],
            dtype=np.float64,
        )


class AdaptiveQuaternionFilter:
    """One-Euro-style adaptive low-pass filter on orientations.

    The cutoff rises with angular speed so fast head movements stay responsive
    while slow jitter is smoothed. Filtering happens with slerp, which keeps the
    quaternion normalised and avoids the discontinuities of filtering Euler
    angles or rotation vectors around a fixed reference.
    """

    def __init__(self, min_cutoff: float = 2.0, beta: float = 0.1):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.reset()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self._initialized = False
        self._last_time = 0.0
        self._last = IDENTITY.copy()
        self._pending: Optional[np.ndarray] = None
        self._pending_count = 0

    def filter(self, quaternion, timestamp: float) -> np.ndarray:
        q = normalize(quaternion)
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            return q

        if not self._initialized:
            self._initialized = True
            self._last_time = timestamp
            self._last = q
            self._pending = None
            self._pending_count = 0
            return q

        dt = timestamp - self._last_time
        if dt <= 0.0:
            return self._last
        if dt > 1.0:
            self.reset()
            return self.filter(q, timestamp)

        angle = angle_between(self._last, q)
        if angle > math.radians(140.0):
            # A >140 degree single-frame jump is almost always a decoding or
            # tracking glitch.  Never raise alpha to chase it: hold the last
            # good orientation and wait for a short run of consistent samples
            # before accepting that the device really was teleported.
            if (
                self._pending is not None
                and angle_between(self._pending, q) <= math.radians(25.0)
            ):
                self._pending_count += 1
                self._pending = q
            else:
                self._pending = q
                self._pending_count = 1
            self._last_time = timestamp
            if self._pending_count >= 3:
                self._last = q
                self._pending = None
                self._pending_count = 0
            return self._last

        self._pending = None
        self._pending_count = 0
        angular_speed = angle / dt
        cutoff = self.min_cutoff + self.beta * angular_speed
        if cutoff <= 0.0:
            alpha = 1.0
        else:
            alpha = clamp(1.0 - math.exp(-2.0 * math.pi * cutoff * dt), 0.0, 1.0)

        self._last = slerp(self._last, q, alpha)
        self._last_time = timestamp
        return self._last

    @property
    def value(self) -> np.ndarray:
        return self._last


class LandmarkSmoother:
    """One-Euro smoothing for a fixed-length set of 2D or 3D landmarks.

    MediaPipe already has optional temporal smoothing, but it is deliberately
    conservative and does not stop per-frame IPPE/pose jitter. Filtering the
    landmark coordinates before geometry also stabilises the depth estimated
    from shoulder/palm widths. Missing or low-visibility landmarks can be
    marked non-finite so the caller can skip them without resetting the filter
    for every other point.
    """

    def __init__(
        self,
        count: int,
        dimensions: int = 2,
        min_cutoff: float = 6.0,
        beta: float = 0.6,
        derivative_cutoff: float = 1.5,
    ):
        self.count = max(1, int(count))
        self.dimensions = max(1, int(dimensions))
        self.filters = [
            [
                OneEuroFilter(min_cutoff, beta, derivative_cutoff)
                for _ in range(self.dimensions)
            ]
            for _ in range(self.count)
        ]

    def reset(self) -> None:
        for row in self.filters:
            for item in row:
                item.reset()

    def filter(
        self,
        points,
        timestamp: float,
        weights: Optional[np.ndarray] = None,
        min_weight: float = 0.0,
    ) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.count:
            raise ValueError(f"expected {self.count} landmarks")
        if values.shape[1] < self.dimensions:
            raise ValueError(f"expected at least {self.dimensions} coordinates")
        output = values[:, : self.dimensions].copy()
        for index in range(self.count):
            if weights is not None:
                try:
                    weight = float(weights[index])
                except (TypeError, ValueError, IndexError):
                    weight = 0.0
                if not math.isfinite(weight) or weight < min_weight:
                    output[index] = np.nan
                    continue
            for axis in range(self.dimensions):
                output[index, axis] = self.filters[index][axis].filter(
                    output[index, axis], timestamp
                )
        return output
