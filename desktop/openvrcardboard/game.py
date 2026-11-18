"""Built-in head-tracked stereo game used by the phone video pipeline.

The game is a small deterministic flight scene: rings rush toward the player,
stars stream past and a HUD reports score and speed. It is intentionally
dependency-free (NumPy + Pillow) and renders one view per eye, so the Android
app receives true stereo video rather than a duplicated mono frame.

The desktop passes a ``view_provider`` returning :class:`GameInput`. Phone
orientation is treated as relative to the first sample (auto-recenter), and a
left gamepad stick/trigger steers and boosts the flight. Without a phone or
gamepad the camera drifts gently so the scene is always alive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mathutil import inverse, multiply, normalize, to_euler_zyx_deg

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class GameInput:
    """Latest input snapshot consumed by the demo game."""

    orientation: Optional[np.ndarray] = None
    steer_x: float = 0.0
    steer_y: float = 0.0
    trigger: float = 0.0


def _font(size: int, bold: bool = False):
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _project(
    x: float,
    y: float,
    z: float,
    yaw: float,
    pitch: float,
    eye_offset: float,
    width: int,
    height: int,
) -> Optional[tuple[float, float, float]]:
    """Rotate a world point into head space and project it."""
    if z <= 0.25:
        return None
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1 = x * cy - z * sy
    z1 = x * sy + z * cy
    cp, sp = math.cos(pitch), math.sin(pitch)
    y1 = y * cp - z1 * sp
    z2 = y * sp + z1 * cp
    if z2 <= 0.25:
        return None
    x1 -= eye_offset
    focal = width * 0.95
    return (
        width * 0.5 + x1 * focal / z2,
        height * 0.5 - y1 * focal / z2,
        z2,
    )


class StereoGameSource:
    """Render the demo flight game directly in the requested stereo layout."""

    name = "game"
    stereo = True
    __test__ = False

    STAR_COUNT = 70
    RING_COUNT = 9

    def __init__(
        self,
        layout: str = "sbs",
        width: int = 640,
        height: int = 360,
        view_provider: Optional[Callable[[], GameInput]] = None,
        seed: int = 20260101,
    ):
        self.layout = layout if layout in ("mono", "sbs", "top_bottom") else "sbs"
        self.width = max(160, int(width))
        self.height = max(90, int(height))
        self.view_provider = view_provider
        self._rng = np.random.default_rng(seed)
        self._font = _font(max(12, self.height // 22), bold=True)
        self._small = _font(max(9, self.height // 38))
        self._hud = _font(max(10, self.height // 30), bold=True)
        self._background = self._make_background()
        self._stars = self._make_stars()
        self._rings = self._make_rings()
        self._reference = None
        self._yaw = 0.0
        self._pitch = 0.0
        self._steer = np.zeros(2, dtype=np.float64)
        self._trigger = 0.0
        self._started = time.monotonic()
        self._last_read = self._started
        self._frames = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _make_background(self) -> Image.Image:
        width, height = self.width, self.height
        horizon = int(height * 0.46)
        rows = np.arange(height, dtype=np.float32)[:, None]
        t = np.clip(rows / max(1, horizon), 0.0, 1.0)
        top = np.array([8, 12, 38], dtype=np.float32)
        mid = np.array([38, 40, 92], dtype=np.float32)
        low = np.array([16, 10, 34], dtype=np.float32)
        sky = top + (mid - top) * t
        ground_t = np.clip((rows - horizon) / max(1, height - horizon), 0.0, 1.0)
        ground = low + (np.array([4, 3, 12], dtype=np.float32) - low) * ground_t
        colors = np.where(rows < horizon, sky, ground).astype(np.uint8)
        image = Image.fromarray(
            np.repeat(colors[:, None, :], width, axis=1), mode=None
        )
        draw = ImageDraw.Draw(image)
        # Sun and horizon glow.
        draw.ellipse(
            [width * 0.62, horizon - height * 0.16, width * 0.78, horizon + height * 0.02],
            fill=(255, 158, 66),
        )
        draw.rectangle([0, horizon, width, horizon + max(1, height // 180)], fill=(255, 196, 120))
        for band in range(3):
            y = horizon + int(height * (0.04 + band * 0.035))
            draw.line([(0, y), (width, y)], fill=(45, 38, 70), width=max(1, height // 240))
        return image

    def _make_stars(self) -> np.ndarray:
        count = self.STAR_COUNT
        return np.column_stack(
            [
                self._rng.uniform(-12.0, 12.0, count),
                self._rng.uniform(-7.0, 7.0, count),
                self._rng.uniform(1.0, 46.0, count),
                self._rng.uniform(0.35, 1.0, count),
            ]
        )

    def _make_rings(self) -> np.ndarray:
        count = self.RING_COUNT
        return np.column_stack(
            [
                self._rng.uniform(-2.8, 2.8, count),
                self._rng.uniform(-1.8, 1.8, count),
                self._rng.uniform(0.0, 42.0, count),
                self._rng.uniform(0.55, 1.0, count),
            ]
        )

    # ------------------------------------------------------------------
    # Source API
    # ------------------------------------------------------------------
    def open(self) -> None:
        self._started = time.monotonic()
        self._last_read = self._started
        self._frames = 0

    def close(self) -> None:
        pass

    def read(self) -> Image.Image:
        now = time.monotonic()
        elapsed = max(0.0, now - self._started)
        self._update_view(elapsed)
        self._frames += 1

        if self.layout == "sbs":
            output = Image.new("RGB", (self.width * 2, self.height))
            output.paste(self._render_eye(-1.0, elapsed), (0, 0))
            output.paste(self._render_eye(1.0, elapsed), (self.width, 0))
        elif self.layout == "top_bottom":
            output = Image.new("RGB", (self.width, self.height * 2))
            output.paste(self._render_eye(-1.0, elapsed), (0, 0))
            output.paste(self._render_eye(1.0, elapsed), (0, self.height))
        else:
            output = self._render_eye(0.0, elapsed)
        self._last_read = now
        return output

    # ------------------------------------------------------------------
    # Input and rendering
    # ------------------------------------------------------------------
    def _update_view(self, elapsed: float) -> None:
        if self.view_provider is not None:
            try:
                game_input = self.view_provider()
            except Exception:
                game_input = None
            if game_input is not None:
                self._steer[0] = float(np.clip(game_input.steer_x, -1.0, 1.0))
                self._steer[1] = float(np.clip(game_input.steer_y, -1.0, 1.0))
                self._trigger = float(np.clip(game_input.trigger, 0.0, 1.0))
                quaternion = game_input.orientation
                if quaternion is not None:
                    quaternion = normalize(quaternion)
                    if self._reference is None:
                        self._reference = quaternion
                    relative = multiply(inverse(self._reference), quaternion)
                    yaw, pitch, _roll = to_euler_zyx_deg(relative)
                    target_yaw = math.radians(max(-55.0, min(55.0, yaw)))
                    target_pitch = math.radians(max(-40.0, min(40.0, pitch)))
                    self._yaw += (target_yaw - self._yaw) * 0.18
                    self._pitch += (target_pitch - self._pitch) * 0.18

        if self._reference is None:
            self._yaw = 0.22 * math.sin(elapsed * 0.33) + self._steer[0] * 0.6
            self._pitch = 0.09 * math.sin(elapsed * 0.21) + self._steer[1] * 0.4
        else:
            self._yaw += self._steer[0] * 0.55 * (1.0 / 60.0)
            self._pitch += self._steer[1] * 0.35 * (1.0 / 60.0)

    def _render_eye(self, eye_sign: float, elapsed: float) -> Image.Image:
        width, height = self.width, self.height
        image = self._background.copy()
        draw = ImageDraw.Draw(image)
        yaw, pitch = self._yaw, self._pitch
        eye_offset = eye_sign * 0.032
        speed = 5.0 + self._trigger * 7.0

        # Stars stream toward the camera; the modulo wraps each star.
        for x, y, z0, weight in self._stars:
            z = (z0 - elapsed * speed) % 46.0
            z = max(1.2, z)
            point = _project(x, y, z, yaw, pitch, eye_offset, width, height)
            if point is None:
                continue
            px, py, depth = point
            if not (-8 <= px <= width + 8 and -8 <= py <= height + 8):
                continue
            size = max(1, int(weight * 3.0 * min(1.0, 8.0 / depth)))
            shade = int(150 + 105 * min(1.0, 4.0 / depth))
            draw.ellipse(
                [px - size, py - size, px + size, py + size],
                fill=(shade, shade, min(255, shade + 30)),
            )

        # Rings approach the player and wrap around.
        ring_total = 42.0
        for x, y, z0, weight in self._rings:
            z = ring_total - ((elapsed * speed + z0) % ring_total)
            if z < 1.0:
                continue
            point = _project(x, y, z, yaw, pitch, eye_offset, width, height)
            if point is None:
                continue
            px, py, depth = point
            radius = max(2.0, width * 0.42 * weight / depth)
            if px + radius < 0 or px - radius > width or py + radius < 0 or py - radius > height:
                continue
            near = max(0.0, 1.0 - depth / 8.0)
            color = (
                int(70 + 185 * near),
                int(200 + 55 * near),
                int(255 - 120 * near),
            )
            line_width = max(2, int(2 + 8 * near))
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                outline=color,
                width=line_width,
            )
            # Cross ticks make the ring orientation readable.
            tick = max(2.0, radius * 0.18)
            draw.line([(px, py - radius - tick), (px, py - radius + tick)], fill=color, width=line_width)
            draw.line([(px, py + radius - tick), (px, py + radius + tick)], fill=color, width=line_width)

        self._draw_hud(draw, elapsed)
        return image

    def _draw_hud(self, draw: ImageDraw.ImageDraw, elapsed: float) -> None:
        width, height = self.width, self.height
        margin = max(6, width // 64)
        draw.rectangle(
            [margin, margin, width - margin, margin + max(20, height // 10)],
            fill=(5, 8, 18),
        )
        draw.text(
            (margin * 2, margin + margin // 2),
            "OPENVR CARDBOARD  //  SKYWAY",
            font=self._hud,
            fill=(120, 230, 255),
        )
        draw.text(
            (margin * 2, margin + height // 22),
            "DEMO FLIGHT",
            font=self._small,
            fill=(190, 205, 230),
        )

        score = int(elapsed * 120.0)
        boost = int(self._trigger * 100)
        draw.text(
            (margin * 2, height - margin - height // 14),
            f"SCORE {score:05d}",
            font=self._font,
            fill=(255, 220, 90),
        )
        draw.text(
            (width // 2, height - margin - height // 15),
            f"SPEED {int(60 + self._trigger * 45)}  BOOST {boost:03d}",
            font=self._small,
            fill=(150, 220, 255),
            anchor="ma",
        )

        # Center reticle.
        cx, cy = width // 2, height // 2
        arm = max(8, width // 40)
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(255, 255, 255), width=2)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(255, 255, 255), width=2)
        draw.ellipse([cx - arm * 2, cy - arm * 2, cx + arm * 2, cy + arm * 2], outline=(90, 180, 220), width=1)

        # Subtle scanlines keep the image lively and make motion obvious.
        step = max(6, height // 36)
        for y in range(0, height, step):
            shade = 18 + int(10 * math.sin(elapsed * 2.0 + y))
            draw.line([(0, y), (width, y)], fill=(shade, shade, min(255, shade + 16)))
