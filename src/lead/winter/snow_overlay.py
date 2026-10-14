"""Image-space snow overlay for camera frames.

Purpose: approximate how a camera sees winter conditions without changing the
simulator. The overlay is applied to the RGB frame before it reaches the driving
model. It does NOT change scene geometry, LiDAR/radar, or vehicle physics, so it
must be paired with a friction change (see the RQ2 design) to test dynamics.

Components, each controlled by a single intensity value in [0, 1]:
    1. Atmosphere: desaturation, cold colour shift, lifted blacks, horizon haze.
    2. Ground cover: low-saturation, dark-to-mid pixels in the lower image are
       blended toward white. A semantic road mask can be passed in; without one,
       a colour heuristic is used and should be treated as a rough fallback.
    3. Falling snow: particles in three depth layers with motion streaks. The
       particle state persists across frames so closed-loop video stays coherent.

Usage:
    overlay = SnowOverlay(height=384, width=384, intensity=0.6, seed=0)
    for frame in frames:                  # uint8 BGR images
        out = overlay(frame, road_mask=None)
"""

from __future__ import annotations

import cv2
import numpy as np


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class SnowOverlay:
    """Stateful snow overlay. One instance per camera per episode."""

    # (relative size, relative speed, brightness, share of particles)
    _LAYERS = (
        (0.8, 0.5, 0.55, 0.55),  # far: small, slow, dim
        (1.6, 1.0, 0.80, 0.30),  # mid
        (3.2, 1.8, 1.00, 0.15),  # near: large, fast, bright
    )

    def __init__(
        self,
        height: int,
        width: int,
        intensity: float = 0.5,
        seed: int = 0,
        fps: float = 20.0,
        wind: float = 0.15,
    ) -> None:
        self.h = int(height)
        self.w = int(width)
        self.intensity = _clip01(intensity)
        self.dt = 1.0 / float(fps)
        self.wind = float(wind)
        self.rng = np.random.default_rng(seed)

        # Particle count scales with area and intensity.
        base = self.h * self.w / 1600.0
        n_total = int(base * (0.2 + 2.8 * self.intensity))
        self._layers = []
        for size, speed, bright, share in self._LAYERS:
            n = max(1, int(n_total * share))
            self._layers.append(
                {
                    "size": size,
                    "speed": speed,
                    "bright": bright,
                    "x": self.rng.uniform(0, self.w, n),
                    "y": self.rng.uniform(0, self.h, n),
                    "phase": self.rng.uniform(0, 2 * np.pi, n),
                }
            )

        # Static multi-octave noise for patchy ground cover, stored as a z-score
        # so the cover threshold can be set from a target coverage fraction.
        field = np.zeros((self.h, self.w), dtype=np.float32)
        for sigma, weight in ((3.0, 0.5), (12.0, 0.35), (40.0, 0.15)):
            n = self.rng.standard_normal((self.h, self.w)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sigma)
            n /= max(float(n.std()), 1e-6)
            field += weight * n
        field = (field - field.mean()) / max(float(field.std()), 1e-6)
        self._cover_noise = field

        # Cover threshold depends only on intensity, so compute it once.
        c = _clip01((self.intensity - 0.15) / 0.85)
        target = 0.15 + 0.80 * c
        self._cover_thr = float(np.quantile(field, 1.0 - target))

    # ------------------------------------------------------------------
    def _atmosphere(self, img: np.ndarray) -> np.ndarray:
        i = self.intensity
        f = img.astype(np.float32) / 255.0

        # Desaturate and shift toward a cold tone (BGR order: blue channel first).
        gray = f.mean(axis=2, keepdims=True)
        sat = 1.0 - 0.45 * i
        f = gray + (f - gray) * sat
        f[..., 0] *= 1.0 + 0.10 * i
        f[..., 2] *= 1.0 - 0.06 * i

        # Lift blacks and reduce contrast (overcast sky).
        f = 0.12 * i + f * (1.0 - 0.12 * i)

        # Horizon haze: stronger near the top of the lower half of the image.
        rows = np.linspace(0.0, 1.0, self.h, dtype=np.float32)[:, None, None]
        haze_w = np.exp(-((rows - 0.45) ** 2) / (2 * 0.18**2)) * (0.15 + 0.35 * i)
        haze_color = np.array([0.86, 0.85, 0.83], dtype=np.float32)
        f = f * (1.0 - haze_w) + haze_color * haze_w
        return np.clip(f, 0.0, 1.0)

    def _ground_cover(self, f: np.ndarray, road_mask: np.ndarray | None) -> np.ndarray:
        i = self.intensity
        c = _clip01((i - 0.15) / 0.85)  # normalised cover level
        if c <= 0.0:
            return f

        if road_mask is not None:
            mask = (road_mask > 0).astype(np.float32)
        else:
            # Heuristic fallback: low saturation, mid brightness (asphalt-like),
            # lower 55 % of the image. Very dark pixels (vehicle bodies, deep
            # shadow) are excluded. Replace with a semantic road mask when possible.
            hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1].astype(np.float32) / 255.0
            val = hsv[..., 2].astype(np.float32) / 255.0
            rows = np.arange(self.h, dtype=np.float32)[:, None] / self.h
            mask = (
                (sat < 0.30) & (val > 0.18) & (val < 0.70) & (rows > 0.45)
            ).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 3)

        # Patchy cover: threshold the noise at the quantile that gives a target
        # covered fraction, with a soft edge. Cover grows from patches to full.
        patch = 1.0 / (1.0 + np.exp(-(self._cover_noise - self._cover_thr) / 0.35))
        depth = 0.25 + 0.65 * patch
        alpha = (mask * depth * min(1.0, 0.35 + c))[..., None]
        snow_color = np.array([0.93, 0.93, 0.95], dtype=np.float32)
        return f * (1.0 - alpha) + snow_color * alpha

    def _particles(self, f: np.ndarray) -> np.ndarray:
        layer_img = np.zeros((self.h, self.w), dtype=np.float32)
        for lay in self._layers:
            fall = lay["speed"] * 160.0 * self.dt * (self.h / 900.0)
            drift = self.wind * fall + 0.6 * np.sin(lay["phase"])
            lay["y"] = (lay["y"] + fall) % self.h
            lay["x"] = (lay["x"] + drift) % self.w
            lay["phase"] = lay["phase"] + 0.15

            streak = max(1, int(round(lay["speed"] * 3.0 * self.h / 900.0)))
            radius = max(1, int(round(lay["size"] * self.h / 900.0)))
            tmp = np.zeros_like(layer_img)
            xs = lay["x"].astype(np.int32)
            ys = lay["y"].astype(np.int32)
            tmp[ys, xs] = lay["bright"]
            # Motion streak along the fall direction, then soften.
            kernel = np.zeros((streak * 2 + 1, streak * 2 + 1), dtype=np.float32)
            cv2.line(kernel, (streak, 0), (streak + int(self.wind * streak), 2 * streak), 1.0, 1)
            kernel /= max(kernel.sum(), 1.0)
            tmp = cv2.filter2D(tmp, -1, kernel * (streak + 1))
            tmp = cv2.GaussianBlur(tmp, (0, 0), radius * 0.6)
            layer_img = np.maximum(layer_img, np.clip(tmp * (1.5 + radius), 0.0, 1.0))

        a = (layer_img * (0.35 + 0.55 * self.intensity))[..., None]
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return f * (1.0 - a) + white * a

    # ------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray, road_mask: np.ndarray | None = None) -> np.ndarray:
        if img_bgr.shape[:2] != (self.h, self.w):
            raise ValueError(f"expected {(self.h, self.w)}, got {img_bgr.shape[:2]}")
        f = self._atmosphere(np.ascontiguousarray(img_bgr))
        f = self._ground_cover(f, road_mask)
        f = self._particles(f)
        return (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
