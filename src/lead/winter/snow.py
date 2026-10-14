"""Image-space snow model v2.1 for closed-loop camera perturbation.

Changes with respect to v1 (`snow_overlay.py`):
    * Ground cover is restricted to ground classes from CARLA's semantic
      segmentation camera instead of a colour heuristic.
    * Ground cover is anchored in the world: every ground pixel is projected to a
      world position using the camera geometry and the ego pose, and the cover
      pattern is sampled there, so patches stay on the asphalt while driving.
    * The snow colour is modulated by the scene luminance, so shadows survive.
    * Falling snow is drawn at a size that stays visible at 384x384.
    * Falling snow (with atmosphere) and ground cover have separate levels, so
      their effects can be attributed separately.

The semantic image and the ego pose are privileged simulator information. They
are used only to render the perturbation and are never passed to the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# CARLA 0.9.14+ semantic tags (CityScapes-based palette).
TAG_ROAD = 1
TAG_SIDEWALK = 2
TAG_TERRAIN = 10
TAG_ROADLINE = 24
TAG_GROUND = 25

# Maximum cover strength per class. Roads are compacted and partly cleared by
# traffic, so they never reach full cover; sidewalks and terrain can.
_COVER_LUT = np.zeros(256, dtype=np.float32)
_COVER_LUT[TAG_ROAD] = 0.80
_COVER_LUT[TAG_ROADLINE] = 0.80
_COVER_LUT[TAG_SIDEWALK] = 1.00
_COVER_LUT[TAG_TERRAIN] = 1.00
_COVER_LUT[TAG_GROUND] = 1.00
_ROAD_LUT = np.zeros(256, dtype=np.float32)
_ROAD_LUT[TAG_ROAD] = 1.0
_ROAD_LUT[TAG_ROADLINE] = 1.0

# Named severity levels: (fall, cover). Fall controls falling snow and the
# atmosphere; cover controls snow on the ground.
PRESETS: dict[str, tuple[float, float]] = {
    "off": (0.0, 0.0),
    "light": (0.3, 0.3),
    "medium": (0.6, 0.6),
    "heavy": (0.9, 0.9),
    "fall_only": (0.6, 0.0),
    "cover_only": (0.0, 0.6),
}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass(frozen=True)
class EgoPose:
    """Ego pose in the CARLA world frame (metres, degrees)."""

    x: float
    y: float
    yaw_deg: float


class CameraGeometry:
    """Ground-plane projection of every pixel of a pinhole camera.

    Uses the CARLA convention: x forward, y right, z up; rotation as
    (roll, pitch, yaw) in degrees. The ground is assumed flat at z = 0 in the ego
    frame (vehicle origin on the ground), which is adequate for texturing.
    """

    def __init__(self, width: int, height: int, fov_deg: float, pos: tuple, rot: tuple,
                 max_range_m: float = 120.0) -> None:
        f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float32) + 0.5,
                             np.arange(height, dtype=np.float32) + 0.5)
        rx = np.ones_like(uu)
        ry = (uu - width / 2.0) / f
        rz = -(vv - height / 2.0) / f
        pitch, yaw = math.radians(rot[1]), math.radians(rot[2])
        x1 = rx * math.cos(pitch) - rz * math.sin(pitch)
        z1 = rx * math.sin(pitch) + rz * math.cos(pitch)
        x2 = x1 * math.cos(yaw) - ry * math.sin(yaw)
        y2 = x1 * math.sin(yaw) + ry * math.cos(yaw)
        height_m = float(pos[2])
        valid = z1 < -1e-3
        t = np.where(valid, height_m / np.maximum(-z1, 1e-3), 0.0)
        self.gx = (pos[0] + t * x2).astype(np.float32)
        self.gy = (pos[1] + t * y2).astype(np.float32)
        self.dist = np.hypot(self.gx, self.gy).astype(np.float32)
        self.valid = valid & (self.dist < max_range_m)

    def world(self, pose: EgoPose) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(math.radians(pose.yaw_deg)), math.sin(math.radians(pose.yaw_deg))
        wx = pose.x + c * self.gx - s * self.gy
        wy = pose.y + s * self.gx + c * self.gy
        return wx, wy


_NOISE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def periodic_noise(size: int, seed: int, beta: float = 2.2) -> np.ndarray:
    """Seamlessly tiling 1/f^beta noise, z-scored. Shared by all cameras of a seed."""
    key = (size, seed)
    if key not in _NOISE_CACHE:
        rng = np.random.default_rng(seed)
        spectrum = np.fft.fft2(rng.standard_normal((size, size)))
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size)))
        k[0, 0] = 1.0
        spectrum *= k ** (-beta / 2.0)
        spectrum[0, 0] = 0.0
        field = np.real(np.fft.ifft2(spectrum)).astype(np.float32)
        _NOISE_CACHE[key] = (field - field.mean()) / max(float(field.std()), 1e-6)
    return _NOISE_CACHE[key]


class SnowRenderer:
    """Stateful snow renderer. One instance per camera per episode."""

    # (size in px at 384 height, fall speed in px/s, brightness, share of particles)
    _LAYERS = (
        (1.0, 45.0, 0.55, 0.55),   # far
        (1.6, 100.0, 0.80, 0.30),  # mid
        (2.0, 190.0, 0.90, 0.15),  # near
    )

    def __init__(
        self,
        height: int,
        width: int,
        fall: float,
        cover: float,
        world_seed: int = 0,
        particle_seed: int = 0,
        fps: float = 20.0,
        geometry: CameraGeometry | None = None,
        noise_size: int = 1024,
        noise_res_m: float = 0.08,
        tracks: float = 0.45,
        wind: float = 0.15,
    ) -> None:
        self.h, self.w = int(height), int(width)
        self.fall = _clip01(fall)
        self.cover = _clip01(cover)
        self.dt = 1.0 / float(fps)
        self.geometry = geometry
        self.noise = periodic_noise(noise_size, world_seed)
        self.noise_size = noise_size
        self.noise_res_m = noise_res_m
        self.tracks = _clip01(tracks)
        self.wind = wind
        self.scale = self.h / 384.0
        rng = np.random.default_rng(particle_seed)

        n_total = int(self.h * self.w / 1000.0 * (0.3 + 2.0 * self.fall)) if self.fall > 0 else 0
        self._layers = []
        for size, speed, bright, share in self._LAYERS:
            n = max(1, int(n_total * share)) if n_total else 0
            self._layers.append({
                "size": size, "speed": speed, "bright": bright,
                "x": rng.uniform(0, self.w, n), "y": rng.uniform(0, self.h, n),
                "phase": rng.uniform(0, 2 * np.pi, n),
            })

        # Coverage threshold: the noise quantile that leaves the target share uncovered.
        target = 0.25 + 0.70 * self.cover
        self._thr = float(np.quantile(self.noise[::4, ::4], 1.0 - target))

        # Ego-frame wheel-track stripes on the road (compacted, darker snow).
        if geometry is not None:
            lateral = np.minimum(np.abs(geometry.gy - 0.8), np.abs(geometry.gy + 0.8))
            near = np.clip(1.0 - (geometry.dist - 25.0) / 25.0, 0.0, 1.0)
            ahead = (geometry.gx > 0.0).astype(np.float32)
            self._track = (np.exp(-(lateral / 0.40) ** 2) * near * ahead * geometry.valid).astype(np.float32)
        else:
            self._track = None

    # ------------------------------------------------------------------
    def _atmosphere(self, f: np.ndarray) -> np.ndarray:
        a = self.fall
        if a <= 0.0:
            return f
        gray = f.mean(axis=2, keepdims=True)
        f = gray + (f - gray) * (1.0 - 0.45 * a)
        f[..., 0] *= 1.0 + 0.08 * a  # BGR: blue up
        f[..., 2] *= 1.0 - 0.05 * a  # red down
        f = 0.10 * a + f * (1.0 - 0.10 * a)
        rows = np.linspace(0.0, 1.0, self.h, dtype=np.float32)[:, None, None]
        haze_w = np.exp(-((rows - 0.5) ** 2) / (2 * 0.16 ** 2)) * (0.10 + 0.40 * a)
        haze = np.array([0.86, 0.85, 0.83], dtype=np.float32)
        return f * (1.0 - haze_w) + haze * haze_w

    def _cover_pattern(self, pose: EgoPose | None) -> np.ndarray:
        """Patch value in [0, 1] per pixel, world-anchored when geometry and pose exist."""
        if self.geometry is None or pose is None:
            return np.full((self.h, self.w), 1.0 / (1.0 + math.exp(self._thr / 0.45)), np.float32)
        wx, wy = self.geometry.world(pose)
        mx = np.mod(wx / self.noise_res_m, self.noise_size).astype(np.float32)
        my = np.mod(wy / self.noise_res_m, self.noise_size).astype(np.float32)
        n = cv2.remap(self.noise, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        detail = np.clip(1.0 - (self.geometry.dist - 20.0) / 40.0, 0.0, 1.0) * self.geometry.valid
        n = n * detail
        return 1.0 / (1.0 + np.exp(-(n - self._thr) / 0.45))

    def _ground_cover(self, f: np.ndarray, semantic: np.ndarray | None, pose: EgoPose | None) -> np.ndarray:
        c = self.cover
        if c <= 0.0 or semantic is None:
            return f
        strength = _COVER_LUT[semantic]
        patch = self._cover_pattern(pose)
        if self._track is not None and self.tracks > 0.0:
            # Modulate track depth with the cover pattern so tracks are not perfectly regular.
            depth = self.tracks * (0.6 + 0.4 * patch)
            patch = patch * (1.0 - depth * self._track * _ROAD_LUT[semantic])
        alpha = strength * patch * (0.35 + 0.65 * c)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8 * self.scale)[..., None]
        lum = f.mean(axis=2, keepdims=True)
        shade = np.clip(0.55 + 1.0 * lum, 0.50, 1.0)
        snow = np.array([0.97, 0.95, 0.94], dtype=np.float32) * shade  # BGR, slightly blue
        return f * (1.0 - alpha) + snow * alpha

    def _particles(self, f: np.ndarray) -> np.ndarray:
        if self.fall <= 0.0:
            return f
        layer = np.zeros((self.h, self.w), dtype=np.float32)
        for lay in self._layers:
            if len(lay["x"]) == 0:
                continue
            fall_px = lay["speed"] * self.scale * self.dt
            drift = self.wind * fall_px + 0.5 * np.sin(lay["phase"])
            lay["y"] = (lay["y"] + fall_px) % self.h
            lay["x"] = (lay["x"] + drift) % self.w
            lay["phase"] = lay["phase"] + 0.15
            thickness = max(1, int(round(lay["size"] * self.scale)))
            streak = max(1.0, fall_px * 0.8)
            tmp = np.zeros_like(layer)
            for x, y in zip(lay["x"], lay["y"]):
                p0 = (int(x), int(y))
                p1 = (int(x + self.wind * streak), int(y + streak))
                cv2.line(tmp, p0, p1, float(lay["bright"]), thickness, cv2.LINE_AA)
            tmp = cv2.GaussianBlur(tmp, (0, 0), 0.4 + 0.3 * lay["size"] * self.scale)
            layer = np.maximum(layer, tmp)
        a = np.clip(layer * (0.45 + 0.50 * self.fall) * 1.25, 0.0, 0.90)[..., None]
        return f * (1.0 - a) + a

    # ------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray, semantic: np.ndarray | None = None,
                 pose: EgoPose | None = None) -> np.ndarray:
        if img_bgr.shape[:2] != (self.h, self.w):
            raise ValueError(f"expected {(self.h, self.w)}, got {img_bgr.shape[:2]}")
        if semantic is not None and semantic.shape[:2] != (self.h, self.w):
            raise ValueError("semantic image must match the RGB image size")
        f = np.ascontiguousarray(img_bgr).astype(np.float32) / 255.0
        f = self._ground_cover(f, semantic, pose)  # cover first, then atmosphere over it
        f = self._atmosphere(f)
        f = self._particles(f)
        return (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def levels_from_env(preset: str, fall: str, cover: str) -> tuple[float, float]:
    """Resolve (fall, cover) from a preset name and optional explicit overrides."""
    base_fall, base_cover = PRESETS.get(preset or "off", (0.0, 0.0))
    return (float(fall) if fall not in ("", None) else base_fall,
            float(cover) if cover not in ("", None) else base_cover)
