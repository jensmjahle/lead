#!/usr/bin/env python3
"""Render a preview grid of the snow overlay on real CARLA camera frames.

Spawns a static ego vehicle at a few spawn points, attaches a camera that
matches LEAD's front camera (384x384, 60 deg FOV, 2.25 m height), captures a
short frame sequence per location so the falling-snow state is realistic, and
writes one grid image: rows are locations, columns are overlay intensities.
The JPEG round-trip used by the agent is applied to every tile.

Run against a dedicated CARLA server, for example:
    python scripts/winter/snow_preview.py --port 2300 --town Town12 --out outputs/snow_preview
"""

from __future__ import annotations

import argparse
import importlib.util
import queue
from pathlib import Path

import carla
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_overlay_class():
    path = REPO / "src/lead/winter/snow_overlay.py"
    spec = importlib.util.spec_from_file_location("snow_overlay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SnowOverlay


def jpeg_roundtrip(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=2300)
    parser.add_argument("--town", default="Town10HD")
    parser.add_argument("--weather", default="ClearNoon", help="Name of a carla.WeatherParameters preset")
    parser.add_argument("--locations", type=int, default=4)
    parser.add_argument("--intensities", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/snow_preview"))
    args = parser.parse_args()

    SnowOverlay = load_overlay_class()
    intensities = [float(x) for x in args.intensities.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    client = carla.Client("localhost", args.port)
    client.set_timeout(180.0)
    world = client.load_world(args.town)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
    ego_bp.set_attribute("role_name", "hero")  # large maps stream tiles around the hero
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "384")
    cam_bp.set_attribute("image_size_y", "384")
    cam_bp.set_attribute("fov", "60")

    rng = np.random.default_rng(args.seed)
    spawns = world.get_map().get_spawn_points()
    picks = rng.choice(len(spawns), size=min(args.locations, len(spawns)), replace=False)

    rows = []
    try:
        for row_idx, sp_idx in enumerate(picks):
            ego = world.try_spawn_actor(ego_bp, spawns[int(sp_idx)])
            if ego is None:
                continue
            for _ in range(40):  # let streamed tiles and textures load around the ego
                world.tick()
            cam = world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=0.25, z=2.25)), attach_to=ego
            )
            frames: queue.Queue = queue.Queue()
            cam.listen(frames.put)
            try:
                raw = []
                for _ in range(args.warmup_frames):
                    world.tick()
                    img = frames.get(timeout=30.0)
                    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                    raw.append(arr[:, :, :3].copy())
            finally:
                # Batched destroy does not block on the synchronous server.
                cam.stop()
                client.apply_batch([carla.command.DestroyActor(cam.id), carla.command.DestroyActor(ego.id)])
                world.tick()

            tiles = []
            for inten in intensities:
                if inten <= 0.0:
                    out = raw[-1]
                else:
                    ov = SnowOverlay(384, 384, intensity=inten, seed=args.seed * 100 + row_idx, fps=20.0)
                    for frame in raw:  # run the sequence so particles are mid-fall
                        out = ov(frame)
                tile = jpeg_roundtrip(out, args.jpeg_quality)
                cv2.imwrite(str(args.out / f"loc{row_idx}_snow{inten:.1f}.jpg"), tile)
                tiles.append(label(tile, f"loc {row_idx}  snow {inten:.1f}"))
            rows.append(np.hstack(tiles))
            print(f"location {row_idx}: captured {len(raw)} frames", flush=True)
    finally:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

    grid = np.vstack(rows)
    grid_path = args.out / f"grid_{args.town}_{args.weather}.jpg"
    cv2.imwrite(str(grid_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"wrote {grid_path} ({grid.shape[1]}x{grid.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
