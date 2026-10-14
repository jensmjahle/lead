#!/usr/bin/env python3
"""Preview snow model v2 on a short autopilot drive in CARLA.

The ego vehicle (LEAD's Lincoln MKZ 2020) drives on autopilot with an RGB and a
semantic segmentation camera that match LEAD's front camera (384x384, 60 deg
FOV, 2.25 m). The recorded sequence is rendered with every preset; outputs:
    grid_<town>_<weather>.jpg   rows = time points, columns = raw, mask, presets
    video_<town>_<weather>.mp4  raw | light | medium | heavy, side by side
    tags_<town>.txt             semantic tag histogram (checks the tag mapping)
Every tile passes the agent's JPEG round-trip (quality 90).

Run against a dedicated CARLA server, for example:
    python scripts/winter/snow_preview_v2.py --port 2300 --town Town10HD --out outputs/snow_preview_v2
"""

from __future__ import annotations

import argparse
import importlib.util
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FRONT_CAMERA = {"pos": (0.25, 0.0, 2.25), "rot": (0.0, 0.0, 0.0), "fov": 60, "size": 384}
VIDEO_PRESETS = ("off", "light", "medium", "heavy")
GRID_PRESETS = ("light", "medium", "heavy", "fall_only", "cover_only")


def load_snow_module():
    path = REPO / "src/lead/winter/snow.py"
    spec = importlib.util.spec_from_file_location("lead_winter_snow", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lead_winter_snow"] = module
    spec.loader.exec_module(module)
    return module


def jpeg_roundtrip(img_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_vis(semantic: np.ndarray, snow) -> np.ndarray:
    """Cover strength per pixel as a grey image (white = full cover allowed)."""
    strength = snow._COVER_LUT[semantic]
    return cv2.cvtColor((strength * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=2300)
    parser.add_argument("--tm-port", type=int, default=8300)
    parser.add_argument("--town", default="Town10HD")
    parser.add_argument("--weather", default="ClearNoon")
    parser.add_argument("--frames", type=int, default=240, help="Recorded frames at 20 Hz")
    parser.add_argument("--warmup", type=int, default=60, help="Frames before recording starts")
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--spawn-index", type=int, default=-1, help="-1 = random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/snow_preview_v2"))
    args = parser.parse_args()

    snow = load_snow_module()
    args.out.mkdir(parents=True, exist_ok=True)
    size = FRONT_CAMERA["size"]

    client = carla.Client("localhost", args.port)
    client.set_timeout(180.0)
    world = client.load_world(args.town)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)

    actors = []
    rgb_q: queue.Queue = queue.Queue()
    sem_q: queue.Queue = queue.Queue()
    raw, sems, poses = [], [], []
    try:
        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
        ego_bp.set_attribute("role_name", "hero")
        spawns = world.get_map().get_spawn_points()
        rng = np.random.default_rng(args.seed)
        order = [args.spawn_index] if args.spawn_index >= 0 else list(rng.permutation(len(spawns)))
        ego = None
        for idx in order:
            ego = world.try_spawn_actor(ego_bp, spawns[int(idx)])
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("could not spawn ego")
        actors.append(ego)

        tf = carla.Transform(carla.Location(*FRONT_CAMERA["pos"]),
                             carla.Rotation(roll=FRONT_CAMERA["rot"][0], pitch=FRONT_CAMERA["rot"][1],
                                            yaw=FRONT_CAMERA["rot"][2]))
        for kind, q in (("sensor.camera.rgb", rgb_q), ("sensor.camera.semantic_segmentation", sem_q)):
            bp = bp_lib.find(kind)
            bp.set_attribute("image_size_x", str(size))
            bp.set_attribute("image_size_y", str(size))
            bp.set_attribute("fov", str(FRONT_CAMERA["fov"]))
            cam = world.spawn_actor(bp, tf, attach_to=ego)
            cam.listen(q.put)
            actors.append(cam)

        ego.set_autopilot(True, args.tm_port)
        tm.ignore_lights_percentage(ego, 100.0)  # keep moving through junctions

        for step in range(args.warmup + args.frames):
            frame = world.tick()
            rgb = rgb_q.get(timeout=30.0)
            sem = sem_q.get(timeout=30.0)
            while rgb.frame < frame:
                rgb = rgb_q.get(timeout=30.0)
            while sem.frame < frame:
                sem = sem_q.get(timeout=30.0)
            if step < args.warmup:
                continue
            raw.append(np.frombuffer(rgb.raw_data, np.uint8).reshape(size, size, 4)[:, :, :3].copy())
            sems.append(np.frombuffer(sem.raw_data, np.uint8).reshape(size, size, 4)[:, :, 2].copy())
            t = ego.get_transform()
            poses.append(snow.EgoPose(t.location.x, t.location.y, t.rotation.yaw))
    finally:
        client.apply_batch([carla.command.DestroyActor(a.id) for a in actors])
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

    tags, counts = np.unique(np.stack(sems), return_counts=True)
    with open(args.out / f"tags_{args.town}.txt", "w") as fh:
        for tag, cnt in zip(tags, counts):
            fh.write(f"{int(tag):3d} {cnt / counts.sum() * 100:6.2f} %\n")

    geometry = snow.CameraGeometry(size, size, FRONT_CAMERA["fov"], FRONT_CAMERA["pos"], FRONT_CAMERA["rot"])
    rendered = {}
    for name in set(VIDEO_PRESETS) | set(GRID_PRESETS):
        fall, cover = snow.PRESETS[name]
        renderer = snow.SnowRenderer(size, size, fall, cover, world_seed=args.seed,
                                     particle_seed=args.seed, geometry=geometry)
        rendered[name] = [jpeg_roundtrip(renderer(r, s, p) if name != "off" else r)
                          for r, s, p in zip(raw, sems, poses)]

    picks = np.linspace(0, len(raw) - 1, args.grid_rows).astype(int)
    rows = []
    for i in picks:
        tiles = [label(rendered["off"][i], f"t={i / 20:.1f}s raw"), label(mask_vis(sems[i], snow), "cover mask")]
        tiles += [label(rendered[name][i], name) for name in GRID_PRESETS]
        rows.append(np.hstack(tiles))
    grid_path = args.out / f"grid_{args.town}_{args.weather}.jpg"
    cv2.imwrite(str(grid_path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    video_path = args.out / f"video_{args.town}_{args.weather}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                             (size * len(VIDEO_PRESETS), size))
    for i in range(len(raw)):
        writer.write(np.hstack([label(rendered[name][i], name) for name in VIDEO_PRESETS]))
    writer.release()
    print(f"wrote {grid_path}, {video_path}, {len(raw)} frames", flush=True)


if __name__ == "__main__":
    main()
