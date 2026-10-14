#!/usr/bin/env python3
"""Calibrate CARLA tire friction against effective road friction.

For each tire_friction value the ego vehicle (LEAD's Lincoln MKZ 2020) is
spawned on a long straight road, accelerated to a fixed speed, and braked
fully with zero steer. The stopping distance d gives the effective friction
coefficient mu_eff = v0^2 / (2 g d). The applied per-wheel value is read back
from the simulator so silent failures of apply_physics_control are visible.

Run against a dedicated CARLA server, for example:
    python scripts/winter/brake_test.py --port 2200 --out outputs/brake_test.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import carla

G = 9.81


def yaw_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def find_straight_spawn(world_map: carla.Map, length_m: float, max_yaw_dev_deg: float) -> carla.Transform:
    """Return a spawn point followed by at least length_m of straight, junction-free lane."""
    best = None
    for sp in world_map.get_spawn_points():
        wp = world_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        yaw0 = wp.transform.rotation.yaw
        travelled, ok, cur = 0.0, True, wp
        while travelled < length_m:
            nxt = cur.next(5.0)
            if not nxt:
                ok = False
                break
            cur = nxt[0]
            travelled += 5.0
            if cur.is_junction or abs(yaw_diff_deg(cur.transform.rotation.yaw, yaw0)) > max_yaw_dev_deg:
                ok = False
                break
        if ok:
            best = wp.transform
            break
    if best is None:
        raise RuntimeError(f"no straight road of {length_m} m found on {world_map.name}")
    best.location.z += 0.5
    return best


def set_friction(vehicle: carla.Vehicle, friction: float | None) -> list[float]:
    pc = vehicle.get_physics_control()
    if friction is not None:
        wheels = pc.wheels
        for w in wheels:
            w.tire_friction = float(friction)
        pc.wheels = wheels
        vehicle.apply_physics_control(pc)
    return [w.tire_friction for w in vehicle.get_physics_control().wheels]


def speed(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def lane_keep_steer(vehicle: carla.Vehicle, world_map: carla.Map) -> float:
    tf = vehicle.get_transform()
    wp = world_map.get_waypoint(tf.location, project_to_road=True)
    ahead = wp.next(8.0)[0] if wp.next(8.0) else wp
    dx = ahead.transform.location.x - tf.location.x
    dy = ahead.transform.location.y - tf.location.y
    target_yaw = math.degrees(math.atan2(dy, dx))
    err = yaw_diff_deg(target_yaw, tf.rotation.yaw)
    return max(-0.3, min(0.3, 0.02 * err))


def trial(world: carla.World, world_map: carla.Map, spawn: carla.Transform, blueprint,
          friction: float | None, v0: float, brake: float, dt: float, settle_ticks: int) -> dict:
    vehicle = world.try_spawn_actor(blueprint, spawn)
    if vehicle is None:
        raise RuntimeError("spawn failed")
    try:
        for _ in range(settle_ticks):
            world.tick()
        applied = set_friction(vehicle, friction)
        for _ in range(10):
            world.tick()

        # Acceleration phase with lane keeping; throttle limited so low friction does not spin forever.
        t_acc = 0.0
        while speed(vehicle) < v0:
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=lane_keep_steer(vehicle, world_map)))
            world.tick()
            t_acc += dt
            if t_acc > 60.0:
                return {"friction_set": friction, "friction_applied": applied, "error": "did not reach v0"}

        # Braking phase: zero steer, constant brake, integrate distance until standstill.
        v_start = speed(vehicle)
        start = vehicle.get_location()
        dist, t_brake = 0.0, 0.0
        while speed(vehicle) > 0.05:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=brake, steer=0.0))
            world.tick()
            dist += speed(vehicle) * dt
            t_brake += dt
            if t_brake > 60.0:
                break
        straight = vehicle.get_location().distance(start)
        mu = v_start * v_start / (2.0 * G * straight) if straight > 0 else float("nan")
        return {
            "friction_set": friction,
            "friction_applied": applied,
            "v_start": round(v_start, 3),
            "brake": brake,
            "stop_distance_m": round(straight, 3),
            "integrated_distance_m": round(dist, 3),
            "brake_time_s": round(t_brake, 3),
            "mu_eff": round(mu, 4),
            "accel_time_s": round(t_acc, 2),
        }
    finally:
        vehicle.destroy()
        world.tick()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=2200)
    parser.add_argument("--town", default="Town06")
    parser.add_argument("--blueprint", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--frictions", default="default,3.5,2.0,1.0,0.8,0.6,0.4,0.3,0.2,0.1")
    parser.add_argument("--speeds", default="10,13.89", help="Initial speeds in m/s")
    parser.add_argument("--brakes", default="1.0", help="Brake commands to test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.05, help="Fixed delta seconds (LEAD runs at 20 Hz)")
    parser.add_argument("--out", type=Path, default=Path("outputs/brake_test.csv"))
    args = parser.parse_args()

    client = carla.Client("localhost", args.port)
    client.set_timeout(120.0)
    world = client.load_world(args.town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)
    world.tick()
    world_map = world.get_map()
    blueprint = world.get_blueprint_library().find(args.blueprint)
    spawn = find_straight_spawn(world_map, length_m=350.0, max_yaw_dev_deg=3.0)
    print(f"server {client.get_server_version()}, map {world_map.name}, spawn {spawn.location}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["friction_set", "friction_applied", "v_start", "brake", "stop_distance_m",
              "integrated_distance_m", "brake_time_s", "mu_eff", "accel_time_s", "repeat", "error"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        try:
            for f_str in args.frictions.split(","):
                friction = None if f_str == "default" else float(f_str)
                for v0 in (float(s) for s in args.speeds.split(",")):
                    for brake in (float(b) for b in args.brakes.split(",")):
                        for rep in range(args.repeats):
                            res = trial(world, world_map, spawn, blueprint, friction, v0, brake, args.dt, 20)
                            res["repeat"] = rep
                            res["friction_set"] = f_str
                            res["friction_applied"] = "/".join(f"{x:.3f}" for x in res["friction_applied"])
                            writer.writerow(res)
                            fh.flush()
                            print(res, flush=True)
        finally:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)


if __name__ == "__main__":
    sys.exit(main())
