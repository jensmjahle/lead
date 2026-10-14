#!/usr/bin/env python3
"""Install the winter hooks into the driving agent (idempotent).

Adds three methods to `AbstractDrivingAgent` and three call sites:
    * `_winter_step()` at the start of `_cameras`: reads the settings once per
      route, applies ego tire friction, spawns semantic cameras for the snow
      cover mask, and records the ego pose every step.
    * `_winter_effects()` per policy camera, before the storage JPEG round-trip:
      renders snow model v2 (`lead.winter.snow`) and optionally dumps frames.
    * `_winter_cleanup()` at the start of `destroy`: removes the semantic cameras.

Runtime switches (environment variables, read once per route):
    LEAD_SNOW              preset: off | light | medium | heavy | fall_only | cover_only
    LEAD_SNOW_FALL         optional float override of the falling-snow level
    LEAD_SNOW_COVER        optional float override of the ground-cover level
    LEAD_SNOW_SEED         integer seed (world pattern and per-camera particles)
    LEAD_SNOW_DUMP_DIR     directory for sample frames, or "auto" for <SAVE_PATH>/snow_frames
    LEAD_SNOW_DUMP_EVERY   dump every N steps (default 40)
    LEAD_TIRE_FRICTION     per-wheel tire_friction for the ego vehicle (CARLA default 3.5)

With none of these set the agent behaves exactly as before. The semantic images
and the ego pose are privileged and only used to render the perturbation; the
policy never receives them.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "src/lead/api/abstract_driving_agent.py"
MARKER = "def _winter_setup("
V1_MARKER = "def _winter_effects(self, camera_id, image: np.ndarray) -> np.ndarray:\n        \"\"\"Apply the image-space snow overlay"

METHODS = '''    # ------------------------------------------------------------------
    # Winter perturbations (installed by scripts/winter/install_winter_hooks.py)
    # ------------------------------------------------------------------
    def _winter_setup(self) -> None:
        """Read the winter settings once per route; apply friction; spawn mask cameras."""
        from lead.winter import snow as winter_snow

        env = os.environ
        fall, cover = winter_snow.levels_from_env(
            env.get("LEAD_SNOW", "off"), env.get("LEAD_SNOW_FALL", ""), env.get("LEAD_SNOW_COVER", "")
        )
        dump_dir = env.get("LEAD_SNOW_DUMP_DIR", "")
        if dump_dir == "auto":
            dump_dir = os.path.join(env.get("SAVE_PATH", "."), "snow_frames")
        self._winter = {
            "module": winter_snow,
            "fall": fall,
            "cover": cover,
            "seed": int(env.get("LEAD_SNOW_SEED", "0") or 0),
            "dump_dir": dump_dir,
            "dump_every": max(1, int(env.get("LEAD_SNOW_DUMP_EVERY", "40") or 40)),
            "renderers": {},
            "sem_actors": [],
            "sem_latest": {},
            "sem_misses": 0,
            "pose": None,
        }
        hero = CarlaDataProvider.get_hero_actor()

        friction = env.get("LEAD_TIRE_FRICTION", "")
        if friction and hero is not None:
            value = float(friction)
            physics = hero.get_physics_control()
            wheels = physics.wheels
            for wheel in wheels:
                wheel.tire_friction = value
            physics.wheels = wheels
            hero.apply_physics_control(physics)
            LOG.info(f"Winter hook: ego tire_friction set to {value}")

        if fall > 0.0 or cover > 0.0:
            LOG.info(f"Winter hook: snow fall={fall}, cover={cover}, seed={self._winter['seed']}")
        if cover > 0.0 and hero is not None:
            world = CarlaDataProvider.get_world()
            rig = self.lead_config.expert.sensor_rig.cameras
            latest = self._winter["sem_latest"]
            for camera_id in self.policy.input_cameras:
                spec = rig[int(LEAD_INDEX_BY_CAMERA_ID[camera_id]) - 1]
                blueprint = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
                blueprint.set_attribute("image_size_x", str(spec["width"]))
                blueprint.set_attribute("image_size_y", str(spec["height"]))
                blueprint.set_attribute("fov", str(spec["fov"]))
                transform = carla.Transform(
                    carla.Location(x=spec["pos"][0], y=spec["pos"][1], z=spec["pos"][2]),
                    carla.Rotation(roll=spec["rot"][0], pitch=spec["rot"][1], yaw=spec["rot"][2]),
                )
                actor = world.spawn_actor(blueprint, transform, attach_to=hero)

                def _store(img, key=camera_id, store=latest):
                    tags = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                    store[key] = (img.frame, tags[:, :, 2].copy())

                actor.listen(_store)
                self._winter["sem_actors"].append(actor)

    def _winter_step(self) -> None:
        """Per-step bookkeeping: lazy setup and the ego pose for world-anchored cover."""
        if not hasattr(self, "_winter"):
            self._winter_setup()
        winter = self._winter
        if winter["cover"] > 0.0:
            hero = CarlaDataProvider.get_hero_actor()
            if hero is not None:
                t = hero.get_transform()
                winter["pose"] = winter["module"].EgoPose(t.location.x, t.location.y, t.rotation.yaw)

    def _winter_semantic(self, camera_id) -> np.ndarray | None:
        """Semantic tags of the current frame for one camera (waits briefly for the callback)."""
        import time as _time

        winter = self._winter
        target = CarlaDataProvider.get_world().get_snapshot().frame
        deadline = _time.time() + 0.5
        while True:
            entry = winter["sem_latest"].get(camera_id)
            if entry is not None and entry[0] >= target:
                return entry[1]
            if _time.time() > deadline:
                winter["sem_misses"] += 1
                return None if entry is None else entry[1]
            _time.sleep(0.001)

    def _winter_effects(self, camera_id, image: np.ndarray) -> np.ndarray:
        """Render snow on one policy camera before the JPEG round-trip."""
        winter = self._winter
        fall, cover = winter["fall"], winter["cover"]
        if fall <= 0.0 and cover <= 0.0 and not winter["dump_dir"]:
            return image
        image = np.ascontiguousarray(image)
        out, semantic = image, None
        idx = int(LEAD_INDEX_BY_CAMERA_ID[camera_id])
        if fall > 0.0 or cover > 0.0:
            renderer = winter["renderers"].get(camera_id)
            if renderer is None:
                spec = self.lead_config.expert.sensor_rig.cameras[idx - 1]
                geometry = winter["module"].CameraGeometry(
                    spec["width"], spec["height"], spec["fov"], spec["pos"], spec["rot"]
                )
                renderer = winter["module"].SnowRenderer(
                    image.shape[0], image.shape[1], fall, cover,
                    world_seed=winter["seed"], particle_seed=winter["seed"] * 100 + idx,
                    fps=float(self.lead_config.expert.simulation.carla_fps), geometry=geometry,
                )
                winter["renderers"][camera_id] = renderer
            if cover > 0.0:
                semantic = self._winter_semantic(camera_id)
            out = renderer(image, semantic, winter["pose"])
        if winter["dump_dir"] and self.step % winter["dump_every"] == 0:
            os.makedirs(winter["dump_dir"], exist_ok=True)
            stem = os.path.join(winter["dump_dir"], f"{max(self.step, 0):05d}_cam{idx}")
            cv2.imwrite(stem + "_raw.jpg", image)
            if out is not image:
                cv2.imwrite(stem + "_snow.jpg", out)
            if semantic is not None:
                cv2.imwrite(stem + "_mask.png", (winter["module"]._COVER_LUT[semantic] * 255).astype(np.uint8))
        return out

    def _winter_cleanup(self) -> None:
        """Destroy the semantic mask cameras and report missed mask frames."""
        winter = getattr(self, "_winter", None)
        if not winter:
            return
        for actor in winter["sem_actors"]:
            try:
                actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        winter["sem_actors"] = []
        if winter["sem_misses"]:
            LOG.warning(f"Winter hook: {winter['sem_misses']} camera frames used a stale or missing mask")

'''

ANCHOR = "    def _cameras(self, sensor_data: CarlaSensorData) -> list[Camera]:\n"
STEP_OLD = "        cameras: list[Camera] = []\n"
STEP_NEW = STEP_OLD + "        self._winter_step()\n"
CALL_OLD = '            image = sensor_data[f"rgb_{LEAD_INDEX_BY_CAMERA_ID[camera_id]}"]\n'
CALL_NEW = CALL_OLD + "            image = self._winter_effects(camera_id, image)\n"
DESTROY_OLD = "    def destroy(self, results: object = None) -> None:\n"
DESTROY_NEW = DESTROY_OLD + "        self._winter_cleanup()\n"


def main() -> None:
    src = TARGET.read_text()
    if MARKER in src:
        print("winter hooks already installed")
        return
    if V1_MARKER in src:
        raise SystemExit("v1 snow hook found; restore the file with "
                         "'git checkout -- src/lead/api/abstract_driving_agent.py' and rerun")
    for needle in (ANCHOR, STEP_OLD, CALL_OLD, DESTROY_OLD):
        assert src.count(needle) == 1, f"anchor not found exactly once: {needle.strip()}"
    src = (src.replace(STEP_OLD, STEP_NEW)
              .replace(CALL_OLD, CALL_NEW)
              .replace(DESTROY_OLD, DESTROY_NEW)
              .replace(ANCHOR, METHODS + ANCHOR))
    TARGET.write_text(src)
    print(f"winter hooks installed in {TARGET.relative_to(REPO)}")


if __name__ == "__main__":
    main()
