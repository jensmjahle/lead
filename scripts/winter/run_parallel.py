#!/usr/bin/env python3
"""Run LEAD closed-loop evaluation in parallel on a single multi-GPU machine.

Each slot owns one CARLA server (own world, streaming and traffic-manager
ports, own GPU) and pulls routes from a shared queue. Every route gets its own
output directory, so finished routes are skipped on restart and infrastructure
failures can be retried without touching valid results.

Infrastructure failures (simulator crash, agent setup or runtime crash,
invalid sensors, missing results) are retried and logged separately; they are
never counted as driving failures. The Bench2Drive game-time cap (TickRuntime)
is a driving outcome and is not retried.

Example:
    python scripts/winter/run_parallel.py \
        --checkpoint checkpoints/visiononly_resnet34_v1.5.1/seed0 \
        --routes-dir src/lead/routes/benchmark_routes/bench2drive \
        --bench2drive --slots 0:4,3:4 --run-name b2d_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
# Statuses that indicate a broken run rather than a driving outcome. "TickRuntime"
# is deliberately absent: Bench2Drive aborts a route after 4000 ticks (200 s of
# game time) when the agent is too slow or stuck. That is deterministic and a
# driving outcome; LEAD's result_parser keeps its partial score.
INFRA_MARKERS = (
    "Simulation crashed",
    "Agent's sensors were invalid",
    "Agent couldn't be set up",
    "Agent crashed",
)
PRINT_LOCK = threading.Lock()
RESULTS_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_dotenv(key: str) -> str:
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"')
    raise KeyError(f"{key} not found in {REPO / '.env'}")


@dataclass
class Slot:
    index: int
    gpu: int
    port: int
    tm_port: int


def parse_slots(spec: str, base_port: int, base_tm_port: int) -> list[Slot]:
    """Parse 'gpu:count,gpu:count' into slots with non-overlapping ports."""
    slots: list[Slot] = []
    for part in spec.split(","):
        gpu_str, count_str = part.split(":")
        for _ in range(int(count_str)):
            i = len(slots)
            slots.append(Slot(i, int(gpu_str), base_port + 10 * i, base_tm_port + 10 * i))
    return slots


def route_record(out_dir: Path) -> dict | None:
    """Return the single route record from a finished output dir, if any."""
    ckpt = out_dir / "checkpoint_endpoint.json"
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text())
        records = data["_checkpoint"]["records"]
        return records[0] if records else None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def is_infra_failure(record: dict | None) -> bool:
    if record is None:
        return True
    status = record.get("status", "")
    if status in ("", "Started"):
        return True
    return any(marker in status for marker in INFRA_MARKERS)


class CarlaServer:
    def __init__(self, slot: Slot, carla_root: Path, quality: str, log_dir: Path):
        self.slot = slot
        self.carla_root = carla_root
        self.quality = quality
        self.log_path = log_dir / f"carla_slot{slot.index}_port{slot.port}.log"
        self.proc: subprocess.Popen | None = None

    def start(self, boot_timeout: float) -> bool:
        self.stop()
        env = os.environ.copy()
        env["VK_ICD_FILENAMES"] = NVIDIA_ICD
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        cmd = [
            str(self.carla_root / "CarlaUE4.sh"),
            f"-world-port={self.slot.port}",
            "-resx=800",
            "-resy=600",
            "-nosound",
            f"-graphicsadapter={self.slot.gpu}",
            f"-carla-streaming-port={self.slot.port + 1}",
            f"-quality-level={self.quality}",
            "-RenderOffScreen",
        ]
        log_file = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        deadline = time.time() + boot_timeout
        time.sleep(10)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            if self.ready():
                return True
            time.sleep(5)
        return False

    def ready(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts/cli/test_carla_connection"), str(self.slot.port)],
                timeout=30,
                capture_output=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=30)
        self.proc = None
        time.sleep(3)


def run_route(args, slot: Slot, route: Path, out_dir: Path) -> dict | None:
    env = os.environ.copy()
    env["LEAD_RUNTIME_TYPE_CHECKING"] = "false"
    env["LEAD_EVAL_LIDAR"] = "true" if args.lidar else "false"
    env["PYTHONUNBUFFERED"] = "1"
    if args.lead_config:
        env["LEAD_CONFIG"] = args.lead_config
    env["LEAD_SNOW"] = args.snow
    env["LEAD_SNOW_FALL"] = "" if args.snow_fall is None else str(args.snow_fall)
    env["LEAD_SNOW_COVER"] = "" if args.snow_cover is None else str(args.snow_cover)
    env["LEAD_SNOW_SEED"] = str(args.snow_seed)
    env["LEAD_TIRE_FRICTION"] = "" if args.tire_friction is None else str(args.tire_friction)
    if args.snow_dump_every:
        env["LEAD_SNOW_DUMP_DIR"] = "auto"
        env["LEAD_SNOW_DUMP_EVERY"] = str(args.snow_dump_every)
    else:
        env.pop("LEAD_SNOW_DUMP_DIR", None)
    cmd = [
        sys.executable, "-m", "lead",
        "--checkpoint", str(args.checkpoint),
        "--routes", str(route),
        "--port", str(slot.port),
        "--traffic-manager-port", str(slot.tm_port),
        "--traffic-manager-seed", str(args.tm_seed),
        "--gpu", str(slot.gpu),
        "--output-dir", str(out_dir),
    ]
    if args.bench2drive:
        cmd.append("--bench2drive")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    run_log = out_dir.parent / f"{out_dir.name}.log"
    with open(run_log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=args.route_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            log(f"slot {slot.index}: {route.stem} hit wall-clock timeout {args.route_timeout}s")
            return None
    return route_record(out_dir)


def worker(args, slot: Slot, todo: "queue.Queue[tuple[Path, int]]", results_path: Path,
           carla_root: Path, log_dir: Path) -> None:
    server = CarlaServer(slot, carla_root, args.quality, log_dir)
    routes_since_restart = 0
    healthy = False
    while True:
        try:
            route, attempt = todo.get_nowait()
        except queue.Empty:
            break
        if not healthy or (args.restart_every and routes_since_restart >= args.restart_every):
            healthy = server.start(args.boot_timeout)
            routes_since_restart = 0
            if not healthy:
                log(f"slot {slot.index}: CARLA failed to boot on port {slot.port}, requeueing {route.stem}")
                todo.put((route, attempt))
                time.sleep(30)
                continue
        out_dir = args.out_root / route.stem
        t0 = time.time()
        record = run_route(args, slot, route, out_dir)
        wall = time.time() - t0
        routes_since_restart += 1
        infra = is_infra_failure(record)
        entry = {
            "route": route.stem,
            "attempt": attempt,
            "slot": slot.index,
            "gpu": slot.gpu,
            "wall_s": round(wall, 1),
            "infra_failure": infra,
            "status": None if record is None else record.get("status"),
            "scores": None if record is None else record.get("scores"),
            "duration_game": None if record is None else record.get("meta", {}).get("duration_game"),
        }
        with RESULTS_LOCK, open(results_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        ds = None if record is None else record.get("scores", {}).get("score_composed")
        log(f"slot {slot.index} gpu {slot.gpu}: {route.stem} attempt {attempt} "
            f"status={entry['status']} DS={ds} wall={wall:.0f}s infra={infra}")
        if infra:
            healthy = False  # restart CARLA before the next route
            if attempt < args.max_attempts:
                todo.put((route, attempt + 1))
    server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--routes-dir", type=Path, required=True, help="Directory with one XML per route")
    parser.add_argument("--routes-list", type=Path, help="Optional text file with route stems to run")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N routes (0 = all)")
    parser.add_argument("--bench2drive", action="store_true")
    parser.add_argument("--slots", default="0:2", help="Comma-separated gpu:count, e.g. 0:4,3:4,1:2")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-root", type=Path, help="Defaults to outputs/eval/<run-name>")
    parser.add_argument("--base-port", type=int, default=2000)
    parser.add_argument("--base-tm-port", type=int, default=8000)
    parser.add_argument("--tm-seed", type=int, default=0)
    parser.add_argument("--quality", default="Epic")
    parser.add_argument("--lidar", action="store_true", help="Keep the LiDAR sensors (off by default)")
    parser.add_argument("--lead-config", default="", help="Value for the LEAD_CONFIG override variable")
    parser.add_argument("--snow", default="off",
                        help="Snow preset: off, light, medium, heavy, fall_only, cover_only")
    parser.add_argument("--snow-fall", type=float, default=None, help="Override falling-snow level")
    parser.add_argument("--snow-cover", type=float, default=None, help="Override ground-cover level")
    parser.add_argument("--tire-friction", type=float, default=None,
                        help="Ego per-wheel tire_friction (CARLA default 3.5; 0.3 ~ packed snow)")
    parser.add_argument("--snow-seed", type=int, default=0)
    parser.add_argument("--snow-dump-every", type=int, default=0,
                        help="Save raw/overlaid frames every N steps into each route dir (0 = off)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--restart-every", type=int, default=10, help="Restart CARLA after N routes (0 = never)")
    parser.add_argument("--boot-timeout", type=float, default=240.0)
    parser.add_argument("--route-timeout", type=float, default=3600.0, help="Wall-clock limit per route in seconds")
    args = parser.parse_args()

    args.checkpoint = (REPO / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    routes_dir = (REPO / args.routes_dir).resolve() if not args.routes_dir.is_absolute() else args.routes_dir
    args.out_root = args.out_root or (REPO / "outputs/eval" / args.run_name)
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_root / "_logs"
    log_dir.mkdir(exist_ok=True)
    results_path = args.out_root / "results.jsonl"
    carla_root = Path(read_dotenv("CARLA_ROOT"))

    routes = sorted(routes_dir.glob("*.xml"), key=lambda p: (len(p.stem), p.stem))
    if args.routes_list:
        wanted = {line.strip() for line in args.routes_list.read_text().splitlines() if line.strip()}
        routes = [r for r in routes if r.stem in wanted]
    if args.limit:
        routes = routes[: args.limit]

    todo: "queue.Queue[tuple[Path, int]]" = queue.Queue()
    skipped = 0
    for route in routes:
        if not is_infra_failure(route_record(args.out_root / route.stem)):
            skipped += 1
            continue
        todo.put((route, 1))

    slots = parse_slots(args.slots, args.base_port, args.base_tm_port)
    (args.out_root / "run_config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()} | {"routes_total": len(routes), "skipped_done": skipped},
        indent=2))
    log(f"{len(routes)} routes, {skipped} already done, {todo.qsize()} queued, {len(slots)} slots: "
        + ", ".join(f"gpu{s.gpu}:{s.port}" for s in slots))

    threads = []
    for slot in slots:
        t = threading.Thread(target=worker, args=(args, slot, todo, results_path, carla_root, log_dir),
                             daemon=True)
        t.start()
        threads.append(t)
        time.sleep(20)  # stagger CARLA boots
    for t in threads:
        t.join()

    final = {}
    for route in routes:
        rec = route_record(args.out_root / route.stem)
        final[route.stem] = rec
    valid = {k: v for k, v in final.items() if not is_infra_failure(v)}
    if valid:
        ds = [v["scores"]["score_composed"] for v in valid.values()]
        rc = [v["scores"]["score_route"] for v in valid.values()]
        log(f"done: {len(valid)}/{len(routes)} valid routes, mean DS={sum(ds)/len(ds):.2f}, "
            f"mean RC={sum(rc)/len(rc):.2f}, infra failures left={len(routes) - len(valid)}")
    else:
        log("done: no valid routes")


if __name__ == "__main__":
    main()
