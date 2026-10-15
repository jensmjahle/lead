#!/usr/bin/env bash
# Showcase: six short braking / pedestrian routes in four conditions, with videos.
# Conditions: clear, snow (medium), low friction (0.3 ~ packed snow), snow + low friction.
# Usage: bash scripts/winter/run_showcase.sh [slots] [base_port]
set -euo pipefail
cd "$(dirname "$0")/../.."
SLOTS="${1:-1:1,2:1}"
PORT="${2:-2400}"
COMMON=(--checkpoint checkpoints/visiononly_resnet34_v1.5.1/seed0
        --routes-dir src/lead/routes/benchmark_routes/bench2drive
        --routes-list scripts/winter/showcase_routes.txt
        --bench2drive --slots "$SLOTS" --base-port "$PORT" --base-tm-port $((PORT + 6000))
        --winter-video --demo-video --snow-dump-every 40)
python scripts/winter/run_parallel.py "${COMMON[@]}" --run-name showcase_clear
python scripts/winter/run_parallel.py "${COMMON[@]}" --snow medium --run-name showcase_snow
python scripts/winter/run_parallel.py "${COMMON[@]}" --tire-friction 0.3 --run-name showcase_friction
python scripts/winter/run_parallel.py "${COMMON[@]}" --snow medium --tire-friction 0.3 --run-name showcase_snow_friction
