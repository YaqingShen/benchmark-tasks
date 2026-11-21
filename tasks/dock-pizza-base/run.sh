#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
if ! command -v inspect-robots >/dev/null; then
    source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
fi
export DOCK_OUT="$PWD/runs/${1:?new run name}"
mkdir -p runs
mkdir "$DOCK_OUT"
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "${DOCK_POLICY:-yam_scripted}" -E task="$PWD/task/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width="${DOCK_WIDTH:-960}" -E cam_height="${DOCK_HEIGHT:-540}" --seed 0 --instruction 'Dock the whole thin pizza base evenly with the pin roller. Return the tool to its rest.' --scorer task_score --max-steps "${DOCK_STEPS:-570}" --store-frames --log-dir "$DOCK_OUT/capture" > "$DOCK_OUT/log" 2>&1
