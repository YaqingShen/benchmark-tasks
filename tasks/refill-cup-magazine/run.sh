#!/usr/bin/env bash
set -eo pipefail
if ! command -v inspect-robots >/dev/null; then
    source "${CUP_ENV:-$HOME/IsaacLab-Arena/yam.env}"
fi
cd "$(dirname "$0")"
export CUP_RUN_DIR="$PWD/runs/${1:?Supply a new run name}"
mkdir -p "$PWD/runs"
mkdir "$CUP_RUN_DIR"
instruction='Load all eight paper cups into the hot-drink dispenser without touching the cups with the grippers. Leave the wrapper outside the magazine so one service request dispenses one cup.'
# Reference controller settings from the verified takeout-clearance run.
export CUP_HANDOFF="${CUP_HANDOFF:-.144}"
export CUP_TAB_DEPTH="${CUP_TAB_DEPTH:-.002}"
export CUP_PULL_LEAD="${CUP_PULL_LEAD:-.008}"
export CUP_YAW_FOLLOW="${CUP_YAW_FOLLOW:-0}"
policy="${CUP_POLICY:-yam_scripted}"
width="${CUP_WIDTH:-640}"
height="${CUP_HEIGHT:-360}"
limit="${CUP_STEPS:-1200}"
echo "$policy ${width}x${height} 20Hz seed=0 limit=$limit" > "$CUP_RUN_DIR/settings.txt"
date -u +%s > "$CUP_RUN_DIR/start.epoch"
trap 'date -u +%s > "$CUP_RUN_DIR/end.epoch"' EXIT
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" -E task="$PWD/task/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width="$width" -E cam_height="$height" --seed 0 --instruction "$instruction" --scorer task_score --max-steps "$limit" --store-frames --log-dir "$CUP_RUN_DIR/capture" > "$CUP_RUN_DIR/log" 2>&1
