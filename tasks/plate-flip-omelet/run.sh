#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
tag=${1:?run name}; mode=${2:-debug}
mkdir -p runs
mkdir "runs/$tag"
export OMELET_TRACE=1
export OMELET_AUDIT_FILE="$PWD/runs/$tag/trace.jsonl"
sha256sum task.py labkit.py arena_assets_fix.py run.sh >"runs/$tag/input.sha256"
extra=()
w=320; h=180
policy=yam_scripted
case "$mode" in debug) ;; video) w=640; h=360; extra=(--store-frames);; dream) w=640; h=360; policy=dream; extra=(--store-frames);; *) exit 2;; esac
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" -E task="$PWD/task.py" -E control_hz=20 -E object_jitter=0 -E cam_width="$w" -E cam_height="$h" --seed 0 --scorer task_score --max-steps "${3:-1200}" --instruction "Use the plate to turn the whole omelet over. Slide it back into the skillet, browned side up, and set the skillet down." "${extra[@]}" --log-dir "runs/$tag" >"runs/$tag/run.log" 2>&1
