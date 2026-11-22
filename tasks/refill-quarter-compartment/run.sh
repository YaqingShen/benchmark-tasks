#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
tag=${1:?fresh run name}; mode=${2:-debug}
mkdir -p runs
mkdir "runs/$tag"
export QUARTER_TRACE="$PWD/runs/$tag/trace.jsonl"
sha256sum task.py geometry.py labkit.py scoring.py packed.py packed_runtime.py arena_assets_fix.py run.sh > "runs/$tag/inputs.sha256"
w=224; h=128; policy=yam_scripted; extra=()
case "$mode" in debug) ;; video) w=640; h=360; extra=(--store-frames);; dream) w=640; h=360; policy=dream; extra=(--store-frames);; *) exit 2;; esac
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" -E task="$PWD/task.py" -E control_hz=20 -E object_jitter=0 -E cam_width="$w" -E cam_height="$h" --seed 0 --scorer task_score --max-steps "${3:-720}" --instruction "Empty the wrapped roll of quarters into the middle coin compartment. Keep change out of the neighboring compartments and off the counter. Leave the empty wrapper on the counter." "${extra[@]}" --log-dir "runs/$tag" >"runs/$tag/run.log" 2>&1
