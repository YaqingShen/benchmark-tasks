#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
set -u
tag=${1:?Run name}
test ! -e "runs/$tag"
mkdir -p "runs/$tag"
mode=${2:-debug}
policy=yam_scripted
width=320; height=180
args=()
if [[ $mode == video || $mode == dream ]]; then args+=(--store-frames); width=640; height=360; fi
if [[ $mode == dream ]]; then policy=dream; fi
sha256sum material.py task.py visual.py arena_assets_fix.py labkit.py > "runs/$tag/input.sha256"
instruction=$(python -c "import json; print(json.load(open('task.json'))['instruction'])")
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" -E task="$PWD/task.py" -E control_hz=20 -E object_jitter=0 -E cam_width="$width" -E cam_height="$height" --max-steps 465 --instruction "$instruction" --scorer task_score "${args[@]}" --log-dir "runs/$tag"
