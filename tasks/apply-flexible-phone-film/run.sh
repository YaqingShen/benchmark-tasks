#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
set -u
tag=${1:?Run name}
test ! -e "runs/$tag"
mkdir -p "runs/$tag"
mode=${2:-debug}
case "$mode" in debug|video|dream) ;; *) echo "Use debug, video, or dream" >&2; exit 2;; esac
policy=yam_scripted
width=320; height=180
args=()
if [[ $mode == video || $mode == dream ]]; then args+=(--store-frames); width=640; height=360; fi
if [[ $mode == dream ]]; then policy=dream; fi
if [[ $mode != debug ]]; then export FILM_CLOSEUP=1; fi
export FILM_TRACE="$PWD/runs/$tag/trace.jsonl"
inputs=(inspection_camera.py bench-config.json laminate.py task.py pockets.py inspection.py measured.py arena_assets_fix.py labkit.py task.json run.sh)
sha256sum "${inputs[@]}" > "runs/$tag/input.sha256"
tar -czf "runs/$tag/source.tar.gz" "${inputs[@]}"
instruction=$(python -c "import json; print(json.load(open('task.json'))['instruction'])")
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" -E task="$PWD/task.py" -E control_hz=20 -E object_jitter=0 -E cam_width="$width" -E cam_height="$height" --seed 0 --max-steps 465 --instruction "$instruction" --scorer task_score "${args[@]}" --log-dir "runs/$tag"
