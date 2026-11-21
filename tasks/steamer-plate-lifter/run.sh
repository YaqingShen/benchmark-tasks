#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
if [[ -f "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}" ]]; then
    source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
fi
set -u
command -v inspect-robots >/dev/null || { echo "Activate the YAM Arena environment first." >&2; exit 1; }
tag=${1:?Provide a new run name}
mode=${2:-debug}
steps=${3:-720}
run="runs/$tag"
if [[ -e "$run" ]]; then
    echo "Run directory already exists: $run" >&2
    exit 1
fi
mkdir -p "$run"
sha256sum task.py task.json labkit.py arena_assets_fix.py > "$run/input.sha256"
instruction=$(python -c "import json; print(json.load(open('task.json'))['instruction'])")
export STEAM_TRACE=1
policy=yam_scripted
frames=()
width=320
height=180
case "$mode" in
    debug) ;;
    video) frames=(--store-frames); width=640; height=360 ;;
    dream) policy=dream; frames=(--store-frames); width=640; height=360 ;;
    *) echo "Expected debug, video, or dream" >&2; exit 2 ;;
esac
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" \
    -E task="$PWD/task.py" -E control_hz=20 -E object_jitter=0 \
    -E cam_width="$width" -E cam_height="$height" --max-steps "$steps" \
    --instruction "$instruction" --scorer task_score "${frames[@]}" --log-dir "$run"
