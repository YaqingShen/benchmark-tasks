#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
run="runs/${1:?Provide a new run name}"
if [[ -e "$run" ]]; then exit 2; fi
mkdir -p "$run/source"
cp task.py cable.py labkit.py arena_assets_fix.py "$run/source/"
sha256sum task.py cable.py run.sh labkit.py arena_assets_fix.py > "$run/inputs.sha256"
export BRAKE_TRACE_CSV="$PWD/$run/measurement.csv"
mode=${3:-debug}
policy=yam_scripted
frames=()
width=320
height=180
case "$mode" in
  video) frames=(--store-frames); width=640; height=360 ;;
  dream) policy=dream; frames=(--store-frames); width=640; height=360 ;;
  debug) ;;
  *) exit 2 ;;
esac
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" \
    -E task="$PWD/$run/source/task.py" -E control_hz=20 -E object_jitter=0 \
    -E cam_width="$width" -E cam_height="$height" --max-steps "${2:-850}" \
    --instruction "Reconnect the front V-brake, then squeeze and release the lever. The wheel should turn freely, stop, and turn freely again." \
    --scorer task_score "${frames[@]}" --log-dir "$run" > "$run/run.log" 2>&1
