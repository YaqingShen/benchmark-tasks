#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
out=${1:?Choose a new output directory}
mkdir -p runs
mkdir "runs/$out"
export OMELET_COUPON_RESULT="$PWD/runs/$out/slide.json"
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy yam_scripted -E task="$PWD/slide_test.py" -E control_hz=20 -E object_jitter=0 -E cam_width=160 -E cam_height=96 --scorer task_score --max-steps 26 --seed 0 --instruction "Hold still during the material check." --log-dir "runs/$out" >"runs/$out/run.log" 2>&1
test -s "$OMELET_COUPON_RESULT"
