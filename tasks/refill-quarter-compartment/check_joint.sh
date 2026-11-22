#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
tag=${1:?fresh name}
mkdir -p runs
mkdir "runs/$tag"
export CALIBRATION_OUT="$PWD/runs/$tag/joint-calibration.json"
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy yam_scripted -E task="$PWD/fixture.py" -E control_hz=20 -E object_jitter=0 -E cam_width=160 -E cam_height=96 --seed 0 --scorer task_score --max-steps 55 --instruction "Remain clear of the calibration fixture." --log-dir "runs/$tag" >"runs/$tag/run.log" 2>&1
