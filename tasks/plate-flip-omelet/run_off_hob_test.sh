#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
out=${1:?Choose a new run name}
mkdir -p runs
mkdir "runs/$out"
export OMELET_TRACE=1 OMELET_AUDIT_FILE="$PWD/runs/$out/trace.jsonl"
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy yam_scripted -E task="$PWD/off_hob_test.py" -E control_hz=20 -E object_jitter=0 -E cam_width=160 -E cam_height=96 --scorer task_score --max-steps 64 --seed 0 --instruction "Hold still for the terminal support check." --log-dir "runs/$out" >"runs/$out/run.log" 2>&1
