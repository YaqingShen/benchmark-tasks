#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
source "${YAM_ENV:-$HOME/IsaacLab-Arena/yam.env}"
tag=${1:?fresh run directory}; mode=${2:-fixture}
mkdir -p runs
mkdir "runs/$tag"
export LOCK_TRACE="$PWD/runs/$tag/trace.jsonl"
export LOCK_FIXTURE="$PWD/runs/$tag/interlock.json"
sha256sum labkit.py arena_assets_fix.py mechanism.py task.py fixture.py scoring.py run.sh >"runs/$tag/inputs.sha256"
policy=yam_scripted; w=320; h=180; extra=(); limit=${3:-700}
case "$mode" in
fixture) file=fixture.py; limit=45; goal='Remain clear of the four suitcase fixtures.';;
debug|video|dream) file=task.py; goal='Open the suitcase with the code 4-7-2. Let the lid spring open on its own.'
  if [[ $mode != debug ]]; then w=640; h=360; extra=(--store-frames); fi
  if [[ $mode == dream ]]; then policy=dream; fi;;
*) exit 2;;
esac
printf 'skip\n\n' | inspect-robots run --embodiment yam_arena --policy "$policy" -E task="$PWD/$file" -E object_jitter=0 -E control_hz=20 -E cam_width="$w" -E cam_height="$h" --instruction "$goal" --scorer task_score --seed 0 --max-steps "$limit" "${extra[@]}" --log-dir "runs/$tag" >"runs/$tag/run.log" 2>&1
