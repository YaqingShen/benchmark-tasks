#!/bin/bash
# Negative control: same right-arm sweep, left arm keeps the aligned paddle still.
set -eu
src=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?Choose a new output directory}
mkdir "$out"
cd "$out"
cp "$src"/{task.py,task.json,labkit.py,laminate.py,measured.py,pockets.py,inspection.py,inspection_camera.py,arena_assets_fix.py,bench-config.json,run.sh} .
cat >> task.py <<'PY'

nominal_demo = demo
def demo():
    from dataclasses import replace
    plan = nominal_demo()
    aligned = plan['left'][4]
    plan['left'][5:] = [replace(w, pos=aligned.pos, finger=aligned.finger)
                        for w in plan['left'][5:]]
    return plan
PY
exec bash run.sh control debug
