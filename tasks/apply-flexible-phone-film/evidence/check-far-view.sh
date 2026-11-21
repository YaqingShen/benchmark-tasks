#!/bin/bash
# Optical replay. Retain the two arms, action plan, colliders, scorer and policy cameras.
set -eu
src=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?Choose a new output directory}
mkdir "$out"
cd "$out"
cp "$src"/{task.py,task.json,labkit.py,laminate.py,measured.py,pockets.py,inspection.py,inspection_camera.py,arena_assets_fix.py,bench-config.json,run.sh} .
python3 - <<'PY'
from pathlib import Path
p=Path('inspection_camera.py')
s=p.read_text()
a="pos=(x, y-.68, glass_z+.014),\n                      rot=(2**-.5,0.,0.,2**-.5)"
b="pos=(x, y+.68, glass_z+.175),\n                      rot=(0.,((1-.24)/2)**.5,((1+.24)/2)**.5,0.)"
if s.count(a)!=1:
    raise ValueError('Unexpected inspection-camera pose')
p.write_text(s.replace(a,b))
PY
exec bash run.sh opposite video
