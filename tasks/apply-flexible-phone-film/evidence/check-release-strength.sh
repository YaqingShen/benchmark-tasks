#!/bin/bash
# Sensitivity check on the simulator only. Use the nominal action plan.
set -eu
src=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?Choose a new output directory}
mkdir "$out"
cd "$out"
cp "$src"/{task.py,task.json,labkit.py,laminate.py,measured.py,pockets.py,inspection.py,inspection_camera.py,arena_assets_fix.py,bench-config.json,run.sh} .
python3 - <<'PY'
from pathlib import Path
p = Path('laminate.py')
s = p.read_text()
if s.count('RELEASE_N=.25\n') != 1:
    raise ValueError('Unexpected nominal interface strength')
p.write_text(s.replace('RELEASE_N=.25\n', 'RELEASE_N=1.0\n'))
PY
exec bash run.sh stronger debug
