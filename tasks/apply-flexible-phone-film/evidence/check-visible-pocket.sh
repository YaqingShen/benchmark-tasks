#!/bin/bash
# Optical/scorer regression: inject a known closed wetting contour, then run the actual script.
set -eu
src=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?Choose a new output directory}
mkdir "$out"
cd "$out"
cp "$src"/{task.py,task.json,labkit.py,laminate.py,measured.py,pockets.py,inspection.py,inspection_camera.py,arena_assets_fix.py,bench-config.json,run.sh} .
cat >> task.py <<'PY'

def known_pocket(env):
    if trial.step != 1:
        return
    from isaaclab.sim.utils.stage import get_current_stage
    air=trial.meters.air
    rectangle=(abs(air.x-70)<=3.5) & (abs(air.y-32)<=3.0)
    interior=(abs(air.x-70)<=2.5) & (abs(air.y-32)<=2.0)
    assert air.touch(rectangle & ~interior)
    scene=env.unwrapped.scene
    air.display(get_current_stage(),scene.env_prim_paths[0]+'/AirPockets',O)
labkit.on_step(known_pocket)
PY
exec bash run.sh control video
