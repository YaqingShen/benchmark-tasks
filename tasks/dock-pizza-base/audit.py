"""Independently reconstruct endpoint coverage from recorded physical pin entries."""
import json, sys
from pathlib import Path
import numpy as np

run=Path(sys.argv[1])
trace=[json.loads(s) for s in (run/'trace.jsonl').read_text().splitlines()]
holes=np.array([pt for row in trace for pt in row['added']]).reshape(-1,2)
assert np.isfinite(holes).all() and len(holes)==trace[-1]['holes']
assert np.all(np.linalg.norm(holes,axis=1)<=.07601)
# Trace stores coordinates rounded to 10 micrometres. Allow a small difference at the boundary.
sample=np.arange(-.074,.075,.003)
x,y=np.meshgrid(sample,sample)
region=x*x+y*y<=.071**2
nearest=(np.min(np.hypot(x[region,None]-holes[:,0],y[region,None]-holes[:,1]),axis=1)
         if len(holes) else np.full(region.sum(),np.inf))
final=json.loads((run/'summary.json').read_text())
assert abs(np.mean(nearest<=.012)-final['coverage'])<.003
delta=np.ptp([row['q'] for row in trace])
print(json.dumps(dict(holes=len(holes),coverage_from_recorded_holes=round(np.mean(nearest<=.012),5),
    wheel_angle_range_rad=round(float(delta),3),last_recorded_step=trace[-1]['step'],
    clear_at_full_score=(all(row['grip_clear'] for row in trace if row['returned'])
                         if any(row['returned'] for row in trace) else None),
    full_score=final['metrics']['task_score']==1.),indent=2))
