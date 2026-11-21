import json
import numpy as np
import task
from yam_arena.demo import ArmPlan
from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS
results={}
for arm,waypoints in task.demo().items():
    plan=ArmPlan(waypoints,LEFT_BASE_POS if arm=='left' else RIGHT_BASE_POS)
    bad=[]
    from yam_arena import ik
    for t in np.arange(0,plan.duration+.01,.10):
        p,R,f,label=plan.target(t)
        plan.action7(float(t))
        pp,RR=ik.tcp_pose(plan.q,plan.base_pos)
        pe=float(np.linalg.norm(pp-p)*1000)
        if pe>4:bad.append(dict(t=round(float(t),2),mm=round(pe,2),label=label))
    results[arm]=dict(duration=plan.duration,worst_mm=plan.worst_ik_mm,worst_deg=plan.worst_ik_deg,bad_count=len(bad),worst=sorted(bad,key=lambda x:x['mm'],reverse=True)[:4])
print(json.dumps(results,indent=2))
if any(r['worst_mm']>4 or r['worst_deg']>4 for r in results.values()):raise SystemExit(1)
