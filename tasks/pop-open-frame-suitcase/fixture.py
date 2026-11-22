"""Press each case with the same force. The spring always acts on the lid."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import labkit
import mechanism as m
from yam_arena.task_file import TABLE_TOP
SCENE='station'
CODES={'correct':m.CODE,'wrong_left':(5,7,2),'wrong_middle':(4,6,2),'wrong_right':(4,7,3)}
ORIGINS=[(.20,-.08),(.50,-.08),(.20,.21),(.50,.21)]
OBJECTS=[m.mechanism(labkit,name,(*xy,TABLE_TOP[SCENE]),code)
         for (name,code),xy in zip(CODES.items(),ORIGINS)]
rows=[]

def tick(env):
    force=2.0 if 10<=len(rows)<32 else 0.
    row={}
    for name,code in CODES.items():
        obj=env.unwrapped.scene[name]
        names=list(obj.joint_names)
        button=names.index('button_joint')
        lid=names.index('lid_joint')
        # A preload shift F/k in the slider's own return spring produces an
        # extra constant F without overriding the travel or the wheel contacts.
        obj.set_joint_position_target(torch.tensor([[force/150.]],device=obj.device),joint_ids=[button])
        obj.set_joint_position_target(torch.tensor([[m.LID_REST]],device=obj.device),joint_ids=[lid])
        obj.write_data_to_sim()
        p=obj.data.joint_pos[0].tolist()
        row[name]={'button_mm':p[button]*1000,'gap_mm':m.gap_mm(p[lid]),
                   'digits':[m.digit(p[names.index(f'wheel{i}_joint')]+n*m.STEP) for i,n in enumerate(code)]}
    rows.append(row)
    if len(rows) in (9,25,39): print('FRAME_CASE',len(rows),row,flush=True)
    if len(rows)==43:
        checks={}
        for name,code in CODES.items():
            seated=all(abs(r[name]['gap_mm'])<2. for r in rows[4:9])
            aligned=all(tuple(r[name]['digits'])==code for r in rows[10:])
            pushed=rows[22:30]
            if name=='correct':
                response=(max(r[name]['button_mm'] for r in pushed)>=m.UNLATCH_MM and
                          all(r[name]['gap_mm']>=20 and r[name]['button_mm']<=1 for r in rows[35:]))
            else:
                response=all(r[name]['gap_mm']<2 and r[name]['button_mm']<2 for r in rows[12:])
            checks[name]=seated and aligned and response
        Path(os.environ['LOCK_FIXTURE']).write_text(json.dumps({'checks':checks,'rows':rows},indent=2))

labkit.on_reset(lambda *args:rows.clear())
labkit.on_step(tick)
def demo():
    import numpy as np
    from yam_arena import station,ik
    from yam_arena.constants import READY
    from yam_arena.demo import Waypoint
    return {a:[Waypoint(3.,tuple(ik.tcp_pose(np.array(READY),station.BASES[a])[0]),-.032,'stand clear')] for a in ('left','right')}
def score(objects):return 0.
def success(objects):return False
