import dataclasses
import importlib.util
import json
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('task.py')
spec=importlib.util.spec_from_file_location('checked_hook',p)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
from labkit import PartState

def state(pos, quat=(0,0,0,1), velocity=(0,0,0), spin=(0,0,0)):
 return PartState(pos=np.array(pos,dtype=float),quat_xyzw=np.array(quat,dtype=float),lin_vel=np.array(velocity,dtype=float),ang_vel=np.array(spin,dtype=float),low=np.zeros(3),high=np.zeros(3))
def objects():
 return {'hook':state((t.TRAY_X,t.TRAY_Y,t.TOP+t.FOAM_TOP+.030)),
  'screw':state((t.TRAY_X,t.TRAY_Y,t.TOP+t.FOAM_TOP+.005)),
  'washer':state((t.TRAY_X,t.TRAY_Y,t.TOP+t.FOAM_TOP+.005)),
  'screwdriver':state(t.DRIVER_HOME)}
cases=[]
def check(name,obj,expected,hands=None,threaded=False,damage=False):
 t._M.update(threaded=threaded,dropped=damage,hands=[] if hands is None else hands)
 actual=t.score(obj)
 cases.append(dict(case=name,score=actual,expected=expected,passed=actual==expected))
 assert actual==expected,(name,actual,expected)
check('Completed layout',objects(),1.)
o=objects();o['hook']=state((t.TRAY_X,t.TRAY_Y,t.TOP+.075));check('Hook floating 31 mm above foam',o,.5)
o=objects();o['hook']=state((.09,t.TRAY_Y,t.TOP+t.FOAM_TOP+.030));check('Hook crosses tray edge',o,.5)
o=objects();o['screwdriver']=state(t.DRIVER_HOME,(0,0,2**-.5,2**-.5));check('Driver transverse to stand',o,.75)
o=objects();o['hook']=state(o['hook'].pos,spin=(0,0,1));check('Hook spinning',o,.5)
o=objects();lo,hi=t.collision_bounds(o['hook'],t.HOOK);check('Hand still touching hook',o,.5,[(lo,hi)])
o=objects();o['screw']=state((.28,t.TRAY_Y,t.TOP+.020));check('Screw protrudes through tray wall',o,.75)
o=objects();o['screwdriver']=state(np.array(t.DRIVER_HOME)+(0,0,.010));check('Driver held above rests',o,.75)
check('Damage cap',objects(),.25,damage=True)
check('Still threaded',objects(),0.,threaded=True)
print(json.dumps({'all_passed':True,'cases':cases},indent=2))
