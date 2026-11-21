"""Counterexamples for the ordered kitchen-tong extraction and carry requirement."""
from dataclasses import replace
import math
import numpy as np
import task as T
from labkit import PartState
T.phase_conversion.STATE['solid']=True  # synthetic post-bake test states

Z=np.zeros(3);Q=np.array([0.,0.,0.,1.]);far=[(np.ones(3)*2,np.ones(3)*2.1)]
def state(p,q=Q,j=None):
 p=np.array(p,float);return PartState(p,np.array(q),Z.copy(),Z.copy(),p.copy(),p.copy(),j or {})
def scene(spin=0.,lid=0.,food=None,tool=None):
 R=T._Rx(spin);qp=np.array([math.sin(spin/2),0.,0.,math.cos(spin/2)])
 root=np.array(T.HOME)+[0.,0.,T.ZA]-R@np.array([0.,0.,T.ZA])
 return {'waffle':state(T.HOME,j={'spin':spin,'lid':lid,'latch':0.}),
 'belgian_waffle':food or state(root,qp),'dinner_plate':state(T.PLATE_POS),
 'kitchen_tongs':tool or state(T.TONG_HOME,[0.,0.,math.sin(T.TONG_YAW/2),math.cos(T.TONG_YAW/2)],{'jaw':0.}),
 'tong_cup':state([T.TONG_HOME[0],T.TONG_HOME[1],T.TOP],[0.,0.,math.sin(T.TONG_YAW/2),math.cos(T.TONG_YAW/2)])}
def plate_food():return state(np.array(T.PLATE_POS)+[0.,0.,T.PLATE_FLOOR-T._W_LOW+.0008])
def step(tr,o,h=far):T.track_update(tr,o,h);return o

def prefix():
 tr=T.new_track()
 for a in np.linspace(0,math.pi,25):step(tr,scene(a))
 for a in np.linspace(math.pi,0,25):step(tr,scene(a))
 step(tr,scene(lid=T.LID_OPEN))
 return tr

def loosen(tr, lift=.012):
 tool=scene()['kitchen_tongs'];tool=replace(tool,pos=tool.pos+[0.,0.,.1],joints={'jaw':-.003})
 def hand(t):
  g=t.to_world(T.TONG_GRIP);return [(g-.012,g+.012)]
 step(tr,scene(lid=T.LID_OPEN,tool=tool),hand(tool))
 # Nominal insertion between the two front top lugs, followed by small lift.
 pick=np.array(T.HOME)+[-.0125,-.0125,T._W_CORE1+.0015]
 tool=replace(tool,pos=pick+[0.,.012,lift],joints={'jaw':-.024})
 food=state(np.array(T.HOME)+[0.,0.,lift])
 for _ in range(4):step(tr,scene(lid=T.LID_OPEN,tool=tool,food=food),hand(tool))
 return tr

def carry(tr):
 # Preserve the tool/food relative transform while crossing toward the plate.
 lift=.060
 pick=np.array(T.HOME)+[-.0125,-.0125,T._W_CORE1+.0015]
 tool=state(pick+[0.,.012+.150,lift],[0.,0.,math.sin(T.TONG_YAW/2),math.cos(T.TONG_YAW/2)],{'jaw':-.024})
 food=state(np.array(T.HOME)+[0.,.150,lift])
 g=tool.to_world(T.TONG_GRIP)
 for _ in range(4):step(tr,scene(lid=T.LID_OPEN,tool=tool,food=food),[(g-.012,g+.012)])
 return tr

def finish(tr):
 carry(tr)
 o=step(tr,scene(0,T.LID_OPEN,plate_food()))
 return T.breakdown(o,tr),o

tr=loosen(prefix());assert tr['loosened'],tr
b,o=finish(tr);assert b['full_success'],b
print('PASS: tong extraction, carry, plate support and tool stow')
tr=prefix();b,_=finish(tr);assert not b['full_success'];assert b['score']==0.
print('PASS: plating without extraction rejected')
tr=loosen(prefix(),0.);assert not tr['loosened']
o=step(tr,scene(0,T.LID_OPEN,plate_food()))
assert T.breakdown(o,tr)['score']==0.
print('PASS: touching without lifting then plating rejected')
tr=loosen(prefix());b,o=finish(tr)
c=o['belgian_waffle'].to_world([0.,0.,T.ZA]);step(tr,o,[(c-.01,c+.01)])
assert T.breakdown(o,tr)['score']==0.
print('PASS: direct hand-food contact permanently revokes all credit')
tr=loosen(prefix());o=step(tr,scene(0,T.LID_OPEN,plate_food()))
assert T.breakdown(o,tr)['score']==0.
print('PASS: placing without sustained carry evidence rejected')
tr=loosen(prefix());o=step(tr,scene(lid=T.LID_OPEN))
assert 0.<T.breakdown(o,tr)['score']<1.
print('PASS: legitimate incomplete extraction retains stage credit')
tr=loosen(prefix());tr['hand_evidence']=False;b,_=finish(tr);assert not b['full_success']
print('PASS: missing hand evidence rejected')
tr=loosen(prefix());b,o=finish(tr)
# Either face may land downward; actual collision vertices determine the gap.
f=state([0,0,0],[1.,0.,0.,0.]);verts=T.serving_rules.food_points(T,f)
z=T.PLATE_POS[2]+T.PLATE_FLOOR-float(verts[:,2].min())+.0008
f=replace(f,pos=np.array([T.PLATE_POS[0],T.PLATE_POS[1],z]))
o['belgian_waffle']=f;assert T.on_plate(o)
print('PASS: inverted waffle supported by actual top-lug geometry')
T._TRACK.clear();T._TRACK.update(tr)
T._FINAL.update(since=None,time=0.,previous_joints=None,released=False,still=False)
for _ in range(25):T.update_final_release(o,far)
assert T.score(o)==1.
print('PASS: final release and continuous rest can reach 1.0 in synthetic test')

o['kitchen_tongs'].joints['jaw']-=.001
T.update_final_release(o,far)
assert not T._FINAL['still']
print('PASS: moving tong jaw resets final rest hold')

# A prohibited early transfer cannot be erased by putting food back and
# completing the legitimate cooking and serving motions afterward.
tr=T.new_track();step(tr,scene(food=plate_food()))
assert tr['invalid_serving']
for a in np.linspace(0,math.pi,25):step(tr,scene(a))
for a in np.linspace(math.pi,0,25):step(tr,scene(a))
tr=loosen(tr);b,_=finish(tr)
assert b['score']==0. and not b['full_success']
print('PASS: early prohibited plating remains disqualified after later valid motions')
