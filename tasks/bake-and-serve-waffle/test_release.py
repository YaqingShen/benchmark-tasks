"""Native endpoint conditions distinguish resting, held, moving and newly released food."""
from dataclasses import replace
import numpy as np
import task as T
from labkit import PartState
T.phase_conversion.STATE['solid']=True  # synthetic post-bake test states


def state(pos,joints=None):
    p=np.asarray(pos,float);z=np.zeros(3)
    return PartState(p,np.array([0.,0.,0.,1.]),z,z,p-.001,p+.001,joints or {})


pl=np.asarray(T.PLATE_POS)
food=pl+np.array([0.,0.,T.PLATE_FLOOR-T._W_LOW+.0008])
objects={'waffle':state(T.HOME,{'spin':0.,'lid':T.LID_START,'latch':0.}),
         'belgian_waffle':state(food),'dinner_plate':state(pl), 'kitchen_tongs':state(T.TONG_HOME,{'jaw':0.}),
         'tong_cup':state([T.TONG_HOME[0],T.TONG_HOME[1],T.TOP])}
T._TRACK.update(T.new_track())
T._TRACK.update(phase=2,latched_at_turn=True,shut_at_turn=True,loosened=True,tool_stowed=True,tool_carried=True)
assert T.breakdown(objects)['full_success']
far=[(np.array([2.,2.,2.]),np.array([2.1,2.1,2.1]))]
food_centre=food+np.array([0.,0.,T.ZA])
touch=[(food_centre-.01,food_centre+.01)]
T._FINAL.update(since=None,time=0.,previous_joints=None,released=False,still=False)
for _ in range(30): T.update_final_release(objects,touch)
assert T.score(objects)<1., 'stationary hand-held food must not finish'
T.update_final_release(objects,far)
assert T.score(objects)<1., 'newly released state needs a hold'
for _ in range(25): T.update_final_release(objects,far)
assert T.score(objects)==1.
objects['belgian_waffle']=replace(objects['belgian_waffle'], ang_vel=np.array([0.,0.,1.]))
T.update_final_release(objects,far)
assert T.score(objects)<1., 'spinning food is not resting'
objects['belgian_waffle']=replace(objects['belgian_waffle'], ang_vel=np.zeros(3))
T.update_final_release(objects,far)
assert T.score(objects)<1., 'rest timer must restart'
objects['waffle'].joints['lid']+=.02
T.update_final_release(objects,far)
assert not T._FINAL['still'], 'moving lid is not a final resting state'
print('PASS: held-food rejection, release hold, angular motion and moving-lid rejection')

# Articulated handles are outside the root stand bounds and must still count.
handle=np.asarray(T.HOME)+np.array([-.20,0.,T.ZA-.01])
T.update_final_release(objects,[(handle-.005,handle+.005)])
assert not T._FINAL['released'], 'holding appliance handle cannot count as released'
T._TRACK['max_shift']=.010001
assert not T.breakdown(objects)['full_success']
assert T.score(objects)<1., 'rounded partial credit must not become 1.0'
