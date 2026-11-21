"""Opening in the declared home zone must not contradict the lid guard."""
import math
import numpy as np
import task as T
from labkit import PartState

def observe(phase, spin, lid):
    z=np.zeros(3)
    w=PartState(np.array(T.HOME),np.array([0.,0.,0.,1.]),z,z,z,z,
                {'spin':math.radians(spin),'lid':math.radians(lid)})
    tr=T.new_track();tr['phase']=phase
    T.track_update(tr,{'waffle':w},None)
    return tr

for angle in (0.,5.,5.6,7.):
    assert observe(2,angle,100.)['max_lid_turned']==0.,angle
assert observe(2,8.,100.)['max_lid_turned']>math.radians(99.)
assert observe(0,8.,10.)['max_lid_turned']>math.radians(9.)
assert observe(1,90.,10.)['max_lid_turned']>math.radians(9.)
print('PASS: consistent home tolerance; open first turn and later re-turn still detected')
