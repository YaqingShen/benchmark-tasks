import numpy as np
from fill_model import Dispenser,CAP,plane_for,volume_below
from cup_geometry import FILL_POSITION,level_for_volume
pose=np.eye(4);pose[:3,3]=FILL_POSITION
m=Dispenser()
for i in range(120):m.advance(pose,1 if i<106 else 0,.05)
q=m.metrics();assert abs(m.inside-318)<1e-6 and q['spill_ml']==0 and q['airborne_ml']==0
z=plane_for(m.inside,pose[2,:3]);assert abs(z-level_for_volume(318))<.0003
print('centred fixed cup: mL, fill %, surface mm, independent frustum mm', m.inside,100*m.inside/CAP,z*1000,level_for_volume(318)*1000)
pose[0,3]+=.12
for i in range(30):m.advance(pose,1 if i<10 else 0,.05)
assert abs(m.spill-30)<1e-8
pose[:3,:3]=np.diag([1,-1,-1]);m.advance(pose,0,.05);assert m.inside==0 and abs(m.spill-m.emitted)<1e-6
print('off-centre discharge, inversion and volume conservation passed')
