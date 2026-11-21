"""A previously valid plated result cannot survive hand contact with a cooking face."""
import math,numpy as np
import test_release as setup
import task as T
from hot_surface_guard import zones
T._TRACK.clear();T._TRACK.update(T.new_track())
T._TRACK.update(phase=2,latched_at_turn=True,shut_at_turn=True,loosened=True,tool_stowed=True,tool_carried=True)
assert T.breakdown(setup.objects)['full_success']
pieces=dict(zones(T,setup.objects['waffle']))
c=pieces['upper'].mean(0)
signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
T.track_update(T._TRACK,setup.objects,setup.far,[c+.002*signs])
assert T._TRACK['inner_surface_contact'] and T._TRACK['inner_contact_surface']=='upper'
assert T.score(setup.objects)==0 and not T.success(setup.objects)
T.track_update(T._TRACK,setup.objects,setup.far,[np.ones(3)*2+.002*signs])
assert T.score(setup.objects)==0,'later withdrawal must not erase interior contact'
T.phase_conversion.STATE['solid']=False
T._TRACK.clear();T._TRACK.update(T.new_track())
T.track_update(T._TRACK,setup.objects,setup.far,[c+.002*signs])
assert T._TRACK['inner_surface_contact'],'guard must work before baking as well'
print('PASS: interior contact revokes prior success, persists after withdrawal, and is active before baking')
