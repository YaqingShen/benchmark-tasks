"""Intersect each slider finger with the actual generated gate polygons."""
import itertools
import json
from pathlib import Path
import numpy as np
import mechanism as m

def overlaps(a,b):
    for polygon in (a,b):
        for delta in np.roll(polygon,-1,axis=0)-polygon:
            normal=np.array([-delta[1],delta[0]])
            x,y=a@normal,b@normal
            if x.max()<=y.min() or y.max()<=x.min():
                return False
    return True

def blocked_at(wheel,reading,travel):
    angle=(reading-m.START[wheel])*m.STEP
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    finger=np.array([[-.0130+x*.0115/2+travel,z*.0008/2] for x,z in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    model=m.mechanism(ShapeCollector,'fixture',(0,0,0))
    for shape in model[wheel]['meshes']:
        coordinates,kind=shape
        if kind!='hull':continue
        gate=(coordinates[:4,1:]-[0,m.Z])@rotation.T
        if overlaps(finger,gate):return True
    return False

class ShapeCollector:
    Mesh=staticmethod(lambda points,tris,color,kind:(points,kind))
    Joint=staticmethod(lambda name,**params:params)
    part=staticmethod(lambda *args,**params:params['joints'])

if __name__=='__main__':
    blocked=[[blocked_at(i,n,.0057) for n in range(10)] for i in range(3)]
    codes=[s for s in itertools.product(range(10),repeat=3) if not any(blocked[i][n] for i,n in enumerate(s))]
    assert codes==[m.CODE],codes
    assert not any(blocked_at(i,n,0.) for i in range(3) for n in range(10))
    Path('evidence/gate-geometry.json').write_text(json.dumps(dict(full_stroke_mm=5.7,permitted_codes=codes,blocked_codes=999,rest_clear=True),indent=2)+'\n')
