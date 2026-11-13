"""Measure authored collision meshes in metres, independently of manifest values."""
from pathlib import Path
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

root=Path(__file__).resolve().parent
checks={
    'iphone17':('/iphone17/frame',[71.5,149.6,7.95]),
    'airpods_pro':('/airpods_pro/shell',[60.6,45.2,21.7]),
    'apple_watch':('/apple_watch/case/frame',[36,42,9.7]),
}
result={'nominal_body_envelopes_mm':{},'excludes':'camera bumps, watch crown, display trims and straps'}
for name,(path,expected) in checks.items():
    stage=Usd.Stage.Open(str(root/'assets'/f'{name}.usda'))
    assert UsdGeom.GetStageMetersPerUnit(stage)==1.
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(path))
    measured=np.ptp(np.asarray(mesh.GetPointsAttr().Get()),axis=0)*1000
    assert np.allclose(measured,expected,atol=.001), (name,measured)
    assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    result['nominal_body_envelopes_mm'][name]=[round(float(x),4) for x in measured]
stage=Usd.Stage.Open(str(root/'assets/belkin.usda'))
colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
sectors=[p for p in colliders if p.GetName().startswith('deck_sector_')]
assert len(sectors)==96
assert not stage.GetPrimAtPath('/belkin/airpods_pad').IsValid()
base=UsdGeom.Mesh(stage.GetPrimAtPath('/belkin/base'))
z=np.asarray(base.GetPointsAttr().Get())[:,2]+.005
assert abs(float(max(z))-.010)<1e-8
result.update({'passed':True,'native_recess_convex_sectors':len(sectors),
               'recess_floor_height_mm':float(max(z)*1000),
               'charger_internals':'user-adjusted approximate dimensions; not certified manufacturer CAD'})
(root/'geometry-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
