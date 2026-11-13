"""Static USD audit; this is not a PhysX or robot validation."""
import hashlib
import json
import math
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics
from pipette_geometry import BULB_SPRING_N_M

ROOT=Path(__file__).resolve().parent


def main():
    folder=ROOT/'assets/mechanisms'
    manifest=json.loads((folder/'native-assets.json').read_text())
    results={}
    for name,spec in manifest['assets'].items():
        path=folder/spec['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256'],name
        stage=Usd.Stage.Open(str(path));root=stage.GetDefaultPrim()
        assert root and root.GetName()==name,name
        assert UsdGeom.GetStageMetersPerUnit(stage)==1.
        joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        reference=Usd.Stage.CreateInMemory()
        reference.DefinePrim('/instance').GetReferences().AddReference(str(path))
        for p in reference.Traverse():
            for rel in p.GetRelationships():
                if rel.GetName().startswith('material:binding') or rel.GetName() in ('physics:body0','physics:body1'):
                    assert all(reference.GetPrimAtPath(t) for t in rel.GetTargets()),str(p.GetPath())
        if name.startswith('pipette_'):
            assert len(bodies)==3 and len(joints)==2
            assert not root.HasAPI(UsdPhysics.RigidBodyAPI)
            for p in joints:
                joint=UsdPhysics.PrismaticJoint(p)
                assert joint.GetAxisAttr().Get()=='X'
                drive=UsdPhysics.DriveAPI(p,'linear')
                assert drive.GetTargetPositionAttr().Get()==0.
                assert abs(drive.GetStiffnessAttr().Get()-BULB_SPRING_N_M)<1e-6
                assert abs(joint.GetUpperLimitAttr().Get()-joint.GetLowerLimitAttr().Get()-.0058)<1e-8
        if name.endswith('_cap'):
            assert stage.GetPrimAtPath('/'+name+'/inner_seal')
            # Bore apothem must clear male crests for every angular phase.
            assert .0232*math.cos(math.pi/24)>.0225
        results[name]={'rigid_bodies':len(bodies),'joints':len(joints),'references_resolve':True}
    report={'status':'STATIC_USD_AUDIT_PASSED','native_simulation_test':False,'assets':results}
    out=ROOT.parent.parent/'outputs/formulation-prototype/static-usd-audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
