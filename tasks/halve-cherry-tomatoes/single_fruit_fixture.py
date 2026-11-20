"""Paired PhysX controls. A free 85 g sled starts at the surface of a restrained fruit."""
import json
import tempfile
from pathlib import Path
import numpy as np
import ovstage
from ovphysx import PhysX, PhysXConfig, TensorType
from produce_contact import contact, RADIUS
USDA='''#usda 1.0
(defaultPrim="World"
 metersPerUnit=1
 upAxis="Z")
def Xform "World" {
 def PhysicsScene "Physics" {
 float physics:gravityMagnitude = 0
 }
 def Cube "sled" (prepend apiSchemas=["PhysicsRigidBodyAPI","PhysicsMassAPI","PhysicsCollisionAPI"]) {
 double size=0.02
 float physics:mass = 0.085
 }
}
'''
def measure(path, name, drive, fruit):
    engine=PhysX(config=PhysXConfig(num_threads=1))
    stage=ovstage.Stage(name)
    ovstage.population.open_usd(stage,str(path),ordinal=1,domains=ovstage.PopulationDomain.PHYSICS)
    stage.advance_write_floor(ordinal=1).wait()
    engine.attach_ovstage(stage,read_ordinal=1)
    engine.wait_all()
    def binding(kind):
        b=engine.create_tensor_binding(pattern='/World/sled',tensor_type=kind)
        assert b.count==1
        return b
    pose=binding(TensorType.RIGID_BODY_POSE)
    vel=binding(TensorType.RIGID_BODY_VELOCITY)
    force=binding(TensorType.RIGID_BODY_FORCE)
    pos=np.zeros((1,7),np.float32); speed=np.zeros((1,6),np.float32)
    peak=0.
    for _ in range(480):
        pose.read(pos);vel.read(speed)
        resistance=contact((0.,RADIUS-pos[0,0],0.),speed[0,0])[0] if fruit else 0.
        peak=max(peak,abs(resistance))
        force.write(np.array([[drive-8*speed[0,0]-resistance,0,0]],dtype=np.float32))
        engine.step_sync(1/960)
    pose.read(pos)
    return dict(displacement_mm=float(pos[0,0]*1000),peak_load_N=float(peak))
PhysX.set_cpu_mode(True)
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'sled.usda';path.write_text(USDA)
    results={name:measure(path,name,drive,present) for name,drive,present in
             [('empty',.4,False),('held_fruit',.4,True),('higher_drive',1.8,True)]}
print(json.dumps(results,indent=2),flush=True)
assert results['empty']['displacement_mm']>20
assert 0<results['held_fruit']['displacement_mm']<2.5
assert results['higher_drive']['displacement_mm']>28
