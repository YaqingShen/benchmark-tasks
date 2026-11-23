"""Measure the shared USD geometry and check enclosure reach in the scored run."""
from pathlib import Path
import hashlib,json,math,sys,statistics
from pxr import Usd,UsdGeom
run=Path(sys.argv[1])
sys.path.insert(0,str((run/'source').resolve()))
from spill_station import build_pan
stage=Usd.Stage.CreateInMemory()
build_pan(stage,'/World',0.,separate_floor=True)
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_],False,True)
def bounds(name):
    prim=stage.GetPrimAtPath('/World/spill_pan/'+name)
    box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return tuple(box.GetMin()),tuple(box.GetMax())
def inside(prefix,names):
    near,far,left,right=[bounds(prefix+n) for n in names]
    return far[0][0]-near[1][0],left[0][1]-right[1][1],near[1][2]
w,l,h=inside('',['arm_side','far_side','left_end','right_end'])
wo,lo,zo=inside('secondary_',['near','far','left','right'])
height=bounds('sealed_lid')[0][2]
ow=bounds('far_side')[1][0]-bounds('arm_side')[0][0]
ol=bounds('left_end')[1][1]-bounds('right_end')[0][1]
primary=(w*l-.120)*.020*1000
secondary=(wo*lo-ow*ol)*.020*1000
assert all(abs(a-b)<1e-6 for a,b in [(w,.600),(l,.800),(h,.040),(wo,.680),(lo,.880),(zo,.027),(height,.550)])
assert primary>7.19 and secondary>2.10
reach={}
for line in (run/'trace.jsonl').read_text().splitlines():
    data=json.loads(line)
    for side,y in [('left',.2),('right',-.2)]:
        tcp=data['robot'][side]['tcp']
        dist=math.dist(tcp,[.095,y,.7475+.295])
        reach[side]=max(reach.get(side,0.),dist)
assert max(reach.values())+.050<=.650
print(json.dumps(dict(source_geometry_sha256=hashlib.sha256((run/'source/spill_station.py').read_bytes()).hexdigest(),
    bounds_from_usd=True,inner_clear_mm=[round(1000*w),round(1000*l)],lowest_rim_mm=round(h*1000),
    outer_clear_mm=[round(1000*wo),round(1000*lo)],hood_height_mm=round(height*1000),
    primary_usable_l=round(primary,3),secondary_annular_l=round(secondary,3),
    max_sampled_port_center_to_tcp_m=reach,reach_sample_every_control_steps=20,
    physical_routing_checked=False),indent=2))
