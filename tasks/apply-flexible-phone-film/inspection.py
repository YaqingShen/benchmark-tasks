"""Shared rubric for calibrated inspection photos and the physical simulation.

All XY points are in mm relative to the near-left target corner. For a bench
trial enter four visible phone corners in the same order as PHONE, and trace
the outer film outline and bubble outlines in the same overhead image.
"""
import json,sys
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np

CFG=json.loads(Path(__file__).with_name('bench-config.json').read_text())
L,W=CFG['target_mm']
PL,PW,_=CFG['phone_mm']
TARGET=np.array([[0,0],[L,0],[L,W],[0,W]],float)
MARGIN=np.array([(PL-L)/2,(PW-W)/2])
PHONE=TARGET+np.array([-MARGIN,[MARGIN[0],-MARGIN[1]],MARGIN,[-MARGIN[0],MARGIN[1]]])
STATIONS=tuple(CFG['checkpoints_mm'])

def area(poly):
    p=np.asarray(poly)
    return 0. if len(p)<3 else abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2

def transform(ref,world):
    a=[];b=[]
    for (x,y),(u,v) in zip(ref,world):
        a.extend(([x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]))
        b.extend((u,v))
    return np.r_[np.linalg.solve(a,b),1.].reshape(3,3)

def apply_h(h,points):
    q=np.c_[points,np.ones(len(points))]@h.T
    return q[:,:2]/q[:,2,None]

def score(corners,coverage,removed,checkpoints,bubbles,clear,first_press):
    err=float(np.linalg.norm(np.asarray(corners)-TARGET,axis=1).max())
    # Bubbles are measured by maximum caliper diameter, not count of cells.
    largest=max((float(np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2).max())
                 for p in bubbles),default=0.)
    bubble_area=sum(area(p) for p in bubbles)
    points=dict(alignment=.20*int(err<=CFG['corner_limit_mm']),coverage=.20*int(first_press)*min(1.,coverage/CFG['full_coverage_fraction']),
                removal=.15*int(removed),
                progressive=.25*sum(bool(v) for v in checkpoints)/3,
                finish=.20*int(first_press and removed and coverage>=CFG['full_coverage_fraction'] and largest<=CFG['bubble_diameter_limit_mm']
                                and bubble_area<=CFG['bubble_total_area_limit_mm2'] and clear))
    return dict(score=round(sum(points.values()),6),points=points,corner_error_mm=err,
                coverage=coverage,first_press=bool(first_press),stations=list(map(bool,checkpoints)),liner_clear=bool(removed),
                max_bubble_mm=largest,bubble_area_mm2=bubble_area,card_clear=bool(clear))

def from_photo(d):
    h=transform(d['phone_corners_px'],PHONE)
    outlines = d.get('flat_outlines_px')
    if outlines is None:
        outlines = [d['film_outline_px']]
    flat = [Polygon(apply_h(h, ring)) for ring in outlines]
    coverage = unary_union(flat).intersection(Polygon(TARGET)).area/(L*W)
    corners=apply_h(h,d['film_corners_px'])
    bubbles=[apply_h(h,p) for p in d['bubble_outlines_px']]
    checkpoints=[g is not None and 0<=g<=CFG['max_fold_gap_mm'] for g in d['video_gaps_mm']]
    if len(checkpoints)!=3:raise ValueError('Record 35, 70 and 105 mm checkpoints.')
    return score(corners,coverage,d['liner_clear'],checkpoints,bubbles,d['card_clear'], d['first_press'])
if __name__=='__main__':print(json.dumps(from_photo(json.loads(Path(sys.argv[1]).read_text())),indent=2))
