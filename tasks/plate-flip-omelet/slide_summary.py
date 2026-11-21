"""Summarize displacement of each independently released diagnostic specimen."""
import argparse
import json
from pathlib import Path

def summarize(record):
    cases=record['cases'];paths=record['travel_mm_by_physics_step'];dt=record['dt_s']
    if len(cases)!=len(paths) or len({tuple(k) for k in cases})!=len(cases):
        raise ValueError('Missing or repeated coupon')
    def crossing(s,mark):
        last=0.
        for i,v in enumerate(s):
            if v>=mark:return (i+(mark-last)/(v-last))*dt
            last=v
        return None
    out={}
    for surface in ('plate','skillet'):
        for face in ('brown','plain'):
            group=sorted([(c[2],s) for c,s in zip(cases,paths) if c[:2]==[surface,face]])
            if not group or any(len(s)!=round(1/dt) for _,s in group): raise ValueError('Missing 1 s of 240 Hz samples')
            holds=[a for a,s in group if max(s)<2.]
            slides=[a for a,s in group if max(s)>=2.]
            if not holds or not slides or max(holds)>=min(slides): raise ValueError(f'Unbracketed onset: {surface}, {face}, {holds}, {slides}')
            fast=dict(group)[35]; t=crossing(fast,15)
            if t is None: raise ValueError('Did not reach 15 mm')
            out[surface+'_'+face]=dict(last_hold_deg=max(holds),first_slide_deg=min(slides),travel_15mm_s=t,
                                        fixed_slope_deg=35,hold_threshold_mm=2,observation_s=1)
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path)
    print(json.dumps(summarize(json.loads(p.parse_args().data.read_text())),indent=2))
