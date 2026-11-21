"""Check physical cold-trial measurements against the shared benchmark configuration."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

CFG = json.loads(Path(__file__).with_name("bench-config.json").read_text())

def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("Enter a finite numeric measurement.")
    return value

def check_record(data, folder):
    if not data.get("trial_id") or data.get("task")!=CFG["task"]:
        raise ValueError("Set the task ID and physical specimen ID.")
    for key in ("wood_source","plate_surface","skillet_surface","grain_direction"):
        if not isinstance(data.get(key),str) or not data[key].strip():
            raise ValueError("Missing specimen or surface description: "+key)
    dimensions={}
    for field,tol in (("mass_g","mass_tolerance_g"),("diameter_mm","diameter_tolerance_mm"),("thickness_mm","thickness_tolerance_mm")):
        value=number(data[field])
        dimensions[field]=abs(value-CFG[field])<=CFG[tol]
    slide=CFG["slide_test"]
    groups={}
    clips={}
    for obs in data["sliding"]:
        surface,face=obs["surface"],obs["face"]
        if surface not in ("plate","skillet") or face not in ("brown","plain"):
            raise ValueError("Unknown surface or face.")
        key=(surface,face)
        angle=number(obs["first_motion_deg"])
        if not 0<angle<90: raise ValueError("Invalid release angle.")
        fps=number(obs["fps"])
        start,end=obs["release_frame"],obs["travel_frame"]
        if fps<slide["minimum_video_hz"] or type(start)!=int or type(end)!=int or start<0 or end<=start:
            raise ValueError("The fixed-slope video requires valid release and arrival frames.")
        files=[]
        for field in ("onset_video","travel_video"):
            name=obs[field]
            path=folder/name
            if not path.is_file() or not path.stat().st_size: raise ValueError("Missing evidence: "+name)
            if name not in clips: clips[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(name)
        if not math.isclose(number(obs["fixed_slope_deg"]),slide["kinetic_slope_deg"],abs_tol=.5):
            raise ValueError("Fixed-slope test angle is outside tolerance.")
        t=(end-start)/fps
        groups.setdefault(key,[]).append((angle,t,files[0],files[1],start,end))
    result={}
    for surface in ("plate","skillet"):
        for face in ("brown","plain"):
            obs=groups.get((surface,face),[])
            if len(obs)!=slide["repeats"] or len({(o[3],o[4],o[5]) for o in obs})!=len(obs):
                raise ValueError("Record distinct repeat trials for both faces on both utensils.")
            angles=[o[0] for o in obs]; times=[o[1] for o in obs]
            limits=slide["acceptance"][surface]
            angle=statistics.median(angles); t=statistics.median(times)
            angle_ok=limits["onset_deg"][0]<=angle<=limits["onset_deg"][1]
            time_ok=limits["travel_s"][0]<=t<=limits["travel_s"][1]
            result[surface+"_"+face]=dict(median_angle_deg=angle,median_time_s=t,
                angle_range_deg=[min(angles),max(angles)],time_range_s=[min(times),max(times)],
                passes=angle_ok and time_ok and max(angles)-min(angles)<=slide["max_repeat_angle_spread_deg"])
    return dict(trial_id=data["trial_id"],dimensions_pass=dimensions,sliding=result,
                matched=all(dimensions.values()) and all(v["passes"] for v in result.values()),files_sha256=clips)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("record",type=Path)
    args=p.parse_args()
    try:
        result=check_record(json.loads(args.record.read_text()),args.record.parent)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(2,"Incomplete physical record: "+str(exc)+"\n")
    print(json.dumps(result,indent=2))
