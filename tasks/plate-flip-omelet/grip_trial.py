"""Summarize optional physical gripper trials from two calibrated pad transducers."""
import argparse
import csv
import json
import math
from pathlib import Path

CFG=json.loads(Path(__file__).with_name("bench-config.json").read_text())
def check(data,folder):
    if data.get("task")!=CFG["task"] or not all(isinstance(data.get(k),str) and data[k].strip() for k in
            ("gripper_model","pad_sensors","calibration_id","skillet_handle")):
        raise ValueError("Identify the real gripper, handle and sensor calibration.")
    unc=data["uncertainty_N"]
    if not isinstance(unc,(float,int)) or not math.isfinite(unc) or unc<0: raise ValueError("Enter force uncertainty.")
    results=[]
    files=set()
    for t in data["trials"]:
        force=folder/t["force_csv"];video=folder/t["video"]
        if force.resolve() in files or not video.is_file() or not video.stat().st_size:
            raise ValueError("Each trial needs an independent force record and its video.")
        files.add(force.resolve())
        with force.open(newline="") as stream:
            rows=[[float(r[k]) for k in ("time_s","left_pad_N","right_pad_N")] for r in csv.DictReader(stream)]
        if len(rows)<2 or any(not math.isfinite(x) for row in rows for x in row) or any(min(r[1:])<0 for r in rows):
            raise ValueError("Invalid force trace.")
        deltas=[b[0]-a[0] for a,b in zip(rows,rows[1:])]
        if min(deltas)<=0 or max(deltas)>1/240+1e-5: raise ValueError("Record continuously at 240 Hz or faster.")
        first,last=t["turn_start_s"],t["turn_end_s"]
        if not rows[0][0]<=first<last<=rows[-1][0]: raise ValueError("The force record must span the filmed turnover.")
        peaks=[max(r[j] for r in rows) for j in (1,2)]
        angle,slip=t["maximum_turn_deg"],t["handle_drift_mm"]
        if not 0<=angle<=180 or not 0<=slip or not math.isfinite(slip): raise ValueError("Invalid angular or slip observation.")
        limit=CFG["simulation"]["comparison_limit_per_finger_N"]
        results.append(dict(pad_peak_N=peaks,passes=(max(peaks)+unc<=limit and angle>CFG["grip_test"]["turnover_angle_deg"]
            and slip<=CFG["grip_test"]["maximum_handle_drift_mm"] and t["rested_and_released"] is True)))
    if len(results)!=CFG["grip_test"]["repeats"]:raise ValueError("Three complete independent physical turnovers are required.")
    return dict(trials=results,passes=all(r["passes"] for r in results))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("record",type=Path);a=p.parse_args()
    try:result=check(json.loads(a.record.read_text()),a.record.parent)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(2,"Incomplete physical grip record: "+str(exc)+"\n")
    print(json.dumps(result,indent=2))
