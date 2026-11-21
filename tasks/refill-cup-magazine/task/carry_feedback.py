"""Object-pose feedback for the scripted demonstration, never an observation to Dream-1."""
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CARRY={"raise wrapped cups","translate together","guide bottom into magazine","hold at opening",
       "release lower grip over guide","clear cup stack"}
EXTRACT={"align with both ends","pinch both ends","draw sleeve off","carry empty sleeve","set wrapper aside"}

def publish(scene,step,pose):
    from yam_arena.runtime import tcp_world
    rotation=Rotation.from_quat(pose[3:])
    offsets=[[0.,-.060,.070],[0.,.052,.067],[0.,-.052,.067]]
    markers=pose[:3]+rotation.apply(offsets)
    p=Path(os.environ["CUP_RUN_DIR"])
    sample=np.r_[step,markers.ravel(),tcp_world(scene["left_robot"]),tcp_world(scene["right_robot"]),pose[3:]]
    tmp=p/"seam.tmp"
    tmp.write_bytes(sample.astype("<f8").tobytes())
    tmp.replace(p/"seam.bin")

def install(demo):
    if getattr(demo.ArmPlan.target,"cup_feedback",False):
        return
    original=demo.ArmPlan.target
    anchors={}
    def target(plan,t):
        position,rotation,finger,label=original(plan,t)
        right=plan.waypoints[0].yaw==10.
        if label not in EXTRACT and (not right or label not in CARRY):
            return position,rotation,finger,label
        f=Path(os.environ["CUP_RUN_DIR"])/"seam.bin"
        if not f.exists():
            return position,rotation,finger,label
        data=np.frombuffer(f.read_bytes(),dtype="<f8")
        if len(data)!=20 or abs(data[0]-t*20.)>4.:
            return position,rotation,finger,label
        if label in CARRY:
            position=position+.82*np.clip(data[1:4]-position,[-.035,-.022,-.012],[.035,.022,.012])
        else:
            if os.environ.get("CUP_YAW_FOLLOW","1")=="1":
                # Keep pad closure normal to the observed seal as it turns.
                normal=Rotation.from_quat(data[16:20].copy()).apply([1.,0.,0.])
                angle=np.arctan2(normal[1],normal[0])
                angle=(angle+np.pi/2)%np.pi-np.pi/2
                rotation=Rotation.from_rotvec([0.,0.,np.clip(angle,-.65,.65)]).as_matrix()@rotation
            marker=(data[7:10] if right else data[4:7]).copy()
            marker[2]-=float(os.environ.get("CUP_TAB_DEPTH",".002"))
            tcp=data[13:16] if right else data[10:13]
            if label=="align with both ends":
                position=tcp+np.clip(marker-tcp,-.004,.004)
                anchors.pop(right,None)
            elif label=="pinch both ends":
                position=marker
                anchors.pop(right,None)
            else:
                if right not in anchors:
                    anchors[right]=(position.copy(),marker.copy())
                nominal,point=anchors[right]
                wanted=point+position-nominal
                # Correct small jaw tracking errors while allowing both arms to
                # pull the film clear against gravity and friction.
                lead=float(os.environ.get("CUP_PULL_LEAD",".008"))
                position=marker+np.clip(wanted-marker,[-.009,-.009,-lead],[.009,.009,lead])
        return position,rotation,finger,label
    target.cup_feedback=True
    demo.ArmPlan.target=target
