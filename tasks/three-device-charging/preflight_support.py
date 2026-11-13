"""Plan a feasible physical support approach before running contact dynamics."""
import json,numpy as np
from yam_arena import ik,station,constants
from yam_arena.urdf import quat_xyzw_to_matrix

_,neutral=ik.tcp_pose(np.array(constants.READY));angle=.75;c,s=np.cos(angle),np.sin(angle)
rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
mount=quat_xyzw_to_matrix(np.array(station.wrist_camera('right').rot_xyzw));c,s=np.cos(.18),np.sin(.18)
up=np.array([[1.,0,0],[0,c,-s],[0,s,c]]);r0=rz@neutral@mount@up@mount.T
q0,_,_=ik.solve([.20,-.25,1.04],ik.matrix_to_quat_xyzw(r0),constants.READY,station.BASES['right'],iters=400,damping=.02)
p0,r0=ik.tcp_pose(q0,station.BASES['right']);results=[]
for theta in [0.,.25,.45,.65]:
    for sign in [1.,-1.]:
        z=np.array([np.sin(theta),0,-np.cos(theta)]);x=np.array([0,sign,0]);r=np.column_stack([x,np.cross(z,x),z])
        p=np.array([.51-.025,-.041,.7475+.012])-.013*z+np.array([0,0,-.004])
        q=q0.copy();start_p,start_r=p0,r0;max_p=max_r=0.
        for target in [np.array([.28,-.16,.92]),p+[0,0,.09],p]:
            for blend in np.linspace(0,1,25):
                desired=start_p*(1-blend)+target*blend;rot=ik.slerp_matrix(start_r,r,float(blend))
                q,ep,er=ik.solve(desired,ik.matrix_to_quat_xyzw(rot),q,station.BASES['right'],iters=100,damping=.018,tol=3e-5)
                max_p=max(max_p,ep);max_r=max(max_r,er)
            start_p,start_r=target,r
        row={'theta':theta,'sign':sign,'point':p.tolist(),'rotation':r.tolist(),'max_position_error':max_p,'max_rotation_error':max_r,'end_joints':q.tolist()}
        results.append(row)
        print(json.dumps(row),flush=True)
print('BEST',json.dumps(min(results,key=lambda r:r['max_position_error']*100+r['max_rotation_error'])))
