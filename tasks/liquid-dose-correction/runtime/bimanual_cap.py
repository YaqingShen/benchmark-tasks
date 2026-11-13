"""Contact-driven two-hand cap diagnostic. No object-state writes."""
import numpy as np
from native_coupling import array,transform
from native_probes import PipetteProbe,CapProbe


class ArmTrajectory:
    command=PipetteProbe.command

    def __init__(self,action,side,waypoints):
        from yam_arena import ik,station
        self.ik=ik;self.side=side;self.offset=0 if side=='left' else 7
        self.base_position=station.BASES[side]
        self.action=np.array(action).copy();self.q=self.action[self.offset:self.offset+6].copy()
        self.start_pos,self.start_rot=ik.tcp_pose(self.q,self.base_position)
        self.waypoints=waypoints;self.duration=sum(w[0]for w in waypoints)
        self.max_position_error=0.;self.max_rotation_error=0.


class JointMove:
    """Smooth native joint targets for an empty-hand clearance move."""
    def __init__(self,action,side,q,duration,label):
        from yam_arena import ik
        self.ik=ik;self.action=np.array(action).copy();self.offset=0 if side=='left' else 7
        self.start=self.action[self.offset:self.offset+6].copy();self.target=np.array(q)
        if np.any(self.target<ik.LOWER) or np.any(self.target>ik.UPPER):raise ValueError('Joint target outside limits')
        self.duration=duration;self.label=label

    def command(self,t):
        s=self.ik.smoothstep(t/self.duration);action=self.action.copy()
        action[self.offset:self.offset+6]=(1-s)*self.start+s*self.target
        return action,self.label


class TwoHandCapProbe:
    def __init__(self,base,initial_action,tool='bottle_A_cap'):
        if tool!='bottle_A_cap':raise ValueError('This diagnostic currently tests bottle A')
        from yam_arena import station
        self.base=base;self.tool=tool;self.action=np.array(initial_action).copy()
        bottle=transform(array(base.scene['bottle_A'].data.root_link_pose_w)[0])
        # The finger pads extend 13 mm beyond the TCP along the approach
        # direction. Center the pads, rather than the TCP, on the bottle.
        self.initial_bottle=bottle.copy();center=bottle[:3,3]+[0,-.013,.095]
        side=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
        parked=self.action[:6].copy();parked[0]=.65
        self.park=JointMove(self.action,'left',parked,3.,'clear cap hand from bottle approach')
        self.prep_started=False
        self.prep=ArmTrajectory(self.action,'right',[
            (3.,np.array([.12,.02,station.TABLE_TOP+.28]),side,1.,'raise bottle hand'),
            (3.,center+[0,0,.165],side,1.,'approach bottle from above'),
            (3.,center,side,1.,'lower around bottle body'),
            (1.,center,side,.076/.0939,'grasp bottle body'),
            (1.,center,side,.076/.0939,'hold bottle body')])
        self.opening_start=self.park.duration+self.prep.duration
        self.cap=None;self.duration=self.opening_start+30.;self.rows=[]
        self.finger_ids,self.finger_names=base.scene['right_robot'].find_bodies(['tip_left','tip_right'],preserve_order=True)
        self.right_joint_ids,_=base.scene['right_robot'].find_joints(['joint'+str(i)for i in range(1,7)],preserve_order=True)

    def command(self,t):
        if t<=self.park.duration:
            self.action,label=self.park.command(t);return self.action,label
        if not self.prep_started:
            self.prep.action=self.action.copy();self.prep_started=True
        if t<=self.opening_start:
            self.action,label=self.prep.command(t-self.park.duration);return self.action,label
        if self.cap is None:
            settled=[r for r in self.rows if r['phase']=='hold bottle body'][-10:]
            if len(settled)<10 or sum(min(r['bottle_jaw_contact_n'])>.05 for r in settled)<8:
                raise RuntimeError('Bottle is not held between both fingers; stop before turning the cap')
            self.cap=CapProbe(self.base,self.action,self.tool)
            if abs(self.cap.duration-30)>1e-9:raise RuntimeError('Opening timeline changed')
        self.action,label=self.cap.command(t-self.opening_start)
        return self.action,label

    def observe(self,t,label):
        row={'time_s':t,'phase':label}
        if t>self.opening_start:
            self.cap.observe(t-self.opening_start,label)
            row.update(self.cap.rows[-1]);row['time_s']=t
        forces=array(self.base.scene['bottle_hold_contacts'].data.normal_force_matrix_w)[0,0]
        row['bottle_jaw_contact_n']=np.linalg.norm(forces,axis=-1).tolist()
        row['right_finger_positions']=array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.finger_ids,:3].tolist()
        row['right_joints_measured']=array(self.base.scene['right_robot'].data.joint_pos)[0,self.right_joint_ids].tolist()
        row['right_joints_commanded']=self.action[7:13].tolist()
        bottle=transform(array(self.base.scene['bottle_A'].data.root_link_pose_w)[0])
        row['bottle_position']=bottle[:3,3].tolist()
        row['bottle_drift_since_start_m']=float(np.linalg.norm(bottle[:3,3]-self.initial_bottle[:3,3]))
        self.rows.append(row)

    def result(self):
        opened=self.cap.result() if self.cap is not None else {}
        holding=[r for r in self.rows if r['time_s']>self.opening_start]
        fraction=sum(min(r['bottle_jaw_contact_n'])>.05 for r in holding)/max(1,len(holding))
        passed=opened.get('passed_probe',False) and fraction>=.95
        return {'type':'TWO_HAND_CAP_OPENING_PROBE','full_task_demo':False,'task_score':None,
            'passed_probe':bool(passed),'passed_opening':opened.get('passed_probe',False),
            'bilateral_bottle_contact_fraction':fraction,
            'max_bottle_drift_m':max((r['bottle_drift_since_start_m']for r in self.rows),default=0.),
            'cap_placed_inside_up':False,'pose_writes_after_reset':0,'rows':self.rows}
