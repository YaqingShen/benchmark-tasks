"""Isolated bottle-A delivery diagnostic using robot joints and balance feedback.

The cap starts on its rest in this diagnostic. This does not validate opening,
pipette correction, disposal or a complete single-liquid task.
"""
import math
import numpy as np
from bimanual_cap import TwoHandCapProbe,ArmTrajectory
from native_coupling import array,transform
from geometry_fluid import INTERIORS


class BottlePourProbe:
    def __init__(self,base,action,coupling):
        from yam_arena import ik,station,constants
        self.ik=ik;self.station=station;self.constants=constants
        self.base=base;self.coupling=coupling;self.action=np.array(action).copy()
        self.prep=TwoHandCapProbe(base,action,'bottle_A_cap')
        self.prep_duration=self.prep.opening_start
        self.duration=self.prep_duration+47.;self.rows=[]
        self.hand_id=base.scene['right_robot'].find_bodies('gripper')[0][0]
        self.joint_ids=base.scene['right_robot'].find_joints(['joint'+str(i)for i in range(1,7)],preserve_order=True)[0]
        self.home=transform(array(base.scene['bottle_A'].data.root_link_pose_w)[0])
        self.relative=None;self.phase=None;self.phase_index=-1
        self.stop=False;self.theta=87.5;self.q=self.action[7:13].copy()
        self.grip=.076/.0939;self.max_position_error=0.;self.max_rotation_error=0.
        beaker=coupling.poses()['beaker']
        self.lip=beaker[:3,3]+[-.015,0,INTERIORS['beaker'].mouth_z+.05]

    def hand_target(self,bottle):
        hand=bottle@np.linalg.inv(self.relative)
        return hand[:3,3]+hand[:3,:3]@np.array(self.constants.TCP_OFFSET),hand[:3,:3]

    def pour_target(self,degrees):
        a=math.radians(degrees);c,s=math.cos(a),math.sin(a)
        rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])@self.home[:3,:3]
        down=-rotation[2,:2];norm=np.linalg.norm(down)
        # Use the positive-X pouring side even when upright.
        direction=down/norm if norm>1e-7 else self.home[:3,:3].T@np.array([1.,0,0])
        local=np.r_[np.asarray(direction)[:2]*.0185,.194]
        bottle=np.eye(4);bottle[:3,:3]=rotation;bottle[:3,3]=self.lip-rotation@local
        return self.hand_target(bottle)

    def trajectory(self,index):
        # Contact and gravity make the actual arm differ from its previous
        # target. Start from measured joints to avoid pushing a held bottle
        # down while interpolating from an unachieved commanded pose.
        self.action[7:13]=array(self.base.scene['right_robot'].data.joint_pos)[0,self.joint_ids]
        if index==0:
            bottle=transform(array(self.base.scene['bottle_A'].data.root_link_pose_w)[0])
            hand=transform(array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.hand_id])
            self.relative=np.linalg.inv(hand)@bottle
            forces=array(self.base.scene['bottle_hold_contacts'].data.normal_force_matrix_w)[0,0]
            if min(np.linalg.norm(forces,axis=-1))<.05:raise RuntimeError('No bilateral bottle grasp before lift')
            target=bottle.copy();target[2,3]+=.14;p,r=self.hand_target(target)
            points=[(3.,p,r,self.grip,'lift bottle'),(2.,p,r,self.grip,'settle lifted bottle')]
        elif index==1:
            bottle=transform(array(self.base.scene['bottle_A'].data.root_link_pose_w)[0])
            if bottle[2,3]-self.home[2,3]<.07:raise RuntimeError('Bottle did not lift at least 70 mm')
            hand=transform(array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.hand_id])
            self.relative=np.linalg.inv(hand)@bottle
            p,r=self.pour_target(70.);points=[(4.,p,r,self.grip,'position tilted bottle over beaker')]
        elif index==2:
            p,r=self.pour_target(87.5);points=[(4.,p,r,self.grip,'approach pouring angle')]
        elif index==4:
            target=self.home.copy();target[2,3]+=.14
            angle=math.radians(70.);c,s=math.cos(angle),math.sin(angle)
            target[:3,:3]=np.array([[c,0,s],[0,1,0],[-s,0,c]])@self.home[:3,:3]
            p,r=self.hand_target(target);points=[(3.,p,r,self.grip,'carry tilted bottle clear of beaker')]
        elif index==5:
            target=self.home.copy();target[2,3]+=.14;p,r=self.hand_target(target)
            points=[(3.,p,r,self.grip,'return upright over bottle home')]
        elif index==6:
            p,r=self.hand_target(self.home)
            points=[(3.,p,r,self.grip,'lower bottle home'),(1.,p,r,1.,'release bottle'),
                    (2.,p+[0,-.10,.025],r,1.,'withdraw bottle hand')]
        else:raise ValueError(index)
        return ArmTrajectory(self.action,'right',points)

    def command(self,t):
        if t<=self.prep_duration:
            self.action,label=self.prep.command(t);return self.action,label
        elapsed=self.prep_duration
        for index,duration in enumerate([5.,4.,4.,22.,3.,3.,6.]):
            if t<=elapsed+duration+1e-8:
                if index!=self.phase_index:
                    if index!=self.phase_index+1:raise RuntimeError('Skipped bottle phase')
                    if index!=3:self.phase=self.trajectory(index)
                    self.phase_index=index;self.q=self.action[7:13].copy()
                if index==3:
                    reading=self.coupling.balance.last.get('net_g')
                    if reading is None:raise RuntimeError('Balance has not tared')
                    if reading>=8.5:self.stop=True
                    self.theta=max(70.,self.theta-1.5) if self.stop else min(89.7,self.theta+.0125)
                    p,r=self.pour_target(self.theta)
                    q,pe,re=self.ik.solve(p,self.ik.matrix_to_quat_xyzw(r),q_init=self.q,
                        base_pos=self.station.BASES['right'],iters=180,damping=.025,tol=5e-5)
                    self.max_position_error=max(self.max_position_error,pe);self.max_rotation_error=max(self.max_rotation_error,re)
                    if pe>.002 or re>.02:raise RuntimeError(f'Pouring IK infeasible: {pe:.5f} m, {re:.5f} rad')
                    self.q=q;self.action[7:13]=q
                    label='stop flow from balance reading' if self.stop else 'pour with balance feedback'
                else:self.action,label=self.phase.command(t-elapsed)
                return self.action,label
            elapsed+=duration
        raise RuntimeError('Bottle pour exceeded duration')

    def observe(self,t,label):
        if t<=self.prep_duration:self.prep.observe(t,label)
        bottle=transform(array(self.base.scene['bottle_A'].data.root_link_pose_w)[0])
        hand=transform(array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.hand_id])
        rel=np.linalg.inv(hand)@bottle
        forces=array(self.base.scene['bottle_hold_contacts'].data.normal_force_matrix_w)[0,0]
        measured_q=array(self.base.scene['right_robot'].data.joint_pos)[0,self.joint_ids]
        fk_p,fk_r=self.ik.tcp_pose(measured_q,self.station.BASES['right'])
        actual_tcp=hand[:3,3]+hand[:3,:3]@np.array(self.constants.TCP_OFFSET)
        ledger=self.coupling.ledger
        row={'time_s':t,'phase':label,'bottle_position':bottle[:3,3].tolist(),
             'bottle_tilt_deg':math.degrees(math.acos(float(np.clip(bottle[2,2],-1,1)))),
             'bottle_jaw_contact_n':np.linalg.norm(forces,axis=-1).tolist(),
             'bottle_contact_vectors_n':forces.tolist(),'right_gripper_pose':array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.hand_id].tolist(),
             'right_joints_measured':measured_q.tolist(),'right_joints_commanded':self.action[7:13].tolist(),
             'fk_native_position_error_m':float(np.linalg.norm(fk_p-actual_tcp)),
             'fk_native_rotation_error_rad':float(np.linalg.norm(self.ik._rot_error(fk_r,hand[:3,:3]))),
             'bottle_in_hand':rel[:3,3].tolist(),'beaker_A_g':ledger.vessels['beaker'].liquid.ingredient('A')*1.03,
             'spill_g':ledger.spill.mass(ledger.cfg['density_g_ml']),
             'balance':dict(self.coupling.balance.last),'flow_stopped_by_balance':self.stop}
        if self.relative is not None and label not in ('release bottle','withdraw bottle hand'):
            row['grasp_slip_m']=float(np.linalg.norm(rel[:3,3]-self.relative[:3,3]))
            # Allow initial grasp settling in this diagnostic, then evaluate
            # the measured lift. Stop well before the bottle can leave the pads.
            if row['grasp_slip_m']>.03:raise RuntimeError(f'Bottle slipped {row["grasp_slip_m"]:.4f} m during {label}')
        self.rows.append(row)

    def result(self):
        end=self.rows[-1] if self.rows else {}
        complete=end.get('time_s',0)>=self.duration-.1
        passed=complete and self.stop and 8.<end.get('beaker_A_g',0)<10. and end.get('spill_g',1)<.01
        return {'type':'BOTTLE_A_COARSE_MASS_DELIVERY_PROBE','full_task_demo':False,'task_score':None,
                'setup':'Bottle A cap starts inside-up on its rest; opening is not tested',
                'passed_probe':bool(passed),'delivered_A_g':end.get('beaker_A_g'),
                'spill_g':end.get('spill_g'),'balance':end.get('balance'),
                'max_bottle_lift_m':max((r['bottle_position'][2]-self.home[2,3]for r in self.rows),default=0.),
                'max_grasp_displacement_m':max((r.get('grasp_slip_m',0.)for r in self.rows),default=0.),
                'pose_writes_after_reset':0,'rows':self.rows}
