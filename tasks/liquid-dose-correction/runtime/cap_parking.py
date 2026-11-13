"""Candidate cap handoff, inversion and released inside-up parking.

All actions are robot joint/gripper targets. No cap attachment or state write.
This remains a mechanism diagnostic and cannot report a formulation score.
"""
import math
import numpy as np
from native_coupling import array,transform
from bimanual_cap import TwoHandCapProbe,ArmTrajectory,JointMove


class CapParkingProbe(TwoHandCapProbe):
    def __init__(self,base,action,tool,manifest):
        super().__init__(base,action,tool)
        from yam_arena import ik,station,constants
        self.ik=ik;self.station=station;self.constants=constants
        self.opening_duration=self.duration;self.opening_result=None
        self.rest=np.array(manifest['assets']['station_fixtures']['cap_rest_centers']['A'])
        self.right_gripper_id=base.scene['right_robot'].find_bodies('gripper')[0][0]
        self.receive_seed=np.array([1.512212642,2.088503136,1.576195536,-.810813660,1.519933549,-1.829926804])
        _,self.receive_rotation=ik.tcp_pose(self.receive_seed,station.BASES['right'])
        self.side_receive_rotation=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
        self.specs=[(5.,self.release_bottle),(3.,self.stage_receiver),(3.,self.move_cap),
                    (5.,self.receive_cap),(6.,self.release_cap_hand),(3.,self.flip_cap),(11.,self.park_cap)]
        self.duration+=sum(d for d,_ in self.specs)
        self.phase_index=-1;self.phase=None

    def arm(self,side,points):return ArmTrajectory(self.action,side,points)

    def tcp(self,side):
        offset=0 if side=='left' else 7
        return self.ik.tcp_pose(self.action[offset:offset+6],self.station.BASES[side])

    def release_bottle(self):
        p,r=self.tcp('right')
        raised=p+np.array([0.,0.,.025]);back=raised+np.array([0.,-.13,0.])
        return self.arm('right',[(1.,p,r,1.,'release stabilized bottle'),
            (1.,raised,r,1.,'clear low cap supports'),
            (1.,back,r,1.,'withdraw bottle hand'),
            (2.,np.array([.20,.02,1.05]),r,1.,'raise receiving hand')])

    def stage_receiver(self):
        return self.arm('right',[(3.,np.array([.24,-.02,.95]),self.side_receive_rotation,1.,'stage empty receiving hand')])

    def move_cap(self):
        from yam_arena.demo import grip_rotation
        r=grip_rotation(0)
        return self.arm('left',[(3.,np.array([.24,.10,.95]),r,self.action[6],'move cap to handoff')])

    def receive_cap(self):
        cap=transform(array(self.base.scene[self.tool].data.root_link_pose_w)[0])
        r=self.side_receive_rotation
        p=cap[:3,3]-r[:,2]*.013+np.array([0,0,.012])
        return self.arm('right',[(3.,p,r,1.,'approach cap from side'),
            (1.,p,r,.046/.0939,'grasp cap sides'),
            (1.,p,r,.046/.0939,'hold cap with receiving hand')])

    def release_cap_hand(self):
        forces=array(self.base.scene['cap_right_contacts'].data.normal_force_matrix_w)[0,0]
        if min(np.linalg.norm(forces,axis=-1))<.03:
            raise RuntimeError('Receiving hand lacks bilateral cap contact; retain the cap with the first hand')
        p,r=self.tcp('left');park=np.array(self.constants.READY);park[0]=.65
        park_p,park_r=self.ik.tcp_pose(park,self.station.BASES['left'])
        angle=math.radians(-20);c,s=math.cos(angle),math.sin(angle)
        clear_r=np.array([[c,0,s],[0,1,0],[-s,0,c]])@r
        return self.arm('left',[(1.,p,r,1.,'release cap from first hand'),
                               (2.,p+[0,0,.05],clear_r,1.,'lift first hand clear of cap'),
                               (3.,park_p,park_r,1.,'withdraw first hand')])

    def flip_cap(self):
        forces=array(self.base.scene['cap_right_contacts'].data.normal_force_matrix_w)[0,0]
        if min(np.linalg.norm(forces,axis=-1))<.03:
            raise RuntimeError('Receiving hand lost the cap before inversion')
        # Move the wrist and elbow together; a 180-degree wrist-only turn
        # would exceed the installed joint limit from this side grasp.
        return JointMove(self.action,'right',self.receive_seed,3.,'turn cap inside-up')

    def park_cap(self):
        cap=transform(array(self.base.scene[self.tool].data.root_link_pose_w)[0])
        hand=transform(array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.right_gripper_id])
        relative=np.linalg.inv(hand)@cap
        _,r=self.tcp('right');cap_r=r@relative[:3,:3]
        if -cap_r[2,2]<math.cos(math.radians(10)):
            raise RuntimeError('Cap did not invert with the receiving hand')
        # Allow the slightly tilted closed face to settle flat after release.
        root=self.rest.copy();root[2]+=.027*abs(cap_r[2,2])+.026*np.linalg.norm(cap_r[2,:2])+.001
        p=root-r@relative[:3,3]+r@np.array(self.constants.TCP_OFFSET)
        above=p+[0,0,.04];retreat=above.copy()
        grip=self.action[13]
        return self.arm('right',[(3.,above,r,grip,'above cap rest'),
            (3.,p,r,grip,'lower inside-up cap'),(1.,p,r,1.,'release cap on rest'),
            (2.,retreat,r,1.,'withdraw from parked cap'),(2.,retreat,r,1.,'check released inside-up cap')])

    def command(self,t):
        if t<=self.opening_duration:return super().command(t)
        if self.opening_result is None:
            self.opening_result=super().result()
            if not self.opening_result['passed_probe']:raise RuntimeError('Two-hand opening failed; do not proceed to cap parking')
        elapsed=self.opening_duration
        for index,(duration,build) in enumerate(self.specs):
            if t<=elapsed+duration+1e-8:
                if index!=self.phase_index:
                    if index!=self.phase_index+1:raise RuntimeError('Skipped cap handling phase')
                    self.phase=build();self.phase_index=index
                    if abs(self.phase.duration-duration)>1e-8:raise RuntimeError('Cap handling duration mismatch')
                self.action,label=self.phase.command(t-elapsed);return self.action,label
            elapsed+=duration
        raise RuntimeError('Cap parking command exceeded duration')

    def observe(self,t,label):
        if t<=self.opening_duration:return super().observe(t,label)
        cap=transform(array(self.base.scene[self.tool].data.root_link_pose_w)[0])
        forces=array(self.base.scene['cap_right_contacts'].data.normal_force_matrix_w)[0,0]
        velocity=array(self.base.scene[self.tool].data.root_link_vel_w)[0,:3]
        hand=transform(array(self.base.scene['right_robot'].data.body_link_pose_w)[0,self.right_gripper_id])
        self.rows.append({'time_s':t,'phase':label,'cap_position':cap[:3,3].tolist(),
            'cap_opening_up_cosine':float(-cap[2,2]),'cap_speed_m_s':float(np.linalg.norm(velocity)),
            'cap_rest_xy_error_m':float(np.linalg.norm(cap[:2,3]-self.rest[:2])),
            'cap_rest_normal_n':float(array(self.base.scene['cap_rest_contacts'].data.normal_force_matrix_w)[...,2].sum()),
            'receiving_finger_contact_n':np.linalg.norm(forces,axis=-1).tolist(),
            'receiving_gripper_opening':float(self.action[13]),
            'cap_in_receiving_hand':(np.linalg.inv(hand)@cap)[:3,3].tolist()})

    def result(self):
        opened=self.opening_result
        if opened is None:opened=super().result()
        settled=[r for r in self.rows if r['phase']=='check released inside-up cap'][-20:]
        parked=(len(settled)==20 and all(r['cap_opening_up_cosine']>math.cos(math.radians(5))
            and r['cap_speed_m_s']<.005 and r['cap_rest_xy_error_m']<.01 and r['cap_rest_normal_n']>.01
            and max(r['receiving_finger_contact_n'])<.01 and r['receiving_gripper_opening']>.9 for r in settled))
        return {'type':'TWO_HAND_CAP_OPEN_AND_PARK_PROBE','full_task_demo':False,'task_score':None,
            'passed_probe':bool(opened.get('passed_probe') and parked),'passed_two_hand_opening':opened.get('passed_probe',False),
            'cap_placed_inside_up':bool(parked),'pose_writes_after_reset':0,'rows':self.rows}
