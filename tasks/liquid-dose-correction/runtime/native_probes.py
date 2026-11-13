"""Short robot-only mechanism probes, never a full task demonstration.

Commands are native arm joints and gripper openings. Object poses are only
read; no grasp attachment or prescribed object motion is used. A probe can
fail and must retain its measurements rather than report a task score.
"""
import math
import numpy as np
from native_coupling import array,transform
from pipette_geometry import BULB_CENTER_M


class PipetteProbe:
    def __init__(self,base,initial_action,tool='pipette_A'):
        from yam_arena import ik,station
        from yam_arena.demo import grip_rotation
        self.ik=ik;self.base=base;self.tool=tool
        self.side='left' if tool.endswith('A') else 'right'
        self.offset=0 if self.side=='left' else 7
        self.base_position=station.BASES[self.side]
        self.action=np.array(initial_action,dtype=float).copy()
        self.q=self.action[self.offset:self.offset+6].copy()
        self.start_pos,self.start_rot=ik.tcp_pose(self.q,self.base_position)
        tool_pose=transform(array(base.scene[tool].data.root_link_pose_w)[0])
        self.initial_tip=tool_pose[:3,3].copy()
        bulb=tool_pose[:3,:3]@np.array([0,0,BULB_CENTER_M])+tool_pose[:3,3]
        down=grip_rotation(0)
        # Gap is total pad separation / the 93.9 mm native opening range.
        grasp=.016/.0939
        self.waypoints=[
            (2.,np.array([.10,bulb[1],station.TABLE_TOP+.15]),down,1.,'approach from front'),
            (2.,bulb+[0,0,.08],down,1.,'above bulb'),
            (2.,bulb,down,1.,'descend'),
            (1.5,bulb,down,grasp,'squeeze'),
            (1.,bulb,down,grasp,'hold compressed'),
            (2.,bulb+[0,0,.06],down,grasp,'lift'),
            (1.,bulb+[0,0,.06],down,grasp,'hold lifted'),
            (2.,bulb,down,grasp,'lower'),
            (1.5,bulb,down,1.,'release'),
            (2.,bulb+[0,0,.08],down,1.,'retreat'),
        ]
        self.duration=sum(w[0] for w in self.waypoints)
        self.rows=[];self.max_position_error=0.;self.max_rotation_error=0.
        arm=base.scene[self.side+'_robot']
        self.gripper_ids,_=arm.find_bodies('gripper')

    def command(self,t):
        p0,r0,g0=self.start_pos,self.start_rot,self.action[self.offset+6]
        elapsed=0.
        for duration,p1,r1,g1,label in self.waypoints:
            if t<=elapsed+duration:
                s=self.ik.smoothstep((t-elapsed)/duration)
                target=(1-s)*p0+s*p1
                rotation=self.ik.slerp_matrix(r0,r1,s)
                q,pe,re=self.ik.solve(target,self.ik.matrix_to_quat_xyzw(rotation),q_init=self.q,
                                      base_pos=self.base_position,iters=180,damping=.025,tol=5e-5)
                self.max_position_error=max(self.max_position_error,pe)
                self.max_rotation_error=max(self.max_rotation_error,re)
                if pe>.002 or re>.02:
                    raise RuntimeError(f'Probe IK infeasible during {label}: {pe:.6f} m, {re:.6f} rad')
                self.q=q
                action=self.action.copy();action[self.offset:self.offset+6]=q
                action[self.offset+6]=(1-s)*g0+s*g1
                return action,label
            elapsed+=duration;p0,r0,g0=p1,r1,g1
        action=self.action.copy();action[self.offset:self.offset+6]=self.q
        action[self.offset+6]=1.
        return action,'complete'

    def observe(self,t,label):
        asset=self.base.scene[self.tool]
        pose=array(asset.data.root_link_pose_w)[0]
        ids,names=asset.find_joints(['squeeze_left','squeeze_right'],preserve_order=True)
        q=array(asset.data.joint_pos)[0,ids]
        row={'time_s':t,'phase':label,'tip_position':pose[:3].tolist(),
             'compression':float(np.clip((q[0]-q[1])/.0116,0,1)),
             'tip_height_change_m':float(pose[2]-self.initial_tip[2])}
        hand=array(self.base.scene[self.side+'_robot'].data.body_link_pose_w)[0,self.gripper_ids[0]]
        hand_pose=transform(hand)
        row['tip_in_gripper_frame']=(hand_pose[:3,:3].T@(pose[:3]-hand_pose[:3,3])).tolist()
        self.rows.append(row)

    def result(self):
        max_lift=max((r['tip_height_change_m'] for r in self.rows),default=0.)
        max_compression=max((r['compression'] for r in self.rows),default=0.)
        held=[r for r in self.rows if r['phase']=='hold lifted']
        held_stable=len(held)>=10 and min(r['tip_height_change_m'] for r in held)>.04
        relative_drift=float(np.max(np.ptp([r['tip_in_gripper_frame'] for r in held],axis=0))) if held else None
        held_stable=held_stable and relative_drift<.005
        return {'type':'PHYSICAL_PIPETTE_PROBE','full_task_demo':False,'task_score':None,
                'passed_probe':held_stable and max_compression>.25,
                'passed_lift_and_squeeze':held_stable and max_compression>.25,
                'max_lift_m':max_lift,'max_compression':max_compression,
                'held_relative_drift_m':relative_drift,
                'max_planned_ik_position_error_m':self.max_position_error,
                'max_planned_ik_rotation_error_rad':self.max_rotation_error,
                'pose_writes_after_reset':0,'rows':self.rows}


class CapProbe:
    """Turn, regrasp and lift a cap through real thread contacts."""
    command=PipetteProbe.command

    def __init__(self,base,initial_action,tool='bottle_A_cap'):
        from yam_arena import ik,station
        from yam_arena.demo import grip_rotation
        self.ik=ik;self.base=base;self.tool=tool
        self.bottle=tool.removesuffix('_cap')
        self.side='left' if '_A_' in tool else 'right';self.offset=0 if self.side=='left' else 7
        self.base_position=station.BASES[self.side]
        self.action=np.array(initial_action,dtype=float).copy();self.q=self.action[self.offset:self.offset+6].copy()
        self.start_pos,self.start_rot=ik.tcp_pose(self.q,self.base_position)
        cap=transform(array(base.scene[tool].data.root_link_pose_w)[0])
        bottle=transform(array(base.scene[self.bottle].data.root_link_pose_w)[0])
        self.initial_bottle=bottle.copy();self.initial_relative=np.linalg.inv(bottle)@cap
        self.previous_yaw=math.atan2(self.initial_relative[1,0],self.initial_relative[0,0])
        self.turn_angle=0.
        center=cap[:3,3]+[0,0,.015]
        down=grip_rotation(0);turned=grip_rotation(math.pi/2)
        grip=.042/.0939;opened=.070/.0939
        self.waypoints=[(2.,center+[0,0,.020],down,opened,'approach cap'),
                        (2.,center,down,opened,'descend to cap'),
                        (1.,center,down,grip,'grasp cap')]
        for k in range(6):
            p=center+[0,0,(k+1)*.00075]
            self.waypoints.append((2.,p,turned,grip,f'turn {k+1}'))
            if k<5:
                self.waypoints.extend([(.5,p,turned,opened,'release for regrasp'),
                                       (1.,p,down,opened,'reset wrist'),
                                       (.5,p,down,grip,'regrasp')])
        self.waypoints.extend([(2.,p+[0,0,.015],turned,grip,'lift unthreaded cap'),
                               (1.,p+[0,0,.015],turned,grip,'hold lifted')])
        self.duration=sum(w[0] for w in self.waypoints)
        self.rows=[];self.max_position_error=0.;self.max_rotation_error=0.
        self.gripper_ids,_=base.scene[self.side+'_robot'].find_bodies('gripper')

    def observe(self,t,label):
        cap=transform(array(self.base.scene[self.tool].data.root_link_pose_w)[0])
        bottle=transform(array(self.base.scene[self.bottle].data.root_link_pose_w)[0])
        relative=np.linalg.inv(bottle)@cap
        yaw=math.atan2(relative[1,0],relative[0,0])
        self.turn_angle+=(yaw-self.previous_yaw+math.pi)%(2*math.pi)-math.pi
        self.previous_yaw=yaw
        lift=float(relative[2,3]-self.initial_relative[2,3])
        hand=transform(array(self.base.scene[self.side+'_robot'].data.body_link_pose_w)[0,self.gripper_ids[0]])
        self.rows.append({'time_s':t,'phase':label,'relative_turns':self.turn_angle/(2*math.pi),
            'cap_lift_m':lift,'screw_pitch_error_m':lift-.003*self.turn_angle/(2*math.pi),
            'cap_center_offset_m':float(np.linalg.norm(relative[:2,3])),
            'cap_relative_tilt_deg':math.degrees(math.acos(float(np.clip(relative[2,2],-1,1)))),
            'cap_speed_m_s':float(np.linalg.norm(array(self.base.scene[self.tool].data.root_link_vel_w)[0,:3])),
            'cap_supported_normal_n':float(array(self.base.scene['cap_contacts'].data.normal_force_matrix_w)[...,2].sum()),
            'bottle_drift_m':float(np.linalg.norm(bottle[:3,3]-self.initial_bottle[:3,3])),
            'bottle_tilt_deg':math.degrees(math.acos(float(np.clip(bottle[2,2],-1,1)))),
            'bottle_yaw_deg':math.degrees(math.atan2(bottle[1,0],bottle[0,0])),
            'cap_yaw_deg':math.degrees(math.atan2(cap[1,0],cap[0,0])),
            'gripper_yaw_deg':math.degrees(math.atan2(hand[1,0],hand[0,0])),
            'cap_in_gripper_frame':(hand[:3,:3].T@(cap[:3,3]-hand[:3,3])).tolist()})

    def result(self):
        held=[r for r in self.rows if r['phase']=='hold lifted']
        drift=float(np.max(np.ptp([r['cap_in_gripper_frame'] for r in held],axis=0))) if held else None
        threaded=[r for r in self.rows if .1<r['relative_turns']<1.25 and r['phase'].startswith('turn')]
        pitch_error=max((abs(r['screw_pitch_error_m']) for r in threaded),default=None)
        passed=(len(held)>=10 and drift<.005 and min(r['cap_lift_m'] for r in held)>.015
                and min(r['relative_turns'] for r in held)>1.25 and pitch_error is not None and pitch_error<.0015
                and max(r['bottle_drift_m'] for r in self.rows)<.01 and max(r['bottle_tilt_deg'] for r in self.rows)<5.)
        return {'type':'PHYSICAL_CAP_PROBE','full_task_demo':False,'task_score':None,'passed_probe':passed,
                'final_relative_turns':self.rows[-1]['relative_turns'] if self.rows else 0.,
                'max_pitch_error_while_threaded_m':pitch_error,'held_relative_drift_m':drift,
                'max_planned_ik_position_error_m':self.max_position_error,
                'max_planned_ik_rotation_error_rad':self.max_rotation_error,
                'pose_writes_after_reset':0,'rows':self.rows}


class CapCycleProbe(CapProbe):
    """Open, lift, replace and close a cap, then release it to settle."""
    def __init__(self,base,initial_action,tool='bottle_A_cap'):
        super().__init__(base,initial_action,tool)
        from yam_arena.demo import grip_rotation
        center=np.array(self.waypoints[1][1]);down=grip_rotation(0);turned=grip_rotation(math.pi/2)
        grip=.042/.0939;opened=.070/.0939
        self.waypoints.append((2.,center+[0,0,.0045],turned,grip,'replace cap'))
        for k in range(6):
            point=center+[0,0,(5-k)*.00075]
            self.waypoints.append((2.,point,down,grip,f'close turn {k+1}'))
            if k<5:
                self.waypoints.extend([(.5,point,down,opened,'release closing regrasp'),
                    (1.,point,turned,opened,'reset closing wrist'),(.5,point,turned,grip,'closing regrasp')])
        self.waypoints.extend([(.5,center,down,opened,'release closed cap'),
            (2.,center+[0,0,.020],down,opened,'retreat from closed cap'),
            (3.,center+[0,0,.020],down,opened,'hold closed')])
        self.duration=sum(w[0] for w in self.waypoints)

    def result(self):
        result=super().result();opened=result['passed_probe']
        settled=[r for r in self.rows if r['phase']=='hold closed'][-20:]
        closed=(len(settled)==20 and all(abs(r['cap_lift_m'])<.0005 and r['cap_center_offset_m']<.001
            and r['cap_relative_tilt_deg']<5. and r['cap_speed_m_s']<.005
            and r['cap_supported_normal_n']>.01 for r in settled))
        result.update(type='PHYSICAL_CAP_OPEN_CLOSE_PROBE',passed_probe=bool(opened and closed),
                      passed_opening=bool(opened),passed_released_closed_state=bool(closed))
        return result


def wet_waypoints(bulb,beaker,waste,down,table_top):
    """Diagnostic with the dedicated pipette initially pointing toward +X."""
    side=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    # Keep contact while the submerged tip produces suction. Opening to
    # 22 mm released the bulb and let the tool slide; 19 mm retains preload
    # while allowing approximately 1 mL of bulb-volume recovery.
    grasp=.016/.0939;relaxed=.019/.0939;squeezed=.0155/.0939
    high=np.array([.25,bulb[1],table_top+.34])
    turn_at=np.array([.25,bulb[1],table_top+.21])
    over=np.array([beaker[0],beaker[1],table_top+.34])
    dip=beaker+np.array([0.,0.,.008+BULB_CENTER_M])
    out=np.array([waste[0],waste[1],table_top+.34])
    dispense=waste+np.array([0.,0.,.07+BULB_CENTER_M+.02])
    return [
        (2.,np.array([.10,bulb[1],table_top+.15]),down,1.,'approach from front'),
        (2.,bulb+[0,0,.08],down,1.,'above bulb'),
        (2.,bulb,down,1.,'descend'),
        (1.5,bulb,down,grasp,'squeeze'),
        (1.,bulb,down,grasp,'hold compressed'),
        (2.,bulb+[0,0,.20],down,grasp,'lift clear'),
        (2.,turn_at,down,grasp,'move clear of vessels'),
        (2.5,turn_at,side,grasp,'orient tip downward'),
        (2.,high,side,grasp,'raise vertical pipette'),
        (2.,over,side,grasp,'over beaker'),
        (3.,dip,side,grasp,'immerse tip'),
        (1.5,dip,side,grasp,'settle immersed'),
        (2.,dip,side,relaxed,'aspirate'),
        (2.,dip,side,relaxed,'hold aspirated'),
        (3.,over,side,relaxed,'withdraw tip'),
        (3.,out,side,relaxed,'over waste'),
        (2.,dispense,side,relaxed,'lower over waste'),
        (2.,dispense,side,squeezed,'dispense'),
        (2.,dispense,side,squeezed,'hold dispensed'),
    ]


class PipetteWetProbe(PipetteProbe):
    def __init__(self,base,initial_action,tool,coupling,source='beaker'):
        super().__init__(base,initial_action,tool)
        if tool not in ('pipette_A','pipette_B'):raise ValueError('Unknown dedicated pipette')
        from yam_arena.demo import grip_rotation
        from yam_arena import station
        self.coupling=coupling
        if source not in ('beaker','cylinder_'+tool[-1]):raise ValueError('Unsupported or non-dedicated diagnostic source')
        self.source=source
        poses=coupling.poses()
        bulb=poses[tool][:3,:3]@np.array([0,0,BULB_CENTER_M])+poses[tool][:3,3]
        self.waypoints=wet_waypoints(bulb,poses[source][:3,3],poses['waste'][:3,3],grip_rotation(0),station.TABLE_TOP)
        if source.startswith('cylinder_'):
            dip=poses[source][:3,3].copy()
            dip[2]=coupling.fluid.surface(source,poses[source])-.002+BULB_CENTER_M
            updated=[]
            for d,p,r,g,label in self.waypoints:
                p=p.copy()
                if label in ('immerse tip','settle immersed','aspirate','hold aspirated'):p=dip.copy()
                # Lift while rotating: a high horizontal-tool pose exceeds
                # wrist reach, while a low rotation can strike the bench.
                if label=='orient tip downward':p[2]=station.TABLE_TOP+.31
                if label in ('raise vertical pipette','over beaker','withdraw tip','over waste'):
                    p[2]=station.TABLE_TOP+.43
                updated.append((d,p,r,g,label.replace('over beaker','over cylinder '+tool[-1])))
            self.waypoints=updated
        self.duration=sum(w[0] for w in self.waypoints)
        self.nominal_waypoints=list(self.waypoints);self.dip_adjust=np.zeros(3)
        self.held_tip_in_hand=None

    def command(self,t):
        elapsed=0.;active=False
        for duration,_,_,_,label in self.nominal_waypoints:
            if elapsed<t<=elapsed+duration:
                active=label in ('settle immersed','aspirate','hold aspirated') or (label=='immerse tip' and t>elapsed+duration-.5)
                break
            elapsed+=duration
        if active:
            poses=self.coupling.poses();target=poses[self.source][:3,3].copy()
            target[2]=self.coupling.fluid.surface(self.source,poses[self.source])-.002
            error=target-poses[self.tool][:3,3]
            self.dip_adjust=np.clip(self.dip_adjust+np.clip(error*.3,-.0005,.0005),-.015,.015)
        self.waypoints=[(d,p+self.dip_adjust if label in ('immerse tip','settle immersed','aspirate','hold aspirated') else p,r,g,label)
                        for d,p,r,g,label in self.nominal_waypoints]
        return super().command(t)

    def observe(self,t,label):
        super().observe(t,label)
        if label=='hold compressed':
            self.held_tip_in_hand=np.array(self.rows[-1]['tip_in_gripper_frame'])
        e=self.coupling.ledger
        self.rows[-1].update(pipette_ml=e.pipettes[self.tool].liquid.volume,
            source=self.source,source_ml=e.vessels[self.source].liquid.volume,
            beaker_ml=e.vessels['beaker'].liquid.volume,waste_ml=e.vessels['waste'].liquid.volume,
            spill_ml=e.spill.volume,pressure_pa=self.coupling.last['pipettes'][self.tool]['air_pressure_pa'])
        self.rows[-1].update(dip_adjustment_m=self.dip_adjust.tolist(),
            balance=self.coupling.balance.last if self.coupling.balance else None,
            display_text=self.coupling.balance.screen.previous if self.coupling.balance else None)
        if self.held_tip_in_hand is not None and label in ('lift clear','move clear of vessels','orient tip downward','raise vertical pipette','over beaker','over cylinder A','over cylinder B','immerse tip','settle immersed','aspirate','hold aspirated','withdraw tip','over waste','lower over waste','dispense','hold dispensed'):
            drift=float(np.linalg.norm(np.array(self.rows[-1]['tip_in_gripper_frame'])-self.held_tip_in_hand))
            self.rows[-1]['grasp_slip_m']=drift
            if drift>.005:raise RuntimeError(f'Pipette slipped {drift:.6f} m in gripper during {label}')

    def result(self):
        maximum=max((r.get('pipette_ml',0.)for r in self.rows),default=0.)
        end=self.rows[-1] if self.rows else {}
        self.coupling.ledger.assert_conservation()
        return {'type':'PHYSICAL_WET_PIPETTE_PROBE','full_task_demo':False,'task_score':None,
                'setup':((('10.8' if self.tool.endswith('A') else '30.8')+' mL '+self.tool[-1]+' initially placed in '+self.source) if self.source.startswith('cylinder_') else '40 mL '+self.tool[-1]+' initially placed in beaker')+' for an isolated aspiration/dispense test',
                'passed_probe':maximum>.5 and end.get('waste_ml',0.)>maximum*.9 and end.get('spill_ml',1.)<.01,
                'max_aspirated_ml':maximum,'final_waste_ml':end.get('waste_ml'),
                'final_pipette_ml':end.get('pipette_ml'),'final_spill_ml':end.get('spill_ml'),
                'max_planned_ik_position_error_m':self.max_position_error,
                'max_planned_ik_rotation_error_rad':self.max_rotation_error,
                'pose_writes_after_reset':0,'rows':self.rows}
