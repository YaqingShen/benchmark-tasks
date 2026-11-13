"""Native pipette mass-correction diagnostic; not a submission controller.

Motion alignment uses measured tool/vessel poses. Dose control uses only the
force-based balance reading. No object state or liquid quantity is commanded.
"""
import numpy as np
from native_probes import PipetteProbe,wet_waypoints
from native_coupling import transform,array
from pipette_geometry import BULB_CENTER_M
from geometry_fluid import INTERIORS
from disposal_geometry import pipette_bounds_in_bin,fully_contained


def cycle_phase(label):
    if '::' not in label:return 0,label
    cycle,phase=label.split('::',1)
    return int(cycle),phase


def released_disposal_window(rows):
    """Observable rest: containment, hand clearance and pose stability for 1 s.

    Instantaneous solver velocities can oscillate at contact while the visible
    tool stays still. Keep them in the report; judge rest from its actual pose
    history, including rotation so a pivoting tool cannot count as settled.
    """
    if len(rows)<20:return False
    rows=rows[-20:]
    if rows[-1]['time_s']-rows[0]['time_s']<.94:return False
    if not all(r['pipette_contained_and_released'] for r in rows):return False
    tips=np.array([r['tip_position'] for r in rows])
    axes=np.array([r['tool_axis_world'] for r in rows])
    return bool(np.linalg.norm(np.ptp(tips,axis=0))<=.002 and
                np.min(axes@axes.T)>=np.cos(np.deg2rad(3.)))


class MassPulseControl:
    """Small gripper-gap changes separated by instrument settling periods."""
    def __init__(self,mode,gap_m):
        if mode not in ('add','remove'):raise ValueError(mode)
        self.mode=mode;self.gap=gap_m;self.last_pulse=-2.;self.stopped=False;self.pulses=0
        self.exhausted=False

    def update(self,t,reading):
        value=reading.get('net_g')
        if value is None:raise RuntimeError('Balance has no original tare')
        done=value>=9.9998 if self.mode=='add' else value<=10.0002
        if done and reading.get('stable'):
            self.stopped=True
        if not self.stopped and not self.exhausted and reading.get('stable') and t-self.last_pulse>=1.5:
            step=float(np.clip(abs(value-10.)*.0012,.000001,.00010))
            self.gap+=-step if self.mode=='add' else step
            if not .0155<=self.gap<=.019:
                self.gap=float(np.clip(self.gap,.0155,.019));self.exhausted=True
            self.last_pulse=t;self.pulses+=1
        return self.gap/.0939


class PipetteCorrectionProbe(PipetteProbe):
    def __init__(self,base,initial_action,coupling,mode='remove'):
        super().__init__(base,initial_action,'pipette_A')
        from yam_arena import station
        from yam_arena.demo import grip_rotation
        self.coupling=coupling;self.mode=mode
        self.source='beaker' if mode=='remove' else 'bottle_A'
        poses=coupling.poses();bulb=poses[self.tool][:3,:3]@np.array([0,0,BULB_CENTER_M])+poses[self.tool][:3,3]
        self.waypoints=wet_waypoints(bulb,poses[self.source][:3,3],poses['waste'][:3,3],grip_rotation(0),station.TABLE_TOP)
        surface=coupling.fluid.surface(self.source,poses[self.source])
        floor=poses[self.source][2,3]+INTERIORS[self.source].z[0]
        dip=poses[self.source][:3,3].copy();dip[2]=max(floor+.0005,surface-.0008)+BULB_CENTER_M
        side=self.waypoints[7][2]
        updated=[]
        for d,p,r,g,label in self.waypoints:
            p=p.copy()
            if label in ('immerse tip','settle immersed','aspirate','hold aspirated'):p=dip.copy()
            if label=='orient tip downward':p[2]=station.TABLE_TOP+.31
            # The held tip is about195mm below the TCP. Keep its entire
            # shaft above the240mm bin rim during inter-vessel transfers.
            if label in ('raise vertical pipette','over beaker','withdraw tip','over waste'):p[2]=station.TABLE_TOP+.47
            if label=='over waste':p[2]=station.TABLE_TOP+.32
            if mode=='remove' and label=='aspirate':
                updated.append((120.,p,r,.016/.0939,'fine remove'));continue
            if mode=='remove' and label=='hold aspirated':continue
            if mode=='add' and label=='over waste':
                bp=poses['beaker'][:3,3]
                updated.extend([(3.,np.r_[bp[:2],station.TABLE_TOP+.47],side,.019/.0939,'over receiving beaker'),
                    (2.,bp+[0,0,INTERIORS['beaker'].mouth_z+.025+BULB_CENTER_M],side,.019/.0939,'lower over receiving beaker'),
                    (55.,bp+[0,0,INTERIORS['beaker'].mouth_z+.025+BULB_CENTER_M],side,.019/.0939,'fine add'),
                    (2.,np.r_[bp[:2],station.TABLE_TOP+.47],side,.019/.0939,'lift from receiving beaker')])
            updated.append((d,p,r,g,label))
        # A physically held bulb stroke need not cover the starting error.
        # Repeat after emptying into waste, in either correction direction.
        # Future cycles are skipped as soon as a settled target is reached.
        start=next(i for i,w in enumerate(updated)if w[4]=='over beaker')
        cycle_waypoints=list(updated[start:])
        for cycle in (1,2):
            updated += [(d,p.copy(),r.copy(),g,f'{cycle}::{label}')
                        for d,p,r,g,label in cycle_waypoints]
        # Dispose vertically in a nearby open bin; allow the released tool to
        # settle, then inspect actual containment and hand separation.
        trash=poses['trash_bin'][:3,3]
        above=trash+[0,0,.253+BULB_CENTER_M]
        # Release with the bulb above the rim so the open fingers cannot hit
        # the bin. A slight lean avoids balancing vertically on the narrow tip.
        angle=np.deg2rad(5.);c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[1.,0.,0.],[0.,c,s],[0.,-s,c]])
        leaned=rotation@side;axis=rotation@np.array([0.,0.,1.])
        leaned_above=trash+[0,-.005,.253]+BULB_CENTER_M*axis
        release=trash+[0,-.005,.10]+BULB_CENTER_M*axis
        updated += [(2.,poses['waste'][:3,3]+[0,0,.125+BULB_CENTER_M],side,.0155/.0939,'raise from waste'),
                    (3.,np.array([.20,.05,station.TABLE_TOP+.45]),side,.0155/.0939,'raise before pipette bin'),
                    (3.,above,side,.0155/.0939,'over pipette bin'),
                    (2.,leaned_above,leaned,.0155/.0939,'lean above pipette bin'),
                    (3.,release,leaned,.0155/.0939,'lower into pipette bin'),
                    (2.,release,leaned,.0155/.0939,'hold in pipette bin'),
                    (1.,release,leaned,1.,'release used pipette'),
                    (2.,leaned_above,leaned,1.,'withdraw from pipette bin'),
                    (20.,leaned_above,leaned,1.,'settle final reading')]
        self.waypoints=updated;self.nominal=list(updated);self.duration=sum(w[0]for w in updated)
        self.controller=MassPulseControl('remove' if mode=='remove'else 'add',.016 if mode=='remove'else .019)
        self.controllers={0:self.controller};self.active_cycle=0
        self.dip_adjust=np.zeros(3);self.held_tip_in_hand=None;self.fine_started=False;self.fine_finished=False
        self.remove_state='lift';self.remove_since=0.;self.remove_offset=0.;self.remove_hold_gap=.016

    def removal_control(self,t):
        """Aspirate briefly, lift clear, then read the settled balance."""
        elapsed=t-self.remove_since
        if self.remove_state=='lift':
            self.remove_offset=.020*self.ik.smoothstep(min(elapsed/.6,1.))
            if elapsed>=.6:self.remove_state='read';self.remove_since=t
        elif self.remove_state=='read':
            self.remove_offset=.020
            reading=self.coupling.balance.last
            if elapsed>=1.5 and reading.get('stable'):
                self.controller.update(t,reading)
                if not (self.controller.stopped or self.controller.exhausted):
                    self.remove_state='lower';self.remove_since=t
        elif self.remove_state=='lower':
            self.remove_offset=.020*(1-self.ik.smoothstep(min(elapsed/.8,1.)))
            if elapsed>=.8:self.remove_state='aspirate';self.remove_since=t
        elif self.remove_state=='aspirate':
            self.remove_offset=0.;self.remove_hold_gap=self.controller.gap
            if elapsed>=.8:self.remove_state='lift';self.remove_since=t
        return self.remove_hold_gap/.0939

    def command(self,t):
        elapsed=0.;label='';local=0.
        for d,_,_,_,name in self.nominal:
            if elapsed<t<=elapsed+d+1e-8:label=name;local=t-elapsed;break
            elapsed+=d
        cycle,plain=cycle_phase(label)
        if plain.startswith('fine ') and cycle!=self.active_cycle:
            self.controller=MassPulseControl(self.mode,.019 if self.mode=='add'else .016)
            self.controllers[cycle]=self.controller;self.active_cycle=cycle
            self.remove_state='lift';self.remove_since=0.;self.remove_offset=0.;self.remove_hold_gap=.016
        immersed=plain in ('settle immersed','aspirate','hold aspirated') or(plain=='fine remove'and self.remove_state=='aspirate') or(plain=='immerse tip'and local>2.5)
        if immersed:
            poses=self.coupling.poses();target=poses[self.source][:3,3].copy()
            surface=self.coupling.fluid.surface(self.source,poses[self.source])
            floor=poses[self.source][2,3]+INTERIORS[self.source].z[0]
            target[2]=max(floor+.0007,surface-.00055)
            error=target-poses[self.tool][:3,3]
            self.dip_adjust=np.clip(self.dip_adjust+np.clip(error*.3,-.0003,.0003),-.015,.015)
        if plain.startswith('fine '):
            self.fine_started=True
            grip=self.removal_control(local) if self.mode=='remove'else self.controller.update(local,self.coupling.balance.last)
            if self.controller.stopped or self.controller.exhausted:
                self.nominal=[(local if name==label else d,p,r,g,name)for d,p,r,g,name in self.nominal
                              if not(self.controller.stopped and cycle_phase(name)[0]>cycle)]
                self.duration=sum(w[0]for w in self.nominal)
        else:grip=self.controller.gap/.0939
        self.waypoints=[];after=False
        for d,p,r,g,name in self.nominal:
            phase_cycle,phase=cycle_phase(name)
            controller=self.controllers.get(phase_cycle)
            phase_grip=(controller.gap if controller else (.019 if self.mode=='add'else .016))/.0939
            if phase.startswith('fine '):after=True;g=grip if name==label else phase_grip
            elif after and phase in ('withdraw tip','over waste','lower over waste','lift from receiving beaker'):g=phase_grip
            if phase=='hold dispensed':after=False
            if phase in ('immerse tip','settle immersed','aspirate','hold aspirated','fine remove'):p=p+self.dip_adjust
            if phase=='fine remove':p=p+[0,0,self.remove_offset]
            self.waypoints.append((d,p,r,g,name))
        if plain=='fine remove':
            _,p,r,_,_=next(w for w in self.waypoints if w[4]==label)
            q,pe,re=self.ik.solve(p,self.ik.matrix_to_quat_xyzw(r),q_init=self.q,
                base_pos=self.base_position,iters=180,damping=.025,tol=5e-5)
            if pe>.002 or re>.02:raise RuntimeError('Fine removal lift/read waypoint is unreachable')
            self.q=q;action=self.action.copy();action[self.offset:self.offset+6]=q;action[self.offset+6]=grip
            return action,label
        action,phase=super().command(t)
        if cycle_phase(phase)[1].startswith('fine '):action[self.offset+6]=grip
        return action,phase

    def observe(self,t,label):
        super().observe(t,label)
        e=self.coupling.ledger;row=self.rows[-1]
        if label=='hold compressed':self.held_tip_in_hand=np.array(row['tip_in_gripper_frame'])
        if self.held_tip_in_hand is not None and label not in ('release used pipette','withdraw from pipette bin','settle final reading'):
            drift=float(np.linalg.norm(np.array(row['tip_in_gripper_frame'])-self.held_tip_in_hand))
            row['grasp_slip_m']=drift
            if drift>.005:raise RuntimeError(f'Pipette slipped {drift:.6f} m during {label}')
        row.update(beaker_g=e.vessels['beaker'].liquid.mass(e.cfg['density_g_ml']),
            pipette_ml=e.pipettes[self.tool].liquid.volume,waste_ml=e.vessels['waste'].liquid.volume,
            spill_g=e.spill.mass(e.cfg['density_g_ml']),balance=dict(self.coupling.balance.last),
            pulse_count=self.controller.pulses,gap_m=self.controller.gap,controller_stopped=self.controller.stopped,
            tip_vessel=self.coupling.last['pipettes'][self.tool]['tip_vessel'],dip_adjustment_m=self.dip_adjust.tolist())
        poses=self.coupling.poses();local=np.linalg.inv(poses['trash_bin'])@np.r_[poses[self.tool][:3,3],1.]
        row.update(tool_axis_world=poses[self.tool][:3,2].tolist(),beaker_pose=poses['beaker'].tolist(),
                   removal_phase=self.remove_state,removal_lift_m=self.remove_offset)
        velocity=array(self.base.scene[self.tool].data.root_link_vel_w)[0,:3]
        hand_ids,_=self.base.scene['left_robot'].find_bodies(['tip_left','tip_right'],preserve_order=True)
        fingers=array(self.base.scene['left_robot'].data.body_link_pose_w)[0,hand_ids,:3]
        tool_points=np.array([poses[self.tool][:3,3]+poses[self.tool][:3,2]*z for z in np.linspace(0,.225,40)])
        hand_clearance=float(np.linalg.norm(fingers[:,None,:]-tool_points[None,:,:],axis=-1).min())
        bounds=self.coupling.manifest['assets']['trash_bin']['interior_bounds_m']
        ids,_=self.base.scene[self.tool].find_bodies(['bulb_left','bulb_right'],preserve_order=True)
        bulb_poses=[transform(p)for p in array(self.base.scene[self.tool].data.body_link_pose_w)[0,ids]]
        lower,upper=pipette_bounds_in_bin(poses[self.tool],bulb_poses,poses['trash_bin'])
        inside=fully_contained(lower,upper,bounds)
        from yam_arena import constants
        arm=self.base.scene['left_robot']
        finger_ids,_=arm.find_joints(list(constants.FINGER_JOINTS),preserve_order=True)
        gap=float(abs(array(arm.data.joint_pos)[0,finger_ids].mean())*2)
        contacts=[]
        for name in ('tip_left','tip_right'):
            try:sensor=self.base.scene['pipette_'+name+'_contacts']
            except KeyError:sensor=self.base.scene['left_'+name+'_tool_contact']
            contacts.append(float(np.linalg.norm(array(sensor.data.normal_force_matrix_w),axis=-1).sum()))
        contact=max(contacts)
        released=gap>.05 and contact<.005
        row.update(tip_in_bin=local[:3].tolist(),tool_speed_m_s=float(np.linalg.norm(velocity)),
                   hand_clearance_m=hand_clearance,measured_gripper_gap_m=gap,finger_tool_contact_n=contact,
                   tool_bin_bounds_min_m=lower.tolist(),tool_bin_bounds_max_m=upper.tolist(),
                   pipette_contained_and_released=bool(inside and released),
                   pipette_released_in_bin=bool(inside and released and np.linalg.norm(velocity)<.005))

    def result(self):
        end=self.rows[-1]if self.rows else{}
        complete=end.get('time_s',0)>=self.duration-.1
        mass=end.get('beaker_g',0.)
        mass_ok=bool(complete and abs(mass-10.)<=.05 and end.get('balance',{}).get('stable') and end.get('spill_g',1)<.01)
        settled=[r for r in self.rows if r['phase']=='settle final reading'][-20:]
        disposed=released_disposal_window(settled)
        return {'type':'PIPETTE_MASS_CORRECTION_PROBE','full_task_demo':False,'task_score':None,
            'mode':self.mode,'passed_mass_correction':mass_ok,
            'target_display_reached':self.controller.stopped,
            'disposal_rest_definition':'1 s released and contained, position span <=2 mm and axis change <=3 degrees',
            'passed_instantaneous_disposal_velocity':bool(len(settled)==20 and all(r['pipette_released_in_bin']for r in settled)),
            'passed_probe':bool(mass_ok and disposed),'passed_released_disposal':bool(disposed),
            'initial_mass_g':self.coupling.ledger.initial_beaker_g,'final_mass_g':mass,
            'balance':end.get('balance'),'pose_writes_after_reset':0,'rows':self.rows}


class PipetteDisposalProbe(PipetteCorrectionProbe):
    """Isolate physical pickup and released bin placement; no dose claim."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        first=next(i for i,w in enumerate(self.nominal)if w[4]=='raise vertical pipette')
        last=next(i for i,w in enumerate(self.nominal)if w[4]=='raise before pipette bin')
        self.nominal=self.nominal[:first+1]+self.nominal[last:]
        self.waypoints=list(self.nominal);self.duration=sum(w[0]for w in self.nominal)

    def result(self):
        result=super().result()
        result.update(type='PIPETTE_RELEASED_DISPOSAL_PROBE',passed_mass_correction=False,
                      passed_probe=result['passed_released_disposal'])
        return result
