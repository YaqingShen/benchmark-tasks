"""Unlock a zipperless frame suitcase with the known combination 4-7-2."""
import math
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import labkit
import mechanism as m
from scoring import LockScore
from yam_arena.task_file import TABLE_TOP
from yam_arena.demo import Waypoint

SCENE='station'
TABLE=TABLE_TOP[SCENE]
TOP=TABLE+m.DECK_HEIGHT
XY=np.array([.29, .006])
labkit.CONTACT_REPORT.add('lock')
OBJECTS=[m.mechanism(labkit,'lock',(*XY,TABLE))]
class Trial:
    def reset(self,*args):
        self.joints=None
        self.value=0.
        self.score_state=LockScore()
        self.angles=[n*m.STEP for n in m.START]
        self.button_mm=0.
        self.opened=False
        self.step=0
    def tick(self,env):
        import torch
        lock=env.unwrapped.scene['lock']
        if self.joints is None:
            names=list(lock.joint_names)
            from isaaclab_physx.physics import PhysxManager
            robot=env.unwrapped.scene['left_robot']
            poses=robot.root_physx_view.link_paths[0]
            tips=[poses[list(robot.body_names).index(key)] for key in ('tip_left','tip_right')]
            root=env.unwrapped.scene.env_prim_paths[0]+'/lock/'
            self.contact_views=[PhysxManager.get_physics_sim_view().create_rigid_contact_view([root+link],filter_patterns=[tips],max_contact_data_count=16) for link in ('wheel0','wheel1','wheel2','button')]
            self.joints=[names.index(f'wheel{i}_joint') for i in range(3)]
            self.button=names.index('button_joint')
            self.lid=names.index('lid_joint')
            self.lid_contact=PhysxManager.get_physics_sim_view().create_rigid_contact_view([root+'lid'],filter_patterns=[[p for a in ('left_robot','right_robot') for p in env.unwrapped.scene[a].root_physx_view.link_paths[0]]],max_contact_data_count=64)
        state=lock.data.joint_pos[0]
        angles=state[self.joints].detach()
        # Each spring follows the nearest of ten identical detents.
        detents=torch.round(angles/m.STEP)*m.STEP
        lock.set_joint_position_target(detents[None,:],joint_ids=self.joints)
        lock.set_joint_position_target(torch.full((1,1),m.LID_REST,device=lock.device),joint_ids=[self.lid])
        values=[q+n*m.STEP for q,n in zip(angles.cpu().tolist(),m.START)]
        self.angles=values
        button_mm=1000*state[self.button].item()
        self.button_mm=button_mm
        gap=m.gap_mm(float(state[self.lid]))
        forces=self.lid_contact.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape(-1,3)
        contact=float(np.linalg.norm(forces,axis=1).sum())
        self.opened=self.score_state.released and gap>=20
        self.value=self.score_state.update(values,button_mm,gap,contact)
        if os.environ.get('LOCK_TRACE'):
            import json
            with open(os.environ['LOCK_TRACE'],'a') as stream:
                print(json.dumps(dict(step=self.step, angles=values, digits=[m.digit(v) for v in values],
                    button_mm=button_mm, lid_gap_mm=gap,lid_contact_N=contact, valid=not self.score_state.invalid, score=self.value)),file=stream)
        if self.step%10==0:
            print('LOCK',self.step,[m.digit(v) for v in values],round(float(state[self.button])*1000,2),
                  round(gap,2),self.value,flush=True)
            from yam_arena.runtime import link_poses
            frame=link_poses(env.unwrapped.scene['left_robot'])['gripper']
            tcp=frame[:3,3]+frame[:3,:3]@[0,0,.134]-np.r_[XY,TOP]
            print('DIAL_CONTACT',self.step,np.round(tcp*1000,1).tolist(),[np.round(np.linalg.norm(v.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape(-1,3),axis=1),1).tolist() for v in self.contact_views],flush=True)
        if os.environ.get('LOCK_TRACE') and self.step in (0,40,120,220,310):
            from PIL import Image
            folder=Path(os.environ['LOCK_TRACE']).parent
            for cam in ('top_cam','left_cam','right_cam'):
                raw=env.unwrapped.scene[cam].data.output['rgb']
                raw=raw.torch if hasattr(raw,'torch') else raw
                Image.fromarray(raw[0,...,:3].cpu().numpy()).save(folder/f'{cam}-{self.step:03}.png')
        self.step+=1
trial=Trial()
trial.reset()
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)
def score(objects):return trial.value
def success(objects):return trial.value==1.
def demo():
    from yam_arena import demo as dm,ik,station
    from yam_arena.constants import READY
    arm='left'
    home,home_rot=ik.tcp_pose(np.array(READY),station.BASES[arm])
    raw=dm.ArmPlan.target
    if not getattr(raw,'luggage_dials',False):
        memory={'phase':'select','start':0,'tries':[0,0,0],'index':0,'sign':1,'xyz':(.0,0.,.065)}
        def phase(name):
            memory['phase']=name
            memory['start']=trial.step
        def follow(plan,t):
            pos,rot,opening,label=raw(plan,t)
            if label=='stand clear':return plan.start_pos,plan.start_rot,plan.finger_start,label
            if label!='operate compact lock':return pos,rot,opening,label
            if getattr(trial,'opened',False):
                return np.r_[XY+[m.SWITCH_X,m.SWITCH_Y+.015],TOP+.072],dm.grip_rotation(-math.pi/2),-.014,label
            ph=memory['phase'];age=trial.step-memory['start']
            if ph=='select':
                unmatched=[i for i,v in enumerate(trial.angles) if not m.indexed(v,m.CODE[i])]
                if not unmatched:
                    phase('button hover')
                else:
                    i=unmatched[0]
                    memory['index']=i
                    memory['sign']=1 if math.remainder(m.CODE[i]*m.STEP-trial.angles[i],math.tau)>0 else -1
                    memory['tries'][i]+=1
                    phase('hover')
            i=memory['index'];sign=memory['sign'];x=m.X[i]
            # A fingertip rolls the near flank, then departs radially so it does
            # not drag the wheel back across its detent while lifting.
            y0=-sign*(.0035+.001*(memory['tries'][i]%3))
            ph=memory['phase'];age=trial.step-memory['start']
            aligned=m.indexed(trial.angles[i],m.CODE[i])
            if ph=='hover':
                local=(x,y0,.058)
                if age>=15:phase('touch')
            elif ph=='touch':
                u=min(1.,age/11.)
                local=(x,y0,.058-.0205*u)
                if aligned or age>=14:phase('away' if aligned else 'roll')
            elif ph=='roll':
                u=min(1.,age/14.)
                local=(x,y0-sign*.008*u,.0385-.002*u)
                if aligned or age>=16:phase('away')
            elif ph=='away':
                local=(x,y0-sign*.009,.057)
                if age>=11:phase('select')
            elif ph=='button hover':
                local=(m.SWITCH_X,m.SWITCH_Y-.019,.055)
                if age>=15:phase('button down')
            elif ph=='button down':
                local=(m.SWITCH_X,m.SWITCH_Y-.020,.032)
                if age>=12:phase('button push')
            elif ph=='button push':
                local=(m.SWITCH_X,m.SWITCH_Y-.020+.023*min(age/18.,1.),.032)
                if age>=27:phase('button clear')
            else:
                local=(m.SWITCH_X,m.SWITCH_Y-.009,.061)
                if age>=14:phase('select')
            return np.r_[XY+np.array(local[:2]),TOP+local[2]],dm.grip_rotation(-math.pi/2),-.003,label
        follow.luggage_dials=True
        dm.ArmPlan.target=follow
    plans={}
    for a in ('left','right'):
        idle_pos,_=ik.tcp_pose(np.array(READY),station.BASES[a])
        if a!=arm:plans[a]=[Waypoint(35.,tuple(idle_pos),-.032,'stand clear')]
        else:plans[a]=[Waypoint(35.,(XY[0],XY[1],TOP+.06),-.003,'operate compact lock',-math.pi/2)]
    return plans
