"""Normally closed edge pick: passive tool, dynamic wafer, contacting slider fingers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

SPEC=dict(version='edge_pick_fixture_v1',wafer_diameter=.100,wafer_thickness=.001,
    wafer_flat=.0325,wafer_mass=.020,pillar_radius=.0005,pillar_height=.020,
    heater_height=.020,pedestal_height=.025,pillar_triangle_side=.05543,
    upper_envelope=.003,lower_envelope=.005,pad_width=.0325,land_depth=.001,
    upper_land_inset=.00075,lower_land_inset=.00425,
    jaw_stiffness=.10,jaw_rest=1.0,jaw_damping=.003,
    paddle_stiffness=10.,paddle_rest=.160,paddle_damping=1.,deadband=.0015,
    finger_force_cap=5.,finger_kp=5000.,finger_kv=100.,carriage_force_cap=20.,friction=.5,
    dt=.0005,control_dt=.05,lift=.020,
    nominal_tool_origin=[-.081285,.150,.063],grip_x=-.050,edge_z_offset=-.0005,edge_x_offset=-.0002,hold_half_gap=.0185,open_half_gap=.0155)


def _fmt(v):
    return ' '.join(f'{x:.12g}'for x in v)


def build_model_xml(config=None):
    s=copy.deepcopy(SPEC)
    if config:s.update(config)
    radius=s['wafer_diameter']/2;flat=-math.sqrt(radius**2-(s['wafer_flat']/2)**2)
    tool=np.array([flat-.034,.15,.063]);grip=tool+[s['grip_x'],0.,.006]
    bottom=s['pedestal_height']+s['heater_height']+s['pillar_height'];wafer_z=bottom+s['wafer_thickness']/2
    root=ET.Element('mujoco',model='normally_closed_wafer_edge_pick')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    option=ET.SubElement(root,'option',timestep=str(s['dt']),integrator='implicitfast',solver='Newton',iterations='100',tolerance='1e-10',cone='elliptic',impratio='10')
    ET.SubElement(option,'flag',multiccd='enable',filterparent='disable')
    ET.SubElement(root,'size',njmax='4000',nconmax='500')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='960',offheight='540')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',condim='3',friction=f"{s['friction']} .002 .0001",solref='.003 1',solimp='.95 .99 .0001',margin='0.00002')
    asset=ET.SubElement(root,'asset')
    a=math.acos(flat/radius);angles=np.linspace(-a,a,64)
    vertices=[[radius*math.cos(v),radius*math.sin(v),z]for z in(-s['wafer_thickness']/2,s['wafer_thickness']/2)for v in angles]
    ET.SubElement(asset,'mesh',name='wafer_mesh',vertex=_fmt(np.asarray(vertices).ravel()))
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='-.1 -.2 .5',dir='.2 .3 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'camera',name='overview',pos='-.25 -.34 .27',xyaxes='1 -.6 0 .3 .5 1')
    ET.SubElement(world,'camera',name='edge',pos='-.14 -.22 .12',xyaxes='1 -.3 0 .08 .25 1')
    parts=[]
    def geom(parent,name,kind,pos,size,mass=0.,rgba='.6 .6 .6 1',**kw):
        attrs=dict(name=name,type=kind,pos=_fmt(pos),size=_fmt(size),rgba=rgba,**kw)
        if mass:attrs['mass']=str(mass)
        ET.SubElement(parent,'geom',**attrs)
        if parent.get('name')in('base','jaw','squeeze_slider'):
            parts.append(dict(body=parent.get('name'),name=name,type=kind,pos=list(pos),size=list(size),mass=mass))
    geom(world,'table','plane',[0,0,0],[.5,.5,.02],rgba='.72 .68 .53 1')
    geom(world,'pedestal','cylinder',[0,0,.0125],[.075,.0125],rgba='.45 .45 .48 1')
    geom(world,'heater','cylinder',[0,0,.035],[.075,.01],rgba='.65 .65 .68 1')
    pr=s['pillar_triangle_side']/math.sqrt(3)
    pillar_tops=[]
    for i in range(3):
        xy=[pr*math.cos(i*2*math.pi/3),pr*math.sin(i*2*math.pi/3)]
        geom(world,f'pillar_{i}','cylinder',[*xy,.055],[s['pillar_radius'],.010],rgba='.45 .45 .45 1')
        pillar_tops.append([*xy,bottom])
    for i,(x,z)in enumerate([(tool[0],.059),(tool[0]-.098,.053)]):
        geom(world,f'tool_stand_{i}','box',[x,tool[1],z/2],[.004,.020,z/2],rgba='.25 .25 .27 1')
    wafer_initial=[s.get('wafer_x_offset',0.),s.get('wafer_y_offset',0.),wafer_z+s.get('wafer_z_offset',0.)]
    wafer=ET.SubElement(world,'body',name='wafer',pos=_fmt(wafer_initial),euler=_fmt([0,0,s.get('wafer_yaw',0.)]))
    ET.SubElement(wafer,'freejoint',name='wafer_free')
    ET.SubElement(wafer,'geom',name='wafer_geom',type='mesh',mesh='wafer_mesh',mass=str(s['wafer_mass']),rgba='.35 .43 .62 1')
    base=ET.SubElement(world,'body',name='base',pos=_fmt(tool))
    ET.SubElement(base,'freejoint',name='tool_free')
    for i,side in enumerate((-1,1)):
        geom(base,'base_rail_'+str(i),'box',[-.033,side*.01275,-.001],[.064,.0015,.003],.015)
    geom(base,'fixed_handle','box',[-.065,0.,-.007],[.035,.013,.003],.015,rgba='.20 .22 .25 1')
    geom(base,'lower_bridge','box',[.0335,0.,-.00025],[.0035,.01625,.00075],.002)
    geom(base,'lower_jaw_housing','box',[.0365,0.,.00005],[.0025,.01625,.00095],.003)
    geom(base,'lower_land','box',[.034+s['lower_land_inset'],0.,.0015],[.0005,.01625,.0005],.001,rgba='.2 .2 .22 1')
    # Passive hinge preload closes the front pad; downward rear force opens it.
    jaw=ET.SubElement(base,'body',name='jaw',pos='0 0 .003')
    ET.SubElement(jaw,'joint',name='jaw_hinge',type='hinge',axis='0 1 0',range='-.075 0',stiffness=str(s['jaw_stiffness']),springref=str(s['jaw_rest']),damping=str(s['jaw_damping']),solreflimit='.002 1',solimplimit='.99 .999 .001')
    geom(jaw,'jaw_lever','box',[-.015,0.,.001],[.052,.009,.0015],.015)
    geom(jaw,'upper_jaw_housing','box',[.0355,0.,.001],[.0015,.01625,.0007],.003)
    geom(jaw,'upper_land','box',[.034+s['upper_land_inset'],0.,-.0005],[.0005,.01625,.0005],.001,rgba='.2 .2 .22 1')
    geom(jaw,'release_trigger','box',[-.070,0.,.012],[.013,.010,.0015],.004,rgba='.75 .40 .13 1')
    geom(jaw,'trigger_post','box',[-.062,0.,.0065],[.001,.004,.0041],.001)
    paddle=ET.SubElement(base,'body',name='squeeze_slider',pos='-.065 0 .020')
    ET.SubElement(paddle,'joint',name='squeeze_travel',type='slide',axis='0 0 1',range='-.014 .001',stiffness=str(s['paddle_stiffness']),springref=str(s['paddle_rest']),damping=str(s['paddle_damping']))
    geom(paddle,'upper_handle','box',[0,0,0],[.027,.013,.002],.010,rgba='.20 .22 .25 1')
    # The paddle's underside meets the trigger after the specified deadband.
    carriage=ET.SubElement(world,'body',name='carriage',pos=_fmt(grip),gravcomp='1')
    ET.SubElement(carriage,'inertial',pos='0 0 0',mass='.25',diaginertia='.002 .002 .002')
    for axis,name in enumerate(('x','y','z')):
        vector=[0,0,0];vector[axis]=1
        ET.SubElement(carriage,'joint',name='carriage_'+name,type='slide',axis=_fmt(vector),range='-.3 .3',damping='15')
    for sign,name in [(-1,'lower'),(1,'upper')]:
        finger=ET.SubElement(carriage,'body',name='finger_'+name,gravcomp='1')
        ET.SubElement(finger,'joint',name='finger_'+name,type='slide',axis='0 0 1',range='-.05 .05',damping='2')
        geom(finger,'finger_'+name,'box',[0,0,0],[.015,.012,.003],.030,rgba='.15 .45 .60 1')
    actuator=ET.SubElement(root,'actuator')
    for name in ('x','y','z'):
        ET.SubElement(actuator,'position',name='carriage_'+name,joint='carriage_'+name,kp='1500',kv='70',forcerange=f"-{s['carriage_force_cap']} {s['carriage_force_cap']}")
    for name in ('lower','upper'):
        ET.SubElement(actuator,'position',name='finger_'+name,joint='finger_'+name,kp=str(s['finger_kp']),kv=str(s['finger_kv']),forcerange=f"-{s['finger_force_cap']} {s['finger_force_cap']}")
    metadata=dict(spec=s,tool_origin=tool.tolist(),gripper_origin=grip.tolist(),wafer_bottom=bottom,wafer_initial_center=wafer_initial,pillar_top_centers=pillar_tops,
        primitive_parts=parts,scope='Three-axis force-limited test carriage with two real slider fingers; not a full YAM robot.',
        physics='Dynamic free tool and wafer; passive jaw hinge and squeeze slider; no welds or jaw actuator.')
    return ET.tostring(root,encoding='unicode'),metadata


def contacts(model,data):
    totals={};force=np.zeros(6)
    for i in range(data.ncon):
        c=data.contact[i];mujoco.mj_contactForce(model,data,i,force)
        names=tuple(sorted((mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2))))
        totals[names]=totals.get(names,0.)+max(0.,float(force[0]))
    return totals


def angle_degrees(a,b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T@b)-1)/2,-1.,1.))))


def simulate(seed=0):
    rng=np.random.default_rng(seed)
    changes=dict(friction=float(rng.uniform(.4,.6)),wafer_mass=float(rng.uniform(.018,.022)),
        wafer_yaw=float(rng.uniform(-math.radians(.25),math.radians(.25))),wafer_z_offset=float(rng.uniform(0.,.0001)),
        wafer_x_offset=float(rng.uniform(-.00025,.00025)),wafer_y_offset=float(rng.uniform(-.00025,.00025)))if seed else {}
    xml,meta=build_model_xml(changes);s=meta['spec'];model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    if model.neq or model.nmocap or model.nu!=5:raise ValueError('Unexpected attachment or actuator in the fixture.')
    jid=lambda n:model.joint(n).qposadr[0]
    for name,value in [('finger_lower',-.035),('finger_upper',.035)]:data.qpos[jid(name)]=value
    data.ctrl[:3]=0.;data.ctrl[3:]=[-.035,.035]
    mujoco.mj_forward(model,data)
    ids={n:model.body(n).id for n in ('base','jaw','squeeze_slider','wafer','carriage','finger_lower')}
    origin=np.array(meta['tool_origin']);initial_wafer=np.array(meta['wafer_initial_center']);grip=np.array(meta['gripper_origin'])
    phases=[('settle',.5,origin,.035),('grip',1.5,origin,s['hold_half_gap']),
        ('tool_lift',1.5,origin+[0,0,.025],s['hold_half_gap']),
        ('open',1.5,origin+[0,0,.025],s['open_half_gap']),
        ('transfer',2.5,origin+[-.022,-.15,.025],s['open_half_gap']),
        ('lower_to_edge',2.,origin+[-.022,-.15,s['edge_z_offset']],s['open_half_gap']),
        ('insert',2.,origin+[0,-.15,s['edge_z_offset']],s['open_half_gap']),
        ('seat_lower_pad',1.,origin+[0,-.15,0.],s['open_half_gap']),
        ('close',2.,origin+[0,-.15,0.],s['hold_half_gap']),
        ('lift',2.,origin+[0,-.15,.020],s['hold_half_gap']),
        ('hold',2.,origin+[0,-.15,.020],s['hold_half_gap']),
        ('return',2.,origin+[0,-.15,s['edge_z_offset']],s['hold_half_gap']),
        ('release',2.,origin+[0,-.15,s['edge_z_offset']],s['open_half_gap']),
        ('release_clear',.75,origin+[0,-.15,s['edge_z_offset']],s['open_half_gap']),
        ('withdraw',2.,origin+[-.025,-.15,s['edge_z_offset']],s['open_half_gap']),
        ('restore_grip',2.,origin+[-.025,-.15,s['edge_z_offset']],s['hold_half_gap']),
        ('final_hold',2.,origin+[-.025,-.15,.020],s['hold_half_gap'])]
    records=[];replay=dict(time_s=[0.],qpos=[data.qpos.copy()],qvel=[data.qvel.copy()],ctrl=[data.ctrl.copy()])if seed==0 else None
    audit=dict(contact_band_tolerance_m=.0001,wafer_contact_samples=0,outside_band_contacts=0,
        unexpected_wafer_contact_peak_N=0.,upper_contact_inset_mm=[1e9,-1e9],lower_contact_inset_mm=[1e9,-1e9],
        upper_peak_force_N=0.,lower_peak_force_N=0.,trigger_contact_peak_N=0.,normal_closed_without_fingers=False,
        opened_by_contact=False,opened_gap_max_mm=0.)
    geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)for i in range(model.ngeom)]
    wafer_geom=model.geom('wafer_geom').id;flat=-math.sqrt(.05**2-(.0325/2)**2);force=np.zeros(6)
    peak_jaw=0.;peak_finger=0.;tool_picked=False;hold=[];final=[];failure=None;elapsed=0.;start=origin.copy();start_half=.035;steps=0;alignment=None
    finger_history=np.zeros((round(s['control_dt']/s['dt']),2))
    for phase,duration,target,half in phases:
        if phase not in('settle','grip','tool_lift','open'):
            target=target+np.array(alignment['target_translation_m'])+np.array([s['edge_x_offset'],0.,0.])
        n=round(duration/s['dt']);stride=round(s['control_dt']/s['dt'])
        for k in range(n):
            if k%stride==0:
                a=(k+1)/n;a=a*a*(3-2*a);desired=start+(target-start)*a;gap=start_half+(half-start_half)*a
                if phase in ('settle','grip'):carriage_target=grip
                else:
                    carriage_target=data.xpos[ids['carriage']]+desired-data.xpos[ids['base']]
                    carriage_target[2]+=gap+float(data.ctrl[3])
                    carriage_target[2]+=sum(model.body_mass[ids[b]]for b in('base','jaw','squeeze_slider'))*9.81/1500
                data.ctrl[:3]=np.clip(carriage_target-grip,-.25,.25);data.ctrl[3:]=[-gap,gap]
            mujoco.mj_step(model,data)
            steps+=1
            finger_substep=np.zeros(2)
            for ci in range(data.ncon):
                con=data.contact[ci]
                mujoco.mj_contactForce(model,data,ci,force);load=max(0.,float(force[0]))
                pair={geom_names[con.geom1],geom_names[con.geom2]}
                for fi,name,handle in [(0,'finger_lower','fixed_handle'),(1,'finger_upper','upper_handle')]:
                    if pair=={name,handle}:finger_substep[fi]+=load
                if con.geom1!=wafer_geom and con.geom2!=wafer_geom:continue
                if load<1e-4:continue
                other=geom_names[con.geom2 if con.geom1==wafer_geom else con.geom1]
                local=data.xmat[ids['wafer']].reshape(3,3).T@(con.pos-data.xpos[ids['wafer']]);inset=local[0]-flat
                audit['wafer_contact_samples']+=1
                if other in('upper_land','lower_land'):
                    side=other.split('_')[0];bounds=audit[side+'_contact_inset_mm'];bounds[0]=min(bounds[0],1000*inset);bounds[1]=max(bounds[1],1000*inset)
                    audit[side+'_peak_force_N']=max(audit[side+'_peak_force_N'],load)
                    limit=s[side+'_envelope']
                    if not -.0001<=inset<=limit+.0001 or abs(local[1])>s['pad_width']/2+.0001:audit['outside_band_contacts']+=1
                elif not other.startswith('pillar_'):audit['unexpected_wafer_contact_peak_N']=max(audit['unexpected_wafer_contact_peak_N'],load)
            finger_history[(steps-1)%len(finger_history)]=finger_substep
            if k%100==0 or k==n-1:
                c=contacts(model,data);tool=data.xpos[ids['base']].copy();wafer=data.xpos[ids['wafer']].copy()
                jawload=sum(v for pair,v in c.items()if 'wafer_geom'in pair and('upper_land'in pair or'lower_land'in pair))
                peak_jaw=max(peak_jaw,jawload);peak_finger=max(peak_finger,float(np.max(np.abs(data.actuator_force[3:]))))
                wafer_r=data.xmat[ids['wafer']].reshape(3,3);tool_r=data.xmat[ids['base']].reshape(3,3)
                fixed_finger_r=data.xmat[ids['finger_lower']].reshape(3,3)
                upper=data.geom_xpos[model.geom('upper_land').id];upper_r=data.geom_xmat[model.geom('upper_land').id].reshape(3,3)
                upper_bottom=upper-upper_r[:,2]*.0005;lower_top=data.geom_xpos[model.geom('lower_land').id]+tool_r[:,2]*.0005
                gap=float((upper_bottom-lower_top)@tool_r[:,2]);trigger=c.get(tuple(sorted(('release_trigger','upper_handle'))),0.)
                finger_contact=[sum(v for pair,v in c.items()if 'finger_'+name in pair and any(g in pair for g in ('fixed_handle','upper_handle')))for name in ('lower','upper')]
                finger_gap=[float(mujoco.mj_geomDistance(model,data,model.geom('finger_'+name).id,model.geom(handle).id,.01,None))for name,handle in [('lower','fixed_handle'),('upper','upper_handle')]]
                audit['trigger_contact_peak_N']=max(audit['trigger_contact_peak_N'],trigger)
                if phase=='settle'and k>n/2 and max(finger_contact)<.001 and abs(gap)<.0002:audit['normal_closed_without_fingers']=True
                if phase in('open','release')and min(finger_contact)>.05 and trigger>.05 and gap>.0018:
                    audit['opened_by_contact']=True;audit['opened_gap_max_mm']=max(audit['opened_gap_max_mm'],1000*gap)
                row=dict(time=round(float(data.time),4),phase=phase,tool=tool.tolist(),wafer=wafer.tolist(),jaw_angle=float(data.qpos[jid('jaw_hinge')]),
                    squeeze_mm=float(-1000*data.qpos[jid('squeeze_travel')]),jaw_force=jawload,
                    finger_contact=finger_contact,finger_contact_mean_N=finger_history.mean(axis=0).tolist(),finger_gap_m=finger_gap,jaw_forces=[c.get(tuple(sorted(('wafer_geom',g))),0.)for g in('lower_land','upper_land')],
                    wafer_pin_forces=[c.get(tuple(sorted(('wafer_geom','pillar_'+str(i)))),0.)for i in range(3)],
                    wafer_pin_force=sum(v for pair,v in c.items()if 'wafer_geom'in pair and any(g.startswith('pillar_')for g in pair)),
                    all_tool_wafer_force=sum(v for pair,v in c.items()if 'wafer_geom'in pair and not any(g.startswith('pillar_')for g in pair)),
                    wafer_tilt_deg=math.degrees(math.acos(float(np.clip(wafer_r[2,2],-1,1)))),
                    wafer_rotation=wafer_r.tolist(),tool_rotation=tool_r.tolist(),
                    wafer_in_tool=(tool_r.T@(wafer-tool)).tolist(),tool_in_hand=(tool-data.xpos[ids['carriage']]).tolist(),
                    tool_in_fixed_finger=(fixed_finger_r.T@(tool-data.xpos[ids['finger_lower']])).tolist(),
                    tool_rotation_in_fixed_finger=(fixed_finger_r.T@tool_r).tolist(),
                    trigger_contact_N=trigger,open_gap_mm=1000*gap)
                records.append(row)
                if phase=='tool_lift'and tool[2]-origin[2]>.020:tool_picked=True
                if phase=='hold'and k>=n/2-100:hold.append(row)
                if phase=='final_hold'and k>=n/2-100:final.append(row)
                if np.linalg.norm(tool-data.xpos[ids['carriage']])>.25 or wafer[2]<.04:failure='tool_or_wafer_drop';break
            if replay is not None and steps%100==0:
                replay['time_s'].append(float(data.time));replay['qpos'].append(data.qpos.copy());replay['qvel'].append(data.qvel.copy());replay['ctrl'].append(data.ctrl.copy())
        if failure:break
        if phase=='settle':
            observed_position=data.xpos[ids['wafer']].copy();observed_rotation=data.xmat[ids['wafer']].reshape(3,3).copy()
            observed_flat=observed_position+observed_rotation@np.array([flat,0.,-s['wafer_thickness']/2])
            alignment=dict(source='Measured settled wafer pose; translation-only scripted fixture alignment, not vision.',
                observed_position_m=observed_position.tolist(),observed_rotation=observed_rotation.tolist(),flat_center_world_m=observed_flat.tolist(),
                target_translation_m=(observed_flat-np.array([flat,0.,meta['wafer_bottom']])).tolist())
        start=np.array(target);start_half=half;elapsed+=duration
    lift_pass=bool(hold)and min(r['wafer'][2]-initial_wafer[2]for r in hold)>=.018
    hold_stable=bool(hold)and max(np.linalg.norm(np.array(r['wafer'])-np.array(hold[0]['wafer']))for r in hold)<=.001
    returned=bool(final)and max(np.linalg.norm(np.array(r['wafer'])-initial_wafer)for r in final)<=.001
    supported=bool(final)and min(r['wafer_pin_force']for r in final)>.8*s['wafer_mass']*9.81 and min(min(r['wafer_pin_forces'])for r in final)>.005
    released=bool(final)and max(r['all_tool_wafer_force']for r in final)<.01
    two_sided=bool(hold)and min(min(r['jaw_forces'])for r in hold)>.05
    tilt=bool(hold)and max(r['wafer_tilt_deg']for r in hold)<=2. and bool(final)and max(r['wafer_tilt_deg']for r in final)<=2.
    retention=bool(hold)and max(np.linalg.norm(np.array(r['wafer_in_tool'])-np.array(hold[0]['wafer_in_tool']))for r in hold)<=.001
    tool_retention=bool(hold)and min(min(r['finger_contact'])for r in hold)>.05 and max(np.linalg.norm(np.array(r['tool_in_hand'])-np.array(hold[0]['tool_in_hand']))for r in hold)<=.001
    rotation_stable=bool(hold)and max(angle_degrees(np.array(r['wafer_rotation']),np.array(hold[0]['wafer_rotation']))for r in hold)<=2. and max(angle_degrees(np.array(r['tool_rotation']),np.array(hold[0]['tool_rotation']))for r in hold)<=2.
    motion=dict(wafer_translation_mm=0.,wafer_rotation_deg=0.,tool_translation_mm=0.,tool_rotation_deg=0.)
    close_rows=[r for r in records if r['phase']=='close']
    loaded=[r for r in records if r['phase']in('lift','hold','return')]
    motion_retention=bool(close_rows and loaded)
    if motion_retention:
        reference=close_rows[-1];ref_rotation=np.array(reference['tool_rotation']).T@np.array(reference['wafer_rotation'])
        motion['wafer_translation_mm']=1000*max(np.linalg.norm(np.array(r['wafer_in_tool'])-reference['wafer_in_tool'])for r in loaded)
        motion['wafer_rotation_deg']=max(angle_degrees(ref_rotation,np.array(r['tool_rotation']).T@np.array(r['wafer_rotation']))for r in loaded)
        motion_retention=(motion['wafer_translation_mm']<=1. and motion['wafer_rotation_deg']<=2.
            and min(min(r['jaw_forces'])for r in loaded)>.05 and max(r['wafer_tilt_deg']for r in loaded)<=2.)
    # Track the fixed handle against its contacting finger through every squeeze.
    refs=[r for r in records if r['phase']=='grip'];transport=[r for r in records if r['phase']not in('settle','grip')]
    transport_retention=bool(refs and transport)
    if transport_retention:
        ref=refs[-1]
        motion['tool_translation_mm']=1000*max(np.linalg.norm(np.array(r['tool_in_fixed_finger'])-ref['tool_in_fixed_finger'])for r in transport)
        motion['tool_rotation_deg']=max(angle_degrees(np.array(ref['tool_rotation_in_fixed_finger']),np.array(r['tool_rotation_in_fixed_finger']))for r in transport)
        motion['tool_min_finger_contact_N']=min(min(r['finger_contact'])for r in transport)
        motion['tool_min_finger_contact_mean_N']=min(min(r['finger_contact_mean_N'])for r in transport)
        motion['finger_contact_mean_window_s']=s['control_dt']
        worst=min(transport,key=lambda r:min(r['finger_contact']))
        motion['tool_min_contact_at']=dict(phase=worst['phase'],time_s=worst['time'],forces_N=worst['finger_contact'],gaps_mm=(1000*np.array(worst['finger_gap_m'])).tolist())
        motion['tool_max_finger_gap_mm']=1000*max(max(r['finger_gap_m'])for r in transport)
        transport_retention=(motion['tool_translation_mm']<=2. and motion['tool_rotation_deg']<=3.
            and motion['tool_min_finger_contact_mean_N']>.005 and motion['tool_max_finger_gap_mm']<=.1)
    bands=audit['outside_band_contacts']==0 and audit['unexpected_wafer_contact_peak_N']<.01
    checks=dict(tool_pickup=tool_picked,wafer_lift=lift_pass,hold_stable=hold_stable,returned=returned,pin_supported=supported,released=released,
        both_jaws_loaded=two_sided,wafer_tilt=tilt,wafer_retention=retention,tool_retention=tool_retention,rotation_stable=rotation_stable,
        contact_bands=bands,passive_closed=audit['normal_closed_without_fingers'],contact_opening=audit['opened_by_contact'],
        full_loaded_motion_retention=motion_retention,full_tool_transport_retention=transport_retention)
    result=dict(seed=seed,success=all(checks.values())and failure is None,checks=checks,contact_audit=audit,
        model_sha256=hashlib.sha256(xml.encode()).hexdigest(),actual_spec=s,motion_retention=motion,measured_alignment=alignment,
        failure=failure,peak_combined_jaw_force_N=peak_jaw,peak_finger_actuator_force_N=peak_finger,
        final_wafer=records[-1]['wafer']if records else None,trial_parameters=changes,phase_samples=[r for i,r in enumerate(records)if i==0 or r['phase']!=records[i-1]['phase']or i==len(records)-1],
        hold_summary=None if not hold else dict(min_rise_mm=1000*min(r['wafer'][2]-initial_wafer[2]for r in hold),max_rise_mm=1000*max(r['wafer'][2]-initial_wafer[2]for r in hold),
            duration_s=hold[-1]['time']-hold[0]['time'],max_tilt_deg=max(r['wafer_tilt_deg']for r in hold),
            minimum_lower_jaw_force_N=min(r['jaw_forces'][0]for r in hold),minimum_upper_jaw_force_N=min(r['jaw_forces'][1]for r in hold)))
    result=json.loads(json.dumps(result,default=lambda v:v.item(),allow_nan=False))
    return result,xml,meta,replay


def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=1);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.trials<1:p.error('At least one trial is required.')
    if not 1<=args.workers<=8:p.error('Workers must be between1and8.')
    root=Path(__file__).resolve().parent;results=[];started=time.time();source_before=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();nominal_metadata=None
    with ProcessPoolExecutor(max_workers=min(args.workers,args.trials)) as pool:
        for seed,(result,xml,meta,replay) in enumerate(pool.map(simulate,range(args.trials))):
            results.append(result)
            print(json.dumps({k:result[k]for k in ('seed','success','checks','failure','hold_summary','motion_retention')},allow_nan=False),flush=True)
            if seed==0:
                nominal_metadata=copy.deepcopy(meta)
                (root/'scene.xml').write_text(xml)
                if replay is not None:np.savez_compressed(root/'nominal_replay.npz',**replay)
    source_after=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_before!=source_after:raise RuntimeError('Source changed during the batch; results are not frozen.')
    report=dict(scope=nominal_metadata['scope'],spec=SPEC,criteria='At least16/20 successful varied local mechanism trials; not full robot validation.',
        trials=results,successes=sum(r['success']for r in results),count=len(results),wall_seconds=time.time()-started,
        source_sha256=source_before,source_unchanged_during_batch=True,nominal_scene_sha256=hashlib.sha256((root/'scene.xml').read_bytes()).hexdigest(),
        nominal_replay_sha256=hashlib.sha256((root/'nominal_replay.npz').read_bytes()).hexdigest(),post_initialization_tool_or_wafer_pose_writes=0,
        actuator_scope='Only carriage and two fingers; all tool joints passive.',metadata=nominal_metadata,
        varied_ranges=dict(friction=[.4,.6],wafer_mass_kg=[.018,.022],wafer_x_y_m=[-.00025,.00025],wafer_yaw_deg=[-.25,.25],initial_z_extra_m=[0.,.0001]))
    (root/'batch_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
