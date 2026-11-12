"""Two continuous, separately scored wafer tasks; no return motions."""
from pathlib import Path
import argparse,copy,gzip,hashlib,json,math,subprocess,sys,time,zipfile
import numpy as np
import wafer_retention_isaac_v12 as native
import wafer_full_cycle_isaac as full
from wafer_full_cycle_isaac import FullCycle,TraceRowList,compact_evaluation_rows,compact_guard_rows
from wafer_cycle_protocol import (GunFeedback,gun_motion_timing,plan_scan,GUN_OFF_GRIP_M,
    air_coverage,held_wafer_check,gun_retention_check)
from wafer_finger_preload_v12 import preceding_interval_features,wafer_prefix_phases
from wafer_grasp_protocol import _pose
from wafer_tool_pickup import evaluate_tool_pickup
SUCCESS='TWO_TASKS_SUCCEEDED_IN_THIS_SIMULATION'
PICKUP='WAFER_PICKUP_TASK_SUCCEEDED_IN_THIS_SIMULATION'
BLOW='HELD_WAFER_BLOW_TASK_SUCCEEDED_IN_THIS_SIMULATION'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def mechanical(metadata,rows):
    result=native.evaluate_wafer_grasp_trial(metadata,rows,require_return=False)
    peak=max(r['unexpected_robot_tool_contact_normal_force_N']+r['unexpected_robot_environment_contact_normal_force_N']for r in rows)
    result['checks']['robot_obstacle_contacts']=dict(passed=peak<=result['limits']['maximum_unwanted_contact_N'],maximum_force_N=peak)
    result['checks']['finger_handle_jam_guard']=native.evaluate_finger_handle_load(rows,engine=metadata.get('split_evidence_engine','isaac'))
    result['checks']['passive_pad_motion_guard']=native.evaluate_passive_pad_state(rows,metadata['pad_compliance'])
    result['checks']['passive_hinge_torque_guard']=native.evaluate_passive_hinge_state(rows,metadata['passive_hinge'])
    result['failed_checks']=[k for k,v in result['checks'].items()if not v['passed']]
    result['passed']=not result['failed_checks']and not result.get('errors')
    return {k:result[k]for k in ('checks','failed_checks','passed','errors','limits')if k in result}

def evaluate_blow(metadata,rows):
    rows=compact_evaluation_rows(rows)
    projected=[dict(r,phase='lift_hold')if r['phase'].startswith('gun_')or r['phase']in('air_on','air_scan','air_off')else r for r in rows]
    result=mechanical(metadata,projected);checks=result['checks']
    checks['air_path_coverage']=air_coverage(rows,float(metadata['physics_dt_s']))
    checks['wafer_retained_during_air_path']=held_wafer_check(rows)
    checks['gun_bilateral_retention']=gun_retention_check(rows,float(metadata['physics_dt_s']),metadata['gun']['grip_local_m'])
    contacts=[c[2]for r in rows for c in r.get('unexpected_gun_right_contacts',[])];checks['gun_contacts']=dict(passed=max(contacts,default=0.)<=.01,maximum_force_N=max(contacts,default=0.))
    preair=[r for r in rows if r['phase'].startswith('gun_')and r['phase']!='gun_clear'];checks['air_off_before_aim']=dict(passed=bool(preair)and all(not r['air_on']for r in preair))
    off=[r for r in rows if r['phase']=='air_off'];window=round(.25/metadata['physics_dt_s'])
    checks['air_switched_off']=dict(passed=bool(off)and len(rows)>=window and all(not r['air_on']for r in rows[-window:]),final_air_on=rows[-1]['air_on'],hold_s=.25)
    force=max(abs(r['trigger_spring_force_N'])for r in rows);speed=max(abs(r['trigger_velocity_m_s'])for r in rows)
    checks['passive_trigger_bounds']=dict(passed=force<=6. and speed<=1.,maximum_force_N=force,maximum_speed_m_s=speed)
    result['failed_checks']=[k for k,v in checks.items()if not v['passed']];result['passed']=not result['failed_checks']and not result.get('errors')
    result['result']=BLOW if result['passed']else'FAILED';result['score']=float(result['passed'])
    result['scope']='Held-wafer contact retention, gun pickup,18-row geometric air coverage and air off. No return, evaporation or cleanliness claim.'
    return result

def safety(metadata,rows):
    guards=compact_guard_rows(rows)
    checks=dict(tool_hand_retention=full.evaluate_tool_hand_retention(metadata,rows),
        passive_handle_operating_limits=full.evaluate_handle_paddles(metadata,guards,operating=True),
        tool_environment_contacts=full.evaluate_tool_environment(guards),
        finger_load=native.evaluate_finger_handle_load(guards,engine=metadata.get('split_evidence_engine','isaac')),
        passive_pad=native.evaluate_passive_pad_state(guards,metadata['pad_compliance']),
        passive_hinge=native.evaluate_passive_hinge_state(guards,metadata['passive_hinge']),
        duration=full.workflow_duration_check(metadata,rows[-1]['time_s']))
    peak=max(max(r['forbidden_wafer_contact_force_N'],r['unexpected_robot_tool_contact_normal_force_N']+r['unexpected_robot_environment_contact_normal_force_N'],max((c[2]for c in r['unexpected_gun_right_contacts']),default=0.))for r in rows)
    checks['unwanted_contacts']=dict(passed=peak<=.01,maximum_force_N=peak)
    return dict(passed=all(c['passed']for c in checks.values()),checks=checks)

def task_rows(metadata,rows,end):
    boundary=next(x for x in metadata['full_stage_boundaries']if x['stage']=='wafer_cycle');start=boundary['initial_sample_index'];origin=boundary['global_time_s']
    part=TraceRowList(dict(r,global_time_s=r['time_s'],time_s=r['time_s']-origin)for r in rows[start:end+1])
    part[0]=dict(part[0],phase='initialization',finger_preload_command_index=None)
    return start,part

def score_pickup(metadata,rows,end):
    data=rows[:end+1];start,part=task_rows(metadata,rows,end)
    checks=dict(tool_pickup=evaluate_tool_pickup(metadata,data[:start+1]),wafer_lift=mechanical(metadata,compact_evaluation_rows(part)),safety=safety(metadata,data))
    progress=full.evaluate_task_score(metadata,data);done={x['stage']for x in progress['diagnostics']['completed']}
    checks['measured_milestones']=dict(passed=not progress['invalid_reasons']and {'tool_pickup','wafer_lift'}.issubset(done),completed=sorted(done))
    passed=all(c.get('passed')is True for c in checks.values())
    return dict(result=PICKUP if passed else'FAILED',score=float(passed),passed=passed,checks=checks,start_s=0.,end_s=data[-1]['time_s'],scope='Empty hands, physical pliers pickup, wafer grasp, lift and stable hold.')

class SplitCycle(FullCycle):
    def __init__(self,through='blow'):self.through=through
    def validate(self,bundle,plan,metadata):
        super().validate(bundle,plan,metadata)
        if sha(__file__)!=self.full['files']['wafer_two_tasks.py']:raise ValueError('Split task source differs.')
        metadata['split_evidence_engine']='isaac'
        metadata['split_tasks']=dict(version='continuous_two_tasks_v1',through=self.through,initial_state='Both hands empty; Task2 continues the physical Task1 state.',return_motions=False)
    def run(self,env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix):
        raw=recorder.raw;end=len(raw.rows)-1;metadata['split_tasks']['task1_end_index']=end;metadata['split_tasks']['task1_end_s']=raw.rows[-1]['time_s']
        first=score_pickup(metadata,raw.rows,end);native._json(Path(destination)/'task1_runtime_report.json',first)
        print('TASK1 RUNTIME:',first['result'],flush=True)
        if not first['passed']or self.through=='pickup':return dict(result=first['result'],tasks={'task1':first},errors=[])
        second=self.run_air(env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix)
        second.update(start_s=first['end_s'],end_s=raw.rows[-1]['time_s']);second['checks']['whole_run_safety']=safety(metadata,raw.rows)
        second['passed']=second['passed']and second['checks']['whole_run_safety']['passed'];second['score']=float(second['passed']);second['result']=BLOW if second['passed']else'FAILED'
        native._json(Path(destination)/'task2_runtime_report.json',second)
        return dict(result=SUCCESS if second['passed']else'FAILED',tasks={'task1':first,'task2':second},errors=[],scope='Two continuous tasks; return motions not run.')

    def run_air(self,env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix):
            from yam_arena import ik,station,constants
            start=time.monotonic();action=np.array(previous_action);records=[];gun_servo=None;state={}
            timing=gun_motion_timing(metadata)
            if timing:metadata['gun_motion_timing_profile']=dict(timing)
            origin=native._array(base.scene.env_origins,(1,3))[0]
            def tcp(q,side='left'):return ik.tcp_pose(q,np.asarray(station.BASES[side])+origin)
            def solve_ik(p,r,q,side='left',damping=.012):
                return ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=np.asarray(station.BASES[side])+origin,iters=600,damping=damping,tol=1e-7)
            def solve_pose(p,r,q,side='left'):
                value,pe,ae=solve_ik(p,r,q,side)
                if pe>.0001 or ae>.001:raise ValueError('Cycle IK exceeds its bounds.')
                return value
            def tick(phase,diagnostic):
                if time.monotonic()-start>7200:raise RuntimeError('Cycle operational wall budget exceeded.')
                self.preload_reference['control_index']=diagnostic.get('left_preload',{}).get('control_index')
                recorder.begin(phase);env.step(torch.tensor([action.tolist()],device=base.device,dtype=torch.float32));row=recorder.end()
                records.append(dict(time_s=row['time_s']-.05,phase=phase,action=action.tolist(),**diagnostic))
                if len(records)%20==0:print(f"Cycle {row['time_s']:.1f}s: {phase}; trigger {row['trigger_compression_m']*1000:.2f}mm",flush=True)
                return row
            qgrip=np.array(self.gun['grasp_q_rad']);p,r=tcp(qgrip,'right');qlift=solve_pose(p+[0,0,.025],r,qgrip,'right')
            def motion():
                for phase,duration,q,f in [('gun_settle',2.,action[7:13].copy(),-.035),('gun_approach',4.,qgrip,-.035),
                    ('gun_close',4.,qgrip,GUN_OFF_GRIP_M),('gun_hold',2.,qgrip,GUN_OFF_GRIP_M),('gun_lift',4.,qlift,GUN_OFF_GRIP_M),('gun_lift_hold',2.,qlift,GUN_OFF_GRIP_M)]:
                    yield phase,(dict(timing['pickup_phases'])[phase]if timing else duration),q,f,-1
                seed=action[7:13].copy()if timing else recorder.rows[-1]['measured_right_arm_q_rad']
                commanded=None
                if timing:
                    p,r=tcp(seed,'right');commanded=np.eye(4);commanded[:3,3]=p;commanded[:3,:3]=r
                yield from plan_scan(lambda:recorder.rows[-1],seed,self.gun,solve_pose,
                    ik.slerp_matrix,constants.TCP_OFFSET,state,timing,commanded)
            try:
                for phase,duration,target,grip,row_id in motion():
                    preload.phases.append((phase,duration));self.scan_row=row_id
                    q0=action[7:13].copy();f0=action[13]*constants.FINGER_OPEN
                    for i in range(round(duration*20)):
                        features=preceding_interval_features(recorder.rows,preload.command_index,metadata['physics_dt_s'])
                        previous_fingers,diagnostic=preload.step(phase,features,previous_fingers);action[6]=previous_fingers[0]/constants.FINGER_OPEN
                        s=(i+1)/(duration*20);u=s*s*(3-2*s)
                        nominal=target[i]if target.ndim==2 else target.copy()if phase in ('air_on','air_scan')else q0+(target-q0)*u
                        correction=None
                        if phase in ('air_on','air_scan'):
                            if gun_servo is None:
                                relation=state['gun_to_tcp'];inverse=np.linalg.inv(relation)
                                def fk(q):
                                    p,r=tcp(q,'right');return p+r@inverse[:3,3],r@inverse[:3,:3]
                                def solve(p,r,q):return solve_ik(p+r@relation[:3,3],r@relation[:3,:3],q,'right')
                                gun_servo=GunFeedback(fk,solve,ik.LOWER,ik.UPPER)
                            g=_pose(recorder.rows[-1]['gun_pose_world'],'gun')
                            nominal,correction=gun_servo.step(nominal,g[:3,3],g[:3,:3],action[7:13],integrate=not timing or phase=='air_on')
                        if timing and np.max(np.abs(nominal-action[7:13]))>math.radians(timing['maximum_joint_step_deg']):
                            raise native.TrialAbort('Right-hand command jump exceeded one degree.')
                        action[7:13]=nominal;action[13]=(f0+(grip-f0)*u)/constants.FINGER_OPEN
                        tick(phase,dict(left_preload=diagnostic,gun_feedback=correction,scan_row=row_id))
                self.scan_row=-1
            finally:
                with (Path(destination)/'cycle_commands.jsonl').open('w')as stream:
                    for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
                metadata['cycle_completed_control_actions']=len(records)
                metadata['final_trigger_drive']=native.native_mechanism_readback(self.gunasset,self.export['passive_spring'],0.,expected_dof_count=1)
            result=evaluate_blow(metadata,recorder.rows)
            return result

def audit_commands(metadata,rows,directory,include_air):
    directory=Path(directory);origin=next(x['global_time_s']for x in metadata['full_stage_boundaries']if x['stage']=='wafer_cycle')
    commands=[]
    for filename,clock in [('tool_commands.jsonl',0.),('commands.jsonl',origin)]+([('cycle_commands.jsonl',origin)]if include_air else []):
        with (directory/filename).open()as stream:
            for line in stream:
                item=json.loads(line);t=item['start_time_s']if filename=='commands.jsonl'else item['time_s']
                commands.append((t+clock,item['target']if filename=='commands.jsonl'else item['action'],item['phase']))
    commands.sort(key=lambda x:x[0]);dt=float(metadata['physics_dt_s']);stride=round(.05/dt)
    if len(rows)!=1+stride*len(commands):raise ValueError('Native global command/sample count differs.')
    denominator=float(metadata['controller']['native_finger_open_q_m'])
    for index,(timestamp,action,phase)in enumerate(commands):
        if abs(timestamp-index*.05)>1e-8:raise ValueError('Native global command timestamp gap.')
        action=np.asarray(action,dtype=np.float32).astype(float)
        expected={'commanded_arm_q_rad':action[:6],'commanded_finger_q_m':[action[6]*denominator]*2,
            'commanded_right_arm_q_rad':action[7:13],'commanded_right_finger_q_m':[action[13]*denominator]*2}
        for row in rows[1+index*stride:1+(index+1)*stride]:
            if row['phase']!=phase:raise ValueError('Native command and measured phase differ.')
            for key,value in expected.items():
                if not np.allclose(row[key],value,atol=1e-7,rtol=0):raise ValueError('Native actuator readback differs: '+key)
    return dict(complete=True,commands=len(commands),physics_samples=len(rows),maximum_readback_tolerance=1e-7)

def audit_preload(metadata,stage,directory,include_air):
    directory=Path(directory);dt=metadata['physics_dt_s'];phases=wafer_prefix_phases(metadata)
    records=[json.loads(x)for x in(directory/'finger_preload.jsonl').read_text().splitlines()]
    commands=[json.loads(x)for x in(directory/'commands.jsonl').read_text().splitlines()]
    count=1+round(sum(d for _,d in phases)/dt)
    result=native.audit_finger_preload_evidence(stage[:count],records,commands,metadata['controller']['native_finger_open_q_m'],dt,phases)
    if not result['complete']:raise ValueError('Prefix preload evidence incomplete.')
    if include_air:
        controller=native.FingerPreloadController(phases)
        for record in records:controller.step(record['phase'],record['interval'],record['previous_command_q_m'])
        cycle=[json.loads(x)for x in(directory/'cycle_commands.jsonl').read_text().splitlines()];stride=round(.05/dt)
        for index,command in enumerate(cycle):
            sample=count-1+index*stride
            if abs(command['time_s']-stage[sample]['time_s'])>1e-8:raise ValueError('Cycle preload clock differs.')
            record=command['left_preload'];controller.phases.append((record['phase'],.05))
            _,expected=controller.step(record['phase'],record['interval'],record['previous_command_q_m'])
            if expected!=record or record['interval']!=native.preceding_interval_features(stage,record['control_index'],dt,prefix_length=sample+1):raise ValueError('Cycle preload command or measurement differs.')
        if len(stage)!=count+len(cycle)*stride:raise ValueError('Incomplete cycle evidence.')
    return dict(complete=True)

def verify_run(directory):
    directory=Path(directory);metadata=json.loads((directory/'metadata.json').read_text());report=json.loads((directory/'trial_report.json').read_text())
    identities={n:sha(directory/n)for n in('metadata.json','trial_report.json')};trace=sha(directory/'samples.jsonl.gz')
    with gzip.open(directory/'samples.jsonl.gz','rt')as f:rows=TraceRowList((json.loads(x)for x in f),cache_blocks=1024)
    result=dict(version=1,passed=False,tasks={},errors=[],physics_samples=len(rows),trace_sha256=trace,source_files_sha256=identities)
    from wafer_robot_geometry import evaluate_native_robot_geometry
    try:
        if metadata.get('split_evidence_engine')!='isaac':raise ValueError('Expected native force provenance.')
        full.audit_native_full_provenance(metadata,rows)
        end=metadata['split_tasks']['task1_end_index']
        if not 0<end<len(rows)or abs(rows[end]['time_s']-metadata['split_tasks']['task1_end_s'])>1e-8:raise ValueError('Invalid Task1 boundary.')
        for name,stop in [('task1',end)]+([('task2',len(rows)-1)]if(directory/'task2_runtime_report.json').exists()else[]):
            include_air=name=='task2';data=rows[:stop+1];_,stage=task_rows(metadata,rows,stop)
            task=score_pickup(metadata,rows,stop)if not include_air else evaluate_blow(metadata,stage)
            if include_air:
                task.update(start_s=rows[end]['time_s'],end_s=rows[-1]['time_s']);task['checks']['whole_run_safety']=safety(metadata,data)
                task['passed']=task['passed']and task['checks']['whole_run_safety']['passed'];task['score']=float(task['passed']);task['result']=BLOW if task['passed']else'FAILED'
            saved=json.loads((directory/(name+'_runtime_report.json')).read_text());matches=task==saved
            commands=audit_commands(metadata,data,directory,include_air);preload=audit_preload(metadata,stage,directory,include_air)
            geometry=evaluate_native_robot_geometry(metadata,data)
            complete=commands['complete']and preload['complete']and geometry['evidence_complete']
            passed=task['passed']and matches and complete and geometry['passed']
            result['tasks'][name]=dict(passed=passed,score=float(passed),runtime_score=task['score'],matches_report=matches,
                start_s=task['start_s'],end_s=task['end_s'],result=task['result']if passed else'FAILED',
                evidence=dict(complete=complete,commands=commands,preload=preload,robot_geometry=geometry),checks=task['checks'])
            print('Independent '+name+': '+str(passed),flush=True)
    except Exception as exc:result['errors'].append(type(exc).__name__+': '+str(exc))
    if identities!={n:sha(directory/n)for n in identities}or trace!=sha(directory/'samples.jsonl.gz'):result['errors'].append('Closed evidence changed during audit.')
    expected=['task1']if metadata.get('split_tasks',{}).get('through')=='pickup'else['task1','task2']
    result['passed']=not result['errors']and all(result['tasks'].get(n,{}).get('passed')is True for n in expected)
    result['result']='INDEPENDENT_TWO_TASK_EVIDENCE_CHECKED'if result['passed']else'FAILED_OR_INCOMPLETE'
    return result

def export_replay(directory,audit):
    directory=Path(directory);accepted={k:v for k,v in audit['tasks'].items()if v['passed']}
    if not accepted or audit['errors']:raise ValueError('No independently accepted task.')
    if audit['trace_sha256']!=sha(directory/'samples.jsonl.gz')or audit['source_files_sha256']!={n:sha(directory/n)for n in('metadata.json','trial_report.json')}:raise ValueError('Replay source differs from its audit.')
    metadata=json.loads((directory/'metadata.json').read_text());dt=metadata['physics_dt_s'];stride=round(.05/dt);end=max(v['end_s']for v in accepted.values());last_index=round(end/dt)
    target=directory/'video_samples.jsonl.gz';selected=0
    with gzip.open(directory/'samples.jsonl.gz','rt')as source,gzip.open(target,'xt',compresslevel=5)as output:
        for i,line in enumerate(source):
            if i>last_index:break
            row=json.loads(line)
            if abs(row['time_s']-i*dt)>1e-8:raise ValueError('Replay source cadence differs.')
            if i%stride==0:output.write(line);selected+=1
    if last_index%stride:raise ValueError('Task boundary is not a control boundary.')
    manifest=dict(version=2,scope='Independently accepted task segments from continuous native measurements.',sample_interval_s=.05,
        selected_count=selected,physics_samples=audit['physics_samples'],original_trace_sha256=audit['trace_sha256'],selected_trace_sha256=sha(target),
        source_files={n:sha(directory/n)for n in('metadata.json','trial_report.json','independent_check.json')},
        exporter_file='wafer_two_tasks.py',exporter_sha256=sha(__file__),source_start_s=0.,source_end_s=end,
        tasks={k:{n:v[n]for n in('start_s','end_s','score','result')}for k,v in accepted.items()})
    native._json(directory/'video_replay_manifest.json',manifest)
    archive=directory/'video_replay.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED)as z:
        for n in(*manifest['source_files'],'video_replay_manifest.json','video_samples.jsonl.gz'):z.write(directory/n,n)
    return dict(path=str(archive),sha256=sha(archive),tasks=sorted(accepted))

def batch(bundle,output,count=5):
    bundle=Path(bundle).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((bundle/'bundle_files.json').read_text());result=dict(result='RUNNING',attempted=0,passed=0,requested_trials=count,trials=[])
    native._json(output/'batch_report.json',result)
    for index in range(1,count+1):
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            print('Waiting for shared GPU; no process is stopped.',flush=True);time.sleep(30)
        if any(sha(bundle/n)!=h for n,h in manifest.items()):raise ValueError('Frozen split bundle changed.')
        run=output/f'trial_{index:02d}';print(f'Starting two tasks {index}/{count}',flush=True)
        with(output/f'trial_{index:02d}.log').open('x')as log:
            process=subprocess.run([sys.executable,'-u',str(bundle/'wafer_two_tasks.py'),'--bundle',str(bundle),'--output',str(run),'--through','blow'],stdout=log,stderr=subprocess.STDOUT)
        try:audit=verify_run(run)
        except Exception as exc:audit=dict(passed=False,tasks={},errors=[type(exc).__name__+': '+str(exc)])
        native._json(run/'independent_check.json',audit)
        item=dict(trial=run.name,passed=audit['passed']and process.returncode==0,tasks={k:v['passed']for k,v in audit['tasks'].items()},errors=audit['errors'])
        if any(v['passed']for v in audit['tasks'].values()):
            try:item['video_replay']=export_replay(run,audit)
            except Exception as exc:item['video_error']=str(exc)
        result['trials'].append(item);result['attempted']+=1;result['passed']+=int(item['passed']);native._json(output/'batch_report.json',result)
        if not item['passed']:result['result']='STOPPED_ON_DEVELOPMENT_FAILURE';break
    else:result['result']='CLOUD_FIVE_COMPLETED'
    result['observed_success_fraction']=result['passed']/result['attempted']if result['attempted']else None
    native._json(output/'batch_report.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--through',choices=('pickup','blow'),default='blow');p.add_argument('--check-only',action='store_true');p.add_argument('--batch',type=int);p.add_argument('--audit',action='store_true');a=p.parse_args()
    full.ALLOW_UNQUALIFIED_DEVELOPMENT=True
    if a.check_only:
        plan,exported,models,gate=native.validate_bundle(a.bundle);metadata=native.cloud_model_metadata(exported['metadata'],plan);SplitCycle(a.through).validate(a.bundle,plan,metadata)
        print('TWO_TASK_BUNDLE_CHECKED')
    elif a.output is None:p.error('--output is required')
    elif a.audit:native._json(a.output/'independent_check.json',verify_run(a.output))
    elif a.batch:batch(a.bundle,a.output,a.batch)
    else:native.run_isaac_trial(a.bundle,a.output,extension=SplitCycle(a.through),device='cuda:0')
