"""Five fresh native trials of one frozen complete-cycle bundle."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,sys,time,zipfile
import numpy as np
from wafer_cycle_isaac import evaluate_native_cycle
from wafer_retention_isaac_v12 import validate_bundle


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_run(run,full_cycle=False):
    metadata=json.loads((run/'metadata.json').read_text());report=json.loads((run/'trial_report.json').read_text())
    if full_cycle:
        from wafer_trace_rows import TraceRowList
        with gzip.open(run/'samples.jsonl.gz','rt')as stream:rows=TraceRowList((json.loads(line)for line in stream),cache_blocks=1024)
    else:
        with gzip.open(run/'samples.jsonl.gz','rt')as stream:rows=[json.loads(line)for line in stream]
    if full_cycle:
        from wafer_full_cycle_isaac import evaluate_native_full,SUCCESS
        audit=evaluate_native_full(metadata,rows,directory=run)
        if audit.get('matches_report')is not True or audit.get('evidence',{}).get('complete')is not True:
            raise ValueError('Full-workflow independent trace or command audit is incomplete: '+str(audit.get('errors')))
        return dict(result='INDEPENDENT_FULL_WORKFLOW_TRACE_AND_COMMANDS_MATCH',passed=audit['result']==SUCCESS,
            native_result=audit['result'],matches_report=audit['matches_report'],
            physics_samples=len(rows),trace_sha256=sha(run/'samples.jsonl.gz'),evidence=audit['evidence'],
            task_score=audit.get('task_score'),
            checks={'robot_geometry':audit['checks']['robot_geometry']},source_files_sha256=audit['source_files_sha256'],
            scope='Complete empty-hand native workflow with supported gun release; evaporation is outside scope.')
    expected=evaluate_native_cycle(metadata,rows)
    for name,check in expected['checks'].items():
        if report.get('checks',{}).get(name)!=check:raise ValueError('Independent check mismatch: '+name)
    if report['result']!=expected['result']:raise ValueError('Independent outcome differs.')
    dt=metadata['physics_dt_s'];substeps=round(.05/dt)
    commands=json.loads((run/'commands.json').read_text())
    cycle=[json.loads(line)for line in(run/'cycle_commands.jsonl').open()]
    actions=[c['target']for c in commands]+[c['action']for c in cycle]
    if len(rows)!=1+len(actions)*substeps:raise ValueError('Missing complete physics intervals.')
    for index,row in enumerate(rows[1:]):
        action=actions[index//substeps]
        targets=[*row['commanded_arm_q_rad'],row['commanded_finger_q_m'][0],
            *row['commanded_right_arm_q_rad'],row['commanded_right_finger_q_m'][0]]
        requested=np.array(action);requested[[6,13]]*=-.04695
        if not np.allclose(targets,requested,atol=1e-7,rtol=0):raise ValueError('Saved target buffers differ from issued actions.')
    return dict(result='INDEPENDENT_TRACE_AND_COMMANDS_MATCH',passed=expected['result']=='WAFER_AIR_PATH_AND_RETURN_SUCCEEDED_IN_THIS_SIMULATION',
        physics_samples=len(rows),control_actions=len(actions),trace_sha256=sha(run/'samples.jsonl.gz'),
        scope='Measured native simulation and geometric air coverage; not real drying or a probability guarantee.')


def export_video_replay(run,audit):
    report=json.loads((run/'trial_report.json').read_text())
    if (audit.get('passed')is not True or audit.get('result')!='INDEPENDENT_FULL_WORKFLOW_TRACE_AND_COMMANDS_MATCH'
            or report.get('result')!='FULL_WORKFLOW_SUCCEEDED_IN_THIS_SIMULATION' or report.get('errors')):
        raise ValueError('Video export requires an independently accepted full workflow.')
    task=report.get('task_score',{})
    if task.get('success')is not True or task.get('score')!=1. or audit.get('task_score')!=task:
        raise ValueError('Video export requires matching independent full task scores.')
    geometry=audit.get('checks',{}).get('robot_geometry',{})
    if (audit.get('native_result')!=report['result']or audit.get('matches_report')is not True
            or audit.get('evidence',{}).get('complete')is not True or geometry.get('passed')is not True
            or geometry.get('evidence_complete')is not True or geometry.get('result')!='CLEAR_IN_ALL_RECORDED_PHYSICS_SAMPLES'
            or audit.get('source_files_sha256')!={n:sha(run/n)for n in ('metadata.json','trial_report.json')}):
        raise ValueError('Video export requires the current complete native audit and geometry checks.')
    metadata=json.loads((run/'metadata.json').read_text());dt=float(metadata['physics_dt_s']);stride=round(.05/dt)
    if stride<1 or abs(stride*dt-.05)>1e-10:raise ValueError('Video cadence is incompatible with physics cadence.')
    output=run/'video_samples.jsonl.gz';last=None;selected=0;count=0
    with gzip.open(run/'samples.jsonl.gz','rt')as source,gzip.open(output,'xt',compresslevel=5)as target:
        for index,line in enumerate(source):
            row=json.loads(line);t=float(row['time_s'])
            if (last is None and abs(t)>1e-10)or(last is not None and abs(t-last-dt)>1e-8):
                raise ValueError('Source video trace is discontinuous.')
            if index%stride==0:target.write(line);selected+=1
            last=t;count+=1
    if count!=audit['physics_samples']or(count-1)%stride or sha(run/'samples.jsonl.gz')!=audit['trace_sha256']:
        raise ValueError('Source video trace differs from independent verification.')
    manifest=dict(version=1,selected_count=selected,physics_samples=count,sample_interval_s=.05,
        original_trace_sha256=audit['trace_sha256'],selected_trace_sha256=sha(output),
        source_files={name:sha(run/name)for name in ('metadata.json','trial_report.json','independent_check.json')},
        exporter_sha256=sha(Path(__file__)),source_start_s=0.,source_end_s=last,
        scope='Exact every-20Hz measurements from the independently accepted full physics trace; no interpolation or new simulation.')
    (run/'video_replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=run/'video_replay.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED)as package:
        for name in (*manifest['source_files'],'video_replay_manifest.json','video_samples.jsonl.gz'):
            package.write(run/name,name)
    return dict(trial=run.name,path=str(archive),sha256=sha(archive))


def run_batch(bundle,destination,full_cycle=False,device='cuda:0',development=False):
    if device not in ('cpu','cuda:0')or(device=='cpu'and not full_cycle):
        raise ValueError('CPU selection is available only for the complete-workflow adapter.')
    bundle=Path(bundle).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((bundle/'bundle_files.json').read_text())
    if development:
        import wafer_full_cycle_isaac as full_adapter
        full_adapter.ALLOW_UNQUALIFIED_DEVELOPMENT=True
    plan,_,_,_=validate_bundle(bundle)
    if full_cycle:
        # The adapter performs its complete gate and asset checks before physics.
        if not(bundle/'full_workflow_plan.json').is_file():raise ValueError('Missing qualified full-workflow plan.')
    result=dict(result='RUNNING',attempted=0,passed=0,requested_trials=5,trials=[],
        local_qualification_complete=plan['local_validation'].get('qualification_complete',False),
        full_workflow=full_cycle,physics_device=device,scope='Five fresh nominal cloud repeats; inspect the recorded local qualification status separately.')
    for index in range(1,6):
        while True:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if not active:break
            print('Waiting for shared GPU; no process is stopped.',flush=True);time.sleep(30)
        if any(sha(bundle/name)!=digest for name,digest in manifest.items()):raise ValueError('Frozen bundle changed.')
        path=destination/f'trial_{index:02d}'
        print(f'Starting complete cycle {index}/5',flush=True)
        with (destination/f'trial_{index:02d}.log').open('w')as log:
            script='wafer_full_cycle_isaac.py'if full_cycle else'wafer_cycle_isaac.py'
            command=[sys.executable,'-u',str(bundle/script),'--bundle',str(bundle),'--output',str(path)]
            if full_cycle:command.extend(['--device',device])
            if development:command.append('--development')
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((path/'trial_report.json').read_text())if(path/'trial_report.json').exists()else{'result':'INCOMPLETE'}
        try:audit=verify_run(path,full_cycle)
        except Exception as exc:audit=dict(result='INCOMPLETE',passed=False,error=str(exc))
        (path/'independent_check.json').write_text(json.dumps(audit,indent=2)+'\n')
        result['attempted']+=1;result['passed']+=int(audit.get('passed',False)and p.returncode==0)
        result['trials'].append(dict(name=path.name,result=report['result'],independent=audit,
            failed_checks=report.get('failed_checks'),error=report.get('error')))
        if full_cycle and audit.get('passed')and p.returncode==0 and 'video_replay'not in result:
            try:result['video_replay']=export_video_replay(path,audit)
            except Exception as exc:result['video_export_error']=type(exc).__name__+': '+str(exc)
        (destination/'batch_report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f"Completed {index}/5; passed {result['passed']}",flush=True)
        if report['result']=='INCOMPLETE':
            result['result']='STOPPED_ON_IMPLEMENTATION_OR_INFRASTRUCTURE_ERROR';break
        if development and (not audit.get('passed',False)or p.returncode!=0):
            result['result']='STOPPED_ON_DEVELOPMENT_FAILURE';break
    else:result['result']='CLOUD_FIVE_COMPLETED'
    result['observed_success_fraction']=result['passed']/result['attempted']if result['attempted']else None
    (destination/'batch_report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(destination/'verification.zip','w',compression=zipfile.ZIP_DEFLATED)as archive:
        archive.write(destination/'batch_report.json','batch_report.json')
        for item in result['trials']:
            path=destination/item['name']
            for name in ('metadata.json','trial_report.json','independent_check.json','cycle_commands.jsonl','tool_commands.jsonl','gun_return_commands.jsonl'):
                if(path/name).exists():archive.write(path/name,item['name']+'/'+name)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--full-cycle',action='store_true');p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0');p.add_argument('--development',action='store_true');a=p.parse_args()
    print(json.dumps(run_batch(a.bundle,a.output,a.full_cycle,a.device,a.development),indent=2))
