"""Isolated v12 Isaac runtime with three constant passive mechanical springs.

This is a new isolated scene, not a continuation of another user's environment.
The YAM arm starts at the wafer flat; tool pickup/transport and drying are out of
scope. Importing this module does not import Kit, pxr, torch, or Isaac Lab.
The independent evaluator reads recorded physics-step measurements only.
"""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
import zipfile
import numpy as np
from wafer_grasp_settings import (PHYSICS_DT_S, CONTROL_HZ, PAD_MOUNT_CANT_RAD,
    TCP_FEEDBACK_INTEGRATING_PHASES)
from wafer_grasp_protocol import evaluate_wafer_grasp_trial, _pose
from wafer_retention_jam_guard_v2 import (GUARD_PROFILE, FingerHandleJamStop,
    guard_metadata, guard_finger_handle_load, evaluate_finger_handle_load)

from wafer_passive_pad_guard_v6 import (PAD_GUARD_PROFILE, PadComplianceStop,
    guard_passive_pad_state, evaluate_passive_pad_state, pad_guard_metadata)
from wafer_hinge_guard_v11 import (HINGE_GUARD_PROFILE, HingeComplianceStop,
    guard_passive_hinge_state, evaluate_passive_hinge_state, hinge_guard_metadata)

PASS = 'WAFER_LIFT_RETAINED_IN_THIS_SIMULATION'
from wafer_finger_preload_v12 import (PHASES, PROFILE as FINGER_PRELOAD_PROFILE,
    FingerPreloadController, replay_commands, prelift_readiness, preceding_interval_features, wafer_prefix_phases)
# Operational capture budget only; physical timing and acceptance are unchanged.
TRIAL_WALL_BUDGET_S = 1800.
TRACE_STORAGE=dict(filename='samples.jsonl.gz',format='jsonl',compression='gzip',
    compression_level=1,encoding='utf-8',lossless=True,sample_decimation=1,
    capture_order='Append in-memory row and serialize every physics update before applying the unchanged sample guards.')
EXPECTED_RETENTION_REVISION='parallel_pad_free_closure_candidate'
EXPECTED_GUIDE_PROFILE=dict(handle_width_m=.047,handle_lateral_center_m=.008,
    front_inner_y_m=-.0134,rear_inner_y_m=.029,rail_length_x_m=.024,
    rail_full_height_length_x_m=.020,rail_thickness_y_m=.002,rail_height_m=.004,
    guide_end_height_m=0.)
CORE_PASSIVE_JOINTS=('pliers_hinge','lower_pad_compression','upper_pad_compression')
FULL_PADDLE_JOINTS=('lower_grip_paddle_pitch','upper_grip_paddle_pitch')


def verified_tool_joint_count(springs):
    names=[s['joint_name']for s in springs]
    if len(names)!=len(set(names))or set(names)not in (set(CORE_PASSIVE_JOINTS),set(CORE_PASSIVE_JOINTS+FULL_PADDLE_JOINTS)):
        raise ValueError('Expected three core springs and, only for full workflow, two named passive paddles.')
    return len(names)


class TrialAbort(RuntimeError):
    """A measured physical guard failed; distinguish this from missing APIs."""


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _array(value, shape=None, name='measurement'):
    value = value.torch if hasattr(value, 'torch') else value
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    out = np.asarray(value)
    if out.dtype.kind not in 'fiu' or not np.isfinite(out).all() or (shape is not None and out.shape != shape):
        raise ValueError('Invalid '+name+' shape or finite numeric values.')
    return out.astype(float, copy=True)


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cloud_model_metadata(source,plan):
    """Keep geometry assumptions but separate the unexecuted carriage model."""
    metadata=copy.deepcopy(source);fixture={}
    for key in ('control_actuator_names','control_actuator_ids','actuator_position_units','fixture_lift_force_limit_N',
                'open_finger_target_m','closed_finger_target_m'):
        if key in metadata:fixture[key]=metadata.pop(key)
    old=metadata.get('draft_assumptions',[])
    fixture['draft_assumptions']=copy.deepcopy(old)
    fixture['scope']='Source carriage-fixture metadata only. These actuators/force limits are NOT used in this YAM cloud scene.'
    metadata['source_fixture_only_metadata']=fixture
    metadata['draft_assumptions']=[s for s in old if 'carriage' not in s.lower()]
    metadata['draft_assumptions'] += [
        'Actual YAM articulations execute joint targets; no prescribed lifting carriage exists in this scene.',
        'The internal clamp rest target is fixed once and represents a mechanical opening spring, not a powered closing jaw.',
        'MuJoCo body gravity compensation and Isaac YAM feed-forward are not verified dynamic equivalents.']
    metadata['open_finger_target_m']=float(plan['finger_open_q_m'])
    metadata['closed_finger_target_m']=float(plan['finger_closed_q_m'])
    metadata['closed_finger_target_semantics']='Legacy field is the maximum permitted closure target, not a prescribed final target; measured preload feedback selects the actual common target at each20Hz command.'
    metadata['initial_finger_position_m']=float(plan.get('initial_finger_state_q_m',plan['initial_finger_q_m']))
    metadata['control_actuator_names']=[side+'/'+group for side in ('left_robot','right_robot')
        for group in ('shoulder','wrist','gripper')]+['clamp/'+x['joint_name']+'_spring' for x in source['usd_export']['passive_springs']]
    metadata['actuator_identifier_scope']='Named Isaac actuator groups; measured joint indices are reported per articulation, not as invented global actuator IDs.'
    metadata['action_layout']=dict(left_arm_rad=[0,1,2,3,4,5],left_gripper_normalized=6,
                                   right_arm_rad=[7,8,9,10,11,12],right_gripper_normalized=13)
    metadata['retention_disturbance']=dict(enabled=False,profile='none',
        wrench_world_N_Nm=[0.]*6,post_start_pose_write_count=0,
        scope='Baseline transfer only; development external-wrench probes are excluded.')
    metadata['finger_handle_jam_guard']=guard_metadata('isaac')
    metadata['passive_pad_guard']=pad_guard_metadata()
    metadata['passive_hinge_guard']=hinge_guard_metadata()
    return metadata


def _source_file(filename):
    """Hash the actually imported isolated helper or immutable shared module.

    Development helpers live one directory below the frozen protocol/controller.
    The final minimal bundle places the same modules beside this runtime.
    """
    import importlib
    allowed={'wafer_retention_isaac_v12.py','wafer_retention_usd_v12.py','wafer_retention_checks_v11.py','wafer_retention_checks_v6.py','wafer_hinge_guard_v11.py',
             'wafer_grasp_protocol.py','wafer_grasp_feedback.py','wafer_retention_jam_guard_v2.py',
             'wafer_retention_checks.py','wafer_retention_checks_v2.py','wafer_passive_pad_guard_v6.py','wafer_grasp_settings.py','wafer_finger_preload_v12.py'}
    if filename not in allowed:raise ValueError('Unexpected runtime helper source.')
    if filename=='wafer_retention_isaac_v12.py':return Path(__file__).resolve()
    module=importlib.import_module(filename[:-3])
    return Path(module.__file__).resolve()


def _bundle_file(root, relative, digest=None):
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Bundle file is missing or outside this bundle: '+str(relative))
    if digest is not None and _sha(path) != digest: raise ValueError('Bundle hash mismatch: '+str(relative))
    return path


def _validation_json(root,relative,digest):
    if '!' not in relative:return json.loads(_bundle_file(root,relative,digest).read_text())
    archive,member=relative.split('!',1)
    if archive!='local_validation.zip' or member.startswith('/') or '..' in Path(member).parts:
        raise ValueError('Invalid local validation archive member.')
    with zipfile.ZipFile(_bundle_file(root,archive)) as records:data=records.read(member)
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Local validation member hash differs.')
    return json.loads(data)


def _validate_preload_bound_identity(initial,opening,closed):
    """Ignore sub-picometre JSON rounding, never a materially different bound."""
    expected=(FINGER_PRELOAD_PROFILE['target_lower_bound_m'],
              FINGER_PRELOAD_PROFILE['target_lower_bound_m'],FINGER_PRELOAD_PROFILE['target_upper_bound_m'])
    if not all(math.isclose(actual,wanted,rel_tol=0.,abs_tol=1e-12)
               for actual,wanted in zip((initial,opening,closed),expected)):
        raise ValueError('Native initial/open/maximum finger targets differ from shared controller bounds.')


def validate_bundle(root):
    """Read-only gate for the declared fixed local batch and unchanged criteria."""
    root = Path(root).resolve(); plan = json.loads((root/'cloud_plan.json').read_text())
    manifest_path = _bundle_file(root, plan['export_manifest']['path'], plan['export_manifest']['sha256'])
    exported = json.loads(manifest_path.read_text()); metadata = exported['metadata']
    if (metadata.get('mechanism_source_module')!='wafer_retention_candidate_v12'
            or metadata.get('usd_export',{}).get('actuation',{}).get('mode')!='constant_opening_spring'):
        raise ValueError('User requested a purely mechanical clamp. Powered-jaw bundles are not permitted for this cloud trial.')
    if (metadata.get('mechanism_revision')!=EXPECTED_RETENTION_REVISION or
            metadata.get('pad_mount_cant_rad')!=PAD_MOUNT_CANT_RAD or
            metadata.get('retention_candidate',{}).get('profile')!=EXPECTED_GUIDE_PROFILE):
        raise ValueError('The bundle does not contain the reviewed compliant-pad v12 guide profile.')
    from wafer_retention_usd_v12 import usd_drive_specification,verify_passive_spring_specifications
    verify_passive_spring_specifications(exported)
    drive=metadata['usd_export']['actuation']
    if drive.get('authored_usd_drive') is not True or drive.get('usd_drive')!=usd_drive_specification(drive):
        raise ValueError('Bundle lacks the explicit force DriveAPI with degree-correct passive spring parameters.')
    if _sha(_source_file('wafer_grasp_protocol.py')) != plan['protocol_sha256']:
        raise ValueError('The evaluator differs from the frozen bundle.')
    if _sha(_source_file('wafer_retention_checks_v11.py')) != plan['yam_checks_sha256']:
        raise ValueError('The robot-contact evaluator differs from the frozen bundle.')
    if _sha(_source_file('wafer_retention_checks_v6.py'))!=plan.get('pad_yam_checks_sha256'):
        raise ValueError('The delegated pad/contact evaluator differs from the frozen bundle.')
    if (_sha(_source_file('wafer_hinge_guard_v11.py'))!=plan.get('hinge_guard_sha256') or
            plan.get('hinge_guard_profile')!=HINGE_GUARD_PROFILE):
        raise ValueError('The prospective raw hinge-torque diagnostic differs.')
    if (_sha(_source_file('wafer_finger_preload_v12.py'))!=plan.get('finger_preload_controller_sha256') or
            plan.get('finger_preload',{}).get('enabled') is not True or
            plan['finger_preload'].get('profile')!=FINGER_PRELOAD_PROFILE):
        raise ValueError('The shared scalar finger-preload controller differs from the frozen local source/profile.')
    if _sha(_source_file('wafer_retention_checks_v2.py'))!=plan.get('previous_yam_checks_sha256'):
        raise ValueError('The delegated v2 contact/jam evaluator differs.')
    if (plan.get('pad_guard_profile')!=PAD_GUARD_PROFILE or _sha(_source_file('wafer_passive_pad_guard_v6.py'))!=plan.get('pad_guard_sha256')):
        raise ValueError('The prospective passive-pad diagnostic differs.')
    if _sha(_source_file('wafer_retention_checks.py'))!=plan.get('base_yam_checks_sha256'):
        raise ValueError('The delegated base robot-contact evaluator differs from the frozen bundle.')
    if (plan.get('jam_guard_profile')!=GUARD_PROFILE or
            _sha(_source_file('wafer_retention_jam_guard_v2.py'))!=plan.get('jam_guard_sha256')):
        raise ValueError('The declared v2 finger-load diagnostic differs from the frozen bundle.')
    from wafer_grasp_feedback import PROFILE as FEEDBACK_PROFILE
    if (not isinstance(plan['tcp_feedback'].get('enabled'),bool)
            or plan['tcp_feedback']['profile']!=FEEDBACK_PROFILE
            or plan['tcp_feedback']['integrating_phases']!=list(TCP_FEEDBACK_INTEGRATING_PHASES)+(['lift_hold']if plan.get('cycle_layout')else[])
            or _sha(_source_file('wafer_grasp_feedback.py'))!=plan['tcp_feedback_sha256']):
        raise ValueError('Bounded TCP controller differs from the frozen local implementation.')
    if (_sha(_source_file('wafer_grasp_settings.py')) != plan.get('settings_sha256')
            or metadata['model_dependency_sha256'].get('settings') != plan['settings_sha256']):
        raise ValueError('Local and cloud shared settings differ.')
    if metadata['model_source_sha256'] != plan['model_source_sha256']:
        raise ValueError('Export and local model fingerprints differ.')
    if metadata['model_dependency_sha256'] != plan['model_dependency_sha256']:
        raise ValueError('Plan and exported geometry dependencies differ.')
    for key in ('adapter_sha256','robot_bridge_sha256'):
        digest=plan.get(key)
        if (not isinstance(digest,str) or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Missing or malformed frozen local source hash: '+key)
    models = {}
    for name, item in plan['model_files'].items(): models[name] = _bundle_file(root, item['path'], item['sha256'])
    if set(models) != {'clamp','wafer','heater','pillar_0','pillar_1','pillar_2','pedestal'}:
        raise ValueError('Expected the free clamp/wafer and five independent supports.')
    full_development=plan.get('local_validation',{}).get('policy')=='user_requested_full_development_v1'
    full_policy=full_development or plan.get('local_validation',{}).get('policy')=='full_workflow_fixed20_v1'
    if full_policy:
        from wafer_full_cycle_isaac import validate_full_workflow_evidence
        full_spec,full_nominal,full_gate=validate_full_workflow_evidence(root,plan,metadata)
        trials=full_spec['local_gate']['cases'];passed=full_gate['passed'];evidence=full_gate['evidence'];early_cloud=full_development
        validation=plan['local_validation'];selected_passed=True
    else:
        trials = plan['local_validation']['trials']; passed = 0; evidence = [];selected_passed=False
        validation=plan['local_validation']
        early_cloud=validation.get('policy')=='user_requested_early_cloud_v1'
        if validation.get('policy')=='fixed20_v1':
            if len(trials)!=20 or validation.get('required_count')!=20 or validation.get('minimum_passed')!=16:
                raise ValueError('Exactly20 local cases with at least16 passes are required.')
            archive=_bundle_file(root,'local_validation.zip',validation['archive_sha256'])
            expected_members={item[k].split('!',1)[1] for item in trials for k in ('report','metadata')}
            with zipfile.ZipFile(archive) as records:
                if len(records.namelist())!=40 or set(records.namelist())!=expected_members:
                    raise ValueError('The compact local evidence archive must contain exactly40 case records.')
        elif early_cloud:
            if (validation.get('qualification_complete')is not False or len(trials)!=validation.get('completed_count')
                    or not trials or not validation.get('user_steering')):
                raise ValueError('Incomplete early-cloud provenance.')
            archive=_bundle_file(root,'local_validation.zip',validation['archive_sha256'])
            expected_members={item[k].split('!',1)[1]for item in trials for k in ('report','metadata')}
            with zipfile.ZipFile(archive)as records:
                if len(records.namelist())!=2*len(trials)or set(records.namelist())!=expected_members:
                    raise ValueError('Early-cloud evidence archive differs from completed cases.')
        elif 'policy' in validation:raise ValueError('Unknown local validation policy.')
        elif len(trials)<5:raise ValueError('Legacy evidence contains fewer than five cases.')
        fingerprints = set();case_names=[]
        for item in trials:
            report=_validation_json(root,item['report'],item['report_sha256'])
            local=_validation_json(root,item['metadata'],item['metadata_sha256'])
            case_names.append(local.get('scenario',{}).get('name'))
            disturbance=local.get('retention_disturbance',{})
            if (disturbance.get('enabled') is not False or disturbance.get('profile')!='none'
                    or disturbance.get('wrench_world_N_Nm')!=[0.]*6
                    or disturbance.get('post_start_pose_write_count')!=0):
                raise ValueError('Local transfer gate excludes disturbance probes and post-start pose writes.')
            source = local.get('source_sha256', {}).get('mechanism', local.get('model_source_sha256'))
            evaluator = local.get('source_sha256', {}).get('protocol', local.get('protocol_source_sha256'))
            if source != plan['model_source_sha256'] or evaluator != plan['protocol_sha256']:
                raise ValueError('A local run used another mechanism or evaluator revision.')
            if local.get('source_sha256',{}).get('yam_checks') != plan['yam_checks_sha256']:
                raise ValueError('A local run used another robot-contact evaluator revision.')
            if local.get('source_sha256',{}).get('pad_yam_checks')!=plan['pad_yam_checks_sha256']:
                raise ValueError('A local run used another delegated pad/contact evaluator.')
            if (local.get('source_sha256',{}).get('hinge_guard')!=plan['hinge_guard_sha256'] or
                    local.get('passive_hinge_guard')!=hinge_guard_metadata() or
                    local.get('passive_hinge')!=metadata.get('passive_hinge')):
                raise ValueError('A local run lacks the same raw hinge-torque diagnostic/profile.')
            hinge_check=report.get('checks',{}).get('passive_hinge_torque_guard',{})
            if (hinge_check.get('profile')!=HINGE_GUARD_PROFILE or
                    hinge_check.get('sample_count')!=report.get('trace',{}).get('sample_count') or
                    (report.get('result')==PASS and hinge_check.get('passed') is not True)):
                raise ValueError('Local run lacks consistent full-rate hinge-torque evidence.')
            if local.get('source_sha256',{}).get('parent_yam_checks')!=plan['previous_yam_checks_sha256']:
                raise ValueError('A local run used another delegated v2 contact/jam evaluator.')
            if local.get('source_sha256',{}).get('pad_guard')!=plan['pad_guard_sha256'] or local.get('passive_pad_guard')!=pad_guard_metadata():
                raise ValueError('A local run lacks the same passive-pad diagnostic.')
            pad_check=report.get('checks',{}).get('passive_pad_motion_guard',{})
            if (pad_check.get('profile')!=PAD_GUARD_PROFILE or
                    pad_check.get('sample_count')!=report.get('trace',{}).get('sample_count') or
                    (report.get('result')==PASS and pad_check.get('passed') is not True)):
                raise ValueError('Local run lacks consistent saved-row passive-pad diagnostic evidence.')
            if local.get('pad_compliance')!=metadata.get('pad_compliance'):
                raise ValueError('Local and exported passive pad SI profiles differ.')
            if local.get('source_sha256',{}).get('base_yam_checks')!=plan['base_yam_checks_sha256']:
                raise ValueError('A local run used another delegated base contact evaluator.')
            if (local.get('source_sha256',{}).get('jam_guard')!=plan['jam_guard_sha256'] or
                    local.get('finger_handle_jam_guard')!=guard_metadata('local')):
                raise ValueError('A local run lacks the same prospective v2 finger-load diagnostic.')
            jam=report.get('checks',{}).get('finger_handle_jam_guard',{})
            if (jam.get('profile')!=GUARD_PROFILE or
                    jam.get('measurement_field')!=guard_metadata('local')['measurement_field'] or
                    jam.get('sample_count')!=report.get('trace',{}).get('sample_count') or
                    (report.get('result')==PASS and jam.get('passed') is not True)):
                raise ValueError('A local report lacks consistent saved-row jam-diagnostic evidence.')
            for key in ('adapter','robot_bridge'):
                if local.get('source_sha256',{}).get(key)!=plan[key+'_sha256']:
                    raise ValueError('A local run used another frozen '+key+' revision.')
            if local.get('model_dependency_sha256')!=metadata['model_dependency_sha256']:
                raise ValueError('Local run and exported geometry dependencies differ.')
            if local.get('engine') != 'MuJoCo' or not str(local.get('mujoco_version','')).startswith('3.'):
                raise ValueError('Local gate requires actual recorded MuJoCo runs, not synthetic fixtures.')
            if local.get('physics_dt_s')!=metadata['physics_dt_s'] or report.get('trace',{}).get('physics_dt_s')!=metadata['physics_dt_s']:
                raise ValueError('Local evidence and selected native full-step cadence differ.')
            if report.get('trace', {}).get('sample_count',0) < 1000:
                raise ValueError('Local run has insufficient full-step trace evidence.')
            if report.get('result')==PASS and report['trace']['sample_count']!=1+round(sum(d for _,d in PHASES)/metadata['physics_dt_s']):
                raise ValueError('Passing local run lacks the complete shared phase schedule.')
            if report.get('result')==PASS:
                ready=report.get('prelift_readiness',{});replay=report.get('finger_preload_command_replay',{})
                if (ready.get('passed') is not True or ready.get('result')!='READY_FOR_LIFT' or
                        ready.get('protocol_sha256')!=plan['protocol_sha256']):
                    raise ValueError('Passing local run lacks the original-criterion prelift transition evidence.')
                if (replay.get('complete') is not True or replay.get('profile')!=FINGER_PRELOAD_PROFILE or
                        replay.get('command_count')!=round(sum(d for _,d in PHASES)*20)):
                    raise ValueError('Passing local run lacks complete shared finger-controller replay evidence.')
            if local.get('robot_collider_count')!=315 or local.get('arm_gains')!=plan['arm_gains']:
                raise ValueError('Local gate requires the same actual-YAM geometry and compiled drive gains.')
            if plan.get('cycle_layout'):
                if local.get('right_arm_gains')!=plan.get('right_arm_gains'):
                    raise ValueError('Local/native right-arm drive gains differ.')
                for key in ('cycle_layout','cycle_handle_remap'):
                    if local.get(key)!=metadata.get(key):raise ValueError('Local/native cycle geometry differs: '+key)
                if local.get('initial_finger_q_m')!=[plan['initial_finger_state_q_m']]*2:
                    raise ValueError('Local/native initial finger state differs.')
            if (local.get('controller',{}).get('tcp_feedback')!=plan['tcp_feedback']
                    or local.get('source_sha256',{}).get('tcp_feedback')!=plan['tcp_feedback_sha256']):
                raise ValueError('A local case used another feedback controller/profile.')
            if (local.get('controller',{}).get('finger_preload')!=plan['finger_preload'] or
                    local.get('source_sha256',{}).get('finger_preload')!=plan['finger_preload_controller_sha256']):
                raise ValueError('A local case used another scalar finger-preload controller/profile.')
            signature = (item['report_sha256'], item['metadata_sha256'])
            if signature in fingerprints: raise ValueError('Duplicated local evidence is not another trial.')
            fingerprints.add(signature)
            passed += report.get('result') == PASS
            if local.get('controller',{}).get('phases')!=[list(p) for p in PHASES]:
                raise ValueError('A local run used another phase schedule.')
            if local.get('scenario')==metadata['scenario'] and report.get('result')==PASS:
                for plan_key,local_key in [('initial_left_q_rad','initial_arm_q_rad'),
                        ('seated_left_q_rad','seated_arm_q_rad'),('final_left_q_rad','final_arm_q_rad'),
                        ('finger_open_q_m','open_finger_target_m'),('finger_closed_q_m','closed_finger_target_m')]:
                    if plan[plan_key]!=local[local_key]:
                        raise ValueError('Selected nominal command differs from its passing local run: '+plan_key)
                selected_passed=True
            evidence.append(dict(result=report.get('result'), scenario=local.get('scenario'),
                                 report_sha256=item['report_sha256'], metadata_sha256=item['metadata_sha256']))
        if validation.get('policy')=='fixed20_v1':
            expected_names=['nominal','offset_positive','offset_negative','lower_friction','heavier_wafer']+[f'combined_{i:02d}' for i in range(1,16)]
            if case_names!=expected_names:raise ValueError('Required20 local cases differ or were reordered.')
        if not early_cloud and passed/len(trials) < .8: raise ValueError('Fewer than 80% of the frozen local trials passed.')
        if not selected_passed:raise ValueError('The selected cloud scenario did not itself pass the frozen local actual-YAM test.')
    dt = float(metadata['physics_dt_s']); decimation = round(.05/dt)
    if dt!=FINGER_PRELOAD_PROFILE['physics_dt_s'] or dt!=PHYSICS_DT_S or decimation!=round(1/(CONTROL_HZ*PHYSICS_DT_S)) or abs(decimation*dt-.05)>1e-12:
        raise ValueError('Unsupported full-step cadence at 20 Hz control.')
    for key in ('initial_left_q_rad','seated_left_q_rad','final_left_q_rad','right_q_rad'):
        _array(plan[key],(6,),key)
    opening = float(plan['finger_open_q_m']); closed = float(plan['finger_closed_q_m'])
    initial=float(plan['initial_finger_q_m']);actuation=metadata['usd_export']['actuation']
    _validate_preload_bound_identity(initial,opening,closed)
    expected_initial_state=full_spec['tool_pickup']['profile']['open_finger_m']if full_policy else-.026
    if plan.get('cycle_layout')and plan.get('initial_finger_state_q_m')!=expected_initial_state:
        raise ValueError('Unexpected left-side initial open-finger state.')
    if not -.047<initial<0 or not -.047 < opening <= closed < 0 or closed-opening > .006:
        raise ValueError('Invalid bounded two-finger closure targets.')
    if plan['mechanism_source_module']!=metadata['mechanism_source_module']:
        raise ValueError('Requested mechanism and exported assets disagree.')
    if actuation['mode']=='powered_position':
        if opening!=closed:raise ValueError('Powered clamp holds the robot fingers fixed; only its separate jaw closes.')
        low,high=actuation['target_limits']
        if not low<=plan['jaw_open_target_m']<plan['jaw_closed_target_m']<=high:
            raise ValueError('Powered-jaw targets exceed the exported actuator range.')
    elif actuation['mode']!='constant_opening_spring':raise ValueError('Unsupported mechanism actuator mode.')
    expected_phases=wafer_prefix_phases(full_nominal)if full_policy else PHASES
    if plan.get('control_hz') != 20 or plan.get('phases') != [list(p) for p in expected_phases]:
        raise ValueError('Cloud controller differs from the frozen 20 Hz phase schedule.')
    _array(plan['world_translation_m'],(3,),'world translation')
    if plan.get('world_yaw_deg') not in (90,270) or plan.get('pedestal_height_m') != .025:
        raise ValueError('Unsupported frozen fixture layout.')
    if plan['world_yaw_deg']==270:
        from wafer_cycle_protocol import LAYOUT
        if metadata.get('cycle_layout')!=LAYOUT or plan.get('cycle_layout')!=LAYOUT:
            raise ValueError('Left-side geometry and source layout differ.')
        if plan['world_translation_m'][:2]!=LAYOUT['center_xy_m']:
            raise ValueError('Left-side work-station location differs.')
    return plan, exported, models, dict(status='UNQUALIFIED_DEVELOPMENT_CONFIGURATION_CHECKED'if full_development else'FULL_WORKFLOW_FIXED20_VERIFIED'if full_policy else'EARLY_CLOUD_EVIDENCE_VERIFIED'if early_cloud else'VERIFIED',
        qualification_complete=not early_cloud,count=len(trials),passed=passed,
        success_fraction=passed/len(trials)if trials else None,selected_scenario_passed=not full_development,evidence=evidence,physics_dt_s=dt,decimation=decimation,
        scope='Local physical evidence gate, not proof of transfer to PhysX or real hardware.')


def command_schedule(plan):
    """Exactly the local YAM adapter's cubic joint targets, held for 50 ms."""
    q0 = _array(plan['initial_left_q_rad'],(6,)); q1 = _array(plan['final_left_q_rad'],(6,))
    seated=_array(plan.get('seated_left_q_rad',plan['initial_left_q_rad']),(6,))
    right = _array(plan['right_q_rad'],(6,)); opening = float(plan['finger_open_q_m'])
    closed = float(plan['finger_closed_q_m']); denominator = float(plan['native_finger_open_q_m'])
    if not denominator < opening <= closed < 0: raise ValueError('Invalid native gripper normalization.')
    for phase, duration in plan.get('phases',PHASES):
        for index in range(round(duration*20)):
            fraction = (index+1)/(duration*20); u = fraction*fraction*(3-2*fraction)
            q = q0 if phase=='settle' else q0+(seated-q0)*u if phase=='close' else seated if phase=='grip_hold' else seated+(q1-seated)*u if phase=='lift' else q1
            finger = opening if phase=='settle' else opening+u*(closed-opening) if phase=='close' else closed
            yield phase, np.r_[q,finger/denominator,right,1.].tolist()


def mechanism_target_schedule(plan,actuation):
    """Explicit independent jaw motor, or a constant mechanical spring rest."""
    for phase,duration in plan.get('phases',PHASES):
        for index in range(round(duration*20)):
            if actuation['mode']=='constant_opening_spring':yield float(actuation['constant_target']);continue
            if actuation['mode']!='powered_position':raise ValueError('Unknown tool actuator mode.')
            fraction=(index+1)/(duration*20);u=fraction*fraction*(3-2*fraction)
            opening=float(plan['jaw_open_target_m']);closed=float(plan['jaw_closed_target_m'])
            yield opening if phase=='settle' else opening+(closed-opening)*u if phase=='close' else closed


def apply_arm_gains(left_config,records):
    """Match frozen local K/D and force caps while retaining native speeds."""
    if [x.get('joint') for x in records]!=['joint'+str(i) for i in range(1,9)]:
        raise ValueError('Expected exactly eight ordered local YAM gain records.')
    groups={'shoulder':records[:3],'wrist':records[3:6],'gripper':records[6:]};out={}
    for name,rows in groups.items():
        first=rows[0];force=_array(first['force_range'],(2,),'local force range')
        expected_cap={'shoulder':27.,'wrist':7.,'gripper':5.}[name]
        expected_unit='N' if name=='gripper' else 'Nm'
        if not np.allclose(force,[-expected_cap,expected_cap],atol=1e-10,rtol=0):
            raise ValueError('Unexpected local YAM torque/force limits.')
        if any((r['stiffness'],r['damping'],r['force_range'],r['force_unit'])!=
               (first['stiffness'],first['damping'],first['force_range'],expected_unit) for r in rows):
            raise ValueError('Group drive values differ or use unexpected units.')
        k=float(first['stiffness']);d=float(first['damping'])
        if not math.isfinite(k+d) or k<=0 or d<0:raise ValueError('Invalid local actuator gains.')
        cfg=left_config.actuators[name];before={key:getattr(cfg,key) for key in ('stiffness','damping','joint_effort_limit','joint_velocity_limit')}
        if name!='gripper' and abs(float(cfg.joint_effort_limit)-expected_cap)>1e-10:
            raise ValueError('Native YAM effort limit changed.')
        out[name]=dict(joint_names=[r['joint'] for r in rows],before=before,
            applied=dict(stiffness=k,damping=d,joint_effort_limit=expected_cap,joint_velocity_limit=before['joint_velocity_limit']))
    # Validate all groups before changing any new-scene settings.
    for name,row in out.items():
        cfg=left_config.actuators[name]
        for key,value in row['applied'].items():setattr(cfg,key,value)
    return out


def robot_collision_prims(root,expected_count=315):
    """Require the source collision inventory before authoring new-scene data."""
    from pxr import Usd,UsdPhysics
    if not root or not root.IsValid():raise RuntimeError('Robot spawn did not return a valid prim.')
    colliders=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders)!=expected_count or any(p.IsInstanceProxy() for p in colliders):
        raise RuntimeError('Robot collider inventory differs or is still instanced.')
    if any(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not True for p in colliders):
        raise RuntimeError('A source robot collider is disabled.')
    return colliders


def verify_robot_material(root,expected_count=315):
    """Read composed physics-purpose bindings; visual bindings are independent."""
    from pxr import UsdShade,UsdPhysics
    colliders=robot_collision_prims(root,expected_count)
    path=str(root.GetPath())+'/WaferTrialPhysicsMaterial';values=[]
    for collider in colliders:
        material,_=UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        if not material or str(material.GetPath())!=path:
            raise RuntimeError('Robot physics material was shadowed: '+str(collider.GetPath()))
        api=UsdPhysics.MaterialAPI(material.GetPrim())
        row=[api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),api.GetRestitutionAttr().Get()]
        if not np.allclose(_array(row,(3,)),[.6,.6,0.],atol=1e-7,rtol=0):
            raise RuntimeError('Robot friction/restitution differs from the local model.')
        if material.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get()!='max':
            raise RuntimeError('Robot friction-combine mode is not max.')
        values.append(row)
    return dict(status='COMPOSED_BINDINGS_VERIFIED',root_path=str(root.GetPath()),
        collider_count=len(colliders),material_path=path,static_friction=.6,dynamic_friction=.6,restitution=0.,
        friction_combine_mode='max')


def match_robot_material(root,expected_count=315):
    """Author only the freshly spawned scene, before any PhysX shape exists.

    USD physics-purpose binding is separate from preview/all-purpose materials:
    https://openusd.org/dev/api/class_usd_shade_material_binding_a_p_i.html
    """
    from pxr import UsdShade,UsdPhysics,Sdf
    colliders=robot_collision_prims(root,expected_count);native={}
    for collider in colliders:
        material,_=UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        path=str(material.GetPath()) if material else 'UNBOUND'
        native[path]=native.get(path,0)+1
    path=str(root.GetPath())+'/WaferTrialPhysicsMaterial'
    if root.GetStage().GetPrimAtPath(path):raise RuntimeError('Trial material path already exists.')
    material=UsdShade.Material.Define(root.GetStage(),path);prim=material.GetPrim()
    api=UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(.6);api.CreateDynamicFrictionAttr(.6);api.CreateRestitutionAttr(0.)
    # Author the documented schema token so CPU-only USD audits do not require
    # NVIDIA's schema plugin. PhysX resolves it when the scene is instantiated.
    prim.AddAppliedSchema('PhysxMaterialAPI')
    prim.CreateAttribute('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token).Set('max')
    prim.CreateAttribute('physxMaterial:restitutionCombineMode',Sdf.ValueTypeNames.Token).Set('max')
    for collider in colliders:
        if not UsdShade.MaterialBindingAPI.Apply(collider).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics'):
            raise RuntimeError('Could not bind robot physics material.')
    result=verify_robot_material(root,expected_count)
    result.update(native_resolved_material_counts=native,authored_during='Robot spawn, before PhysX initialization',
        scope='Sliding coefficients matched. MuJoCo and PhysX friction solvers are not asserted equivalent.')
    return result


def install_robot_material_match(spawn_config):
    """Wrap one new config's existing spawn callback without touching source USD."""
    original=spawn_config.func;state={'status':'NOT_SPAWNED','callback_count':0}
    def spawn(*args,**kwargs):
        state['callback_count']+=1
        if state['callback_count']!=1:raise RuntimeError('Expected one robot spawn in the isolated environment.')
        root=original(*args,**kwargs)
        state.update(match_robot_material(root));return root
    spawn_config.func=spawn
    return state


def verify_live_robot_material(asset,expected_count=315):
    """Read PhysX-created shapes, not merely authored USD material attributes.

    ArticulationView.get_material_properties returns static/dynamic/restitution:
    https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html
    """
    for name in ('root_view','root_physx_view'):
        view=getattr(asset,name,None)
        getter=getattr(view,'get_material_properties',None)
        if callable(getter):break
    else:raise RuntimeError('Public live robot material readback is unavailable; no motion allowed.')
    raw=getter()
    if hasattr(raw,'numpy') and not hasattr(raw,'detach'):raw=raw.numpy()
    values=_array(raw,(1,expected_count,3),'live robot material properties')[0]
    if not np.allclose(values,np.array([.6,.6,0.]),atol=1e-6,rtol=0):
        raise RuntimeError('Live PhysX robot friction/restitution differs from the matched source.')
    return dict(status='LIVE_PHYSX_SHAPES_VERIFIED',source=name+'.get_material_properties',
        shape_count=len(values),minimum=values.min(axis=0).tolist(),maximum=values.max(axis=0).tolist(),
        column_order=['static_friction','dynamic_friction','restitution'])


def _shape_materials(root):
    from pxr import Usd,UsdPhysics,UsdShade
    result={}
    for collider in root.GetStage().Traverse(Usd.TraverseInstanceProxies()):
        if not collider.GetPath().HasPrefix(root.GetPath()) or not collider.HasAPI(UsdPhysics.CollisionAPI):continue
        material,_=UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        if not material:raise RuntimeError('Missing physics material: '+str(collider.GetPath()))
        api=UsdPhysics.MaterialAPI(material.GetPrim())
        values=[api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),api.GetRestitutionAttr().Get()]
        prim=material.GetPrim()
        stiffness=float(prim.GetAttribute('physxMaterial:compliantContactStiffness').Get()or 0.)
        damping=float(prim.GetAttribute('physxMaterial:compliantContactDamping').Get()or 0.)
        acceleration=bool(prim.GetAttribute('physxMaterial:compliantContactAccelerationSpring').Get()or False)
        if stiffness>0:values[2]=-stiffness
        if material.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get()!='max':
            raise RuntimeError('Generated asset friction-combine mode differs.')
        name=collider.GetName()
        if name in result:raise RuntimeError('Ambiguous generated collider name: '+name)
        result[name]=dict(material=_array(values,(3,),'resolved material').tolist(),compliance=[stiffness,damping,float(acceleration)],
            contact_offset=float(collider.GetAttribute('physxCollision:contactOffset').Get()),
            rest_offset=float(collider.GetAttribute('physxCollision:restOffset').Get()))
    if not result:raise RuntimeError('Generated asset has no collision shapes.')
    return result


def verify_live_generated_materials(stage,models,assets,spawned):
    """Compare source USD, composed bindings, and live shape properties before motion."""
    from pxr import Usd
    result={}
    for name in ('clamp','wafer'):
        source=Usd.Stage.Open(str(models[name]));expected=_shape_materials(source.GetDefaultPrim())
        path=spawned['bodies'][name][0]
        root=stage.GetPrimAtPath(str(Path(path).parent) if name=='clamp' else path)
        actual=_shape_materials(root)
        if set(actual)!=set(expected) or any(not np.allclose(actual[n][k],expected[n][k],rtol=0,atol=1e-7)
                for n in expected for k in expected[n]):
            raise RuntimeError('Composed generated materials differ from source: '+name)
        for view_name in ('root_view','root_physx_view'):
            view=getattr(assets[name],view_name,None)
            if callable(getattr(view,'get_material_properties',None)):break
        else:raise RuntimeError('Live generated material readback unavailable: '+name)
        record={};count=len(expected)
        for key,getter,width in [('material','get_material_properties',3),
                                 ('contact_offset','get_contact_offsets',None),('rest_offset','get_rest_offsets',None)]:
            raw=getattr(view,getter)()
            if hasattr(raw,'numpy') and not hasattr(raw,'detach'):raw=raw.numpy()
            shape=(1,count,3) if width else (1,count)
            values=_array(raw,shape,'live '+name+' '+key)[0]
            wanted=np.asarray([v[key] for v in expected.values()])
            ordered=lambda a:np.asarray(sorted(a.tolist()))
            if not np.allclose(ordered(values),ordered(wanted),rtol=0,atol=1e-6 if width else 1e-8):
                raise RuntimeError('Live generated properties differ: '+name+' '+key)
            record[key]=values.tolist()
        result[name]=dict(status='SOURCE_COMPOSED_AND_LIVE_PROPERTIES_MATCH',shape_count=count,
            composed_by_collider=actual,live_shape_values=record,source_view=view_name,
            scope='Composed collider bindings and live property multiset; not cross-engine contact equivalence.')
    return result


def verify_live_arm_gains(asset,configuration):
    result={}
    for group,row in configuration.items():
        ids,names=asset.find_joints(row['joint_names'],preserve_order=True)
        if list(names)!=row['joint_names']:raise RuntimeError('Live actuator joint mapping changed.')
        result[group]={}
        for field,key in {'joint_stiffness':'stiffness','joint_damping':'damping',
                          'joint_effort_limits':'joint_effort_limit','joint_vel_limits':'joint_velocity_limit'}.items():
            actual=_array(getattr(asset.data,field),name=field)[0,ids]
            if not np.allclose(actual,row['applied'][key],rtol=1e-6,atol=1e-6):
                raise RuntimeError('Live left '+group+' '+key+' differs from the frozen plan.')
            result[group][field]=actual.tolist()
    return result


def feedback_command(servo,nominal_action,measured_hand_pose,previous_q,tcp_offset,phase,integrating_phases=TCP_FEEDBACK_INTEGRATING_PHASES):
    """Use the measured actor-link pose; never replace it with target FK."""
    action=_array(nominal_action,(14,),'nominal action');hand=_pose(measured_hand_pose,'measured hand')
    offset=_array(tcp_offset,(3,),'TCP offset');tcp=hand[:3,3]+hand[:3,:3]@offset
    command,diagnostic=servo.step(action[:6],tcp,hand[:3,:3],previous_q,
                                  integrate=phase in integrating_phases)
    action[:6]=command
    return action.tolist(),diagnostic


def finger_preload_command(controller,action,phase,rows,previous_target,denominator,physics_dt):
    """Consume the completed interval without new reads, steps or state writes."""
    command=_array(action,(14,),'native14-dimensional action')
    substeps=round(.05/physics_dt)
    if abs(substeps*physics_dt-.05)>1e-12 or not rows or (len(rows)-1)%substeps:
        raise ValueError('Preload command lacks a complete preceding physics interval.')
    index=(len(rows)-1)//substeps
    features=preceding_interval_features(rows,index,physics_dt)
    targets,diagnostic=controller.step(phase,features,previous_target)
    targets=_array(targets,(2,),'symmetric finger command').tolist()
    lo=FINGER_PRELOAD_PROFILE['target_lower_bound_m'];hi=FINGER_PRELOAD_PROFILE['target_upper_bound_m']
    if targets[0]!=targets[1] or not lo<=targets[0]<=hi or not float(denominator)<lo:
        raise ValueError('Preload controller requested unsupported asymmetric/out-of-range native gripper targets.')
    command[6]=targets[0]/float(denominator)
    if diagnostic['control_index']!=index or diagnostic['interval']!=features:
        raise ValueError('Shared controller did not preserve its measured source interval.')
    return command.tolist(),targets,diagnostic


def finger_target_readback(asset,expected):
    """Read original target buffers; never use rounded values as controller history."""
    ids,names=asset.find_joints(['joint7','joint8'],preserve_order=True)
    if list(names)!=['joint7','joint8'] or len(set(ids))!=2:raise RuntimeError('Native finger joint order differs.')
    actual=_array(asset.data.joint_pos_target,name='native finger position targets')[0,ids]
    if not np.allclose(actual,expected,atol=1e-7,rtol=0):raise RuntimeError('Native scalar gripper target readback differs from the requested command.')
    return dict(joint_names=list(names),joint_indices=[int(i)for i in ids],requested_target_q_m=list(expected),
        measured_target_q_m=actual.tolist(),absolute_tolerance_m=1e-7,
        scope='Public float32 command-buffer readback only; actual finger pose and force are recorded separately.')


def audit_finger_preload_evidence(rows,records,commands,denominator,physics_dt,phases=None):
    """Bind replayed controller inputs to prior measurements and actual targets."""
    replay=replay_commands(records,phases);substeps=round(.05/physics_dt)
    if abs(substeps*physics_dt-.05)>1e-12 or len(records)!=len(commands) or not rows:
        raise ValueError('Finger controller/source recording cadence differs.')
    for index,(record,command)in enumerate(zip(records,commands)):
        source_index=index*substeps
        if source_index>=len(rows):
            raise ValueError('Preload interval extends beyond saved measurements.')
        features=preceding_interval_features(rows,index,physics_dt,prefix_length=source_index+1)
        if json.dumps(record['interval'],sort_keys=True,allow_nan=False)!=json.dumps(features,sort_keys=True,allow_nan=False):
            raise ValueError('Preload interval statistics differ from the actual saved physics measurements.')
        target=_array(record['commanded_finger_q_m'],(2,))
        if (command['finger_preload_control_index']!=index or
                not np.allclose(command['requested_finger_target_q_m'],target,atol=1e-12,rtol=0) or
                len(command['target'])!=14 or abs(command['target'][6]*denominator-target[0])>1e-12):
            raise ValueError('Issued native scalar action differs from shared controller output.')
    for index,row in enumerate(rows):
        if index==0:
            if row.get('finger_preload_command_index') is not None:raise ValueError('Initialization predates any preload command.')
            continue
        control=(index-1)//substeps
        if control>=len(records) or row.get('finger_preload_command_index')!=control:
            raise ValueError('Physics sample has no matching active preload command.')
        if not np.allclose(_array(row['commanded_finger_q_m'],(2,)),records[control]['commanded_finger_q_m'],atol=1e-7,rtol=0):
            raise ValueError('Actual native target buffers differ during the held control interval.')
    return dict(status='SAVED_NATIVE_PRELOAD_INPUTS_OUTPUTS_AND_REPLAY_MATCH',sample_count=len(rows),
        command_count=len(records),complete=replay['complete'] and len(rows)==1+len(records)*substeps,
        native_target_readback_tolerance_m=1e-7,controller_replay=replay,
        scope='All preceding physics samples at the verified cadence, exact requested controller history and float32 native target buffers; no new physical test.')


def contact_measurement(matrix,net,partner_names):
    """Only the two dedicated pad bodies count as intended wafer contact."""
    expected=['lower_pad','upper_pad','lower_half','upper_half','pillar_0','pillar_1','pillar_2','heater','pedestal']
    dust=[f'dust_{i:02d}' for i in range(12)]
    base=partner_names[:-12] if partner_names[-12:]==dust else partner_names
    if base not in (expected,expected+['lower_grip_paddle','upper_grip_paddle']):raise ValueError('Dedicated pad/lever/support filter order differs.')
    matrix=_array(matrix,(1,1,len(partner_names),3),'wafer normal-force matrix')[0,0]
    net=_array(net,(1,1,3),'wafer net normal force')[0,0]
    magnitudes=np.linalg.norm(matrix,axis=1);residual=float(np.linalg.norm(net-matrix.sum(axis=0)))
    return dict(jaw_contact_normal_force_N=magnitudes[:2].tolist(),pillar_contact_normal_force_N=magnitudes[4:7].tolist(),
        forbidden_wafer_contact_force_N=float(magnitudes[2:4].sum()+magnitudes[7:len(base)].sum()+residual),
        wafer_nonpad_lever_contact_normal_force_N=magnitudes[2:4].tolist(),
        filtered_normal_force_vectors_world_N=matrix.tolist(),net_normal_force_world_N=net.tolist(),
        unmatched_net_normal_force_residual_N=residual,contact_force_scope='wafer_partner_filtered')


def robot_contact_measurement(left_net,left_names,right_net,right_names,tip_matrices,powered):
    """Known wrong partners are explicit; unmatched totals remain residuals.

    Aggregate vectors can cancel, so this is not a per-contact force sum.
    Recorded-pose collision screening remains an independent requirement.
    """
    left=_array(left_net,(1,len(left_names),3),'left arm normal forces')[0]
    right=_array(right_net,(1,len(right_names),3),'right arm normal forces')[0]
    if any(left_names.count(name)!=1 for name in ('tip_left','tip_right')):
        raise ValueError('Robot tip contact bodies are ambiguous.')
    indices=[left_names.index(name) for name in ('tip_left','tip_right')]
    other=[i for i in range(len(left_names)) if i not in indices]
    unexpected_environment=float(np.linalg.norm(left[other],axis=1).sum()+np.linalg.norm(right,axis=1).sum())
    unexpected_tool=0.;holding=[];residuals=[];vectors=[]
    for tip,index in enumerate(indices):
        matrix=_array(tip_matrices[tip],(1,1,2,3),'tip/clamp normal-force matrix')[0,0]
        intended=0 if powered else 1-tip
        holding.append(float(np.linalg.norm(matrix[intended])))
        unexpected_tool+=float(np.linalg.norm(matrix[1-intended]))
        residual=float(np.linalg.norm(left[index]-matrix.sum(axis=0)))
        unexpected_environment+=residual;residuals.append(residual);vectors.append(matrix.tolist())
    return dict(finger_handle_partner_normal_force_N=holding,
        unexpected_robot_tool_contact_normal_force_N=unexpected_tool,
        unexpected_robot_environment_contact_normal_force_N=unexpected_environment,
        tip_unmatched_normal_force_residual_N=residuals,tip_tool_normal_force_vectors_world_N=vectors,
        left_body_normal_force_vectors_world_N=left.tolist(),right_body_normal_force_vectors_world_N=right.tolist())


def measured_jaw_gap(link_poses, faces):
    faces = _array(faces,(2,4,3),'contact faces'); world = []
    for pose,face in zip(link_poses,faces):
        t = _pose(pose,'jaw link pose'); world.append(face@t[:3,:3].T+t[:3,3])
    normal = np.cross(world[0][1]-world[0][0],world[0][3]-world[0][0]); length=np.linalg.norm(normal)
    if length < 1e-10: raise ValueError('Degenerate lower contact face.')
    normal/=length
    # Authored lower face winding points toward the wafer.
    return float((world[1].mean(axis=0)-world[0].mean(axis=0))@normal)


def guard_physics_jaw_load(row):
    """Apply the existing 15 N limit at every saved physics sample."""
    forces=_array(row['jaw_contact_normal_force_N'],(2,),'physics-step jaw forces')
    if np.any(forces<0):raise ValueError('Normal-force norms cannot be negative.')
    if float(forces.max())>15.:
        raise TrialAbort(f"Jaw load exceeded the declared 15 N simulation bound at {row['time_s']:.6f} s: {forces.tolist()}")


def guard_physics_loads_v12(row,pad_compliance,passive_hinge):
    """Persist first; retain every original guard and the prospective pad guard."""
    guard_physics_jaw_load(row)
    try:
        guard_finger_handle_load(row,engine='isaac')
        guard_passive_pad_state(row,pad_compliance)
        guard_passive_hinge_state(row,passive_hinge)
    except (FingerHandleJamStop,PadComplianceStop,HingeComplianceStop) as exc:raise TrialAbort(str(exc)) from exc


class PhysicsRecorder:
    """Wrap one scene instance; original update is called exactly once.

    Every pose is read after the original update refreshed public buffers.
    The wrapper never calls a physics step, sensor update, reset, or state write.
    A count/cadence mismatch makes the diagnostic incomplete before another
    action can be submitted. The original method is restored even on errors.
    """
    def __init__(self, scene, reader, destination, physics_dt, control_dt=.05,sample_guard=None,rows=None):
        self.scene=scene; self.reader=reader; self.dt=float(physics_dt);self.control_dt=float(control_dt)
        self.sample_guard=sample_guard
        self.expected=round(control_dt/physics_dt)
        if self.expected<1 or abs(self.expected*physics_dt-control_dt)>1e-12: raise ValueError('Bad recorder cadence.')
        self.original=scene.update;self.original_instance=scene.__dict__.get('update');self.had_instance='update' in scene.__dict__
        self.active=None;self.count=0;self.total=0;self.rows=[]if rows is None else rows;self.closed=False
        self.stream=gzip.open(Path(destination)/TRACE_STORAGE['filename'],'xt',
            compresslevel=TRACE_STORAGE['compression_level'],encoding=TRACE_STORAGE['encoding'])
        def updated(dt,*args,**kwargs):
            result=self.original(dt,*args,**kwargs)
            if self.active is not None:
                if abs(float(dt)-self.dt)>1e-12:raise RuntimeError('Unexpected scene.update interval.')
                self.count+=1;self.total+=1
                if self.count>self.expected:raise RuntimeError('Too many physics updates for one action.')
                self._record(self.active,self.total*self.dt)
            return result
        self.wrapper=updated;scene.update=updated

    def _record(self,phase,time_s):
        row=self.reader(phase);row['time_s']=float(time_s);row['phase']=phase
        encoded=json.dumps(row,allow_nan=False)
        if hasattr(self.rows,'append_encoded'):self.rows.append_encoded(encoded.encode('utf-8'),validate=False)
        else:self.rows.append(row)
        self.stream.write(encoded+'\n')
        if self.sample_guard is not None:
            try:self.sample_guard(row)
            except Exception:
                self.stream.flush()
                raise
        return row

    def initial(self,phase='initialization'):
        if self.rows or self.active is not None:raise RuntimeError('Initial sample can only be read once.')
        row=self._record(phase,0.);self.stream.flush();return row

    def begin(self,phase):
        if self.active is not None:raise RuntimeError('Previous action was not completed.')
        self.active=phase;self.count=0

    def end(self):
        self.active=None;self.stream.flush()
        if self.count!=self.expected:raise RuntimeError('Missing full-step samples for an action.')
        return self.rows[-1]

    def close(self):
        if self.closed:return
        self.closed=True;self.active=None
        try:self.stream.flush();self.stream.close()
        finally:
            if self.scene.update is not self.wrapper:raise RuntimeError('Scene update hook changed unexpectedly.')
            if self.had_instance:self.scene.update=self.original_instance
            else:del self.scene.__dict__['update']


def _body_pose(asset, name):
    ids,names=asset.find_bodies(name)
    if list(names)!=[name] or len(ids)!=1:raise RuntimeError('Ambiguous body: '+name)
    poses=_array(asset.data.body_link_pose_w,name='body link poses')
    pose=poses[0,int(ids[0])];_pose(pose,name);return pose.tolist()


def _joint_positions(asset,names):
    ids,found=asset.find_joints(list(names),preserve_order=True)
    if list(found)!=list(names):raise RuntimeError('Joint mapping differs from requested names.')
    return _array(asset.data.joint_pos,name='joint positions')[0,ids].tolist()


def _joint_velocities(asset,names):
    ids,found=asset.find_joints(list(names),preserve_order=True)
    if list(found)!=list(names):raise RuntimeError('Joint velocity mapping differs.')
    return _array(asset.data.joint_vel,name='joint velocities')[0,ids].tolist()


def _joint_targets(asset,names):
    ids,found=asset.find_joints(list(names),preserve_order=True)
    if list(found)!=list(names):raise RuntimeError('Joint target measurement mapping differs.')
    return _array(asset.data.joint_pos_target,name='public joint target buffers')[0,ids].tolist()


def mechanism_target_readback(asset,joint_name,expected,targets=None):
    """Public command-buffer readback; no physics or state update is invoked."""
    ids,names=asset.find_joints([joint_name],preserve_order=True)
    if list(names)!=[joint_name]:raise RuntimeError('Mechanism target mapping differs.')
    target=float(_array(asset.data.joint_pos_target if targets is None else targets,name='mechanism position target')[0,ids[0]])
    if not math.isclose(target,float(expected),rel_tol=0,abs_tol=1e-7):
        raise RuntimeError('Mechanism target differs from the constant mechanical spring rest.')
    return target


def verify_mechanism_usd_drive(root,springs):
    """Check all constant force drives before native articulation creation."""
    from pxr import Usd,UsdPhysics
    from wafer_retention_usd_v12 import verify_drive_prim
    joints={p.GetName():p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint)}
    verified_tool_joint_count(springs)
    if set(joints)!={x['joint_name'] for x in springs}:raise RuntimeError('Passive tool joint topology differs.')
    records={x['joint_name']:verify_drive_prim(joints[x['joint_name']],x) for x in springs}
    return dict(status='COMPOSED_FORCE_DRIVE_VERIFIED',passive_joints=records,constant_target_count=len(springs))


def install_mechanism_drive_check(spawn_config,actuation,collision_profile=None):
    """Validate one isolated spawn without editing its source asset or stepping."""
    original=spawn_config.func;state={'status':'NOT_SPAWNED','callback_count':0}
    def spawn(*args,**kwargs):
        state['callback_count']+=1
        if state['callback_count']!=1:raise RuntimeError('Expected one isolated mechanism spawn.')
        root=original(*args,**kwargs);state.update(verify_mechanism_usd_drive(root,actuation))
        if collision_profile is not None:
            from wafer_retention_usd_v12 import verify_tool_collision_prim
            state['tool_self_collision']=verify_tool_collision_prim(root,collision_profile)
        return root
    spawn_config.func=spawn
    return state


def native_mechanism_readback(asset,actuation,expected_target,include_parameters=True,expected_dof_count=3,target_snapshot=None):
    """Read public native PhysX buffers after command upload, without stepping.

    Native gain/target values alone do not prove drive presence. A force DriveAPI
    must separately pass the spawn-time USD check. The installed public API was
    inspected: get_drive_types is required to prove native force-drive presence.
    """
    ids,names=asset.find_joints([actuation['joint_name']],preserve_order=True)
    if list(names)!=[actuation['joint_name']] or len(ids)!=1:
        raise RuntimeError('Expected exactly the named passive spring DOF.')
    for attribute in ('root_view','root_physx_view'):
        view=getattr(asset,attribute,None)
        if view is not None:break
    else:raise RuntimeError('Public native mechanism view is unavailable.')
    expected={'position_targets':float(expected_target),'velocity_targets':0.}
    if target_snapshot is not None and include_parameters:raise ValueError('Target snapshots cannot replace parameter readback.')
    if include_parameters:expected.update(stiffnesses=actuation['stiffness'],dampings=actuation['damping'],
        max_forces=actuation['effort_limit'],max_velocities=actuation['velocity_limit'])
    measured={};drive_type=dict(status='NOT_RECHECKED',reason='Checked with full parameters before the first trial step.')
    if include_parameters:
        type_getter=getattr(view,'get_drive_types',None)
        if callable(type_getter):
            raw=type_getter()
            if hasattr(raw,'numpy') and not hasattr(raw,'detach'):raw=raw.numpy()
            code=float(_array(raw,(1,expected_dof_count),'native mechanism drive type')[0,int(ids[0])])
            # NVIDIA DofDriveType: None=0, Force=1, Acceleration=2.
            if code!=1.:raise RuntimeError('Native mechanism drive is absent or is not a force drive.')
            drive_type=dict(status='NATIVE_FORCE_DRIVE_VERIFIED',source=attribute+'.get_drive_types',code=1,type='force')
        else:raise RuntimeError('Required public native get_drive_types is unavailable; no trial motion allowed.')
    for key,value in expected.items():
        getter=getattr(view,'get_dof_'+key,None)
        if not callable(getter):raise RuntimeError('Public native mechanism get_dof_'+key+' is unavailable.')
        raw=getter()if target_snapshot is None else target_snapshot[key]
        if hasattr(raw,'numpy') and not hasattr(raw,'detach'):raw=raw.numpy()
        actual=float(_array(raw,(1,expected_dof_count),'native mechanism '+key)[0,int(ids[0])])
        if not math.isclose(actual,float(value),rel_tol=1e-6,abs_tol=1e-7):
            raise RuntimeError('Native mechanism '+key+' differs from the fixed SI source profile.')
        measured[key]=actual
    return dict(status='NATIVE_PARAMETERS_AND_TARGET_VERIFIED' if include_parameters else 'NATIVE_TARGET_VERIFIED',
        source=attribute,values=measured,joint_name=actuation['joint_name'],native_joint_index=int(ids[0]),position_unit=actuation['position_unit'],effort_unit=actuation['effort_unit'],
        native_drive_type_readback=drive_type,
        scope='Native SI buffer readback, not a physical spring-response or grasp result.')


def passive_target_snapshot(asset,count):
    """Acquire each target buffer once at the current completed physics boundary."""
    public=_array(asset.data.joint_pos_target,(1,count),'passive public targets')
    for attribute in ('root_view','root_physx_view'):
        view=getattr(asset,attribute,None)
        if view is not None:break
    else:raise RuntimeError('Public native mechanism view is unavailable.')
    native={}
    for key in ('position_targets','velocity_targets'):
        getter=getattr(view,'get_dof_'+key,None)
        if not callable(getter):raise RuntimeError('Public native mechanism get_dof_'+key+' is unavailable.')
        value=getter()
        if hasattr(value,'numpy')and not hasattr(value,'detach'):value=value.numpy()
        native[key]=_array(value,(1,count),'native passive '+key)
    return public,native


def upload_initial_mechanism_target(asset,actuation,target):
    """Flush an already-set target through the public API; no physics step."""
    mechanism_target_readback(asset,actuation['joint_name'],target)
    writer=getattr(asset,'write_data_to_sim',None)
    if not callable(writer):raise RuntimeError('Public mechanism command upload is unavailable.')
    writer()
    return native_mechanism_readback(asset,actuation,target)


def _rigid_link_pose(asset):
    """Never substitute a center-of-mass pose for the authored wafer frame."""
    if hasattr(asset.data,'root_link_pose_w'):
        pose=_array(asset.data.root_link_pose_w,(1,7),'wafer root link pose')[0]
        source='root_link_pose_w'
    elif hasattr(asset.data,'body_link_pose_w'):
        pose=_array(asset.data.body_link_pose_w,(1,1,7),'wafer body link pose')[0,0]
        source='body_link_pose_w'
    else:raise RuntimeError('No explicit public wafer link-pose buffer; COM/root-pose substitution is forbidden.')
    _pose(pose,'wafer link pose');return pose.tolist(),source


def _gather_public_arrays(fields,tensor_backend):
    """One explicit device-to-host copy of fresh public buffers per snapshot.

    No tensor values are cached. Dtype/shape/device inspection uses tensor
    metadata only; finite-value and pose checks happen after the CPU transfer.
    Public getters may synchronize internally: this does not promise one total
    backend synchronization or a measured speedup.
    """
    names=[];shapes=[];sizes=[];flat=[];device=None
    for name,raw,expected in fields:
        value=getattr(raw,'torch',raw)
        if not tensor_backend.is_tensor(value):raise ValueError('Public snapshot field is not a tensor: '+name)
        if value.dtype not in (tensor_backend.float16,tensor_backend.float32,tensor_backend.float64):
            raise ValueError('Unsupported non-real public snapshot dtype: '+name)
        shape=tuple(value.shape)
        if expected is not None and shape!=expected:raise ValueError('Public snapshot shape differs: '+name)
        if not shape or any(type(n)is not int or n<=0 for n in shape):raise ValueError('Empty or invalid public buffer shape: '+name)
        if device is None:device=value.device
        elif value.device!=device:raise ValueError('Public snapshot fields occupy different devices.')
        names.append(name);shapes.append(shape);sizes.append(math.prod(shape));flat.append(value.detach().reshape(-1))
    if len(names)!=len(set(names))or not names:raise ValueError('Ambiguous public snapshot field inventory.')
    # torch.cat promotes float32/float64 without discarding float64 precision.
    # There is exactly one explicit host copy. All following work is CPU-only.
    packed=tensor_backend.cat(flat,dim=0).detach().cpu().numpy()
    values=_array(packed,(sum(sizes),),'packed public snapshot')
    arrays={};offset=0
    for name,shape,size in zip(names,shapes,sizes):
        arrays[name]=values[offset:offset+size].reshape(shape);offset+=size
    return arrays


def snapshot_tensor_self_check():
    """Small real-PyTorch CPU preflight; no simulator, GPU job or state change."""
    import torch
    first=torch.tensor([[.125,-2.,3.]],dtype=torch.float32,device='cpu')
    precise=1.+2.**-45
    second=torch.tensor([[precise,7.]],dtype=torch.float64,device='cpu')
    class PublicWrapper:
        @property
        def torch(self):return second
    actual=_gather_public_arrays([('first',first,(1,3)),('second',PublicWrapper(),(1,2))],torch)
    if not np.array_equal(actual['first'],np.array([[.125,-2.,3.]],dtype=float)):
        raise RuntimeError('Real tensor snapshot value mismatch.')
    if actual['second'][0,0]!=precise or actual['second'].dtype!=np.float64:
        raise RuntimeError('Real tensor snapshot lost float64 precision.')
    invalid=torch.tensor([[float('nan')]],dtype=torch.float32,device='cpu')
    try:_gather_public_arrays([('invalid',invalid,(1,1))],torch)
    except ValueError:pass
    else:raise RuntimeError('Nonfinite public tensor was accepted.')
    return dict(result='REAL_TORCH_CPU_SNAPSHOT_PARITY_PASSED',torch_version=torch.__version__,
        mixed_float32_float64_preserved=True,wrapped_public_buffer_supported=True,nonfinite_rejected=True,
        scope='Small CPU tensor API/type/precision check only; no CUDA timing or simulator equivalence claimed.')


def make_reader(base,metadata,partner_names,preload_reference=None,*,tensor_backend=None,extension=None):
    """Read fresh public tensors after each existing update; preserve all fields."""
    if tensor_backend is None:
        import torch as tensor_backend
    clamp=base.scene.articulations['clamp'];wafer=base.scene.rigid_objects['wafer'];sensor=base.scene.sensors['wafer_contacts']
    if len(sensor.body_names)!=1:raise RuntimeError('Wafer force sensor must contain one sensing body.')
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    left=base.scene.articulations['left_robot'];left_sensor=base.scene.sensors['left_contacts'];right_sensor=base.scene.sensors['right_contacts']
    left_names=list(left_sensor.body_names);right_names=list(right_sensor.body_names)
    tips=[base.scene.sensors['tip_left_tool_contacts'],base.scene.sensors['tip_right_tool_contacts']]
    metadata['robot_contact_sensor_body_names']=dict(left=left_names,right=right_names)
    for name,sensor_part in zip(('tip_left','tip_right'),tips):
        if list(sensor_part.body_names)!=[name]:raise RuntimeError('Tip-specific filtered sensor body differs.')
    all_springs=metadata['usd_export']['passive_springs'];verified_tool_joint_count(all_springs)
    by_name={x['joint_name']:x for x in all_springs};springs=[by_name[name]for name in CORE_PASSIVE_JOINTS]
    joint_names=list(CORE_PASSIVE_JOINTS)
    def body_id(asset,name):
        ids,names=asset.find_bodies(name)
        if list(names)!=[name]or len(ids)!=1:raise RuntimeError('Ambiguous body: '+name)
        return int(ids[0])
    def joint_ids(asset,names):
        ids,found=asset.find_joints(list(names),preserve_order=True)
        if list(found)!=list(names)or len(ids)!=len(names)or len(set(ids))!=len(ids):
            raise RuntimeError('Joint mapping differs from requested names.')
        return [int(i)for i in ids]
    clamp_ids={name:body_id(clamp,name)for name in ('lower_half','upper_half','lower_pad','upper_pad')}
    hand_id=body_id(left,'gripper');passive_ids=joint_ids(clamp,joint_names)
    arm_ids=joint_ids(left,ARM_JOINTS);finger_ids=joint_ids(left,FINGER_JOINTS)
    def body_pose(values,index,name):
        if values.ndim!=3 or values.shape[0]!=1 or values.shape[2]!=7 or not 0<=index<values.shape[1]:
            raise ValueError('Invalid public body-pose layout: '+name)
        pose=values[0,index];_pose(pose,name);return pose.tolist()
    def joint_values(values,ids):
        if values.ndim!=2 or values.shape[0]!=1 or any(not 0<=i<values.shape[1]for i in ids):
            raise ValueError('Invalid public joint-buffer layout.')
        return values[0,ids].tolist()
    metadata['observation_readback']=dict(mode='fresh_public_tensors_one_explicit_host_copy',
        public_buffer_count=13 if extension is None else 13+len(extension.fields()),explicit_cpu_transfers_per_snapshot=1,cached_values='Static name-to-index mappings only',
        sample_timing='After the existing original scene.update returns; no added update or physics step.',
        precision='Retain float16/float32/float64 values via torch.cat dtype promotion, then CPU float64 checks.',
        limitation='Public getters may synchronize internally. No measured native speedup or total-sync count is claimed.')
    def reader(phase):
        try:wafer_pose=wafer.data.root_link_pose_w;pose_source='root_link_pose_w';pose_shape=(1,7)
        except AttributeError:
            try:wafer_pose=wafer.data.body_link_pose_w;pose_source='body_link_pose_w';pose_shape=(1,1,7)
            except AttributeError:raise RuntimeError('No explicit public wafer link-pose buffer; COM/root-pose substitution is forbidden.')
        # Acquire every buffer anew after scene.update; never cache tensor data.
        arrays=_gather_public_arrays([
            ('wafer_pose',wafer_pose,pose_shape),('clamp_poses',clamp.data.body_link_pose_w,None),
            ('clamp_q',clamp.data.joint_pos,None),('clamp_v',clamp.data.joint_vel,None),
            ('wafer_matrix',sensor.data.normal_force_matrix_w,(1,1,len(partner_names),3)),
            ('wafer_net',sensor.data.net_normal_forces_w,(1,1,3)),
            ('left_net',left_sensor.data.net_normal_forces_w,(1,len(left_names),3)),
            ('right_net',right_sensor.data.net_normal_forces_w,(1,len(right_names),3)),
            ('tip_left_matrix',tips[0].data.normal_force_matrix_w,(1,1,2,3)),
            ('tip_right_matrix',tips[1].data.normal_force_matrix_w,(1,1,2,3)),
            ('left_q',left.data.joint_pos,None),('left_targets',left.data.joint_pos_target,None),
            ('left_poses',left.data.body_link_pose_w,None)]+([]if extension is None else extension.fields()),tensor_backend)
        pose=arrays['wafer_pose'].reshape(7);_pose(pose,'wafer link pose');pose=pose.tolist()
        clamp_poses={name:body_pose(arrays['clamp_poses'],index,name)for name,index in clamp_ids.items()}
        links=[clamp_poses[name]for name in ('lower_pad','upper_pad')]
        q=joint_values(arrays['clamp_q'],passive_ids);v=joint_values(arrays['clamp_v'],passive_ids)
        forces=contact_measurement(arrays['wafer_matrix'],arrays['wafer_net'],partner_names)
        right_net=arrays['right_net']if extension is None else extension.unmatched_right(arrays,right_names)
        robot_forces=robot_contact_measurement(arrays['left_net'],left_names,right_net,right_names,
            [arrays['tip_left_matrix'],arrays['tip_right_matrix']],False)
        pad_force=[-x['stiffness']*(pos-x['constant_target'])-x['damping']*vel for x,pos,vel in zip(springs[1:],q[1:],v[1:])]
        hinge=springs[0];hinge_torque=-hinge['stiffness']*(q[0]-hinge['constant_target'])-hinge['damping']*v[0]
        row=dict(phase=phase,wafer_pose_world=pose,wafer_pose_buffer=pose_source,
            finger_preload_command_index=None if preload_reference is None else preload_reference['control_index'],
            clamp_pose_world=clamp_poses['lower_half'],jaw_link_poses_world=links,
            lever_link_poses_world=[clamp_poses[name] for name in ('lower_half','upper_half')],
            measured_jaw_gap_m=measured_jaw_gap(links,metadata['contact_faces_local_m']),
            mechanism_joint_position=q[0],mechanism_joint_position_unit='rad',mechanism_joint_velocity=v[0],
            hinge_angle_rad=q[0],hinge_velocity_rad_s=v[0],passive_hinge_torque_Nm=hinge_torque,
            passive_hinge_torque_scope='Uncapped SI spring/damper law from actual hinge q/qdot; not saturated drive or contact torque.',
            passive_joint_names=joint_names,passive_joint_positions=q,passive_joint_velocities=v,
            pad_compression_m=q[1:],pad_compression_velocity_m_s=v[1:],passive_pad_spring_force_N=pad_force,
            passive_pad_force_scope='Uncapped SI spring/damper law evaluated from measured q/qdot; not measured contact or saturated actuator force.',
            measured_arm_q_rad=joint_values(arrays['left_q'],arm_ids),measured_finger_q_m=joint_values(arrays['left_q'],finger_ids),
            commanded_finger_q_m=joint_values(arrays['left_targets'],finger_ids),
            measured_hand_pose_world=body_pose(arrays['left_poses'],hand_id,'gripper'),**forces,**robot_forces)
        if extension is not None:row.update(extension.measurement(arrays,row,phase))
        return row
    return reader


def _world_pose(pose,translation,yaw_deg=90):
    if yaw_deg not in (90,270):raise ValueError('Unsupported fixture yaw.')
    angle=math.radians(yaw_deg);c=math.cos(angle);s=math.sin(angle)
    p=_array(pose,(7,));t=_pose(p,'source pose');r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    x,y,z,w=p[3:]/np.linalg.norm(p[3:]);c=math.cos(angle/2);s=math.sin(angle/2)
    q=np.array([c*x-s*y,c*y+s*x,c*z+s*w,c*w-s*z])
    return [*(r@t[:3,3]+translation),*q]


def prepare_native_initial_poses(local_poses,translation,pose_type,torch_module,yaw_deg=90):
    """Normalize scalar types at the Arena Pose boundary; preserve native reset.

    Installed Pose.to_tensor calls torch.tensor without an explicit dtype.
    NumPy float64 tuple members therefore produce doubles, which the installed
    PhysX articulation transformf writer rejects. Pose.from_dict deliberately
    converts each scalar to a Python float, so the existing float32 default is
    used. Never change that process-wide default or replace Arena reset events.
    """
    expected={'clamp','wafer','heater','pillar_0','pillar_1','pillar_2','pedestal'}
    if set(local_poses)!=expected:raise ValueError('Expected seven initial object poses.')
    if torch_module.get_default_dtype()!=torch_module.float32:
        raise RuntimeError('Arena reset requires the existing torch.float32 default; no scene was constructed.')
    translation=_array(translation,(3,),'initial pose translation');poses={};checks={}
    for name,local in local_poses.items():
        world=_array(_world_pose(local,translation,yaw_deg),(7,),'transformed initial pose')
        pose=pose_type.from_dict({'position_xyz':world[:3].tolist(),'rotation_xyzw':world[3:].tolist()})
        if any(type(v) is not float for v in (*pose.position_xyz,*pose.rotation_xyzw)):
            raise RuntimeError('Arena Pose did not retain native Python float scalars: '+name)
        tensor=pose.to_tensor(device='cpu')
        if tensor.dtype!=torch_module.float32:
            raise RuntimeError('Arena initial pose tensor is not float32: '+name)
        actual=_array(tensor,(7,),'native initial pose '+name)
        if not np.allclose(actual,world,atol=1e-7,rtol=0):
            raise RuntimeError('Native pose conversion changed the declared layout: '+name)
        poses[name]=pose;checks[name]=dict(dtype=str(tensor.dtype),shape=list(tensor.shape),
            maximum_scalar_rounding_error=float(np.max(np.abs(actual-world))),world_pose=actual.tolist())
    return poses,dict(status='ALL_SEVEN_NATIVE_POSE_TENSORS_FLOAT32',default_dtype=str(torch_module.get_default_dtype()),
        objects=checks,scope='Initialization representation only. Native Arena reset, environment origins and velocity reset are retained.')


def ensure_gpu_idle():
    out=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],
        check=True,capture_output=True,text=True,timeout=10)
    desktop='/usr/libexec/gnome-remote-desktop-daemon'
    lines=[];ignored=[]
    for line in out.stdout.splitlines():
        if not line.strip():continue
        pid=line.split(',')[0].strip()
        try:exe=str(Path('/proc')/pid/'exe');exe=str(Path(exe).resolve(strict=True))
        except OSError:exe=''
        (ignored if exe==desktop else lines).append(line)
    if lines:raise RuntimeError('Another GPU process is active. This trial did not start: '+'; '.join(lines))
    return dict(status='NO_OTHER_COMPUTE_JOB_AT_CHECK',desktop_processes=ignored,scope='Snapshot only; another user can start a job afterward.')


def verify_spawned_assets(exported):
    """Four dynamic tool bodies, three internal joints, and an unattached wafer."""
    import omni.usd
    from pxr import Usd,UsdPhysics
    from wafer_retention_usd_v12 import verify_drive_prim,verify_passive_spring_specifications,verify_joint_frame_prim
    verify_passive_spring_specifications(exported)
    stage=omni.usd.get_context().get_stage();prims=list(stage.Traverse(Usd.TraverseInstanceProxies()))
    actual={};dynamic_paths=[]
    for name in exported['models']:
        subset=[p for p in prims if name in str(p.GetPath()).split('/')]
        bodies=[p for p in subset if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        expected_names=exported['tool_link_names'] if name=='clamp' else [name]
        if sorted(p.GetName() for p in bodies)!=sorted(expected_names):raise RuntimeError('Unexpected composed rigid bodies: '+name)
        for body in bodies:
            rb=UsdPhysics.RigidBodyAPI(body);kinematic=name not in ('clamp','wafer')
            if rb.GetRigidBodyEnabledAttr().Get() is not True or rb.GetKinematicEnabledAttr().Get()!=kinematic:
                raise RuntimeError('Disabled or wrongly kinematic body: '+str(body.GetPath()))
            if not kinematic:
                dynamic_paths.append(str(body.GetPath()))
                expected=exported['metadata']['usd_export']['body_inertials'][body.GetName() if name=='clamp' else name]['mass_kg']
                if not math.isclose(float(UsdPhysics.MassAPI(body).GetMassAttr().Get()),expected,rel_tol=1e-6,abs_tol=1e-9):raise RuntimeError('Spawned mass differs.')
                if body.GetAttribute('physxRigidBody:disableGravity').Get() is not False:raise RuntimeError('Gravity disabled on a free trial body.')
        actual[name]=[str(p.GetPath()) for p in bodies]
    attached=[]
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim);links=[str(t) for rel in (joint.GetBody0Rel(),joint.GetBody1Rel()) for t in rel.GetTargets()]
        if any(target in dynamic_paths for target in links):attached.append((prim,links))
    springs=exported['metadata']['usd_export']['passive_springs'];specs=exported['metadata']['usd_export']['joints']
    count=verified_tool_joint_count(springs)
    if len(attached)!=count:raise RuntimeError('Unexpected attachment/world joint involving wafer or clamp.')
    by_name={prim.GetName():(prim,links) for prim,links in attached};body_paths={Path(path).name:path for path in actual['clamp']};records={}
    if set(by_name)!={x['name'] for x in specs}:raise RuntimeError('Unexpected passive mechanism joint names.')
    for spec,spring in zip(specs,springs,strict=True):
        prim,links=by_name[spec['name']];joint_type=UsdPhysics.PrismaticJoint if spec['type']=='slide' else UsdPhysics.RevoluteJoint
        if not prim.IsA(joint_type) or set(links)!={body_paths[spec['parent']],body_paths[spec['child']]}:
            raise RuntimeError('The wafer must be unattached; all tool joints must connect their intended tool bodies.')
        records[spec['name']]=verify_drive_prim(prim,spring)
        if 'local_pos0_m' in spec:records[spec['name']]['joint_frames']=verify_joint_frame_prim(prim,spec)
    result=dict(status='COMPOSED_USD_VERIFIED',bodies=actual,wafer_attachments=0,clamp_internal_joints=count,
        passive_mechanism_drives=records,scope='Composed free topology/mass only; no physical contact or retention claim.')
    profile=exported['metadata']['usd_export'].get('tool_self_collision')
    if profile is not None:
        from wafer_retention_usd_v12 import verify_tool_collision_prim
        root=stage.GetPrimAtPath(actual['clamp'][0]).GetParent()
        result['tool_self_collision']=verify_tool_collision_prim(root,profile)
    return result


def configure_mechanism_reset(cfg,clamp_object):
    from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
    name,event=clamp_object.get_event_cfg()
    if name!='clamp' or event is None or event.mode!='reset' or event.params['asset_cfg'].name!='clamp':
        raise ValueError('Expected one configured clamp reset event.')
    # Object caches its reset pose before the full workflow moves the tool stand.
    event.func=reset_articulation_pose_and_joints
    event.params['pose']=clamp_object.initial_pose
    setattr(cfg.events,name,event)


def run_isaac_trial(bundle,destination,extension=None,device='cuda:0'):
    """Create/reset once; use YAM targets and the explicit mechanism drive."""
    if device not in ('cpu','cuda:0'):raise ValueError('Unsupported physics device.')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    app=env=recorder=raw_recorder=command_stream=feedback_stream=finger_stream=None;metadata={};commands=[];finger_records=[];prelift_diagnostic=None;stage='bundle verification';result={'result':'INCOMPLETE'}
    try:
        plan,exported,models,gate=validate_bundle(bundle);metadata=cloud_model_metadata(exported['metadata'],plan)
        if metadata.get('full_cycle_handle_geometry')and not getattr(extension,'full_workflow_adapter',False):
            raise ValueError('Full-workflow geometry requires its own complete-workflow gate and adapter.')
        if verified_tool_joint_count(metadata['usd_export']['passive_springs'])==5 and not getattr(extension,'full_workflow_adapter',False):
            raise ValueError('Passive handle paddles require the full-workflow adapter and gate.')
        if extension is not None:extension.validate(bundle,plan,metadata)
        metadata.update(engine='Isaac Lab / PhysX',physics_device_requested=device,local_validation=gate,contact_force_scope='wafer_partner_filtered',
            trace_storage=copy.deepcopy(TRACE_STORAGE),
            wafer_free_joint_representation='Unconstrained dynamic PhysX rigid body with six free degrees of freedom; no attachment joint.',
            initial_condition='Fresh scene with actual YAM and open hinged pliers prealigned at the flat. No transport.',
            source_files={name:_sha(_source_file(name)) for name in ('wafer_retention_isaac_v12.py','wafer_retention_usd_v12.py','wafer_retention_checks_v11.py','wafer_hinge_guard_v11.py','wafer_retention_checks_v6.py','wafer_retention_checks.py','wafer_retention_jam_guard_v2.py','wafer_retention_checks_v2.py','wafer_passive_pad_guard_v6.py','wafer_grasp_settings.py','wafer_finger_preload_v12.py','wafer_grasp_protocol.py','wafer_grasp_feedback.py')},
            controller=dict(command_hz=20,phases=[list(p) for p in PHASES],interpolation='Joint-space cubic arm targets; shared bounded scalar finger preload at20Hz.',tcp_feedback=plan['tcp_feedback'],finger_preload=plan['finger_preload']),
            force_measurement_limitations=['Each jaw force is a wafer/dedicated-pad-body filtered normal vector norm. Non-pad lever contacts are separately forbidden; opposing pad contacts can still cancel.',
                'Unmatched net force residual can hide cancelling unfiltered contacts; it is not a bound on every unwanted contact.'])
        metadata['gpu_snapshot']=ensure_gpu_idle();_json(destination/'metadata.json',metadata)
        stage='application startup';os.environ['ENABLE_CAMERAS']='0'
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=False,device=device).app
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from yam_arena import constants,station,ik
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        if abs(plan['native_finger_open_q_m']-constants.FINGER_OPEN)>1e-9:raise RuntimeError('Native gripper opening changed.')
        if extension is None and not np.allclose(plan['right_q_rad'],constants.READY,atol=1e-12,rtol=0):
            raise RuntimeError('Right-arm parked pose differs from the native READY configuration.')
        if not math.isclose(plan['world_translation_m'][2],station.TABLE_TOP+.025,abs_tol=1e-12):
            raise RuntimeError('Native station height differs from the checked pedestal layout.')
        embodiment=BimanualYamEmbodiment(enable_cameras=False,rig=station.rig(),robot_contact_offset=.0001,contact_sensors=True)
        embodiment.camera_config=None
        left=embodiment.scene_config.left_robot
        left.init_state.joint_pos={**dict(zip(constants.ARM_JOINTS,plan['initial_left_q_rad'])),'joint[7-8]':plan.get('initial_finger_state_q_m',plan['initial_finger_q_m'])}
        right=embodiment.scene_config.right_robot
        right.init_state.joint_pos={**dict(zip(constants.ARM_JOINTS,plan['right_q_rad'])),'joint[7-8]':constants.FINGER_OPEN}
        if extension is not None:
            right.init_state.joint_pos['joint[7-8]']=-.035
            metadata['right_actuators']=apply_arm_gains(right,plan['right_arm_gains'])
        metadata['robot_materials']={name:install_robot_material_match(cfg.spawn)
            for name,cfg in [('left_robot',left),('right_robot',right)]}
        metadata['left_actuators']=apply_arm_gains(left,plan['arm_gains'])
        # K/D and force caps match local records. Gravity compensation differs:
        # MuJoCo uses body compensation; Isaac uses the native YAM feed-forward.
        objects={};translation=_array(plan['world_translation_m'],(3,))
        stage='initial pose dtype validation'
        poses,metadata['initial_pose_tensor_validation']=prepare_native_initial_poses(
            exported['initial_local_poses'],translation,Pose,torch,plan['world_yaw_deg'])
        for name,path in models.items():
            obj=Object(name=name,usd_path=str(path),object_type=ObjectType.ARTICULATION if name=='clamp' else ObjectType.RIGID,
                initial_pose=poses[name])
            objects[name]=obj
        actuation=metadata['usd_export']['actuation'];joint_name=actuation['joint_name'];springs=metadata['usd_export']['passive_springs']
        objects['clamp'].object_cfg.actuators={x['joint_name']+'_spring':ImplicitActuatorCfg(
            joint_names_expr=[x['joint_name']],stiffness=x['stiffness'],damping=x['damping'],
            joint_effort_limit=x['effort_limit'],joint_velocity_limit=x['velocity_limit']) for x in springs}
        objects['clamp'].object_cfg.init_state.joint_pos={x['joint_name']:0. for x in springs}
        objects['clamp'].object_cfg.init_state.joint_vel={x['joint_name']:0. for x in springs}
        metadata['mechanism_drive_spawn']=install_mechanism_drive_check(objects['clamp'].object_cfg.spawn,springs,
            metadata['usd_export'].get('tool_self_collision'))
        partners=[*exported['contact_link_names'],*exported['lever_link_names'],'pillar_0','pillar_1','pillar_2','heater','pedestal']
        filters=[objects['clamp'].prim_path+'/'+name for name in partners[:4]]+[objects[name].get_contact_sensor_prim_path() for name in partners[4:]]
        if extension is not None and hasattr(extension,'wafer_contact_links'):
            extra=extension.wafer_contact_links();partners.extend(extra)
            filters.extend(objects['clamp'].prim_path+'/'+name for name in extra)
        sensor_cfg=ContactSensorCfg(prim_path=objects['wafer'].get_contact_sensor_prim_path(),filter_prim_paths_expr=filters,
            update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=256)
        finger_filters=filters[2:4]
        if extension is not None and hasattr(extension,'finger_contact_links'):
            finger_filters=[objects['clamp'].prim_path+'/'+name for name in extension.finger_contact_links()]
        tip_cfgs={name+'_tool_contacts':ContactSensorCfg(
            prim_path=left.prim_path+embodiment.link_paths[name],filter_prim_paths_expr=finger_filters,
            update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=256)
            for name in ('tip_left','tip_right')}
        if extension is not None:tip_cfgs.update(extension.configure(objects,embodiment,Pose,Object,ObjectType,ImplicitActuatorCfg,ContactSensorCfg,torch))
        if extension is not None and hasattr(extension,'extra_wafer_contacts'):
            for name,path in extension.extra_wafer_contacts():
                partners.append(name);filters.append(path)
            sensor_cfg.filter_prim_paths_expr=filters
        original=scenes.env_cfg_callback(20);callback_count=0
        def configure(cfg):
            nonlocal callback_count
            callback_count+=1;cfg=original(cfg)
            if callback_count!=1 or abs(cfg.sim.dt-1/240)>1e-12 or cfg.decimation!=12:
                raise RuntimeError('Unexpected environment baseline configuration.')
            cfg.sim.dt=metadata['physics_dt_s'];cfg.decimation=gate['decimation'];cfg.sim.render_interval=gate['decimation']
            if not hasattr(cfg.sim.physics,'solve_articulation_contact_last'):raise RuntimeError('Native contact ordering is unavailable.')
            cfg.sim.physics.solve_articulation_contact_last=False
            cfg.sim.physics.solver_type=0
            cfg.scene.wafer_contacts=sensor_cfg
            configure_mechanism_reset(cfg,objects['clamp'])
            for name,tip_cfg in tip_cfgs.items():setattr(cfg.scene,name,tip_cfg)
            return cfg
        description=IsaacLabArenaEnvironment(name='yam_prealigned_wafer_grasp',
            scene=Scene([*scenes.task_room('station',embodiment.rig.top_cam),*objects.values()]),
            embodiment=embodiment,env_cfg_callback=configure)
        stage='environment construction';env=ArenaEnvBuilder(description,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device=device)).make_registered()
        if callback_count!=1:raise RuntimeError('Environment callback count differs.')
        import omni.usd
        usd_stage=omni.usd.get_context().get_stage()
        scene_prim=usd_stage.GetPrimAtPath(env.unwrapped.cfg.sim.physics_prim_path)
        contact_last=scene_prim.GetAttribute('physxScene:solveArticulationContactLast').Get()
        solver_type=scene_prim.GetAttribute('physxScene:solverType').Get()
        if contact_last is not False or solver_type!='PGS':raise RuntimeError('Native solver readback differs.')
        metadata['native_solver_contact_order']=dict(solver_type=solver_type,solve_articulation_contact_last=contact_last,source='Composed physics scene before initial reset')
        drive_spawn=metadata['mechanism_drive_spawn']
        if drive_spawn['callback_count']!=1 or drive_spawn['status']!='COMPOSED_FORCE_DRIVE_VERIFIED':
            raise RuntimeError('Mechanism force drive was not verified before PhysX initialization.')
        for side,material in metadata['robot_materials'].items():
            if material['callback_count']!=1 or material['status']!='COMPOSED_BINDINGS_VERIFIED':
                raise RuntimeError('Robot physics material was not matched during spawn: '+side)
            material['before_reset']=verify_robot_material(usd_stage.GetPrimAtPath(material['root_path']))
        reset_event=env.unwrapped.cfg.events.clamp
        if reset_event.func is not reset_articulation_pose_and_joints or reset_event.mode!='reset' or reset_event.params['asset_cfg'].name!='clamp':
            raise RuntimeError('Passive mechanism pose/joint-reset event differs.')
        expected_pose=objects['clamp'].initial_pose
        if not np.allclose(reset_event.params['pose'].to_tensor(device='cpu'),expected_pose.to_tensor(device='cpu'),atol=1e-9,rtol=0):
            raise RuntimeError('Cached reset pose differs from the approved tool layout.')
        metadata['mechanism_initialization_reset']=dict(mode='reset',function='isaaclab_arena.terms.events.reset_articulation_pose_and_joints',
            scope='Restore the approved stand pose, configured passive joints and zero default velocities during the initial scene reset; no trial-time state writes.')
        stage='initial reset';env.reset();base=env.unwrapped
        metadata['physics_device_actual']=str(base.device)
        if metadata['physics_device_actual']!=device:raise RuntimeError('Live physics device differs from the requested device.')
        for side,material in metadata['robot_materials'].items():
            material['live_physics']=verify_live_robot_material(base.scene[side])
        stage='spawned free-body verification';metadata['spawned_assets']=verify_spawned_assets(exported)
        metadata['generated_materials']=verify_live_generated_materials(usd_stage,models,
            {name:base.scene[name] for name in ('clamp','wafer')},metadata['spawned_assets'])
        if abs(float(base.physics_dt)-metadata['physics_dt_s'])>1e-12 or abs(float(base.step_dt)-.05)>1e-12:
            raise RuntimeError('Live simulation timing differs from the plan.')
        if set(base.scene.sensors)!= {'left_contacts','right_contacts','wafer_contacts',*tip_cfgs}:
            raise RuntimeError('Unexpected sensors; camera-free trial required.')
        if base.observation_manager.active_terms != {'policy':['joint_pos']}:
            raise RuntimeError('Unexpected observations; image streams are forbidden.')
        metadata['verified_live_left_actuators']=verify_live_arm_gains(base.scene['left_robot'],metadata['left_actuators'])
        if extension is not None:extension.ready(base,metadata,torch)
        metadata['actuator_joint_map']=[]
        for side in ('left_robot','right_robot'):
            for group,names,unit in [('shoulder',list(constants.ARM_JOINTS[:3]),'rad'),
                                     ('wrist',list(constants.ARM_JOINTS[3:]),'rad'),
                                     ('gripper',list(constants.FINGER_JOINTS),'m')]:
                indices,found=base.scene[side].find_joints(names,preserve_order=True)
                if list(found)!=names:raise RuntimeError('Live actuator group joint order differs.')
                metadata['actuator_joint_map'].append(dict(asset=side,actuator_group=group,joint_names=names,
                    joint_indices=[int(i) for i in indices],position_unit=unit,effort_unit='Nm' if unit=='rad' else 'N'))
        for side in ('left','right'):
            target=plan['initial_left_q_rad'] if side=='left' else plan['right_q_rad']
            if not np.allclose(_joint_positions(base.scene[side+'_robot'],constants.ARM_JOINTS),target,atol=1e-4,rtol=0):
                raise RuntimeError('Initial arm joints did not match the source plan.')
        # Fixed spring-rest targets only: no adaptive pad command or powered jaw.
        clamp=base.scene.articulations['clamp'];joint_names=[x['joint_name'] for x in springs]
        ids,names=clamp.find_joints(joint_names,preserve_order=True)
        if list(names)!=joint_names or len(set(ids))!=verified_tool_joint_count(springs):raise RuntimeError('Passive spring joints must map uniquely.')
        for index,profile in zip(ids,springs,strict=True):
            metadata['actuator_joint_map'].append(dict(asset='clamp',actuator_group=profile['joint_name']+'_spring',
                joint_names=[profile['joint_name']],joint_indices=[int(index)],position_unit=profile['position_unit'],
                effort_unit=profile['effort_unit'],command_mode='Fixed mechanical spring rest; no adaptive command.'))
        metadata['verified_live_mechanism_drive']=verify_live_arm_gains(clamp,{x['joint_name']:dict(
            joint_names=[x['joint_name']],applied=dict(stiffness=x['stiffness'],damping=x['damping'],
                joint_effort_limit=x['effort_limit'],joint_velocity_limit=x['velocity_limit'])) for x in springs})
        spring_targets=[x['constant_target'] for x in springs]
        clamp.actuators.target_command.set_position_index(value=torch.tensor([spring_targets],device=base.device,dtype=torch.float32),joint_ids=list(ids))
        metadata['initial_passive_target_readback']={x['joint_name']:mechanism_target_readback(clamp,x['joint_name'],x['constant_target']) for x in springs}
        metadata['initial_mechanism_target_readback']=metadata['initial_passive_target_readback'][joint_name]
        writer=getattr(clamp,'write_data_to_sim',None)
        if not callable(writer):raise RuntimeError('Public passive spring target upload is unavailable.')
        writer()
        metadata['initial_native_passive_drives']={x['joint_name']:native_mechanism_readback(clamp,x,x['constant_target'],expected_dof_count=len(springs)) for x in springs}
        metadata['initial_native_mechanism_drive']=metadata['initial_native_passive_drives'][joint_name]
        metadata['trial_wall_budget_s']=TRIAL_WALL_BUDGET_S
        angle=math.radians(plan['world_yaw_deg']);c,s=math.cos(angle),math.sin(angle)
        r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        origin=_array(base.scene.env_origins,(1,3))[0]
        metadata['pillar_top_centers_world_m']=(np.asarray(metadata['pillar_top_centers_world_m'])@r.T+translation+origin).tolist()
        if 'fixture_pivot_world_m' in metadata.get('geometry',{}):
            point=np.asarray(metadata['geometry']['fixture_pivot_world_m'])
            metadata['geometry']['fixture_pivot_world_m']=(r@point+translation+origin).tolist()
        metadata['filter_partner_names']=partners;metadata['filter_prim_paths_expr']=filters
        metadata['mechanism_actuation']=actuation
        metadata['passive_spring_actuations']=springs
        _json(destination/'metadata.json',metadata)
        preload_reference={'control_index':None}
        if extension is not None:extension.preload_reference=preload_reference
        reader=make_reader(base,metadata,partners,preload_reference,extension=extension);recorder=PhysicsRecorder(base.scene,reader,destination,
            metadata['physics_dt_s'],sample_guard=lambda row:(guard_physics_loads_v12(row,metadata['pad_compliance'],metadata['passive_hinge']),None if extension is None else extension.guard(row)),
            rows=extension.trace_rows()if extension is not None and hasattr(extension,'trace_rows')else None)
        metadata['jaw_load_guard']=dict(maximum_N=15.,checked_at='Every recorded physics step after persisting its measurements',
            capture_hz=1/metadata['physics_dt_s'])
        _json(destination/'metadata.json',metadata)
        raw_recorder=recorder
        stage='initial measured snapshot';initial_row=recorder.initial(getattr(extension,'initial_phase','initialization'));started=time.monotonic();last_phase=None
        if extension is not None and hasattr(extension,'before_prefix'):
            stage='tool_pickup'
            recorder=extension.before_prefix(env,base,metadata,raw_recorder,plan,destination,torch)
            initial_row=recorder.rows[0]
            started=time.monotonic()
        from wafer_grasp_feedback import BoundedTCPFeedback
        robot_base=np.asarray(station.BASES['left'])+origin
        servo=BoundedTCPFeedback(lambda q:ik.tcp_pose(q,robot_base),
            lambda p,r,q:ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=robot_base,iters=100,damping=.015,tol=1e-7),
            ik.LOWER,ik.UPPER)
        previous_command=_array(plan['initial_left_q_rad'],(6,))
        if extension is not None and hasattr(extension,'initialize_prefix_feedback'):
            previous_command=extension.initialize_prefix_feedback(servo,plan,metadata)
        if extension is not None and hasattr(extension,'configure_prefix_feedback'):
            servo=extension.configure_prefix_feedback(servo,plan,metadata,initial_row,lambda:recorder.rows[-1])
        finger_controller=FingerPreloadController(wafer_prefix_phases(metadata))
        previous_finger_command=[float(plan['initial_finger_q_m'])]*2
        command_stream=(destination/'commands.jsonl').open('x',encoding='utf-8')
        finger_stream=(destination/'finger_preload.jsonl').open('x',encoding='utf-8')
        if plan['tcp_feedback']['enabled']:feedback_stream=(destination/'tcp_feedback.jsonl').open('x',encoding='utf-8')
        for (phase,action),mechanism_target in zip(command_schedule(plan),mechanism_target_schedule(plan,actuation),strict=True):
            if time.monotonic()-started>TRIAL_WALL_BUDGET_S:raise RuntimeError('1200 second trial wall budget exceeded before next action.')
            if phase=='lift' and prelift_diagnostic is None:
                stage='prelift_readiness'
                prelift_diagnostic=prelift_readiness(metadata,recorder.rows)
                _json(destination/'prelift_readiness.json',prelift_diagnostic)
                if prelift_diagnostic.get('passed') is not True:
                    raise TrialAbort('Original prelift readiness criteria did not pass; no lift command was issued.')
            if phase!=last_phase:print('Wafer phase: '+phase,flush=True);last_phase=phase
            nominal_action=copy.deepcopy(action);feedback=None
            if extension is not None:action[13]=-.035/constants.FINGER_OPEN
            if plan['tcp_feedback']['enabled']:
                action,feedback=feedback_command(servo,action,recorder.rows[-1]['measured_hand_pose_world'],
                    previous_command,constants.TCP_OFFSET,phase,plan['tcp_feedback']['integrating_phases'])
                feedback.update(time_s=recorder.total*recorder.dt,phase=phase,commanded_q_rad=action[:6])
                feedback_stream.write(json.dumps(feedback,allow_nan=False)+'\n');feedback_stream.flush()
            previous_command=np.asarray(action[:6])
            action,previous_finger_command,finger_diagnostic=finger_preload_command(finger_controller,action,phase,
                recorder.rows,previous_finger_command,constants.FINGER_OPEN,metadata['physics_dt_s'])
            finger_records.append(copy.deepcopy(finger_diagnostic))
            finger_stream.write(json.dumps(finger_diagnostic,allow_nan=False)+'\n');finger_stream.flush()
            preload_reference['control_index']=finger_diagnostic['control_index']
            command=dict(command_index=len(commands),start_time_s=recorder.total*recorder.dt,
                         phase=phase,nominal_target=nominal_action,target=action,mechanism_target=mechanism_target,
                         passive_spring_targets={x['joint_name']:x['constant_target'] for x in springs},
                         finger_preload_control_index=finger_diagnostic['control_index'],
                         requested_finger_target_q_m=previous_finger_command.copy(),
                         tcp_feedback_command_index=None if feedback is None else feedback['command_index'])
            commands.append(command);command_stream.write(json.dumps(command,allow_nan=False)+'\n');command_stream.flush()
            stage=phase;recorder.begin(phase)
            env.step(torch.tensor([action],device=base.device,dtype=torch.float32))
            row=recorder.end();command['completed_time_s']=row['time_s']
            command['finger_target_readback']=finger_target_readback(base.scene['left_robot'],previous_finger_command)
            public_targets,native_targets=passive_target_snapshot(clamp,len(springs))
            command['mechanism_target_readback']=mechanism_target_readback(clamp,joint_name,mechanism_target,public_targets)
            command['native_mechanism_target_readback']=native_mechanism_readback(clamp,actuation,mechanism_target,include_parameters=False,expected_dof_count=len(springs),target_snapshot=native_targets)
            command['passive_target_readbacks']={x['joint_name']:mechanism_target_readback(clamp,x['joint_name'],x['constant_target'],public_targets) for x in springs}
            command['native_passive_target_readbacks']={x['joint_name']:native_mechanism_readback(clamp,x,x['constant_target'],include_parameters=False,expected_dof_count=len(springs),target_snapshot=native_targets) for x in springs}
            if max(row['jaw_contact_normal_force_N'])>15:raise TrialAbort('Jaw load exceeded the declared simulation bound.')
            if actuation['mode']=='constant_opening_spring' and abs(row['mechanism_joint_velocity'])>actuation['velocity_limit']+1e-6:
                raise TrialAbort('Passive hinge speed exceeded the range used to bound its spring effort.')
            if not -.004<row['measured_jaw_gap_m']<.015:raise TrialAbort('Measured jaw gap left the bounded mechanism range.')
            if row['wafer_pose_world'][2]<min(x[2] for x in metadata['pillar_top_centers_world_m'])-.005:
                raise TrialAbort('Wafer dropped below the support plane.')
            if len(commands)%20==0:print(f"Wafer trial: {row['time_s']:.1f} s; jaw forces {row['jaw_contact_normal_force_N']}",flush=True)
        evaluation_rows=recorder.rows
        if extension is not None and getattr(extension,'full_workflow_adapter',False):
            from wafer_full_cycle_isaac import compact_evaluation_rows
            evaluation_rows=compact_evaluation_rows(evaluation_rows)
        result=evaluate_wafer_grasp_trial(metadata,evaluation_rows)
        result['backend']='Isaac Lab / PhysX with YAM; prealigned-stage only'
        limit=result['limits']['maximum_unwanted_contact_N']
        maxima={key:max(r[key] for r in evaluation_rows) for key in ('unexpected_robot_tool_contact_normal_force_N',
            'unexpected_robot_environment_contact_normal_force_N')}
        maximum_combined=max(r['unexpected_robot_tool_contact_normal_force_N']+
            r['unexpected_robot_environment_contact_normal_force_N'] for r in evaluation_rows)
        clear=maximum_combined<=limit
        result['checks']['robot_obstacle_contacts']=dict(passed=clear,maximum_unwanted_contact_N=limit,maxima=maxima,
            maximum_combined_unwanted_normal_residual_N=maximum_combined,
            scope='Whole-body force-vector and filtered-partner residuals; vector cancellation can hide unmatched contacts. Recorded-pose geometry screening is separate.')
        if not clear:
            result.setdefault('failed_checks',[]).append('robot_obstacle_contacts');result['result']=result['lift_result']='FAILED'
        if actuation['mode']=='constant_opening_spring':
            maximum_speed=max(abs(r['mechanism_joint_velocity']) for r in evaluation_rows)
            hinge_clear=maximum_speed<=actuation['velocity_limit']+1e-6
            result['checks']['passive_spring_speed_bound']=dict(passed=hinge_clear,maximum_measured_rad_s=maximum_speed,
                maximum_allowed_rad_s=actuation['velocity_limit'],constant_rest_target_rad=actuation['constant_target'],
                effort_limit_Nm=actuation['effort_limit'])
            if not hinge_clear:
                result.setdefault('failed_checks',[]).append('passive_spring_speed_bound');result['result']=result['lift_result']='FAILED'
        del evaluation_rows
        if extension is not None and result.get('result')==PASS:
            stage='air_return_cycle'
            result=extension.run(env,base,metadata,recorder,commands[-1]['target'],finger_controller,
                previous_finger_command,destination,torch,result)
    except Exception as exc:
        result=dict(result='FAILED' if isinstance(exc,TrialAbort) else 'INCOMPLETE',
                    lift_result='FAILED' if isinstance(exc,TrialAbort) else 'NOT_EVALUATED',return_result='NOT_EVALUATED',
                    stage=stage,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if raw_recorder is not None and raw_recorder.rows:result['stopped_phase']=raw_recorder.rows[-1]['phase']
        print('Wafer trial stopped during '+stage+': '+str(exc),flush=True)
    finally:
        if prelift_diagnostic is not None:
            result['prelift_readiness']=prelift_diagnostic
            if prelift_diagnostic.get('passed') is not True:
                result.setdefault('failed_checks',[]).append('prelift_not_ready')
                result.setdefault('checks',{})['prelift_not_ready']=dict(passed=False,evidence=prelift_diagnostic)
                if result.get('result')!='INCOMPLETE':result['result']=result['lift_result']='FAILED'
        if finger_stream is not None:finger_stream.flush();finger_stream.close()
        if feedback_stream is not None:feedback_stream.flush();feedback_stream.close()
        if command_stream is not None:command_stream.flush();command_stream.close()
        if commands:_json(destination/'commands.json',commands)
        if finger_records:
            try:
                result['finger_preload_command_replay']=replay_commands(finger_records,wafer_prefix_phases(metadata))
                prefix_rows=recorder.rows if extension is None else recorder.rows[:1+round(sum(d for _,d in wafer_prefix_phases(metadata))/metadata['physics_dt_s'])]
                result['finger_preload_input_output_audit']=audit_finger_preload_evidence(prefix_rows,finger_records,commands,
                    plan['native_finger_open_q_m'],metadata['physics_dt_s'],wafer_prefix_phases(metadata))
            except (KeyError,TypeError,ValueError)as exc:
                result['finger_preload_evidence_error']=str(exc);result['result']='INCOMPLETE'
        if recorder is not None:
            global_rows=(raw_recorder or recorder).rows
            if raw_recorder is not None and recorder is not raw_recorder:
                result['wafer_stage_trace']=dict(origin_global_s=recorder.origin_s,initial_global_sample_index=recorder.initial_index,stage_sample_count=len(recorder.rows))
            result['recorded_physics_samples']=len(global_rows)
            result['physics_steps_in_current_action']=recorder.count
            result['completed_control_actions']=sum('completed_time_s' in c for c in commands)
            if global_rows:
                result['trace']=dict(sample_count=len(global_rows),physics_dt_s=metadata['physics_dt_s'],
                    start_time_s=global_rows[0]['time_s'],end_time_s=global_rows[-1]['time_s'])
                try:
                    check=evaluate_finger_handle_load(global_rows,engine='isaac')
                    result.setdefault('checks',{})['finger_handle_jam_guard']=check
                    if not check['passed']:
                        result.setdefault('failed_checks',[]).append('finger_handle_jam_guard')
                        if result.get('result')!='INCOMPLETE':result['result']=result['lift_result']='FAILED'
                except ValueError as exc:
                    result.setdefault('checks',{})['finger_handle_jam_guard']=dict(passed=False,reason=str(exc))
                    result['jam_guard_evidence_error']=str(exc);result['result']='INCOMPLETE'
            if global_rows:
                try:
                    check=evaluate_passive_hinge_state(global_rows,metadata['passive_hinge'])
                    result.setdefault('checks',{})['passive_hinge_torque_guard']=check
                    if not check['passed']:
                        result.setdefault('failed_checks',[]).append('passive_hinge_torque_guard')
                        if result.get('result')!='INCOMPLETE':result['result']=result['lift_result']='FAILED'
                except (KeyError,ValueError) as exc:
                    result.setdefault('checks',{})['passive_hinge_torque_guard']=dict(passed=False,reason=str(exc))
                    result['hinge_guard_evidence_error']=str(exc);result['result']='INCOMPLETE'
            if global_rows:
                try:
                    check=evaluate_passive_pad_state(global_rows,metadata['pad_compliance'])
                    result.setdefault('checks',{})['passive_pad_motion_guard']=check
                    if not check['passed']:
                        result.setdefault('failed_checks',[]).append('passive_pad_motion_guard')
                        if result.get('result')!='INCOMPLETE':result['result']=result['lift_result']='FAILED'
                except (KeyError,ValueError) as exc:
                    result.setdefault('checks',{})['passive_pad_motion_guard']=dict(passed=False,reason=str(exc))
                    result['pad_guard_evidence_error']=str(exc);result['result']='INCOMPLETE'
            try:recorder.close()
            except Exception as exc:result['recorder_cleanup_error']=type(exc).__name__+': '+str(exc);result['result']='INCOMPLETE'
        # Save before closing Kit: some versions exit during application cleanup.
        _json(destination/'metadata.json',metadata);_json(destination/'trial_report.json',result)
        print('Wafer result: '+str(result.get('result')),flush=True)
        if env is not None:
            try:env.close()
            except Exception as exc:print('Environment cleanup: '+str(exc),flush=True)
        if app is not None:app.close()
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--check-only',action='store_true',help='Validate frozen files/local gate without importing the simulator.')
    args=parser.parse_args()
    if args.check_only:print(json.dumps(validate_bundle(args.bundle)[3],indent=2))
    else:run_isaac_trial(args.bundle,args.output,device=args.device)
