"""Fresh-process Inspect bridge to the shared native MCP2 world."""
from __future__ import annotations

import atexit
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def bundle_paths(bundle):
    bundle=Path(bundle).expanduser().resolve()
    edge,baseline=bundle/'edge_native',bundle/'baseline'
    required=(edge/'native/parity.json',edge/'run_edge_workflow.py',edge/'mixed_debris_assets.py',
              baseline/'cloud_plan.json',baseline/'full_workflow_plan.json')
    required+=tuple(edge/'native'/('debris_'+kind+'.usda')for kind in ('dust_grain','fiber','flake','wear_fragment'))
    for path in required:
        if not path.is_file():raise ValueError('Missing new-tool bundle file: '+str(path))
    return edge,baseline


def bundle_manifest(bundle):
    edge,baseline=bundle_paths(bundle)
    groups={'edge_native':[ *edge.glob('*.py'), *edge.glob('native/*'),edge/'scene.xml',edge/'batch_report.json'],
            'baseline':[ *baseline.glob('*.py'), *baseline.glob('benchmark/*.py'), *baseline.glob('assets/*.usda'),
                        baseline/'cloud_plan.json',baseline/'full_workflow_plan.json']}
    return {prefix+'/'+str(p.relative_to(edge if prefix=='edge_native'else baseline)):_sha(p)
            for prefix,paths in groups.items()for p in sorted(paths)if p.is_file()and not p.name.startswith('._')}


def bundle_revision(bundle):
    return _digest(bundle_manifest(bundle))


def _layout_seed(value):
    if value is None:return None
    if isinstance(value,bool)or not isinstance(value,(int,np.integer))or int(value)<0:
        raise ValueError('layout_seed must be a nonnegative integer or None.')
    return int(value)


def environment_revision(bundle,layout_seed=None):
    return _digest(dict(bundle_revision=bundle_revision(bundle),layout_seed=_layout_seed(layout_seed)))


def _action(value):
    target=np.asarray(value,dtype=float)
    if target.shape!=(14,)or not np.isfinite(target).all()or np.any(target[[6,13]]<0.)or np.any(target[[6,13]]>1.):
        raise ValueError('Expected14 finite joint targets and normalized grippers in[0,1].')
    return target


def _command_matches(row,command,finger_open):
    action=_action(command['action']).astype(np.float32)
    for start,arm,fingers in ((0,'commanded_arm_q_rad','commanded_finger_q_m'),
                              (7,'commanded_right_arm_q_rad','commanded_right_finger_q_m')):
        actual=np.asarray(row[arm],dtype=np.float32);f=np.asarray(row[fingers],dtype=np.float32)
        expected=np.float32(action[start+6]*np.float32(finger_open))
        if actual.shape!=(6,)or f.shape!=(2,)or not np.array_equal(actual,action[start:start+6])or not np.allclose(f,expected,rtol=0.,atol=1e-8):
            raise ValueError('Recorded joint command differs from native actuator readback.')


def _candidate_bridge(metadata,bundle):
    edge,baseline=bundle_paths(bundle)
    matched={}
    excluded={prefix+'/'+name:digest for prefix,values in (('edge_native',metadata['source_files']),
        ('baseline',metadata['baseline_source_sha256']))for name,digest in values.items()if Path(name).name.startswith('._')}
    for name in ('native_adapter.py','native_air.py','native_evidence.py','wafer_debris_score.py','mixed_debris_assets.py'):
        expected=metadata['source_files'][name]
        if _sha(edge/name)!=expected:raise ValueError('Candidate native component differs: '+name)
        matched['edge_native/'+name]=expected
    dust=metadata['dust']
    if dust.get('shape_schema')!='mixed_convex_debris_v1':raise ValueError('The candidate requires mixed cleanroom-surrogate debris.')
    helper_spec=importlib.util.spec_from_file_location('_mixed_debris_bridge',edge/'mixed_debris_assets.py')
    if helper_spec is None or helper_spec.loader is None:raise ValueError('Cannot load the bound debris asset verifier.')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    for name,(_,digest)in helper.particle_asset_bindings(edge,dust).items():
        matched['edge_native/'+name]=digest
    native=metadata['native_adapter']
    for name,digest in native['native_assets'].items():
        if Path(name).name!=name or _sha(edge/'native'/name)!=digest:raise ValueError('Candidate native asset differs: '+name)
        matched['edge_native/native/'+name]=digest
    for name,digest in (('native/parity.json',native['parity_sha256']),('scene.xml',native['source_scene_sha256'])):
        if _sha(edge/name)!=digest:raise ValueError('Candidate mechanical source differs: '+name)
        matched['edge_native/'+name]=digest
    source=metadata['baseline_source_sha256']
    names=[name for name in source if not Path(name).name.startswith('._')and ('/'not in name or name.startswith('assets/'))]
    names+=['benchmark/collision_geometry.py','benchmark/wafer_robot_geometry.py']
    for name in names:
        path=Path(name)
        if path.is_absolute()or '..'in path.parts:raise ValueError('Invalid baseline source path.')
        if _sha(baseline/path)!=source[name]:raise ValueError('Candidate baseline physics differs: '+name)
        matched['baseline/'+name]=source[name]
    return dict(candidate_revision=bundle_revision(bundle),matched_files=matched,excluded_nonexecuted_appledouble_sidecars=excluded,
                scope='Matching physical assets including mixed-debris meshes and generator, adapter, air, evidence, scorer, plans and baseline physics. '
                      'Camera and shared orchestration changes require a fresh scored execution; source success is not target success.')


def load_recorded_targets(run,candidate_bundle=None):
    run=Path(run).expanduser().resolve()
    names=('metadata.json','trial_report.json','independent_check.json','commands.jsonl','observations.jsonl.gz')
    hashes={name:_sha(run/name)for name in names}
    metadata,report,audit=(_read(run/name)for name in names[:3])
    task=report.get('task_score',{})
    if (report.get('result')!='FULL_DEMONSTRATION_SUCCEEDED'or metadata.get('stage')!='full'
            or metadata.get('version')!='mcp2_native_development_v1'or metadata.get('engine')!='Isaac Lab / PhysX'
            or metadata.get('post_start_pose_write_count')!=0
            or metadata.get('dust',{}).get('post_start_particle_pose_writes')!=0
            or metadata.get('dust',{}).get('shape_schema')!='mixed_convex_debris_v1'
            or metadata.get('dust',{}).get('geometry_evidence',{}).get('status')!='COMPOSED_COLLIDERS_VERIFIED'
            or metadata.get('sources_unchanged_during_run')is not True
            or task.get('score') != 1. or task.get('success')is not True or task.get('evidence_complete')is not True
            or task.get('rubric')!='visible_debris_practical_v6'or task.get('invalid_reasons')):
        raise ValueError('Replay source must be a complete successful mixed-debris native run.')
    provenance=audit.get('source_provenance',{})
    if (audit.get('passed')is not True or audit.get('matches_report')is not True or audit.get('task_score')!=task
            or audit.get('classification')!='NATIVE_EDGE_PICK_SAMPLED_RESCORE'
            or audit.get('original_result')!='FULL_DEMONSTRATION_SUCCEEDED'or audit.get('original_report_unchanged')is not True
            or provenance.get('status')!='FILES_VERIFIED'or provenance.get('baseline_status')!='FILES_VERIFIED'
            or provenance.get('source_files')!=metadata.get('source_files')
            or audit.get('trace_sha256')!=hashes['observations.jsonl.gz']):
        raise ValueError('Replay source lacks a matching complete-file independent audit.')
    for name in ('metadata.json','trial_report.json','commands.jsonl','observations.jsonl.gz'):
        if audit.get('source_files_sha256',{}).get(name)!=hashes[name]:raise ValueError('Audited source file changed: '+name)
    if metadata.get('physics_dt_s')!=.0005 or metadata.get('control_hz')!=20 or metadata.get('native_finger_open_q_m')!=-.04695:
        raise ValueError('Replay source cadence or gripper mapping differs.')
    commands=[json.loads(line)for line in (run/'commands.jsonl').read_text().splitlines()]
    if not 0<len(commands)<=2400:raise ValueError('Replay requires1..2400 native actions.')
    for index,item in enumerate(commands):
        if abs(float(item['time_s'])-index*.05)>1e-8:raise ValueError('Native commands have a time gap or duplicate.')
        _action(item['action'])
    if (report.get('control_actions')!=len(commands)or metadata.get('physics_updates')!=len(commands)*100
            or abs(float(report.get('simulation_seconds',-1))-len(commands)*.05)>1e-8):
        raise ValueError('Source action, duration and physics counts disagree.')
    initial=None;count=0
    with gzip.open(run/'observations.jsonl.gz','rt')as stream:
        for index,line in enumerate(stream):
            row=json.loads(line)
            if index>len(commands)or abs(float(row['time_s'])-index*.05)>1e-8:raise ValueError('Recorded observations have a gap or extra frame.')
            if index:_command_matches(row,commands[index-1],-.04695)
            else:
                arms=[]
                for arm,fingers in (('measured_arm_q_rad','measured_finger_q_m'),('measured_right_arm_q_rad','measured_right_finger_q_m')):
                    a=np.asarray(row[arm],dtype=float);f=np.asarray(row[fingers],dtype=float)
                    if a.shape!=(6,)or f.shape!=(2,)or not np.isfinite(a).all()or not np.isfinite(f).all():raise ValueError('Missing initial measured robot state.')
                    arms.append(np.r_[a,np.clip(f.mean()/-.04695,0.,1.)])
                initial=np.r_[*arms]
            count+=1
    if count!=len(commands)+1 or audit.get('scoring_samples')!=count:raise ValueError('Source final observation is missing.')
    source_manifest={'edge_native/'+name:digest for name,digest in metadata['source_files'].items()}
    source_manifest.update({'baseline/'+name:digest for name,digest in metadata['baseline_source_sha256'].items()})
    source_manifest.update({'edge_native/native/'+name:digest for name,digest in metadata['native_adapter']['native_assets'].items()})
    source_manifest['edge_native/native/parity.json']=metadata['native_adapter']['parity_sha256']
    source_manifest['edge_native/scene.xml']=metadata['native_adapter']['source_scene_sha256']
    assets=metadata['dust'].get('assets',{})
    if set(assets)!={'dust_grain','fiber','flake','wear_fragment'}:raise ValueError('Missing four bound mixed-debris assets.')
    for kind,item in assets.items():
        name=item['asset_file']
        if name!='native/debris_'+kind+'.usda':raise ValueError('Invalid mixed-debris asset path.')
        source_manifest['edge_native/'+name]=item['sha256']
    bridge=None if candidate_bundle is None else _candidate_bridge(metadata,candidate_bundle)
    if any(_sha(run/name)!=digest for name,digest in hashes.items()):raise ValueError('Replay source changed while loading.')
    return dict(targets=[_action(c['action'])for c in commands],initial=initial,source_hashes=hashes,
                source_revision=_digest(source_manifest),execution_revision=None if bridge is None else environment_revision(candidate_bundle,metadata['dust']['seed']),
                bridge=bridge,seed=metadata['dust']['seed'])
class _GeometryIntegrity:
    """Track USD geometry/topology changes; this is not a material-damage model."""
    def __init__(self, stage, roots):
        from pxr import Tf, Usd
        self.stage, self.roots = stage, roots
        self.initial = {name: self.signature(path) for name, path in roots.items()}
        if any(not value for value in self.initial.values()):
            raise ValueError("A declared task object is absent from the composed scene.")
        self.valid = dict.fromkeys(roots, True)
        self.dirty = set()
        self.listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.changed, stage)

    @staticmethod
    def tracked_property(prim, name):
        from pxr import UsdPhysics
        if name in {"physics:velocity", "physics:angularVelocity"}:
            return False
        return (name.startswith(("physics:", "physx", "drive:", "material:binding")) or
                name in {"points", "faceVertexCounts", "faceVertexIndices", "size", "radius", "height", "extent"} or
                (name.startswith("xformOp") and prim and prim.HasAPI(UsdPhysics.CollisionAPI) and
                 not prim.HasAPI(UsdPhysics.RigidBodyAPI)))

    def signature(self, root):
        from pxr import Usd
        prim = self.stage.GetPrimAtPath(root)
        if not prim or not prim.IsActive():
            return None
        values = []
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            properties = []
            for attr in child.GetAttributes():
                name = attr.GetName()
                if self.tracked_property(child, name):
                    properties.append((name, str(attr.Get())))
            relationships = [(r.GetName(), [str(x) for x in r.GetTargets()], str(r.GetAllMetadata()))
                             for r in child.GetRelationships() if r.GetName().startswith(("physics:", "material:binding"))]
            values.append((str(child.GetPath()), child.GetTypeName(), child.IsActive(), properties, relationships))
        return hashlib.sha256(repr(values).encode()).hexdigest()

    def changed(self, notice, sender):
        paths = []
        for path in (*notice.GetResyncedPaths(), *notice.GetChangedInfoOnlyPaths()):
            if not path.IsPropertyPath() or self.tracked_property(self.stage.GetPrimAtPath(path.GetPrimPath()), path.name):
                paths.append(str(path.GetPrimPath()))
        for name, root in self.roots.items():
            if any(p == root or p.startswith(root + "/") or root.startswith(p + "/") for p in paths):
                self.dirty.add(name)

    def snapshot(self):
        for name in self.dirty:
            self.valid[name] = self.valid[name] and self.signature(self.roots[name]) == self.initial[name]
        self.dirty.clear()
        present = {name: bool(self.stage.GetPrimAtPath(root)) and self.stage.GetPrimAtPath(root).IsActive()
                   for name, root in self.roots.items()}
        return present, {name: bool(present[name] and self.valid[name]) for name in self.roots}

    def close(self):
        self.listener.Revoke()


def _worker(connection,config):
    world=None
    try:
        edge,baseline=bundle_paths(config['bundle'])
        sys.path[:0]=[str(edge),str(baseline/'benchmark'),str(baseline)]
        from run_edge_workflow import NativeEdgeWorld
        world=NativeEdgeWorld(edge,baseline,Path(config['output']),stage='full',device=config['device'],
                             seed=config['seed'],enable_cameras=True,width=config['width'],height=config['height'],
                             defer_application_close=True)
        world.start()
        world.metadata['inspect_trial']=dict(scene_id=config['scene_id'],seed=config['seed'],
            requested_trial_seed=config['requested_trial_seed'],layout_seed=config['layout_seed'],
            environment_revision=config['revision'],observations='Three RGB cameras and measured14-D joints only.')
        def snapshot():
            images,state,timestamp=world.observe()
            return dict(ok=True,observation=(images,state,timestamp),score=world.score(),
                diagnostics=dict(physics_updates=world.count*100,
                    particle_force_audit=copy.deepcopy(world.metadata.get('particle_force_audit',{})),
                    scope='Native engineering checks; excluded from policy observations.'),
                evidence=dict(directory=config['output'],trace='observations.jsonl.gz',commands='commands.jsonl',
                              observation_hz=20,physics_hz=2000))
        connection.send(snapshot())
        while True:
            operation,value=connection.recv()
            if operation=='step':
                world.step_action(_action(value),'policy')
                connection.send(snapshot())
            elif operation=='finish':
                report=world.finish()
                connection.send(dict(ok=True,report=report))
                break
            else:raise ValueError('Unknown owned-worker request.')
    except EOFError:
        if world is not None:world.close(dict(result='INCOMPLETE',error='Inspect parent disconnected.'))
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        if world is not None:
            try:world.close(dict(result='INCOMPLETE',error=error))
            except Exception:error+='; cleanup failed: '+traceback.format_exc()
        try:connection.send(dict(ok=False,error=error,traceback=traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        connection.close()
        if world is not None:
            try:world.close()
            finally:world.shutdown_application()


class NativeWaferWorld:
    def __init__(self,bundle,evidence_dir,width,height,device='cpu',layout_seed=None):
        self.bundle=Path(bundle).expanduser().resolve();_,baseline=bundle_paths(self.bundle)
        if _sha(__file__)!=_sha(baseline/'benchmark/wafer_benchmark_backend.py'):
            raise ValueError('Installed backend differs from the selected bundle.')
        self.evidence_dir=Path(evidence_dir).expanduser().resolve()
        self.width,self.height=int(width),int(height)
        self.device=str(device)
        if self.device!='cpu':raise ValueError('This candidate is qualified only for the native CPU PhysX configuration.')
        self.layout_seed=_layout_seed(layout_seed)
        self.bundle_revision=bundle_revision(self.bundle)
        self.revision=environment_revision(self.bundle,self.layout_seed)
        self.process=self.connection=self.destination=None
        self.cached=None;self.report=None
        atexit.register(self.close)

    def _receive(self,timeout):
        deadline=time.monotonic()+timeout
        while True:
            if self.connection.poll(min(1.,max(0.,deadline-time.monotonic()))):
                try:packet=self.connection.recv()
                except EOFError:raise RuntimeError('Native worker closed without a response.')from None
                if not isinstance(packet,dict)or packet.get('ok')is not True:
                    raise RuntimeError(packet.get('error','Invalid native worker response.')if isinstance(packet,dict)else'Invalid native worker response.')
                return packet
            if not self.process.is_alive():raise RuntimeError('Native worker exited with code'+str(self.process.exitcode))
            if time.monotonic()>=deadline:raise TimeoutError('Native worker response timed out; only the owned worker will be closed.')

    def reset(self,*,seed,scene_id):
        self.close()
        if bundle_revision(self.bundle)!=self.bundle_revision:raise ValueError('Immutable benchmark bundle changed.')
        requested_trial_seed=_layout_seed(seed)
        seed=self.layout_seed if self.layout_seed is not None else(17 if requested_trial_seed is None else requested_trial_seed)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        self.destination=self.evidence_dir/stamp
        context=multiprocessing.get_context('spawn')
        parent,child=context.Pipe(duplex=True)
        config=dict(bundle=str(self.bundle),output=str(self.destination),width=self.width,height=self.height,
                    device=self.device,seed=seed,requested_trial_seed=requested_trial_seed,layout_seed=self.layout_seed,
                    scene_id=str(scene_id),revision=self.revision)
        self.process=context.Process(target=_worker,args=(child,config),name='wafer-edge-owned-world')
        self.connection=parent;self.report=None;self.cached=None
        self.process.start();child.close()
        try:self.cached=self._receive(600.)
        except Exception:
            self.close();raise

    def step(self,target):
        if self.process is None or self.report is not None:raise RuntimeError('Reset before stepping the native world.')
        self.connection.send(('step',_action(target)))
        self.cached=self._receive(180.)

    def observe(self):
        if self.cached is None:raise RuntimeError('No native observation is available.')
        images,state,timestamp=self.cached['observation']
        if set(images)!={'top_cam','left_cam','right_cam'}:raise ValueError('Incomplete native camera inventory.')
        for name,rgb in images.items():
            rgb=np.asarray(rgb)
            if rgb.shape!=(self.height,self.width,3)or rgb.dtype!=np.uint8:raise ValueError('Invalid RGB camera frame: '+name)
        return {k:np.ascontiguousarray(v)for k,v in images.items()},_action(state).copy(),float(timestamp)

    def score(self):
        if self.cached is None:raise RuntimeError('No measured score is available.')
        result=copy.deepcopy(self.report['task_score']if self.report is not None else self.cached['score'])
        if self.report is not None and self.report.get('result')=='INCOMPLETE':
            result.update(score=0.,success=False,evidence_complete=False)
            result['invalid_reasons']=[*result.get('invalid_reasons',[]),'native_close_incomplete']
        return result

    def diagnostics(self):
        return copy.deepcopy(self.cached['diagnostics'])

    def evidence_reference(self):
        return dict(self.cached['evidence'])

    def finish(self):
        if self.report is not None:return self.report
        if self.process is None:return None
        self.connection.send(('finish',None))
        report=self._receive(180.)['report']
        self.process.join(timeout=180.)
        if self.process.is_alive():raise TimeoutError('Owned worker application shutdown timed out.')
        if self.process.exitcode!=0:raise RuntimeError('Owned worker application shutdown failed with code '+str(self.process.exitcode))
        if report.get('task_score')is None:raise RuntimeError('Native close omitted the final measured score.')
        if _read(self.destination/'trial_report.json')!=report:raise RuntimeError('Final native report changed during application shutdown.')
        self.report=report
        return self.report

    def close(self):
        if self.process is None:return
        try:
            if self.process.is_alive()and self.report is None:
                try:self.finish()
                except (RuntimeError,EOFError,BrokenPipeError,OSError,TimeoutError):pass
            self.process.join(timeout=10.)
            if self.process.is_alive():
                self.process.terminate();self.process.join(timeout=10.)
            if self.process.is_alive():raise RuntimeError('Owned native worker did not stop; no new world may start.')
        finally:
            if self.connection is not None:self.connection.close()
            if not self.process.is_alive():
                self.process.close();self.process=None;self.connection=None
