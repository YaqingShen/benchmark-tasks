"""Render one recorded Isaac run with its USD assets; physics stays disabled."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,sys,traceback
import numpy as np
from verify_develop import validate_development_scene,validate_development_row


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_rows(run,preview):
    meta=json.loads((run/'metadata.json').read_text())
    compact=run/'video_samples.jsonl.gz'
    source=compact if compact.exists() else run/'samples.jsonl.gz'
    if not preview:
        report=json.loads((run/'trial_report.json').read_text())
        audit=json.loads((run/'independent_check.json').read_text())
        if (report['result']!='WAFER_DEVELOP_SUCCEEDED_IN_THIS_SIMULATION' or report.get('errors')
            or audit.get('passed') is not True or audit.get('matches_report') is not True
            or report.get('score')!=1 or report.get('success')is not True):
            raise ValueError('A complete independently checked development run is required.')
        if compact.exists():
            manifest=json.loads((run/'video_replay_manifest.json').read_text())
            if sha(source)!=manifest['selected_trace_sha256'] or audit['trace_sha256']!=manifest['original_trace_sha256']:
                raise ValueError('Replay trace differs from its independent verification.')
        elif sha(source)!=audit['trace_sha256']:raise ValueError('Full trace differs from its audit.')
        for name,digest in audit['source_files_sha256'].items():
            if sha(run/name)!=digest:raise ValueError('Audited result changed: '+name)
    rows=[];next_t=0.;last=None
    try:
        with gzip.open(source,'rt') as f:
            for line in f:
                row=json.loads(line);t=float(row['time_s'])
                validate_development_row(row)
                if last is not None and t<=last:raise ValueError('Non-monotonic recording.')
                last=t
                if t>=next_t-1e-8:rows.append(row);next_t+=.05
    except EOFError:
        if not preview:raise
    if not rows:raise ValueError('Empty recording.')
    if preview:rows=[rows[-1]]
    elif abs(rows[0]['time_s'])>1e-8 or rows[-1]['time_s']>120.:
        raise ValueError('Expected a complete continuous run within 120 seconds.')
    return meta,rows,source


def render(args):
    run=args.run.resolve();bundle=args.bundle.resolve()
    meta,rows,trace=read_rows(run,args.preview)
    validate_development_scene(meta,bundle)
    assets={p.name:sha(p) for p in (bundle/'assets').glob('*.usda') if not p.name.startswith('.')}
    for name,digest in assets.items():
        expected={**meta['full_workflow_sources'],**meta['develop_sources']}.get('assets/'+name)
        if digest!=expected:raise ValueError('Recorded asset changed: '+name)
    sys.path.insert(0,str(bundle))
    from wafer_retention_isaac_v12 import ensure_gpu_idle
    ensure_gpu_idle()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':640,'height':360,'renderer':'PathTracing',
                       'samples_per_pixel_per_frame':4,'denoiser':True,
                       'multi_gpu':False,'disable_viewport_updates':True})
    try:
        import carb
        import omni.usd,omni.timeline,omni.replicator.core as rep
        rep.settings.set_render_pathtraced(samples_per_pixel=8)
        settings=carb.settings.get_settings()
        settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled',True)
        settings.set_bool('/rtx/pathtracing/optixDenoiser/temporalMode/enabled',False)
        settings.set_float('/rtx/pathtracing/optixDenoiser/blendFactor',0.)
        settings.set_bool('/omni/replicator/captureMotionBlur',False)
        from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,Gf
        import isaaclab.sim as sim
        from yam_arena import station
        from yam_arena.robot import yam_arm_cfg,asset_link_paths
        from PIL import Image
        import imageio.v2 as imageio
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
        transforms={}

        def bind(path):
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid():raise ValueError('Missing replay prim: '+path)
            prim.SetInstanceable(False)
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.SetResetXformStack(True)
            op=xf.AddTransformOp();transforms[path]=op
            return path

        def set_pose(path,p):
            p=np.asarray(p,dtype=float)
            if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>1e-4:
                raise ValueError('Invalid measured pose.')
            q=Gf.Quatd(float(p[6]),Gf.Vec3d(*p[3:6]))
            matrix=Gf.Matrix4d(1.);matrix.SetRotate(q);matrix.SetTranslateOnly(Gf.Vec3d(*p[:3]))
            transforms[path].Set(matrix)

        def reference(name,file):
            root='/World/'+name
            stage.DefinePrim(root,'Xform').GetReferences().AddReference(str(file))
            return root

        links=asset_link_paths();robot_paths=[]
        for side in ('left','right'):
            root='/World/'+side+'_robot'
            cfg=yam_arm_cfg(root,station.BASES[side],dream_look=True)
            stage.DefinePrim(root,'Xform').GetReferences().AddReference(cfg.spawn.usd_path)
        for side,name in meta['robot_geometry']['link_keys']:
            robot_paths.append(bind('/World/'+side+'_robot'+links[name]))
        objects={}
        for name in ('wafer','heater','pillar_0','pillar_1','pillar_2','pedestal','tool_stand_handle','tool_stand_hinge'):
            file=bundle/'assets'/f'{name}.usda'
            if file.exists():objects[name]=bind(reference(name,file))
        clamp=reference('clamp',bundle/'assets/clamp.usda')
        clamp_groups={key:[bind(clamp+'/'+name) for name in names] for key,names in (
            ('lever_link_poses_world',('lower_half','upper_half')),
            ('jaw_link_poses_world',('lower_pad','upper_pad')),
            ('handle_paddle_poses_world',('lower_grip_paddle','upper_grip_paddle')))}
        beaker=bind(reference('beaker',bundle/'assets/beaker.usda'))
        timer=reference('timer',bundle/'assets/timer.usda');timer_base=bind(timer+'/timer_base');timer_button=bind(timer+'/timer_button')
        coating=reference('wafer/coating',bundle/'assets/wafer_coating.usda')
        pattern={name:UsdGeom.Imageable(stage.GetPrimAtPath(coating+'/'+name)).CreateVisibilityAttr()for name in ('BEFORE','UNDERDEVELOPED','GOOD','OVERDEVELOPED')}
        digits={(d,s):UsdGeom.Imageable(stage.GetPrimAtPath(timer+f'/timer_base/screen_panel/digit_{d}_{s}')).CreateVisibilityAttr()for d in range(3)for s in range(7)}
        lit=({0,1,2,3,4,5},{1,2},{0,1,3,4,6},{0,1,2,3,6},{1,2,5,6},{0,2,3,5,6},{0,2,3,4,5,6},{0,1,2},{0,1,2,3,4,5,6},{0,1,2,3,5,6})
        report=json.loads((run/'trial_report.json').read_text())
        entered=report.get('immersion_start_s');left=report.get('immersion_end_s')
        quality=report.get('pattern_quality','BEFORE')
        table=UsdGeom.Cube.Define(stage,'/World/table');table.CreateSizeAttr(1.)
        table.CreateDisplayColorAttr([(0.55,0.44,0.25)])
        table.AddTranslateOp().Set(Gf.Vec3d(*station.TABLE_CENTER));table.AddScaleOp().Set(Gf.Vec3d(*station.TABLE_SIZE))
        UsdLux.DomeLight.Define(stage,'/World/fill').CreateIntensityAttr(600.)
        key=UsdLux.DistantLight.Define(stage,'/World/key');key.CreateIntensityAttr(1800.)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set((315.,20.,30.))
        for prim in list(stage.Traverse(Usd.TraverseInstanceProxies())):
            if prim.IsInstanceProxy():continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        cameras=[];annotators=[];products=[]
        for name,resolution in [('overview',(640,360)),('detail',(320,200)),('timer_view',(320,160))]:
            path='/World/'+name;cam=UsdGeom.Camera.Define(stage,path)
            cam.CreateClippingRangeAttr((.005,100.));cam.CreateFocalLengthAttr(18. if name=='overview' else 24.)
            cam.CreateHorizontalApertureAttr(24.);cam.CreateVerticalApertureAttr(24.*resolution[1]/resolution[0])
            cameras.append(bind(path))
            product=rep.create.render_product(path,resolution);products.append(product)
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach(product);annotators.append(annotator)

        def aim(path,eye,target,up):
            view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(*up))
            transforms[path].Set(view.GetInverse())

        aim(cameras[0],(-.35,-.06,2.05),(.32,-.10,.82),(0.,0.,1.))
        rep.orchestrator.set_capture_on_play(False)
        omni.timeline.get_timeline_interface().pause()
        args.output.mkdir(parents=True,exist_ok=True)
        video=args.output/'wafer_develop_isaac.mp4';writer=None
        if not args.preview:writer=imageio.get_writer(str(video),fps=20,codec='libx264',quality=8,macro_block_size=8)
        try:
            for index,row in enumerate(rows):
                for path,p in zip(robot_paths,row['robot_link_poses_world']):set_pose(path,p)
                set_pose(objects['wafer'],row['wafer_pose_world'])
                for name,p in row['fixture_poses_world'].items():
                    if name in objects:set_pose(objects[name],p)
                for field,paths in clamp_groups.items():
                    for path,p in zip(paths,row[field]):set_pose(path,p)
                set_pose(beaker,row['beaker_pose_world']);set_pose(timer_base,row['timer_base_pose_world']);set_pose(timer_button,row['timer_button_pose_world'])
                value=f"{min(999,math.ceil(row['timer_remaining_s']*10-1e-6)):03d}"
                for (d,s),attr in digits.items():attr.Set('inherited'if s in lit[int(value[d])]else'invisible')
                state=quality if left is not None and row['time_s']>=left else'BEFORE'
                for name,attr in pattern.items():attr.Set('inherited'if name==state else'invisible')
                tp=np.asarray(row['timer_base_pose_world']);tr=Gf.Rotation(Gf.Quatd(float(tp[6]),Gf.Vec3d(*tp[3:6])))
                panel=tp[:3]+np.asarray(tr.TransformDir(Gf.Vec3d(-.013,.008,.055)))
                normal=np.asarray(tr.TransformDir(Gf.Vec3d(0.,-.8660254038,.5)))
                aim(cameras[2],panel+normal*.24,panel,(0.,0.,1.))
                wp=np.asarray(row['wafer_pose_world']);wrot=Gf.Rotation(Gf.Quatd(float(wp[6]),Gf.Vec3d(*wp[3:6])))
                eye=wp[:3]+np.array(wrot.TransformDir(Gf.Vec3d(0.,0.,.16)))
                up=np.array(wrot.TransformDir(Gf.Vec3d(0.,1.,0.)))
                aim(cameras[1],eye,wp[:3],up)
                rep.orchestrator.step(delta_time=0.,pause_timeline=True,rt_subframes=1)
                images=[np.asarray(a.get_data()) for a in annotators]
                for warm_attempt in range(10):
                    if all(im.ndim==3 and im.shape[2]>=3 and im[...,:3].any() for im in images):break
                    print(f'Waiting for camera output at frame {index}: {warm_attempt+1}',flush=True)
                    rep.orchestrator.step(delta_time=0.,pause_timeline=True,rt_subframes=4)
                    images=[np.asarray(a.get_data()) for a in annotators]
                if any(im.ndim!=3 or im.shape[2]<3 or not im[...,:3].any() for im in images):raise ValueError('Empty rendered frame after bounded camera warmup.')
                canvas=Image.new('RGB',(960,360))
                canvas.paste(Image.fromarray(images[0][...,:3]),(0,0))
                canvas.paste(Image.fromarray(images[1][...,:3]),(640,0))
                canvas.paste(Image.fromarray(images[2][...,:3]),(640,200))
                if index==0:canvas.save(args.output/('diagnostic.png' if args.preview else 'preview.png'))
                if writer is not None:writer.append_data(np.asarray(canvas))
                if index%100==0:print(f'Rendered {index+1}/{len(rows)} frames',flush=True)
        finally:
            if writer is not None:writer.close()
        if any(sha(bundle/'assets'/name)!=digest for name,digest in assets.items()):raise ValueError('Assets changed during rendering.')
        proof=dict(source=str(run),source_trace_sha256=sha(trace),frame_count=len(rows),fps=20,
            source_start_s=rows[0]['time_s'],source_end_s=rows[-1]['time_s'],asset_sha256=assets,
            renderer='Isaac Sim RTX PathTracing; 960x360 review export, 8 samples per pixel',renderer_sha256=sha(__file__),layout='Overview, wafer pattern, and audience-facing physical timer',
            scope='Exact recorded body poses; no physics steps or new success judgement. Review export changes only image resolution and path-tracing sample count.',preview_only=args.preview)
        (args.output/'replay.json').write_text(json.dumps(proof,indent=2)+'\n')
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');render(p.parse_args())
