"""Render measured native body poses; no simulation steps or text overlays."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Invalid evidence or asset path: ' + str(name))
    return path


def read_run(run, preview=False, times=()):
    metadata = json.loads((run / 'metadata.json').read_text())
    trace = run / 'observations.jsonl.gz'
    source_hashes = {name: sha(run / name) for name in ('metadata.json', 'trial_report.json', trace.name)}
    if metadata.get('version') != 'mcp2_native_development_v1':
        raise ValueError('This renderer requires the new three-body native tool run.')
    if metadata.get('sources_unchanged_during_run') is not True:
        raise ValueError('Recorded source integrity was not verified.')
    if not preview:
        audit_path = run / 'independent_check.json'
        audit = json.loads(audit_path.read_text())
        score = audit.get('task_score', {})
        if (audit.get('passed') is not True or audit.get('matches_report') is not True
                or score.get('score') != 1. or score.get('success') is not True):
            raise ValueError('A new independently verified full-score run is required.')
        if audit.get('trace_sha256') != source_hashes[trace.name]:
            raise ValueError('The actual replay trace differs from the audit.')
        for name in ('metadata.json', 'trial_report.json'):
            if audit.get('source_files_sha256', {}).get(name) != source_hashes[name]:
                raise ValueError('The closed evidence differs from its audit: ' + name)
        source_hashes[audit_path.name] = sha(audit_path)
    rows = []
    with gzip.open(trace, 'rt') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            t = float(row['time_s'])
            if not np.isfinite(t) or abs(t - index * .05) > 1e-8 or t > 120.00000001:
                raise ValueError('Replay must contain continuous 20 Hz observations from time zero.')
            if set(row['tool_pose_world']) != {'base', 'jaw', 'squeeze_slider'}:
                raise ValueError('Missing actual three-body tool poses.')
            if len(row['gun_link_poses_world']) != 2 or len(row['dust_pose_world']) != 12:
                raise ValueError('Missing actual gun-trigger or visible marker poses.')
            rows.append(row)
    if not rows:
        raise ValueError('Empty replay trace.')
    full_count = len(rows)
    if preview:
        if times:
            indices = []
            for t in times:
                index = round(float(t) * 20)
                if index < 0 or index >= len(rows):
                    raise ValueError('Requested still time is outside the actual recording.')
                indices.append(index)
            rows = [rows[i] for i in sorted(set(indices))]
        else:
            rows = [rows[-1]]
    elif times:
        raise ValueError('Selected still times require --preview.')
    return metadata, rows, source_hashes, full_count


def asset_bindings(root, baseline, metadata):
    bindings = {}
    native = metadata['native_adapter']
    for name, expected in native['native_assets'].items():
        path = checked_path(root / 'native', name)
        if sha(path) != expected:
            raise ValueError('Recorded native asset changed: ' + name)
        bindings['native/' + name] = (path, expected)
    parity_path = checked_path(root / 'native', 'parity.json')
    if sha(parity_path) != native['parity_sha256']:
        raise ValueError('Native asset parity metadata changed.')
    bindings['native/parity.json'] = (parity_path, native['parity_sha256'])
    parity = json.loads(parity_path.read_text())
    exported = metadata['gun_native_export']
    baseline_assets = [exported['models']['gun'], exported['models']['gun_stand']]
    if metadata['dust'].get('shape_schema'):
        from mixed_debris_assets import particle_asset_bindings
        bindings.update(particle_asset_bindings(root, metadata['dust']))
    else:
        baseline_assets.append('assets/dust.usda')
    for name in baseline_assets:
        path = checked_path(baseline, name)
        expected = metadata['baseline_source_sha256'].get(name)
        if expected is None or sha(path) != expected:
            raise ValueError('Recorded baseline asset changed: ' + name)
        bindings['baseline/' + name] = (path, expected)
    if metadata.get('gun_link_pose_names') != ['blow_gun', 'gun_trigger']:
        raise ValueError('Unknown measured gun link ordering.')
    return bindings, parity


def render(args):
    root, baseline, run = args.root.resolve(), args.baseline.resolve(), args.run.resolve()
    metadata, rows, evidence_hashes, full_count = read_run(run, args.preview, args.time)
    bindings, parity = asset_bindings(root, baseline, metadata)
    sys.path.insert(0, str(baseline))
    from wafer_retention_isaac_v12 import ensure_gpu_idle
    ensure_gpu_idle()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 1280, 'height': 720, 'renderer': 'PathTracing',
                         'samples_per_pixel_per_frame': 32, 'denoiser': True,
                         'multi_gpu': False, 'disable_viewport_updates': True})
    writer = None
    try:
        import carb
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
        from yam_arena import station
        from yam_arena.robot import yam_arm_cfg, asset_link_paths
        from PIL import Image
        import imageio.v2 as imageio

        rep.settings.set_render_pathtraced(samples_per_pixel=32)
        settings = carb.settings.get_settings()
        settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled', True)
        settings.set_bool('/rtx/pathtracing/optixDenoiser/temporalMode/enabled', False)
        settings.set_float('/rtx/pathtracing/optixDenoiser/blendFactor', 0.)
        settings.set_bool('/omni/replicator/captureMotionBlur', False)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        transforms = {}
        expected_transforms = {}
        maximum_transform_error = 0.

        def bind(path):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError('Missing measured body in the rendered asset: ' + path)
            prim.SetInstanceable(False)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            transforms[path] = xf.AddTransformOp()
            return path

        def pose(path, value):
            value = np.asarray(value, dtype=float)
            if value.shape != (7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:]) - 1.) > 1e-4:
                raise ValueError('Invalid measured XYZ/XYZW pose.')
            matrix = Gf.Matrix4d(1.)
            matrix.SetRotate(Gf.Quatd(float(value[6]), Gf.Vec3d(*value[3:6])))
            matrix.SetTranslateOnly(Gf.Vec3d(*value[:3]))
            transforms[path].Set(matrix)
            expected_transforms[path] = matrix

        def reference(name, path):
            prim = stage.DefinePrim('/World/' + name, 'Xform')
            prim.GetReferences().AddReference(str(path))
            return str(prim.GetPath())

        robot_paths, robot_asset_hashes = [], {}
        links = asset_link_paths()
        for side in ('left', 'right'):
            cfg = yam_arm_cfg('/World/' + side + '_robot', station.BASES[side], dream_look=True)
            reference(side + '_robot', cfg.spawn.usd_path)
            path = Path(cfg.spawn.usd_path)
            if path.is_file():
                robot_asset_hashes[side] = sha(path)
        for side, name in metadata['robot_geometry']['link_keys']:
            robot_paths.append(bind('/World/' + side + '_robot' + links[name]))
        clamp = reference('clamp', root / 'native' / parity['asset_file'])
        tool_paths = {name: bind(clamp + '/' + name) for name in ('base', 'jaw', 'squeeze_slider')}
        appearance = metadata['native_adapter']['spawn_checks']['clamp'].get('appearance_only', {})
        collider_paths = {g['name']: g['path'].removeprefix('/Asset') for g in parity['colliders']}
        for name, color in appearance.items():
            UsdGeom.Gprim(stage.GetPrimAtPath(clamp + collider_paths[name])).CreateDisplayColorAttr([tuple(color)])
        wafer = bind(reference('wafer', root / 'native' / parity['wafer_asset']['asset_file']))
        support_root = reference('supports', root / 'native' / parity['support_asset']['asset_file'])
        fixture_paths = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(support_root), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                fixture_paths[prim.GetName()] = bind(str(prim.GetPath()))
        exported = metadata['gun_native_export']
        gun = reference('gun', baseline / exported['models']['gun'])
        gun_paths = [bind(gun + '/' + name) for name in metadata['gun_link_pose_names']]
        fixture_paths['gun_stand'] = bind(reference('gun_stand', baseline / exported['models']['gun_stand']))
        dust = metadata['dust']
        markers = []
        if dust.get('shape_schema'):
            for particle in dust['particles']:
                marker = reference(particle['id'], root / particle['asset_file'])
                mesh = UsdGeom.Mesh(stage.GetPrimAtPath(marker + '/Particle'))
                if not mesh:
                    raise ValueError('Missing actual convex particle mesh: ' + particle['id'])
                scale = np.asarray(particle['scale_m'], dtype=float)
                # Measured root poses omit spawn scale; keep that scale on the mesh.
                UsdGeom.Xformable(mesh).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
                mesh.CreateDisplayColorAttr([tuple(particle['color_rgb'])])
                if not np.allclose(np.asarray(mesh.GetPointsAttr().Get()) * scale,
                                   particle['vertices_local_m'], rtol=0, atol=1e-10):
                    raise ValueError('Rendered particle dimensions differ from recorded geometry.')
                markers.append(bind(marker))
        else:
            markers = [bind(reference(name, baseline / 'assets/dust.usda')) for name in dust['ids']]
        if len(fixture_paths) != 8 or set(fixture_paths) != set(rows[0]['fixture_poses_world']):
            raise ValueError('Rendered fixed-body inventory differs from actual measured fixtures.')

        table = UsdGeom.Cube.Define(stage, '/World/table')
        table.CreateSizeAttr(1.)
        table.CreateDisplayColorAttr([(.55, .44, .25)])
        table.AddTranslateOp().Set(Gf.Vec3d(*station.TABLE_CENTER))
        table.AddScaleOp().Set(Gf.Vec3d(*station.TABLE_SIZE))
        UsdLux.DomeLight.Define(stage, '/World/fill').CreateIntensityAttr(600.)
        key = UsdLux.DistantLight.Define(stage, '/World/key')
        key.CreateIntensityAttr(1800.)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set((315., 20., 30.))
        for prim in list(stage.Traverse(Usd.TraverseInstanceProxies())):
            if prim.IsInstanceProxy():
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        cameras, annotators = [], []
        for name, size in [('overview', (1280, 720)), ('wafer_detail', (640, 720))]:
            path = '/World/' + name
            camera = UsdGeom.Camera.Define(stage, path)
            camera.CreateClippingRangeAttr((.005, 100.))
            camera.CreateFocalLengthAttr(18. if name == 'overview' else 24.)
            camera.CreateHorizontalApertureAttr(24.)
            camera.CreateVerticalApertureAttr(24. * size[1] / size[0])
            cameras.append(bind(path))
            product = rep.create.render_product(path, size)
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach(product)
            annotators.append(annotator)

        def aim(path, eye, target, up):
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
            matrix = view.GetInverse()
            transforms[path].Set(matrix)
            expected_transforms[path] = matrix

        aim(cameras[0], (-.35, -.06, 2.05), (.32, -.10, .82), (0., 0., 1.))
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        initial_time = timeline.get_current_time()

        def verify_paused_scene():
            if timeline.is_playing() or timeline.get_current_time() != initial_time:
                raise RuntimeError('Replay unexpectedly advanced or resumed the timeline.')
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            error = 0.
            for path, expected in expected_transforms.items():
                actual = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                error = max(error, float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
            if error > 1e-7:
                raise RuntimeError(f'Rendered USD scene differs from measured transforms: {error:g}.')
            return error
        args.output.mkdir(parents=True, exist_ok=False)
        if not args.preview:
            writer = imageio.get_writer(str(args.output / 'wafer_edge_pick_isaac.mp4'), fps=20,
                codec='libx264', quality=8, macro_block_size=16, output_params=['-movflags', '+faststart', '-pix_fmt', 'yuv420p'])
        for index, row in enumerate(rows):
            if len(row['robot_link_poses_world']) != len(robot_paths):
                raise ValueError('Recorded robot-body count differs.')
            for path, value in zip(robot_paths, row['robot_link_poses_world']):
                pose(path, value)
            for name, path in tool_paths.items():
                pose(path, row['tool_pose_world'][name])
            pose(wafer, row['wafer_pose_world'])
            if set(fixture_paths) != set(row['fixture_poses_world']):
                raise ValueError('A measured fixture body disappeared.')
            for name, path in fixture_paths.items():
                pose(path, row['fixture_poses_world'][name])
            for path, value in zip(gun_paths, row['gun_link_poses_world']):
                pose(path, value)
            for path, value in zip(markers, row['dust_pose_world']):
                pose(path, value)
            wp = np.asarray(row['wafer_pose_world'])
            rotation = Gf.Rotation(Gf.Quatd(float(wp[6]), Gf.Vec3d(*wp[3:6])))
            eye = wp[:3] + np.asarray(rotation.TransformDir(Gf.Vec3d(0., 0., .18)))
            up = np.asarray(rotation.TransformDir(Gf.Vec3d(0., 1., 0.)))
            aim(cameras[1], eye, wp[:3], up)
            # Flush authored transforms before capturing the same paused state.
            timeline.pause()
            maximum_transform_error = max(maximum_transform_error, verify_paused_scene())
            app.update()
            maximum_transform_error = max(maximum_transform_error, verify_paused_scene())
            rep.orchestrator.step(delta_time=0., pause_timeline=True, rt_subframes=4, wait_for_render=True)
            maximum_transform_error = max(maximum_transform_error, verify_paused_scene())
            images = [np.asarray(a.get_data()) for a in annotators]
            if any(im.ndim != 3 or im.shape[2] < 3 or not im[..., :3].any() for im in images):
                raise RuntimeError('Empty rendered frame.')
            canvas = Image.new('RGB', (1920, 720))
            canvas.paste(Image.fromarray(images[0][..., :3]), (0, 0))
            canvas.paste(Image.fromarray(images[1][..., :3]), (1280, 0))
            if args.preview:
                canvas.save(args.output / f"still_{row['time_s']:07.2f}s.png")
            elif index == 0:
                canvas.save(args.output / 'preview.png')
            if writer is not None:
                writer.append_data(np.asarray(canvas))
            if index % 100 == 0:
                print(f'Rendered {index + 1}/{len(rows)} frames', flush=True)
        if writer is not None:
            writer.close()
            writer = None
        if any(sha(path) != digest for path, digest in bindings.values()):
            raise ValueError('A source asset changed during replay.')
        if any(sha(run / name) != digest for name, digest in evidence_hashes.items()):
            raise ValueError('Recorded evidence changed during replay.')
        proof = dict(source_trace_sha256=evidence_hashes['observations.jsonl.gz'],
            source_files_sha256=evidence_hashes, asset_sha256={k: v[1] for k, v in bindings.items()},
            native_robot_visual_asset_sha256=robot_asset_hashes, frame_count=len(rows),
            original_observation_count=full_count, fps=20, source_start_s=rows[0]['time_s'],
            source_end_s=rows[-1]['time_s'], renderer_sha256=sha(__file__),
            renderer='Isaac Sim PathTracing, 32 samples, spatial denoising, no temporal denoising',
            capture_sync=dict(app_update_before_capture=True, rt_subframes=4, wait_for_render=True,
                timeline_time_unchanged=True, maximum_usd_transform_matrix_error=maximum_transform_error,
                scope='USD body/camera transform readback; pixel correspondence requires checkpoint inspection.'),
            preview_only=bool(args.preview), layout='Separate overview and wafer panels; no overlays.',
            particle_shape_schema=dust.get('shape_schema', 'legacy_uniform_cube'),
            particle_geometry_source='Hash-bound collision mesh assets with recorded physical scale and measured body orientation.',
            scope='Measured body-pose replay with physics disabled; no new simulation or outcome judgement.')
        video = args.output / 'wafer_edge_pick_isaac.mp4'
        if video.exists():
            proof['video_sha256'] = sha(video)
        (args.output / 'replay.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    finally:
        if writer is not None:
            writer.close()
        app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--time', type=float, action='append', default=[])
    render(parser.parse_args())
