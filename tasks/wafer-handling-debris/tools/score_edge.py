"""Score closed new-pick measurements; no simulator or legacy-report aliases."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'edge_native'), str(ROOT / 'baseline'), str(ROOT / 'baseline/benchmark')]
import wafer_debris_score
from wafer_debris_score import VisibleDebrisScorer
from wafer_robot_geometry import RobotGeometryGuard, compact_result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _binding(metadata, source_dir, baseline_dir=None):
    sources = metadata['source_files']
    required = {'run_edge_workflow.py', 'native_adapter.py', 'native_controller.py',
                'native_air.py', 'native_evidence.py'}
    if not required.issubset(sources):
        raise ValueError('Missing native source-file bindings.')
    for name, digest in sources.items():
        if Path(name).name != name or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid native source binding: ' + name)
        if source_dir is not None and sha(source_dir / name) != digest:
            raise ValueError('Recorded native source bytes differ: ' + name)
    if sources['native_evidence.py'] != metadata['evidence_source_sha256']:
        raise ValueError('Actual evidence reader differs from the initial source inventory.')
    adapter = metadata['native_adapter']
    baseline_sources = metadata.get('baseline_source_sha256', {})
    if not baseline_sources:
        raise ValueError('Missing baseline source bindings.')
    for name, digest in baseline_sources.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid baseline source binding: ' + name)
        if baseline_dir is not None and sha(baseline_dir / path) != digest:
            raise ValueError('Recorded baseline source differs: ' + name)
    if source_dir is not None:
        if sha(source_dir / 'scene.xml') != adapter['source_scene_sha256']:
            raise ValueError('Original mechanical source scene differs.')
        if sha(source_dir / 'native/parity.json') != adapter['parity_sha256']:
            raise ValueError('Native asset parity report differs.')
        for name, digest in adapter['native_assets'].items():
            if Path(name).name != name or sha(source_dir / 'native' / name) != digest:
                raise ValueError('Native physical asset differs: ' + name)
    provenance = dict(status='FILES_VERIFIED' if source_dir is not None else 'RECORDED_HASHES_ONLY',
                      baseline_status='FILES_VERIFIED' if baseline_dir is not None else 'RECORDED_HASHES_ONLY',
                      source_files=sources, native_assets=adapter['native_assets'])
    if 'shape_schema' in metadata.get('dust', {}):
        if source_dir is None:
            raise ValueError('Mixed-debris verification requires its exact native source directory.')
        import mixed_debris_assets
        if sha(mixed_debris_assets.__file__) != sources.get('mixed_debris_assets.py'):
            raise ValueError('The mixed-asset verifier differs from the recorded source.')
        bindings = mixed_debris_assets.particle_asset_bindings(source_dir, metadata['dust'])
        dust = metadata['dust']; evidence = dust.get('geometry_evidence', {})
        if evidence.get('status') != 'COMPOSED_COLLIDERS_VERIFIED':
            raise ValueError('Mixed debris lacks actual composed-collider evidence.')
        if set(evidence.get('particles', {})) != set(dust['ids']):
            raise ValueError('Mixed-debris actual collider inventory differs.')
        for index, particle in enumerate(dust['particles']):
            actual = evidence['particles'][particle['id']]
            expected_vertices = np.asarray(particle['vertices_local_m'], dtype=float)
            actual_vertices = np.asarray(actual['vertices_local_m'], dtype=float)
            if (actual_vertices.shape != expected_vertices.shape or not np.isfinite(actual_vertices).all()
                    or not np.allclose(actual_vertices, expected_vertices, rtol=0., atol=1e-8)):
                raise ValueError('Mixed-debris actual collider geometry differs: ' + particle['id'])
            expected_mass = float(particle['mass_kg'])
            masses = [float(actual['mass_kg']), float(dust['profile']['mass_kg'][index])]
            if not all(np.isfinite(m) and np.isclose(m, expected_mass, rtol=1e-6, atol=1e-15) for m in masses):
                raise ValueError('Mixed-debris measured/configured masses differ: ' + particle['id'])
        provenance['mixed_debris_assets'] = {name: digest for name, (_, digest) in bindings.items()}
        provenance['mixed_geometry_status'] = 'SOURCE_MESH_AND_COMPOSED_COLLIDER_MATCH'
    return provenance


def _command_matches(row, command, finger_open):
    action = np.asarray(command['action'], dtype=np.float32)
    if action.shape != (14,) or not np.isfinite(action).all() or np.any(action[[6, 13]] < 0) or np.any(action[[6, 13]] > 1):
        raise ValueError('Invalid recorded 14-dimensional robot command.')
    for start, arm, fingers in ((0, 'commanded_arm_q_rad', 'commanded_finger_q_m'),
                               (7, 'commanded_right_arm_q_rad', 'commanded_right_finger_q_m')):
        measured = np.asarray(row[arm], dtype=np.float32)
        measured_fingers = np.asarray(row[fingers], dtype=np.float32)
        expected = np.float32(action[start + 6] * np.float32(finger_open))
        if (measured.shape != (6,) or measured_fingers.shape != (2,)
                or not np.array_equal(measured, action[start:start + 6])
                or not np.allclose(measured_fingers, expected, rtol=0., atol=1e-8)):
            raise ValueError('Recorded action differs from measured actuator targets.')


def evaluate(run, source_dir=None, baseline_dir=None):
    run = Path(run).resolve()
    source_dir = None if source_dir is None else Path(source_dir).resolve()
    baseline_dir = None if baseline_dir is None else Path(baseline_dir).resolve()
    files = ('metadata.json', 'trial_report.json', 'observations.jsonl.gz', 'commands.jsonl')
    closed = {name: sha(run / name) for name in files}
    metadata = json.loads((run / 'metadata.json').read_text())
    report = json.loads((run / 'trial_report.json').read_text())
    if (metadata.get('version') != 'mcp2_native_development_v1'
            or metadata.get('engine') != 'Isaac Lab / PhysX'
            or metadata.get('post_start_pose_write_count') != 0
            or metadata.get('dust', {}).get('post_start_particle_pose_writes') != 0):
        raise ValueError('Unsupported native run or post-start object writes.')
    if metadata.get('physics_dt_s') != .0005 or metadata.get('control_hz') != 20:
        raise ValueError('The required native cadence differs.')
    provenance = _binding(metadata, source_dir, baseline_dir)
    with (run / 'commands.jsonl').open() as stream:
        commands = [json.loads(line) for line in stream]
    if len(commands) > 2400:
        raise ValueError('The action count exceeds 120 seconds.')
    for i, command in enumerate(commands):
        if abs(float(command['time_s']) - i * .05) > 1e-8:
            raise ValueError('Missing or reordered robot commands.')
    if (report.get('control_actions') != len(commands)
            or abs(float(report.get('simulation_seconds', -1)) - len(commands) * .05) > 1e-8
            or metadata.get('physics_updates') != len(commands) * 100):
        raise ValueError('Closed command, duration and physics-update counts differ.')
    if metadata.get('robot_geometry_capture') != 'sampled_20hz_native_measured_link_poses_v1':
        raise ValueError('Unknown robot collision sampling scope.')
    guard = RobotGeometryGuard.from_evidence(metadata['robot_geometry'])
    monitor = VisibleDebrisScorer(metadata, .05)
    samples = 0
    with gzip.open(run / 'observations.jsonl.gz', 'rt') as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            if i > len(commands) or abs(float(row['time_s']) - i * .05) > 1e-8:
                raise ValueError('Missing, extra or reordered native observations.')
            if i:
                _command_matches(row, commands[i - 1], metadata['native_finger_open_q_m'])
            poses = row['robot_link_poses_world']
            if not isinstance(poses, list) or len(poses) != len(guard.keys):
                raise ValueError('Incomplete measured robot link inventory.')
            checked = compact_result(guard.check(dict(zip(guard.keys, poses))))
            if checked != row['robot_geometry_check']:
                raise ValueError('Robot collision result differs from independent geometry replay.')
            monitor.update(row, report=False)
            samples += 1
    if samples != len(commands) + 1:
        raise ValueError('The final measured observation is missing.')
    result = monitor.result()
    task_score = dict(result)
    matches_report = report.get('task_score') == task_score
    force_audit = metadata.get('particle_force_audit', {})
    force_ok = (force_audit.get('status') == 'READBACKS_MATCHED' and
                force_audit.get('completed_intervals') == len(commands) * 100 and
                force_audit.get('maximum_readback_error', 1.) <= 1e-10 and
                force_audit.get('maximum_unexpected_wrench', 1.) == 0.)
    if metadata.get('stage') != 'full' or report.get('result') != 'FULL_DEMONSTRATION_SUCCEEDED' or report.get('error'):
        result['success'] = False
        result['full_demonstration'] = False
    else:
        result['full_demonstration'] = bool(result['success'])
    if any(sha(run / name) != digest for name, digest in closed.items()):
        raise ValueError('Closed input files changed during evaluation.')
    if source_dir is not None or baseline_dir is not None:
        _binding(metadata, source_dir, baseline_dir)
    result.update(passed=bool(result.get('full_demonstration') and matches_report and force_ok and metadata.get('sources_unchanged_during_run')),
                  matches_report=matches_report, task_score=task_score,
                  trace_sha256=closed['observations.jsonl.gz'], particle_force_audit=force_audit,
                  classification='NATIVE_EDGE_PICK_SAMPLED_RESCORE', original_report_unchanged=True,
                  original_result=report.get('result'), scoring_samples=samples,
                  recorded_physics_updates=metadata['physics_updates'], source_files_sha256=closed,
                  source_provenance=provenance,
                  scorer_sha256=sha(wafer_debris_score.__file__), evaluator_sha256=sha(__file__),
                  scope='20 Hz primary score and independently replayed robot self/inter-arm geometry. '
                        'Counts bind recorded physics updates; this is not a full-physics collision, '
                        'particle-force, per-shape contact or hardware validation.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Choose a new output; existing evidence is preserved.')
    try:
        result = evaluate(args.run, args.source_dir, args.baseline_dir)
    except Exception as exc:
        result = dict(score=0., success=False, evidence_complete=False,
                      error=type(exc).__name__ + ': ' + str(exc))
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload)
    return 0 if result.get('evidence_complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
