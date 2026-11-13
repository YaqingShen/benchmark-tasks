"""Record exact task code/asset checksums before cross-host evaluations."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
files=[*root.glob('*.py'),root/'task_spec.json',*root.glob('assets/*.usda'),*root.glob('assets/*_screen.png'),root/'assets/manifest.json']
manifest={'revision':'charging011','status':'development candidate; see run evidence for validation',
          'files':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)},
          'model_limitations':['Charger dimensions, magnetic force curves and strap stiffness are not measured hardware calibrations','Charging is a geometry/contact/dwell proxy; no electromagnetic or battery model'],
          'policy_inputs':['three original RGB cameras','14 measured robot joint positions'],
          'no_object_pose_writes_after_reset':True,'no_grasp_attachment_joints':True}
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen',len(files),'files')
