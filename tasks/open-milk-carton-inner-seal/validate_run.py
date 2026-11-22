"""Validate runtime evidence; process exit 0 is insufficient for Inspect Robots."""
import sys,json,pathlib,numpy as np
root=pathlib.Path(sys.argv[1]);errors=[];logs=list((root/'log').glob('*.json'));steps=0
if len(logs)!=1:errors.append('expected one Inspect run record')
else:
 x=json.loads(logs[0].read_text());steps=x.get('stats',{}).get('total_steps',0)
 if x.get('error'):errors.append(x['error'])
 if x.get('status')=='error' or x.get('results',{}).get('errored_trials',0):errors.append('framework reports an errored trial')
 if steps<=0:errors.append('no executed control steps')
frames={c:sorted((root/'log/frames').glob('*/*~'+c+'_*.npy')) for c in ['top_cam','left_cam','right_cam']}
counts={k:len(v) for k,v in frames.items()}
if not min(counts.values()) or len(set(counts.values()))!=1:errors.append('missing or unequal camera frames')
if frames['top_cam']:
 a=np.load(frames['top_cam'][0])[...,:3]
 if float(np.std(a))<2:errors.append('flat or blank initial top camera')
trace=root/'trace.jsonl'
if not trace.exists() or not trace.stat().st_size:errors.append('missing native state trace')
contact_file=root/'contacts.jsonl'
if not contact_file.exists() or not contact_file.stat().st_size:errors.append('missing native contact reports')
else:
 c=json.loads(contact_file.read_text().splitlines()[-1])
 if c.get('callbacks',0)<=0 or not c.get('pairs'):errors.append('native contact reports contain no contact evidence')
result={'runtime_valid':not errors,'errors':errors,'executed_steps':steps,'camera_frames':counts,'task_success_not_assessed':True}
(root/'run_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(0 if not errors else 76)
