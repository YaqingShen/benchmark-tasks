"""Explicit bridge from audited baseline joint commands to a hashed candidate."""
from pathlib import Path
import hashlib, json

def revision(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_candidate_bridge(source_plan,bundle):
    root=Path(bundle).expanduser().resolve()
    provenance=json.loads((root/'candidate_provenance.json').read_text())
    manifest=json.loads((root/'bundle_files.json').read_text())
    candidate=json.loads((root/'full_workflow_plan.json').read_text())
    if revision(source_plan)!=provenance['baseline_plan_revision']:
        raise ValueError('Recorded commands do not originate from the declared baseline plan.')
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Candidate source differs from its manifest: '+name)
    if {k:v for k,v in source_plan.items() if k!='files'}!={k:v for k,v in candidate.items() if k!='files'}:
        raise ValueError('Candidate changes a physical or workflow plan parameter.')
    for name,digest in candidate['files'].items():
        if manifest.get(name)!=digest:
            raise ValueError('Candidate plan source differs from the bundle manifest: '+name)
        baseline=source_plan['files'].get(name)
        if baseline!=digest:
            change=provenance['changed_files'].get(name,{})
            if change.get('baseline')!=baseline or change.get('candidate')!=digest:
                raise ValueError('Candidate source change is not declared: '+name)
    if set(source_plan['files'])-set(candidate['files']):
        raise ValueError('Candidate omitted a baseline source check.')
    return revision(candidate)
