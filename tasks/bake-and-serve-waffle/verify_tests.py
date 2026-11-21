"""Run every regression in a fresh process and retain the complete result log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('runs/test-results.json'))
args=parser.parse_args()
root=Path(__file__).resolve().parent
results=[]
for test in sorted(root.glob('test_*.py')):
    result=subprocess.run([sys.executable,str(test)],cwd=root,capture_output=True,text=True,
                          env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    results.append(dict(test=test.name,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print(('PASS' if result.returncode==0 else 'FAIL'),test.name,flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(results,indent=2)+'\n')
print(f"{sum(r['exit_code']==0 for r in results)}/{len(results)} test files passed")
raise SystemExit(int(any(r['exit_code'] for r in results)))
