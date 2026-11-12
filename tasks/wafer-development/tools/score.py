"""Rescore recorded poses with the practical submission rubric; no simulator."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path

TASK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(TASK/'runtime'),str(TASK/'runtime/benchmark')]
from wafer_develop_score import DevelopmentTaskScorer
from wafer_robot_geometry import RobotGeometryGuard


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--trace',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();meta_path=a.run/'metadata.json'
    trace=a.trace or a.run/'samples.jsonl.gz'
    meta=json.loads(meta_path.read_text());scorer=DevelopmentTaskScorer(meta)
    guard=RobotGeometryGuard.from_evidence(meta['robot_geometry']);selected=0
    opening=gzip.open if trace.suffix=='.gz'else open
    with opening(trace,'rt')as stream:
        for line in stream:
            row=json.loads(line);t=float(row['time_s'])
            if abs(t/.05-round(t/.05))>1e-7:continue
            poses=dict(zip(guard.keys,row['robot_link_poses_world']))
            row['robot_geometry_check']=guard.check(poses)
            scorer.update(row);selected+=1
    result=scorer.result()
    result['provenance']=dict(evaluated_trace_sha256=digest(trace),metadata_sha256=digest(meta_path),
        scorer_sha256=digest(TASK/'runtime/benchmark/wafer_develop_score.py'),
        sampled_robot_geometry_recomputed=True,selected_rows=selected,
        note='Prospective submission rubric applied to saved measurements; original native reports are unchanged.')
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x')as stream:stream.write(text)
    print(text,end='')


if __name__=='__main__':main()
