"""Create an evidence summary from the official run log and its synchronized trace."""
import argparse
import json
from pathlib import Path

import numpy as np

from load_report import summarize


def audit(path):
    files = list(path.glob('adhoc_????????.json'))
    if len(files) != 1:
        raise ValueError('One completed official run is required.')
    log = json.loads(files[0].read_text())
    rows = [json.loads(s) for s in (path/'trace.jsonl').read_text().splitlines()]
    if not rows or [r['step'] for r in rows] != list(range(len(rows))):
        raise ValueError('The physical trace is incomplete.')
    if len(rows) != log['stats']['total_steps'] or len(rows) > log['eval']['max_steps']:
        raise ValueError('Trace length must match the official action count and fit within the limit.')
    final = rows[-1]
    sim = final['inspection']
    score = log['results']['metrics']['task_score']
    if abs(score-sim['score']) > 1e-6 or log['results']['errored_trials']:
        raise ValueError('The scorer and the completed physical trace disagree.')
    checkpoints = []
    for j, station in enumerate((35,70,105)):
        first = next((r for r in rows if r['inspection']['gaps_mm'][j] is not None), None)
        checkpoints.append(dict(station_mm=station, step=first['step'] if first else None,
                                gap_mm=sim['gaps_mm'][j], earned=sim['stations'][j]))
    cf = np.array([r['card_force'][:10] for r in rows])
    gf = np.array([r['glass_force'][:10] for r in rows])
    touches = (cf>.03) & (gf>.004)
    result = dict(policy=log['eval']['policy'], checkpoint=log['eval']['policy_checkpoint'],
                  run_log=files[0].name, official_score=score, actions=log['stats']['total_steps'],
                  limit=log['eval']['max_steps'], trace_samples=len(rows), final_inspection=sim,
                  stations=checkpoints, film_bands_loaded_on_glass=np.flatnonzero(touches.any(axis=0)).tolist(),
                  start_edges_mm=np.round(1000*np.array(rows[0]['film'])[0],3).tolist(),
                  final_edges_mm=np.round(1000*np.array(final['film'])[0],3).tolist(),
                  release_steps=[r[:2] for r in final['releases']],
                  same_source_verification='input.sha256')
    if max(r['liner'][0][0]-rows[0]['liner'][0][0] for r in rows) >= .11:
        result['paddle_load'] = summarize(rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_folder), indent=2))
