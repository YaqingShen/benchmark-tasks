"""Exit successfully only when a complete, spill-free native quarter-roll run is logged."""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
logs = list(run.glob('adhoc_*.json'))
logs = [path for path in logs if not path.name.endswith('.live.json')]
if len(logs) != 1:
    sys.exit('Missing completed Inspect log')
log = json.loads(logs[0].read_text())
if log['status'] != 'success' or log['error'] or log['eval']['policy'] != 'yam_scripted':
    sys.exit('The reference evaluation did not finish normally')
rows = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
if not rows or rows[0]['contained'] != 40 or rows[0]['broken']:
    sys.exit('Invalid initial roll')
if any(row['escaped_closed'] or row['spilled'] for row in rows):
    sys.exit('Coin lost during reference run')
if log['results']['metrics']['task_score'] != 1 or rows[-1]['delivered'] != 40 or rows[-1]['score'] != 1:
    sys.exit('Reference run did not deliver 40 coins and park both paper halves')
# The demonstration uses the steel lip. Verify the fracture occurred within
# one sleeve radius plus 2 mm of the steel, independently of the task score.
fracture = next((row for row in rows if row['broken']), None)
if not fracture or fracture.get('fracture_world_mm') is None:
    sys.exit('Missing fracture position')
from geometry import TARGET, BORE
x, y, z = fracture['fracture_world_mm']
offset = (abs(x - (TARGET[0]*1000+37))-21.5, abs(y-TARGET[1]*1000)-1.5,
          abs(z-36)-15.5)
distance_mm = sum(max(v, 0)**2 for v in offset)**0.5
if distance_mm > BORE*1000+2:
    sys.exit(f'Paper fractured {distance_mm:.1f} mm from the steel lip')
