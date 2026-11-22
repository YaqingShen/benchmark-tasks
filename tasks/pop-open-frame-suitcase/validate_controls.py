import json
import sys
from pathlib import Path
checks=json.loads(Path(sys.argv[1]).read_text())['checks']
print(checks)
if set(checks)!={'correct','wrong_left','wrong_middle','wrong_right'} or not all(checks.values()):
    sys.exit('One or more case interlock controls failed.')
