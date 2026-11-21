from pathlib import Path
import json, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'task'))
from imprints import Sheet, TIP_RADIUS

def exercise(lanes, clearance=0., locked=False):
    sheet = Sheet()
    for y in lanes:
        sheet.update(np.array([-.09,y,.05]),np.eye(3),0.)
        for x in np.linspace(-.09,.09,361):
            sheet.update(np.array([x,y,TIP_RADIUS-.0036+clearance]),np.eye(3),
                         0. if locked else (x+.09)/TIP_RADIUS)
    return dict(holes=len(sheet.holes),coverage=round(sheet.fraction,4))

if __name__=='__main__':
    lanes=[-.048,0.,.048]
    results=dict(hover=exercise(lanes,.015),one_lane=exercise([0.]),
                 same_lane=exercise([0.]*3),locked=exercise(lanes,locked=True),
                 three_lanes=exercise(lanes))
    assert results['hover']['holes']==0
    assert results['one_lane']==results['same_lane']
    assert 0 < results['one_lane']['coverage'] < .7
    assert results['locked']['coverage'] < .3
    assert results['three_lanes']['coverage'] >= .96
    print(json.dumps(results,indent=2))
