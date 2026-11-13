"""Screen-glyph rendering test; displayed test values are not measured masses."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
from formulation_backend import FormulationWorld
from PIL import Image
world=None
try:
    world=FormulationWorld(ROOT,a.output)
    _,q,_=world.reset(initial_mass=10.5)
    # Isolated display test: stop automatic readout painting while leaving
    # actual liquid, forces and instrument measurements unchanged.
    screen=world.coupling.balance.screen
    world.coupling.balance.render=lambda:None
    images=[]
    for value in (8.6,10.5,9.952,10.0):
        screen.set_value(value)
        for _ in range(12):world._step(q)
        rgb,_,_=world.observe()
        name=f'display-{value:.3f}.png'
        Image.fromarray(rgb['top_cam']).save(a.output/name)
        images.append({'file':name,'test_reading_g':value,'authored_digits':screen.previous})
    (a.output/'display-test.json').write_text(json.dumps({'type':'SCREEN_GLYPH_RENDERING_TEST',
        'full_task_demo':False,'task_score':None,'actual_liquid_mass_g':10.5,'images':images,
        'note':'These isolated display test values are not robot measurements or task results.'},indent=2))
finally:
    if world is not None:world.close()
