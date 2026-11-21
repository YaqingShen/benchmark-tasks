import math
import numpy as np
from cooking_state import State,baking_configuration
from phase_conversion import eligibility
s=State()
for _ in range(299):s.advance(.05,closed=True,scale=20.)
assert not s.cooked
s.advance(.05,closed=True,scale=20.)
assert s.cooked and abs(s.robot_time_s-15)<1e-8 and abs(s.bake_equivalent_s-300)<1e-8
s=State()
for _ in range(5999):s.advance(.05,closed=True)
assert not s.cooked
s.advance(.05,closed=True)
assert s.cooked and abs(s.closed_robot_s-300.)<1e-7
s=State()
for _ in range(200):s.advance(.05,closed=True)
s.advance(.05,closed=False)
assert not s.cooked and s.bake_equivalent_s==0
for _ in range(600):s.advance(.05,closed=True,heater_on=False)
assert not s.cooked
args=dict(cooked=True,lid=0,spin=math.pi,points_local=np.array([[.02,.03,0]]))
assert eligibility(**args) is None
assert eligibility(**dict(args,cooked=False))
assert eligibility(**dict(args,lid=.2))
assert eligibility(**dict(args,lid=np.radians(4)))
assert eligibility(**dict(args,points_local=np.array([[.12,0,0]])))
print('PASS:15s/300s separate clocks; open or disabled heater cannot complete timer; no premature/open state swap; no disappearing visible spill')

assert not baking_configuration(0.,0.), 'Closed upright waiting must not pre-credit inverted baking'
assert not baking_configuration(0.,math.radians(170.))
assert baking_configuration(0.,math.pi)
assert not baking_configuration(math.radians(5.),math.pi)
s=State()
for _ in range(8500):s.advance(.05,closed=baking_configuration(0.,0.))
assert not s.cooked and s.bake_equivalent_s==0
print('PASS: closed upright idle cannot earn baking time; five-minute clock requires inverted cooking position')
