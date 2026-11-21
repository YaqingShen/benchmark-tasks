import math
from mechanism import *

def q(c): return -PROFILE.angles[SYMBOLS.index(c)]

def cycle(m,c):
    for _ in range(4): m.update(q(c),0)
    m.update(q(c),-PROFILE.end)
    for _ in range(5): m.update(q(c),0)

m=Mechanism()
for c in 'ROBOT  ': cycle(m,c)
cycle(m,'CUT')
assert m.result()['success']
assert m.labels == ['ROBOT  ']
# Holding the handle down does not generate repeated letters.
m=Mechanism()
for _ in range(100): m.update(q('R'),-PROFILE.end)
assert m.ribbon=='R' and m.full_strokes==1
# Partial squeezes don't produce glyphs.
m=Mechanism()
for _ in range(10): m.update(q('R'),-0.8*PROFILE.complete['emboss']);m.update(q('R'),0)
assert not m.ribbon
# Incorrect letter / no cut / failed release are not success.
for text in ('RABOT','ROBOTX','ROBO'):
    m=Mechanism()
    for c in text:cycle(m,c)
    cycle(m,'CUT')
    assert not m.result()['success']
m=Mechanism()
for c in 'ROBOT': cycle(m,c)
assert not m.result()['success']
m.update(q('CUT'),-PROFILE.end)
assert not m.result()['success']
# Between detents and rotation during a squeeze invalidate a run.
m=Mechanism();m.update(q('R')+.045,-PROFILE.end)
assert m.errors
m=Mechanism();m.update(0,-PROFILE.end*.7);m.update(.1,-PROFILE.end*.7)
assert m.errors
m=Mechanism()
for c in "ROBOT":cycle(m,c)
cycle(m,"CUT")
assert not m.result()["success"], "cutting without a trailing margin must not pass"

# A correct cut still requires full return; neither a held squeeze nor a brief
# return may receive the release credit.
m=Mechanism()
for c in 'ROBOT  ':cycle(m,c)
m.update(q('CUT'),-PROFILE.end)
assert m.labels == ['ROBOT  '] and m.result()['score']==.9
for _ in range(20):m.update(q('CUT'),-PROFILE.end)
assert not m.result()['success']
m.update(q('CUT'),0);m.update(q('CUT'),0)
assert not m.result()['success']
m.update(q('CUT'),0)
assert m.result()['success']

# A second strip cannot be ignored just because the first label is correct.
cycle(m,'X')
assert m.result()['score']==0 and not m.result()['success']
cycle(m,'CUT')
assert m.result()['score']==0 and len(m.labels)==2
print('PASS: exact text, full stroke/return, cut margin, alignment, extra strip, and negative cases')
