"""State of a manual embosser. Inputs are measured hinge angles, never desired letters.
Prototype wheel layout and force parameters are not measured product specifications.
"""
import math
from dataclasses import dataclass, field

TARGET = 'ROBOT'
# 49 equally spaced detents; prototype ordering, not a measured Omega wheel layout.
SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/&'ÄÖÜÅÉ? ") + ('CUT',)
assert len(SYMBOLS) == 49
PITCH = 2*math.pi / len(SYMBOLS)
FULL_STROKE = .255
FULL_RELEASE = .018
ALIGN_TOL = math.radians(1.8)

def selected(q):
    k = int(round(-q/PITCH))
    return SYMBOLS[k % len(SYMBOLS)], abs(q + k*PITCH)

@dataclass
class Mechanism:
    ribbon: str = ''
    labels: list = field(default_factory=list)
    armed: bool = True
    release_s: float = 0.
    stroke_wheel: float | None = None
    errors: list = field(default_factory=list)
    events: list = field(default_factory=list)
    full_strokes: int = 0
    released_cycles: int = 0
    cut_released: bool = False
    time_s: float = 0.

    def update(self, wheel, trigger, dt=.05):
        self.time_s += dt
        depth = max(0., -trigger)
        if depth > .05:
            if self.stroke_wheel is None:
                self.stroke_wheel = wheel
            elif abs(wheel-self.stroke_wheel) > math.radians(3.5):
                if 'wheel turned during squeeze' not in self.errors:
                    self.errors.append('wheel turned during squeeze')
        else:
            self.stroke_wheel = None
        if depth <= FULL_RELEASE:
            self.release_s += dt
            if self.release_s >= .15 and not self.armed:
                self.armed = True
                self.released_cycles += 1
                if self.labels:
                    self.cut_released = True
        else:
            self.release_s = 0.
        if depth >= FULL_STROKE and self.armed:
            self.armed = False
            self.full_strokes += 1
            symbol, error = selected(wheel)
            if error > ALIGN_TOL:
                self.errors.append('pressed between detents')
                symbol = '#'
            event = {'t': round(self.time_s,3), 'wheel_rad': wheel,
                     'trigger_rad': trigger, 'symbol': symbol}
            if symbol == 'CUT':
                self.labels.append(self.ribbon)
                self.ribbon = ''
                self.cut_released = False
                event['label'] = self.labels[-1]
            else:
                self.ribbon += symbol
            self.events.append(event)
            return event
        return None

    def result(self):
        text = self.labels[-1] if self.labels else self.ribbon
        exact = self.labels == [TARGET + '  '] and not self.ribbon
        # Extra/wrong glyphs cannot earn prefix credit by later returning to the target.
        clean = text.strip()
        prefix = 0
        for a,b in zip(clean,TARGET):
            if a != b: break
            prefix += 1
        correct = clean == TARGET
        score = .6 * prefix / len(TARGET)
        if correct: score += .15
        if exact: score += .15
        if exact and self.cut_released: score += .1
        if self.errors or len(clean)>len(TARGET) or len(self.labels)>1 or (self.labels and self.ribbon):
            score = 0.
        return {'score': round(score,4), 'success': bool(score>=1.),
                'ribbon':self.ribbon, 'cut_labels':list(self.labels),
                'full_strokes':self.full_strokes, 'released_cycles':self.released_cycles,
                'errors':list(self.errors), 'cut_released':self.cut_released}
