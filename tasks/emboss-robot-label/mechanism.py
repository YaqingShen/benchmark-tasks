"""Label scoring from observed dial and trigger angles."""
import math
from dataclasses import dataclass, field

TARGET = 'ROBOT'
import os
from omega import Omega, OfficeMateScenario, PrototypeLoad, mode

if os.environ.get("EMBOSS_OFFICEMATE_SCENARIO") == "1":
    PROFILE = OfficeMateScenario()
else:
    PROFILE = PrototypeLoad() if os.environ.get("EMBOSS_ENGINEERING_BASELINE") == "1" else Omega()
SYMBOLS = PROFILE.symbols
ALIGN_TOL = math.radians(1.8)

def selected(q):
    return PROFILE.nearest(q)[:2]

@dataclass
class Mechanism:
    ribbon: str = ''
    labels: list = field(default_factory=list)
    load_phase: str = 'press'
    spring_setting: float = -1.
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
        if depth > PROFILE.release:
            if self.stroke_wheel is None:
                self.stroke_wheel = wheel
            elif abs(wheel-self.stroke_wheel) > math.radians(3.5):
                if 'wheel turned during squeeze' not in self.errors:
                    self.errors.append('wheel turned during squeeze')
        else:
            self.stroke_wheel = None
        if depth <= PROFILE.release:
            self.release_s += dt
            if self.release_s >= .15 and not self.armed:
                self.armed = True
                self.released_cycles += 1
                if self.labels:
                    self.cut_released = True
        else:
            self.release_s = 0.
        if depth >= PROFILE.complete[mode(selected(wheel)[0])] and self.armed:
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
