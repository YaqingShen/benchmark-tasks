"""Pin-tip entry model for a 60 mm wide, 32-pin manual dough docker. SI units."""
import math
import numpy as np

WIDTH = .060
TIP_RADIUS = .021
CORE_RADIUS = .014
BASE_RADIUS = .076
DOUGH_THICKNESS = .0045
MIN_DEPTH = .0022
# Four tracks and eight pins per revolution. These are prototype dimensions.
PINS = [(y, 2*math.pi*(k + (row % 2)*.5)/8)
        for row, y in enumerate((-.0225, -.0075, .0075, .0225)) for k in range(8)]
YS, PHASE = np.transpose(PINS)
GRID = np.array([(x,y) for x in np.arange(-.074,.075,.003)
                 for y in np.arange(-.074,.075,.003) if x*x+y*y <= (.071**2)])
COVER_RADIUS = .012


class Sheet:
    def __init__(self):
        self.holes = []
        self.buried = np.zeros(len(PINS), dtype=bool)
        self.covered = np.zeros(len(GRID), dtype=bool)

    def update(self, axle, orientation, angle):
        phase = PHASE - angle
        local = np.column_stack((TIP_RADIUS*np.sin(phase), YS, -TIP_RADIUS*np.cos(phase)))
        tips = local @ orientation.T + axle
        inside = np.sum(tips[:,:2]**2, axis=1) <= BASE_RADIUS**2
        entering = inside & (tips[:,2] <= -MIN_DEPTH) & ~self.buried
        # A dragged pin makes only one puncture until it clears the surface.
        self.buried = (self.buried | entering) & (tips[:,2] < -.0003) & inside
        added = []
        for xy in tips[entering,:2]:
            if self.holes and np.min(np.linalg.norm(self.holes-xy,axis=1)) < .0025:
                continue
            self.holes.append(xy.copy())
            self.covered |= np.sum((GRID-xy)**2,axis=1) <= COVER_RADIUS**2
            added.append(xy)
        return added

    @property
    def fraction(self):
        return float(self.covered.mean())
