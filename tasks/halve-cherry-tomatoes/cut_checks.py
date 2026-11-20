"""Geometric checks shared by the live station and diagnostic tests. Distances in metres."""
import math
import numpy as np

RADIUS = .014
BLADE_HALF_LENGTH = .0885
PLANE_TOLERANCE = .0035
RETRACTION = .010

def both_tools_held(tips, lid_grip, knife_grip):
    return any(np.linalg.norm(tips[i] - lid_grip) < .024 and
               np.linalg.norm(tips[1-i] - knife_grip) < .025 for i in (0,1))

class BatchPass:
    def __init__(self):
        self.batch_complete = False
        self.cut_ids = set()
        self.passes = []
        self.reset_pass()

    def reset_pass(self):
        self.axis = None
        self.peak = None
        self.enter = {}
        self.previous = {}
        self.pass_ids = set()

    def observe(self, position, rotation, fruit_local, covered, engaged):
        if not engaged:
            self.reset_pass()
            return []
        forward = rotation[:,1]
        if self.axis is not None:
            projected = float(position @ self.axis)
            if projected < self.peak - RETRACTION or self.axis @ forward < math.cos(math.radians(15)):
                self.reset_pass()
        if self.axis is None:
            self.axis = forward.copy()
            self.peak = float(position @ self.axis)
        progress = float(position @ self.axis)
        self.peak = max(self.peak, progress)
        events = []
        for i, (x,y,z) in enumerate(fruit_local):
            if i in self.cut_ids:
                continue
            previous = self.previous.get(i)
            self.previous[i] = (float(y),progress)
            if not covered[i] or abs(z) > PLANE_TOLERANCE:
                self.enter.pop(i,None)
                self.previous.pop(i,None)
                continue
            depth = math.sqrt(RADIUS**2-z*z)
            breadth = math.sqrt(max(0.,depth**2-y*y))
            if abs(x)+breadth > BLADE_HALF_LENGTH:
                self.enter.pop(i,None)
                self.previous.pop(i,None)
                continue
            if previous is not None and previous[0] >= depth > y:
                self.enter[i] = previous[1]
            if i in self.enter and y <= -depth:
                start = self.enter.pop(i)
                # The edge must itself advance at least one radius. This excludes
                # pushing a fruit through a stationary blade. Fruit can still roll slightly.
                if progress-start >= RADIUS:
                    if not self.pass_ids:
                        self.passes.append(self.pass_ids)
                    self.cut_ids.add(i)
                    self.pass_ids.add(i)
                    events.append(i)
        self.batch_complete |= len(self.pass_ids)==8
        return events

    def best_pass(self, retained):
        """Largest single pass, with retained halves breaking ties."""
        return max(((len(ids),sum(retained.get(i,0) for i in ids)) for ids in self.passes),
                   default=(0,0))
