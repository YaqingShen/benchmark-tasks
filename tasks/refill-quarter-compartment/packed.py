"""Gravity-driven sliding of an ordered stack inside a close-fitting cylindrical sleeve."""
import math
import numpy as np
import geometry as g

COUNT=g.COUNT//2
PITCH=g.T+.000075
PAPER_MASS=.0017
INITIAL=np.array([g.L/2-.0006-g.T/2-i*PITCH for i in range(COUNT)])
class Sleeve:
    def __init__(self):
        self.offset=0.
        self.speed=0.
        self.next=0
    def advance(self,outward_acceleration,normal_acceleration,opened,dt=.05):
        if not opened or self.next==COUNT:
            self.speed=0.
            return []
        first=self.next
        # Positive positions point toward the mouth. The intact outer crimp
        # bounds the rear coin. The front coin is free when its rear face clears.
        limit=-g.L/2+.0005+g.T/2-INITIAL[-1]
        for _ in range(12):
            normal=max(0.,normal_acceleration)
            if abs(self.speed)<1e-5 and abs(outward_acceleration)<=.35*normal:
                self.speed=0.;continue
            direction=self.speed if abs(self.speed)>1e-5 else outward_acceleration
            accel=outward_acceleration-math.copysign(.27*normal,direction)
            v=self.speed+accel*dt/12
            if self.speed*v<0 and abs(outward_acceleration)<=.35*normal:v=0.
            self.speed=v
            self.offset+=v*dt/12
            if self.offset<=limit:self.offset=limit;self.speed=max(0.,v)
            while self.next<COUNT and INITIAL[self.next]+self.offset>=g.L/2+g.T/2+.00025:
                self.next+=1
        return list(range(first,self.next))
    @property
    def mass(self):return PAPER_MASS+(COUNT-self.next)*g.MASS
