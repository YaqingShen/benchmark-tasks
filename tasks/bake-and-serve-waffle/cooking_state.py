"""Abstract baking timer with a full300s closed-and-locked interval.

The appliance is preheated in the setup. Opening before completion resets the
continuous timer. This predicts neither temperature nor cooking chemistry.
"""
from dataclasses import dataclass,asdict
TIME_SCALE=1.
RECIPE_S=300.
BATTER_VOLUME_ML=250.
BATTER_DENSITY_KG_M3=1050.
BATTER_MASS_KG=BATTER_VOLUME_ML*1e-6*BATTER_DENSITY_KG_M3

@dataclass
class State:
    robot_time_s:float=0.
    bake_equivalent_s:float=0.
    closed_robot_s:float=0.
    cooked:bool=False
    def advance(self,robot_dt,*,closed,heater_on=True,scale=TIME_SCALE):
        if robot_dt<=0 or scale<1:raise ValueError('Invalid clock parameters')
        self.robot_time_s+=robot_dt
        if not self.cooked:
            if closed and heater_on:
                self.closed_robot_s+=robot_dt
                self.bake_equivalent_s+=robot_dt*scale
                self.cooked=self.bake_equivalent_s>=RECIPE_S-1e-8
            else:
                self.closed_robot_s=0.
                self.bake_equivalent_s=0.
        return self
    def report(self):
        return dict(asdict(self),model='abstract closed-lid baking timer',time_scale=TIME_SCALE,
                    debug_time_compressed=TIME_SCALE>1,physical_cooking_validated=False)
