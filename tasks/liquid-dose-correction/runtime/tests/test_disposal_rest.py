import math
import unittest
from pipette_correction import released_disposal_window


class DisposalRestTests(unittest.TestCase):
    def window(self):
        return [dict(time_s=i*.05,tip_position=[0,0,.01],
                     tool_axis_world=[0,0,1],pipette_contained_and_released=True,
                     tool_speed_m_s=.02 if i%2 else 0.) for i in range(20)]

    def test_visible_rest_does_not_depend_on_instantaneous_solver_velocity(self):
        self.assertTrue(released_disposal_window(self.window()))

    def test_sliding_and_pivoting_are_not_rest(self):
        rows=self.window()
        rows[-1]['tip_position'][0]=.003
        self.assertFalse(released_disposal_window(rows))
        rows=self.window()
        angle=math.radians(4)
        rows[-1]['tool_axis_world']=[math.sin(angle),0,math.cos(angle)]
        self.assertFalse(released_disposal_window(rows))

    def test_requires_release_containment_and_full_window(self):
        rows=self.window()
        self.assertFalse(released_disposal_window(rows[:-1]))
        rows[-1]['pipette_contained_and_released']=False
        self.assertFalse(released_disposal_window(rows))


if __name__=='__main__':unittest.main()
