import unittest
import numpy as np
from disposal_geometry import pipette_bounds_in_bin,fully_contained
from pipette_geometry import BULB_CENTER_M


def pose(x=0,z=0):
    p=np.eye(4);p[:3,3]=[x,0,z];return p


class ContainmentTests(unittest.TestCase):
    def bounds(self):
        return pipette_bounds_in_bin(pose(z=.008),
            [pose(x=-.009,z=.008+BULB_CENTER_M),pose(x=.009,z=.008+BULB_CENTER_M)],np.eye(4))

    def test_tip_inside_does_not_accept_a_protruding_bulb(self):
        lower,upper=self.bounds()
        self.assertFalse(fully_contained(lower,upper,dict(half_x=.046,half_y=.041,floor_z=.008,rim_z=.13)))

    def test_entire_tool_fits_the_deep_bin(self):
        lower,upper=self.bounds()
        self.assertTrue(fully_contained(lower,upper,dict(half_x=.046,half_y=.041,floor_z=.008,rim_z=.24)))

    def test_crossing_wall_is_not_disposal(self):
        lower,upper=self.bounds();lower[0]+=.05;upper[0]+=.05
        self.assertFalse(fully_contained(lower,upper,dict(half_x=.046,half_y=.041,floor_z=.008,rim_z=.24)))


if __name__=='__main__':unittest.main()
