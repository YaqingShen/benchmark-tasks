import math
import unittest
import numpy as np
import geometry as g

class PaperBoundsTest(unittest.TestCase):
    def test_rotated_sleeve_on_table_and_above_it(self):
        angle=math.pi/4
        c,s=math.cos(angle),math.sin(angle)
        rotation=np.array([[c,-s,0],[0,0,-1],[s,c,0]])
        for side in (-1,1):
            low,high=g.paper_bounds(np.zeros(3),rotation,side)
            self.assertLess(abs(low[2]+.014),.0004)
            self.assertAlmostEqual(high[1]-low[1],g.L)
            base=np.array([0.,0.,-low[2]])
            self.assertAlmostEqual(g.paper_bounds(base,rotation,side)[0][2],0.)
            self.assertAlmostEqual(g.paper_bounds(base+[0,0,.010],rotation,side)[0][2],.010)
    def test_upright_sleeve(self):
        for side in (-1,1):
            low,high=g.paper_bounds(np.zeros(3),np.eye(3),side)
            self.assertAlmostEqual(low[2],-g.L/2)
            self.assertAlmostEqual(high[2],g.L/2)
