import unittest
from packed import Sleeve,COUNT,PAPER_MASS,g
class TestSleeve(unittest.TestCase):
    def test_retention_and_release(self):
        s=Sleeve()
        for _ in range(50):
            self.assertEqual(s.advance(9.81,0,False),[])
            self.assertEqual(s.advance(0,9.81,True),[])
            self.assertEqual(s.advance(1.,9.81,True),[])
        emitted=[]
        for _ in range(30):emitted+=s.advance(9.81,0.,True)
        self.assertEqual(emitted,list(range(COUNT)))
        self.assertAlmostEqual(s.mass+len(emitted)*g.MASS,PAPER_MASS+COUNT*g.MASS)
    def test_closed_end(self):
        s=Sleeve()
        for _ in range(40):self.assertEqual(s.advance(-9.81,0.,True),[])
        self.assertEqual(s.next,0)
