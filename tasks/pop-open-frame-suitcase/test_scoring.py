import unittest
from scoring import LockScore
from mechanism import CODE, START, STEP, gap_mm
from math import radians

class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.scorer=LockScore()
        self.goal=[n*STEP for n in CODE]
        self.start=[n*STEP for n in START]
        self.scorer.update(self.start,0,0,0)

    def test_correct_release_then_hand_clear(self):
        self.assertAlmostEqual(self.scorer.update(self.goal,0,0,0),.45)
        self.scorer.update(self.goal,4,12,0)
        for _ in range(10):
            self.assertAlmostEqual(self.scorer.update(self.goal,0,26,0),.45)
        self.assertEqual(self.scorer.update(self.goal,0,26,0),1)
        self.assertAlmostEqual(self.scorer.update(self.goal,0,0,0),.45)

    def test_wrong_release_and_prying(self):
        self.scorer.update(self.start,1,7,0)
        for _ in range(12): self.assertEqual(self.scorer.update(self.goal,0,30,0),0)
        self.setUp()
        self.scorer.update(self.goal,4,8,2)
        for _ in range(12): self.assertEqual(self.scorer.update(self.goal,0,30,0),0)

    def test_gap_reference(self):
        self.assertGreater(gap_mm(radians(8)),20)
