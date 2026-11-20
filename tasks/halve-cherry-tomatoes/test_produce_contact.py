import unittest
from produce_contact import FruitContact, contact, RADIUS
class ProduceControls(unittest.TestCase):
    def test_entry_between_heights(self):
        import numpy as np
        from cut_checks import BatchPass
        batch=BatchPass()
        events=[]
        for y,z in [(0.0164,.0030),(.0139,.0024),(.0090,.0015),(-.015,.0015)]:
            events.extend(batch.observe(np.array([0.,.0164-y,0.]),np.eye(3),[(0.,y,z)],[True],True,[True]))
        self.assertEqual(events,[0])

    def test_weak_drive_and_blank(self):
        def advance(with_fruit):
            x = speed = 0.
            dt = 1/960
            for _ in range(480):
                load = contact((0, RADIUS-x, 0), speed)[0] if with_fruit else 0.
                speed += (.40 - load - 8*speed)/.085 * dt
                x += speed*dt
            return x
        self.assertLess(advance(True), .0025)
        self.assertGreater(advance(False), .020)

    def test_drawing_traversal_and_straight_push(self):
        def trial(draw):
            model=FruitContact()
            for j in range(200):
                model.update(0,(draw*j/200,RADIUS-2*RADIUS*j/200,0),.04,2*RADIUS/.04/200)
            return model
        self.assertTrue(trial(.008).ready()[0])
        self.assertFalse(trial(0).ready()[0])
        m=trial(.008)
        m.update(0,(0,RADIUS+.001,0),-.01,.01)
        self.assertFalse(m.ready()[0])

    def test_no_work_if_fruit_moves_with_tool(self):
        model=FruitContact()
        for _ in range(100):
            model.update(0,(0,RADIUS-.002,0),0.,1/240)
        self.assertEqual(model.work[0],0.)
