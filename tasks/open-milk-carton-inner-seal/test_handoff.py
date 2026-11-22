import unittest
from handoff import Transfer

def pair(a,b):return {'actor0':'/World/envs/env_0/'+a,'actor1':'/World/envs/env_0/'+b,'max_force_n':1.}

class TransferTest(unittest.TestCase):
    def test_hook_returns_before_direct_opening(self):
        t=Transfer()
        tool=[pair('LeftArm/tip_left','hook/body'),pair('hook/body','seal/pullring')]
        fingers=[pair('RightArm/'+n,'seal/pullring') for n in ('tip_left','tip_right')]
        t.observe(tool,-1.,False,False)
        for _ in range(3):t.observe(tool+fingers,-1.,False,False)
        self.assertTrue(t.captured);self.assertFalse(t.parked)
        t.observe(fingers,-1.,False,False);self.assertFalse(t.withdrawn)
        t.observe(fingers+[tool[0]],-1.,False,False);self.assertTrue(t.withdrawn)
        t.observe(fingers,-1.,False,True);self.assertTrue(t.parked)
    def test_receiver_pinching_tool_and_ring_is_not_an_independent_capture(self):
        t=Transfer()
        hook=[pair('LeftArm/tip_left','hook/body'),pair('hook/body','seal/pullring')]
        right=[pair('RightArm/'+tip,'seal/pullring') for tip in ('tip_left','tip_right')]
        for _ in range(4):
            t.observe(hook+right+[pair('RightArm/tip_left','hook/body')],-1.,False,False)
        self.assertTrue(t.raised)
        self.assertFalse(t.captured)
        for _ in range(2):t.observe(hook+right,-1.,False,False)
        self.assertTrue(t.captured)
    def test_numpy_sensor_values_can_be_saved(self):
        import json
        import numpy as np
        t=Transfer()
        for _ in range(4):
            t.observe([pair('LeftArm/tip_left','hook/body'),pair('hook/body','seal/pullring')],np.float64(-1.),False,np.bool_(False))
        self.assertTrue(json.loads(json.dumps(vars(t)))['raised'])
    def test_idle_or_single_finger_cannot_fake_transfer(self):
        t=Transfer()
        for _ in range(10):t.observe([pair('RightArm/tip_left','seal/pullring')],-1.,False,True)
        self.assertFalse(t.captured or t.parked)
