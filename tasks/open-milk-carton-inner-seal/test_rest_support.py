import unittest
import numpy as np
from rest_support import HandleSupport


def contact(index):
    return dict(collider1='/World/envs/env_0/hook/body/box_0',
                collider0=f'/World/envs/env_0/tool_mat/body/box_{index}',max_force_n=.3)


class SupportTest(unittest.TestCase):
    def test_actual_both_posts_then_sleeping_pair(self):
        s=HandleSupport();pose=np.zeros((4,3))
        self.assertFalse(s.observe([contact(0),contact(1)],pose))
        self.assertTrue(s.observe([contact(1),contact(2)],pose))
        self.assertTrue(s.observe([],pose+.0001))
        moved=pose.copy();moved[0,2]=.001
        self.assertFalse(s.observe([],moved))
        self.assertFalse(s.observe([],pose))
