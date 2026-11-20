"""Exercise the live scoring callback with controlled station readings. Run in the YAM environment."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from slicing import Trial

def world(p):
    return np.array((.30+p[1],.071-p[0],.7475+p[2]))

class EndStateTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        envvar=patch.dict(os.environ,TOMATO_RUN_DIR=directory.name)
        envvar.start();self.addCleanup(envvar.stop)
        self.trial=Trial(world)
        self.scene={'left_robot':np.array([0.,.25,1.]),'right_robot':np.array([0.,-.25,1.])}
        self.env=SimpleNamespace(unwrapped=SimpleNamespace(scene=self.scene))
        self.rot=Rotation.from_euler('z',-90,degrees=True).as_quat()
        self.install('upper_lid',(-.117,.012,.110));self.install('knife',(.071,-.11,.094))
        for i in range(8):
            x=.02+.033*(i%4);y=-.026+.052*(i//4)
            self.install(f'tomato_{i}',(x,y,.094))
            for h in (0,1):self.install(f'half_{2*i+h}',(x,y,.094))
        for target,method in [('labkit._np',np.asarray),('yam_arena.runtime.tcp_world',lambda robot:robot)]:
            patched=patch(target,method);patched.start();self.addCleanup(patched.stop)

    def install(self,name,p):
        data=SimpleNamespace(root_link_pose_w=np.r_[world(p),self.rot][None],root_vel_w=np.zeros((1,6)))
        self.scene[name]=SimpleNamespace(data=data)

    def settle(self,cut_ids,batch,groups=None):
        self.trial.cut.update(cut_ids);self.trial.batch.batch_complete=batch
        self.trial.batch.passes=[set(cut_ids)] if groups is None else list(map(set,groups))
        for _ in range(8): self.trial.tick(self.env)
        return self.trial.credit

    def test_parked_unearned_pieces(self):
        self.assertEqual(self.settle([],False),0.)

    def test_half_a_batch(self):
        self.assertAlmostEqual(self.settle(range(4),False),.45)

    def test_all_in_separate_passes(self):
        self.assertAlmostEqual(self.settle(range(8),False,[[i] for i in range(8)]),.1125)

    def test_two_equal_passes(self):
        self.assertAlmostEqual(self.settle(range(8),False,[range(4),range(4,8)]),.45)

    def test_tied_pass_with_retained_halves(self):
        self.install('half_0',(.32,.18,.014))
        self.assertAlmostEqual(self.settle(range(8),False,[range(4),range(4,8)]),.45)

    def test_dropped_half(self):
        self.install('half_3',(.32,.18,.014))
        self.assertAlmostEqual(self.settle(range(8),True),.8875)

    def test_tool_still_held(self):
        self.scene['right_robot']=world((.205,-.121,.094))
        self.assertAlmostEqual(self.settle(range(8),True),.9)

    def test_tipped_lid(self):
        self.scene['upper_lid'].data.root_link_pose_w[0,3:]=Rotation.from_euler('x',30,degrees=True).as_quat()
        self.assertAlmostEqual(self.settle(range(8),True),.9)

    def test_clean_finish_and_loss_of_credit(self):
        self.assertAlmostEqual(self.settle(range(8),True),1.0)
        self.scene['half_9'].data.root_vel_w[0,0]=.05
        self.trial.tick(self.env)
        self.assertFalse(self.trial.final);self.assertAlmostEqual(self.trial.credit,.8875)

if __name__=='__main__': unittest.main()
