"""Reviewer counterexamples through the public trajectory scorer and lock rule."""
import math
import unittest
import numpy as np
import task as T
from labkit import PartState
from rotation_lock import limits, engaged, HOME_ZONE, OPEN_SPIN_STOP, CLOSED_CLEARANCE, LID_MAX


def state(degrees=0.0, lid_degrees=0.0, shift=0.0):
    z = np.zeros(3)
    obj = PartState(np.array(T.HOME) + [shift, 0., 0.],
                    np.array([0., 0., 0., 1.]), z, z, z, z,
                    {'spin': math.radians(degrees), 'lid': math.radians(lid_degrees)})
    return {'waffle': obj}


def trajectory(states):
    T.phase_conversion.reset()
    track = T.new_track()
    for observation in states:
        T.track_update(track, observation)
    return T.breakdown(states[-1], track)


class ReviewRegressions(unittest.TestCase):
    def test_untouched_for_entire_action_limit(self):
        result = trajectory([state(lid_degrees=45.)] * 8500)
        self.assertEqual(result['score'], 0.)
        self.assertEqual(result['stand_score'], 0.)
        self.assertFalse(result['stand_eligible'])

    def test_closing_or_incomplete_turn_does_not_earn_stability(self):
        for angle in (0., 7., 90., 150., 172.999):
            with self.subTest(angle=angle):
                result = trajectory([state(lid_degrees=45.), state(), state(angle), state()])
                self.assertEqual(result['score'], 0.)
                self.assertEqual(result['stand_score'], 0.)

    def test_first_turn_threshold_and_displacement_boundaries(self):
        for shift, credit in ((0., .10), (.009, .10), (.010, .10), (.020, .05), (.030, 0.), (.05, 0.)):
            with self.subTest(shift=shift):
                result = trajectory([state(), state(173., shift=shift)])
                self.assertTrue(result['stand_eligible'])
                self.assertAlmostEqual(result['stand_score'], credit)
                self.assertAlmostEqual(result['score'], .4 + credit)

    def test_returning_base_does_not_erase_prior_motion(self):
        result = trajectory([state(shift=.02), state(), state(180.), state()])
        self.assertEqual(result['stand_score'], .05)
        self.assertEqual(result['score'], .65)

    def test_unsafe_turn_revokes_even_earned_stability(self):
        result = trajectory([state(), state(180.), state(90., 10.)])
        self.assertEqual(result['score'], 0.)
        self.assertEqual(result['stand_score'], 0.)

    def test_rotation_lock_operating_sequence(self):
        self.assertEqual(limits(0., math.radians(45)), ((0., OPEN_SPIN_STOP), (0., LID_MAX)))
        self.assertEqual(limits(0., 0.), ((0., math.pi), (0., LID_MAX)))
        self.assertEqual(limits(math.pi, 0.), ((0., math.pi), (0., CLOSED_CLEARANCE)))
        self.assertEqual(limits(HOME_ZONE, 0.)[1], (0., LID_MAX))
        # An invalid open/inverted state is not silently repaired by a pose write.
        self.assertEqual(limits(math.pi, .5)[1], (0., LID_MAX))

    def test_solver_overshoot_cannot_release_an_engaged_lock(self):
        self.assertTrue(engaged(math.pi, math.radians(.3), True))
        self.assertEqual(limits(math.pi, math.radians(.3), True)[1], (0., CLOSED_CLEARANCE))
        self.assertFalse(engaged(HOME_ZONE, 0., True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
