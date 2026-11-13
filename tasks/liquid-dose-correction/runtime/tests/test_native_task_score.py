import copy
import unittest
from native_task_score import score_measurements


def rows(mass=10.):
    return [{'time_s': (i+1)/20., 'mass_g': mass, 'balance_g': mass,
             'measurement_valid': True, 'pipette_correction_g': .5,
             'spill_g': 0., 'tip_position': [0., 0., 0.],
             'tool_axis_world': [0., 0., 1.],
             'pipette_contained_and_released': True} for i in range(20)]


class NativeScoreTests(unittest.TestCase):
    def test_unchanged_and_no_pipette_cannot_earn_points(self):
        r = rows(10.5)
        self.assertEqual(score_measurements(10.5, r)['score'], 0.)
        r = rows()
        for x in r:
            x['pipette_correction_g'] = 0.
        self.assertEqual(score_measurements(10.5, r)['score'], 0.)

    def test_full_partial_and_missing_disposal(self):
        self.assertEqual(score_measurements(10.5, rows())['score'], 1.)
        self.assertAlmostEqual(score_measurements(10.5, rows(10.275))['score'], .55)
        r = rows()
        for x in r:
            x['pipette_contained_and_released'] = False
        self.assertEqual(score_measurements(10.5, r)['score'], .9)

    def test_invalid_reading_spill_or_support_makes_zero(self):
        r = rows()
        r[-1]['measurement_valid'] = False
        self.assertEqual(score_measurements(10.5, r)['score'], 0.)
        r = rows()
        r[-1]['spill_g'] = .101
        self.assertEqual(score_measurements(10.5, r)['score'], 0.)
        self.assertEqual(score_measurements(10.5, rows(), ['beaker lifted'])['score'], 0.)

    def test_rotating_or_protruding_tool_cannot_finish(self):
        r = rows()
        r[-1]['tool_axis_world'] = [1., 0., 0.]
        self.assertEqual(score_measurements(10.5, r)['score'], .9)
        r = rows()
        r[-1]['pipette_contained_and_released'] = False
        self.assertEqual(score_measurements(10.5, r)['score'], .9)


if __name__ == '__main__':
    unittest.main()
