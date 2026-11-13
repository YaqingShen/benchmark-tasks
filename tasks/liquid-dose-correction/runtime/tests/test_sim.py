import math
import unittest
from unittest.mock import patch
from sim import Formulation
from run_preview import run,aspirate,dispense


def prepared(mass=10.,used=True,disposed=True):
    e=Formulation(initial_beaker_g=9.5);e.cap('bottle_A',-1.5)
    e.vessels['bottle_A'].cap_parked_inside_up=True
    if mass>=9.5:e._receive('beaker',e.take_from('bottle_A',(mass-9.5)/e.cfg['density_g_ml']['A']))
    else:e._receive('waste',e.take_from('beaker',(9.5-mass)/e.cfg['density_g_ml']['A']))
    e.pipettes['pipette_A'].correction_g=.1 if used else 0.
    e.pipettes['pipette_A'].disposed=disposed
    e.assert_conservation();return e


class SingleLiquidTests(unittest.TestCase):
    def test_idle_zero(self):self.assertEqual(Formulation().score()['score'],0.)

    def test_only_one_bottle_and_pipette(self):
        e=Formulation()
        self.assertEqual(set(e.vessels),{'bottle_A','beaker','waste'})
        self.assertEqual(set(e.pipettes),{'pipette_A'})
        self.assertEqual(e.cfg['targets_g'],{'A':10.})

    def test_exact_mass_full_with_open_bottle(self):
        e=prepared();self.assertEqual(e.score()['score'],1.)
        self.assertAlmostEqual(e.score()['dose']['A']['delivered_ml'],10/1.03)
        self.assertEqual(e.balance(),10.);self.assertFalse(e.vessels['bottle_A'].capped)

    def test_meaningful_partial_credit(self):
        self.assertGreater(prepared(9.75).score()['score'],.4)
        self.assertLess(prepared(9.75).score()['score'],1.)
        self.assertEqual(prepared(8.).score()['score'],0.)

    def test_prefilled_idle_cases_score_zero(self):
        for mass in (8.6,9.5,10.5,11.5):
            self.assertEqual(Formulation(initial_beaker_g=mass).score()['score'],0.)

    def test_both_correction_directions_finish(self):
        for mass in (8.6,9.5,10.5,11.5):
            e=run(mass)
            self.assertEqual(e.score()['score'],1.)
            self.assertEqual(e.balance(),10.)
            self.assertFalse(e.vessels['bottle_A'].capped)
            self.assertFalse(any(x['event']=='tare'for x in e.events))

    def test_tolerance_boundaries(self):
        for mass in (9.95,10.05):self.assertEqual(prepared(mass).score()['score'],1.)
        for mass in (9.949,10.051):self.assertLess(prepared(mass).score()['score'],1.)

    def test_no_cylinder_feedback(self):
        with patch.object(Formulation,'cylinder_reading',side_effect=AssertionError('Cylinder used')):
            self.assertEqual(run().score()['score'],1.)

    def test_closed_bottle_blocks_flow(self):
        e=Formulation();e.cap('bottle_A',1.5)
        self.assertEqual(e.pour_step('bottle_A','beaker',100.,.1),0.)
        self.assertEqual(e.balance(),0.)

    def test_pipette_requires_immersed_tip(self):
        e=Formulation();e.pipettes['pipette_A'].compression=1.
        e.set_tip('pipette_A','bottle_A',submerged=False);e.bulb_step('pipette_A',0.,.1)
        self.assertEqual(e.pipettes['pipette_A'].liquid.volume,0.)

    def test_correction_downward_is_conservative(self):
        e=prepared(10.5,used=False,disposed=False)
        aspirate(e,'pipette_A','beaker',.5/1.03);dispense(e,'pipette_A','waste')
        self.assertEqual(e.balance(),10.);self.assertAlmostEqual(e.pipettes['pipette_A'].correction_g,.5)
        e.assert_conservation()

    def test_correction_upward_is_conservative(self):
        e=prepared(9.5,used=False,disposed=False)
        aspirate(e,'pipette_A','bottle_A',.5/1.03);dispense(e,'pipette_A','beaker')
        self.assertEqual(e.balance(),10.);self.assertAlmostEqual(e.pipettes['pipette_A'].correction_g,.5)
        e.assert_conservation()

    def test_worsening_dose_is_not_fine_correction(self):
        e=prepared(10.,used=False,disposed=False)
        aspirate(e,'pipette_A','beaker',.5/1.03)
        self.assertEqual(e.pipettes['pipette_A'].correction_g,0.)

    def test_unused_pipette_cannot_earn_cleanup_credit(self):
        self.assertAlmostEqual(prepared(used=False).score()['score'],.9)

    def test_used_pipette_must_be_discarded(self):
        e=prepared(disposed=False);self.assertAlmostEqual(e.score()['score'],.9)
        e.discard_pipette('pipette_A');self.assertEqual(e.score()['score'],1.)
        with self.assertRaises(ValueError):e.set_tip('pipette_A','bottle_A',True)

    def test_cap_handling_is_not_scored(self):
        e=prepared();e.vessels['bottle_A'].cap_parked_inside_up=False
        self.assertEqual(e.score()['score'],1.)

    def test_bottle_need_not_return_to_precise_home_or_be_recapped(self):
        e=prepared();e.vessels['bottle_A'].home_distance_m=.10
        self.assertEqual(e.score()['score'],1.)
        e.vessels['bottle_A'].supported=False;self.assertAlmostEqual(e.score()['score'],.9)

    def test_spill_and_overflow_conserved(self):
        e=prepared();e._receive(None,e.take_from('bottle_A',.11));self.assertEqual(e.score()['score'],0.)
        e._receive('waste',e.take_from('bottle_A',150.));self.assertAlmostEqual(e.vessels['waste'].liquid.volume,100.)
        e.assert_conservation()

    def test_unstable_beaker_cannot_score(self):
        e=prepared();e.vessels['beaker'].supported=False
        self.assertIsNone(e.balance());self.assertEqual(e.score()['score'],0.)

    def test_loaded_retare_rejected(self):
        e=prepared();e.tare();self.assertEqual(e.score()['score'],0.)
        with self.assertRaises(ValueError):e.checkpoint_A()

    def test_invalid_numeric_inputs(self):
        for value in (-1.,math.nan,math.inf):
            with self.assertRaises(ValueError):Formulation().take_from('bottle_A',value)

    def test_component_sequence_matches_simplified_goal(self):
        e=run();self.assertEqual(e.score()['score'],1.);self.assertEqual(e.balance(),10.)
        self.assertTrue(e.pipettes['pipette_A'].disposed)
        self.assertGreater(e.pipettes['pipette_A'].correction_g,.01)
        self.assertFalse(e.vessels['bottle_A'].capped)
        self.assertFalse(any('B:'in r.get('label','')for r in e.events));e.assert_conservation()

    def test_random_resets_are_reproducible_and_need_real_correction(self):
        from reset_cases import starting_mass
        masses=[starting_mass(seed) for seed in range(30)]
        self.assertEqual(masses,[starting_mass(seed)for seed in range(30)])
        self.assertTrue(any(m<10 for m in masses)and any(m>10 for m in masses))
        for mass in masses:
            self.assertTrue(8.5<=mass<=11.5 and abs(mass-10)>=.15-1e-8)
            self.assertEqual(Formulation(initial_beaker_g=mass).score()['score'],0.)
            e=run(mass)
            self.assertEqual(e.score()['score'],1.)
            self.assertAlmostEqual(e.balance(),10.,places=3)
            e.assert_conservation()

    def test_display_resolution_is_not_tolerance(self):
        e=prepared();self.assertEqual(e.cfg['balance_resolution_g'],.001)
        self.assertEqual(e.cfg['full_tolerance_g'],.05)
        e._receive('beaker',e.take_from('bottle_A',.001/1.03));self.assertEqual(e.balance(),10.001)

if __name__=='__main__':unittest.main()
