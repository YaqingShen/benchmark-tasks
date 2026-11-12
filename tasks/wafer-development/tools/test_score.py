"""Scorer state-machine tests; the geometric helpers have native-trace coverage."""
import copy,math,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'runtime/benchmark')]
from wafer_develop_score import DevelopmentTaskScorer


def pose(x,y,z,vertical=False):
    return [x,y,z,0.,math.sqrt(.5)if vertical else 0.,0.,math.sqrt(.5)if vertical else 1.]


META=dict(pillar_top_centers_world_m=[[.032,0,.9995],[-.016,.0277128,.9995],[-.016,-.0277128,.9995]],
    pillar_radius_m=.0005,wafer_radius_m=.05,wafer_flat_length_m=.0325,wafer_local_bottom_z_m=-.0005)


def sequence(dwell=30.5):
    entered=.5;exited=entered+dwell;started=.55;stopped=exited+.05
    for i in range(round((exited+2.5)/.05)+1):
        t=round(i*.05,8);immersed=entered<=t<exited;returned=t>=exited+1.2
        wafer=pose(.4,0,.87,True)if immersed else pose(0,0,1. if t<.1 or returned else 1.02)
        tool=pose(wafer[0],0,wafer[2]+.02)if t>=.1 and not returned else pose(0,0,1. if t<.1 else 1.04)
        hand=pose(tool[0],0,tool[2]-.134)
        press=(.5<=t<=.6 or exited<=t<=exited+.1)
        elapsed=0. if t<started else min(t,stopped)-started
        yield dict(time_s=t,wafer_pose_world=wafer,clamp_pose_world=tool,measured_hand_pose_world=hand,
            right_hand_pose_world=pose(1,0,1),handle_paddle_poses_world=[tool,tool],
            beaker_pose_world=pose(.4,0,.8),timer_base_pose_world=pose(.6,0,.8),
            timer_button_pose_world=pose(.632,-.02,.834-(.0015 if press else 0.)),
            fixture_poses_world={},objects_present={'wafer':True},objects_intact={'wafer':True},
            develop_objects_intact=True,robot_geometry_intact={'left':True,'right':True},
            robot_geometry_check={'status':'CLEAR','complete':True},
            hand_handle_liquid_clearance_m=.02,measured_jaw_gap_m=.003 if returned or t<.1 else .001,
            timer_elapsed_s=elapsed,timer_remaining_s=max(0.,30.-elapsed),test_held=t>=.1 and not returned)


class ScoreTests(unittest.TestCase):
    def evaluate(self,rows,right_finger=True):
        def pads(row,meta):
            held=row['test_held']
            return dict(both_engaged=held,pads=[dict(overlaps_strip=held,signed_gap_m=0.)]*2)
        scorer=DevelopmentTaskScorer(META)
        with patch('wafer_develop_score.pad_contact_geometry',side_effect=pads),patch.object(DevelopmentTaskScorer,'_right_finger_at_button',return_value=right_finger):
            for row in rows:scorer.update(row)
        return scorer.result()

    def test_complete_and_force_independent(self):
        rows=list(sequence());self.assertTrue(self.evaluate(rows)['success'])
        for row in rows:row['jaw_contact_normal_force_N']=[9999,9999];row['timer_reading_s']=-999
        self.assertEqual(self.evaluate(rows)['score'],1.)

    def test_dwell_boundaries(self):
        for duration,expected in[(27.95,False),(28.,True),(32.,True),(32.05,False)]:
            with self.subTest(duration=duration):
                self.assertEqual(self.evaluate(sequence(duration))['checks']['immersion_time'],expected)

    def test_incomplete_process_keeps_valid_handling_credit(self):
        result=self.evaluate(sequence(27.95))
        self.assertEqual(result['score'],.6)
        self.assertFalse(result['success'])
        self.assertEqual(result['invalid_reasons'],[])

    def test_safe_tool_return_is_allowed_but_drop_is_not(self):
        rows=list(sequence())
        for i in range(1,25):
            row=copy.deepcopy(rows[-1]);row['time_s']=round(rows[-1]['time_s']+.05,8);rows.append(row)
        for row in rows:
            if row['time_s']>=33.3:
                row['clamp_pose_world']=pose(0,0,1.)
                row['handle_paddle_poses_world']=[pose(0,0,1.),pose(0,0,1.)]
        pending=self.evaluate([r for r in rows if r['time_s']<=33.5])
        self.assertEqual(pending['invalid_reasons'],[])
        self.assertTrue(pending['tool_return_settling'])
        self.assertFalse(pending['success'])
        self.assertTrue(self.evaluate(rows)['success'])
        rows[-1]['clamp_pose_world']=pose(0,0,.9)
        rows[-1]['handle_paddle_poses_world']=[pose(0,0,.9),pose(0,0,.9)]
        self.assertEqual(self.evaluate(rows)['score'],0.)

    def test_wrong_hand_and_early_timer_are_invalid(self):
        self.assertIn('timer_not_pressed_by_right_finger',self.evaluate(sequence(),False)['invalid_reasons'])
        rows=list(sequence())
        for row in rows:
            if .5<=row['time_s']<=.6:row['wafer_pose_world']=pose(0,0,1.02)
        self.assertIn('timer_started_before_full_immersion',self.evaluate(rows)['invalid_reasons'])

    def test_drop_and_wet_hand_are_invalid(self):
        rows=list(sequence());rows[30]['test_held']=False
        self.assertEqual(self.evaluate(rows)['score'],0.)
        rows=list(sequence());rows[30]['hand_handle_liquid_clearance_m']=.009
        self.assertIn('hand_or_handle_too_close_to_liquid',self.evaluate(rows)['invalid_reasons'])

    def test_missing_collision_or_sample_is_incomplete(self):
        rows=list(sequence());del rows[40]['robot_geometry_check']
        self.assertFalse(self.evaluate(rows)['evidence_complete'])
        rows=list(sequence());del rows[40]
        self.assertEqual(self.evaluate(rows)['score'],0.)

    def test_visible_timer_must_match_mechanical_operation(self):
        rows=list(sequence());rows[20]['timer_remaining_s']=30.
        self.assertIn('timer_display_does_not_follow_mechanical_presses',self.evaluate(rows)['invalid_reasons'])

    def test_supported_but_not_released_has_no_return_credit(self):
        rows=list(sequence())
        for row in rows:
            if row['time_s']>=32.2:row['test_held']=True;row['measured_jaw_gap_m']=.001
        self.assertFalse(self.evaluate(rows)['checks']['unassisted_return'])

    def test_collision_and_modified_objects_zero_partial_credit(self):
        rows=list(sequence());rows[-1]['robot_geometry_check']['status']='COLLISION'
        self.assertEqual(self.evaluate(rows)['score'],0.)
        rows=list(sequence());rows[-1]['objects_intact']['wafer']=False
        self.assertEqual(self.evaluate(rows)['score'],0.)


if __name__=='__main__':unittest.main()
