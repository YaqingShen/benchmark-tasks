import unittest
from contact_rules import classify,centered_top_grip

def pair(a,b,ca='body/box_0',cb='body/box_0',force=1.):
    prefix='/World/envs/env_0/'
    return dict(actor0=prefix+a+'/body',actor1=prefix+b+'/body',collider0=prefix+a+'/'+ca,collider1=prefix+b+'/'+cb,max_force_n=force)

class ContactTests(unittest.TestCase):
    def test_held_tool_cannot_count_as_released(self):
        x=classify([pair('seal','hook'),pair('hook','discard'),pair('RightArm','hook')])
        self.assertFalse(x['seal_supported_by_tray']);self.assertFalse(x['hands_released'])
    def test_remote_tray_contact_is_not_support(self):
        self.assertFalse(classify([pair('seal','hook'),pair('discard','table')])['seal_supported_by_tray'])
    def test_common_table_does_not_connect_seal_to_tray(self):
        self.assertFalse(classify([pair('seal','hook'),pair('hook','table'),pair('discard','table')])['seal_supported_by_tray'])
    def test_parked_tool_cannot_replace_direct_tray_support(self):
        x=classify([pair('seal','hook'),pair('hook','discard')]);self.assertTrue(x['hands_released']);self.assertFalse(x['seal_supported_by_tray'])
        self.assertTrue(classify([pair('seal','discard')])['seal_supported_by_tray'])
    def test_cap_must_remain_untouched(self):
        self.assertIn('reserved_cap_contact',classify([pair('RightArm','cap')])['violations'])
        self.assertFalse(classify([pair('cap','cap_rest')])['violations'])
    def test_mouth_contact_and_zero_impulse(self):
        p=pair('carton','LeftArm','body/mesh_7',force=1.)
        self.assertIn('gripper_mouth_contact',classify([p])['violations'])
        p['max_force_n']=0.;self.assertFalse(classify([p])['violations'])
    def test_lower_wall_contact_is_allowed(self):
        self.assertFalse(classify([pair('carton','LeftArm','body/box_1')])['violations'])
    def test_arm_collision_is_reported(self):
        self.assertIn('arm_arm_collision',classify([pair('LeftArm','RightArm',force=2.)])['violations'])
    def test_interior_guard_is_not_assumed(self):
        self.assertFalse(classify([])['interior_guard_available'])
    def test_protected_face_contact(self):
        p=pair('cap','RightArm','body/inner_liner');x=classify([p],[p['collider0']]);self.assertIn('protected_interior_contact',x['violations'])
    def test_bimanual_support_uses_opposite_arm(self):
        support=pair('LeftArm','carton',cb='body/box_4')
        grasp=pair('RightArm','hook')
        self.assertFalse(classify([support,grasp])['bimanual_support_active'])
        support['actor0']='/World/envs/env_0/LeftArm/gripper/tip_left'
        grasp['actor0']='/World/envs/env_0/RightArm/gripper/tip_right'
        good=classify([support,grasp])
        support['actor0']=support['actor0'].replace('LeftArm','RightArm')
        bad=classify([support,grasp])
        self.assertTrue(good['bimanual_support_active']);self.assertFalse(bad['bimanual_support_active'])
    def test_shoulder_support_requires_load_on_the_paperboard(self):
        grasp=pair('RightArm','hook')
        grasp['actor0']='/World/envs/env_0/RightArm/gripper/tip_left'
        for collider,force,expected in [('body/box_9',2.,True),('body/box_9',.2,False),('body/mesh_9',5.,False)]:
            hand=pair('LeftArm','carton',cb=collider,force=force)
            hand['actor0']='/World/envs/env_0/LeftArm/gripper/tip_right'
            self.assertEqual(classify([hand,grasp])['bimanual_support_active'],expected)
    def test_fin_support_requires_two_fingertips(self):
        grasp=pair('RightArm','hook');grasp['actor0']='/World/envs/env_0/RightArm/gripper/tip_left'
        a=pair('LeftArm','carton',cb='body/box_5');a['actor0']='/World/envs/env_0/LeftArm/gripper/tip_left'
        b=dict(a,actor0=a['actor0'].replace('tip_left','tip_right'))
        self.assertFalse(classify([a,grasp])['bimanual_support_active'])
        self.assertTrue(classify([a,b,grasp])['bimanual_support_active'])
    def test_centered_pinch_requires_both_loaded_sides(self):
        grasp=pair('RightArm','hook')
        grasp['actor0']='/World/envs/env_0/RightArm/gripper/tip_left'
        left=pair('LeftArm','carton',cb='body/box_5',force=2.)
        left.update(actor0='/World/envs/env_0/LeftArm/gripper/tip_left',last_points=[[.002,0.,.255]])
        right=dict(left,actor0=left['actor0'].replace('tip_left','tip_right'),last_points=[[-.002,0.,.255]])
        def supported(*pairs):
            return centered_top_grip([grasp,*pairs],lambda point:point,.250)[0]
        self.assertTrue(supported(left,right))
        self.assertFalse(supported(left))
        self.assertFalse(supported(left,dict(right,last_points=[[-.002,.03,.255]])))
        self.assertFalse(supported(left,dict(right,last_points=[[.002,0.,.255]])))
        self.assertFalse(supported(left,dict(right,max_force_n=.02)))
        self.assertFalse(supported(left,dict(right,collider1='/World/envs/env_0/carton/body/mesh_7')))
    def test_tool_tip_touching_worktop_is_contamination(self):
        for collider in ('body/box_1','body/box_2','body/box_3','body/box_4','body/box_5','body/mesh_0'):
            self.assertIn('tool_tip_worktop_contact',classify([pair('hook','table',collider)])['violations'])
if __name__=='__main__':unittest.main()
