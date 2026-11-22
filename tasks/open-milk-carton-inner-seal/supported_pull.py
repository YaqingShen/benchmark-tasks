"""Scripted joint position targets for the slow pull. Only the robot is commanded."""
import numpy as np


def compliant_pull(plan, ns, is_left, label, status, generation):
    from yam_arena import ik
    from yam_arena.constants import FINGER_OPEN, SHOULDER, WRIST

    env = ns['labkit'].env()
    arm = env.unwrapped.scene['left_robot' if is_left else 'right_robot']
    q = ns['labkit']._np(arm.data.joint_pos)[0, :6]
    carton = ns['labkit']._part_state(env, ns['CARTON'])
    normal = carton.rotation() @ ns['N']
    if getattr(plan, '_loaded_generation', None) != generation:
        plan._loaded_generation = generation
        plan._loaded_origin = plan.q.copy()
        plan._loaded_demand = 0.
        plan._release_tcp = None
        plan._release_ticks = 0
        plan._contact_offset=np.zeros(3)
        plan._pull_local=ns['N'].copy()
        if not is_left:
            points={}
            for pair in ns['TRACK']['contact_window']['pairs']:
                for a,b in ((0,1),(1,0)):
                    actor=pair[f'actor{a}'];tip=actor.rsplit('/',1)[-1]
                    if ('/RightArm/' in actor and tip in ('tip_left','tip_right')
                        and '/seal/pullring' in pair[f'actor{b}'] and pair['last_points']
                        and pair['max_force_n']>points.get(tip,(0.,))[0]):
                        points[tip]=(pair['max_force_n'],np.mean(pair['last_points'],axis=0))
            if len(points)==2:
                from yam_arena.urdf import quat_xyzw_to_matrix
                actual=ns['labkit']._np(arm.data.body_link_pose_w)[0,list(arm.body_names).index('gripper')]
                wrist=quat_xyzw_to_matrix(actual[3:])
                point=np.mean([obs[1] for obs in points.values()],axis=0)
                plan._contact_offset=wrist.T@(point-actual[:3]-.134*wrist[:,2])
                seal=ns['labkit']._part_state(env,ns['SEAL'])
                origin=ns['labkit']._np(env.unwrapped.scene.env_origins[0])
                line=(point-origin-seal.to_world(ns['anchor']))@(carton.rotation()@ns['S'])
                # Keep the force line on the hinge plane and through the grasp.
                # This avoids twisting the upright ring over the finger's edge.
                slope=float(np.clip(line[0]/max(.001,line[2]),-.30,.30))
                plan._pull_local=ns['S']@np.array([slope,0.,1.])

    if ns['TRACK'].get('seal_material_open'):
        if plan._release_tcp is None:
            # Unload the actuator springs at the actual release pose.
            plan._release_tcp, plan._release_r = ik.tcp_pose(q, plan.base_pos)
            plan.q = q.copy()
        if not is_left:
            plan._release_ticks += 1
            clearance = .060 * ik.smoothstep(min(1., plan._release_ticks / 70.))
            target = plan._release_tcp + clearance * normal
            plan.q, _, _ = ik.solve(target, ik.matrix_to_quat_xyzw(plan._release_r),
                                    plan.q, base_pos=plan.base_pos, iters=60)
        return np.r_[plan.q, 0.], label

    if not all(status.get(k) for k in ('centered_left', 'right_two_fingers', 'parked')):
        plan._loaded_demand = max(0., plan._loaded_demand - .30)
        plan.q += np.clip(plan._loaded_origin-plan.q, -.008, .008)
        return np.r_[plan.q, 0.], label

    # Increase the gripper's commanded tensile load by 3 N per simulated second.
    # The opposing hand balances the previous measured load on the carton.
    measured = np.asarray(ns['TRACK']['contact_window']['external_seal_forces_n'])
    target_force = min(32., max(0., np.mean(measured @ normal))) if is_left else 32.
    plan._loaded_demand += float(np.clip(target_force-plan._loaded_demand, -.15, .15))
    wanted = (carton.rotation() @ plan._pull_local) * plan._loaded_demand
    if is_left:
        # The seam is above the spout. Use its longer horizontal lever arm
        # and press into both folded panels to keep both fingertips engaged.
        wanted = -wanted * np.array([.78,.78,1.15])
    def application_point(joints):
        tcp,rotation=ik.tcp_pose(joints,plan.base_pos)
        return tcp+rotation@plan._contact_offset
    start=application_point(q)
    dq = .00001
    jac = np.array([(application_point(q + dq*axis)-start)/dq for axis in np.eye(6)])
    stiffness = np.repeat([SHOULDER['stiffness'], WRIST['stiffness']], 3)
    limits = np.repeat([SHOULDER['effort'], WRIST['effort']], 3)
    delta = np.clip(jac @ wanted, -.8*limits, .8*limits) / stiffness
    target = np.clip(plan._loaded_origin + delta, ik.LOWER, ik.UPPER)
    plan.q += np.clip(target-plan.q, -.012, .012)
    # Retain the established inner-finger capture as the wire takes the load.
    grip = 0. if is_left else -max(0.,plan._holding_opening-.0003*plan._loaded_demand)/FINGER_OPEN
    return np.r_[plan.q, float(np.clip(grip,0.,1.))], label
