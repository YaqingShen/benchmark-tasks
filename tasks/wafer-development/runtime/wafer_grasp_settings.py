"""Shared simulation draft; material properties are unmeasured."""
PHYSICS_DT_S = 0.0005
CONTROL_HZ = 20
PAD_STIFFNESS_N_M = 2000.0
PAD_DAMPING_N_S_M = 20.0
PAD_MASS_KG = 0.010
PAD_TRAVEL_M = 0.0008
HANDLE_END_RIBS = ('lower_handle_rib_1',)
PAD_MOUNT_CANT_RAD = 0.04
HINGE_CLOSED_LIMIT_RAD = 0.09
TCP_FEEDBACK_INTEGRATING_PHASES = ('settle', 'grip_hold')
