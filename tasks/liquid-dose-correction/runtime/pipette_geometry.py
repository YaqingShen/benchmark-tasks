"""Shared dimensions of the simplified 5 mL transfer pipette."""
# Reach the low 10 mL meniscus inside the 194 mm tall cylinder while
# keeping the gripper above its rim. Capacity remains 5 mL.
STEM_LENGTH_M=.164
BULB_CENTER_M=.195
STEM_DRY_MASS_KG=.0025
STEM_LIQUID_CAPACITY_ML=.30*STEM_LENGTH_M/.104
# Uncalibrated elastic approximation: 2.9 N at full travel per surface.
BULB_SPRING_N_M=500.
