"""Observable stand-stability credit, conditional on actual task progress."""

from appliance_config import MODEL
FULL = MODEL["stand_full_credit_m"]
ZERO = MODEL["stand_zero_credit_m"]


def stand_credit(*, first_turn_complete, max_displacement_m):
    """Return the normalized stability component; multiply by 0.10 once.

    Displacement is the largest distance from the starting base position over
    the entire trial, including motion before the first turn. Returning a moved
    base to its starting mark does not erase the earlier displacement.
    """
    if not first_turn_complete:
        return 0.0
    if max_displacement_m <= FULL:
        return 1.0
    if max_displacement_m >= ZERO:
        return 0.0
    return (ZERO - max_displacement_m) / (ZERO - FULL)
