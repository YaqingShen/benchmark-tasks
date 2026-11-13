"""Repeatable dose-error reset cases; the displayed mass is observable."""
import random

def starting_mass(seed):
    rng=random.Random(seed)
    # Balance the two directions and leave meaningful work outside tolerance.
    direction=-1 if rng.random()<.5 else 1
    error=rng.uniform(.15,1.50)
    return round(10.+direction*error,3)
