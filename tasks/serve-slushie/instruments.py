"""Simulated pickup scale, gravimetric bookkeeping and hopper temperature probe."""
from math import expm1
from cup_geometry import CUP_MASS_KG
from fill_model import HOPPER_ML

DENSITY_G_ML = 1.05  # Assumed bulk density of the prepared mix


class Instruments:
    def __init__(self, initial_c=-1.6, equilibrium_c=-1.3):
        self.product_c = self.probe_c = initial_c
        self.equilibrium_c = equilibrium_c

    def advance(self, dt, model, on_stand, resting, released):
        # Effective refrigerated-batch response. Perfect mixing keeps discharge
        # at the bulk temperature; the sensor has a separate first-order lag.
        self.product_c += (self.equilibrium_c - self.product_c) * -expm1(-dt / 240.)
        self.probe_c += (self.product_c - self.probe_c) * -expm1(-dt / 3.)
        net_g = model.inside * DENSITY_G_ML
        gross_g = 0.
        if on_stand:
            gross_g = round(CUP_MASS_KG * 1000 + net_g, 1) if resting and released else None
        return {
            "pickup_scale_g": gross_g,
            "hopper_net_g": round(max(0., HOPPER_ML - model.emitted) * DENSITY_G_ML, 1),
            "tray_retained_g": round(model.tray_spill * DENSITY_G_ML, 1),
            "other_spill_g": round((model.spill - model.tray_spill) * DENSITY_G_ML, 1),
            "secondary_pan_g": round(model.pan_spill * DENSITY_G_ML, 1),
            "pad_covers_g": round(model.cover_spill * DENSITY_G_ML, 1),
            "unrecovered_g": round((model.spill-model.tray_spill-model.pan_spill-model.cover_spill)*DENSITY_G_ML, 1),
            "probe_c": round(self.probe_c, 1) if model.emitted < HOPPER_ML else None,
        }
