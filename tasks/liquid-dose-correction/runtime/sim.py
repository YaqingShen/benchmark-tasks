"""Conservative liquid and tool component model, not a robot dynamics backend.

Volumes are mL; mass is g. The bulb model relates displacement to displaced
liquid with a bounded rate and a first-order valve response. It does not model
air compressibility, elastic contact, capillary wetting, or a real pipette's
calibration. No action may set a vessel's volume directly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
EPS = 1e-9


@dataclass
class Liquid:
    # Each key preserves ingredient identity and whether it passed a cylinder.
    amounts: dict[tuple[str, bool], float] = field(default_factory=dict)

    @property
    def volume(self):
        return sum(self.amounts.values())

    def ingredient(self, name, measured_only=False):
        return sum(v for (k, measured), v in self.amounts.items()
                   if k == name and (measured or not measured_only))

    def take(self, volume):
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("Volume must be finite and nonnegative")
        amount = min(volume, self.volume)
        fraction = amount / self.volume if self.volume > EPS else 0.0
        result = Liquid({k: v * fraction for k, v in self.amounts.items()})
        self.amounts = {k: max(0., v - result.amounts[k]) for k, v in self.amounts.items()}
        return result

    def add(self, other, measured=False):
        for (ingredient, via_cylinder), amount in other.amounts.items():
            key = ingredient, via_cylinder or measured
            self.amounts[key] = self.amounts.get(key, 0.0) + amount

    def mass(self, density):
        return sum(v * density[k] for (k, _), v in self.amounts.items())

    def snapshot(self):
        return {"A": self.ingredient("A"), "B": self.ingredient("B"),
                "total_ml": self.volume,
                "via_cylinder_ml": sum(v for (_, measured), v in self.amounts.items() if measured)}


@dataclass
class Vessel:
    name: str
    capacity_ml: float
    liquid: Liquid = field(default_factory=Liquid)
    capped: bool = False
    cap_turns: float = 0.0
    home_distance_m: float = 0.0
    tilt_deg: float = 0.0
    supported: bool = True
    speed_m_s: float = 0.0
    broken: bool = False
    cap_parked_inside_up: bool = False


@dataclass
class Pipette:
    name: str
    dedicated_to: str
    capacity_ml: float = 5.0
    liquid: Liquid = field(default_factory=Liquid)
    compression: float = 0.0
    tip_target: str | None = None
    tip_submerged: bool = False
    contamination: bool = False
    disposed: bool = False
    correction_g: float = 0.


class Formulation:
    def __init__(self, config=None, initial_beaker_g=0.):
        self.cfg = dict(CONFIG if config is None else config)
        self.vessels = {
            "beaker": Vessel("beaker", 600.),
            "waste": Vessel("waste", 100.),
        }
        for ingredient in self.cfg['targets_g']:
            name='bottle_'+ingredient
            closed=not self.cfg.get('initial_bottle_open',False)
            self.vessels[name]=Vessel(name,500.,Liquid({(ingredient,False):200.}),closed,1.5 if closed else 0.)
        for name in self.cfg.get('cylinders',[]):
            self.vessels[name]=Vessel(name,self.cfg['cylinder_brim_ml'])
        self.pipettes = {'pipette_'+ingredient:Pipette('pipette_'+ingredient,ingredient)for ingredient in self.cfg['targets_g']}
        self.spill = Liquid()
        self.inflight = []
        self.stock_contaminated = False
        self.measuring_contaminated = False
        self.time = 0.0
        self.events = []
        self.trace = []
        self.stage = "Ready"
        self.initial = {ingredient:200. if ingredient in self.cfg['targets_g']else 0. for ingredient in ('A','B')}
        self.tare_g = 0.0
        self.balance_loaded = "beaker"
        self.a_checkpoint_g=None
        self.b_started=False
        self.mass_order_violation=False
        if not math.isfinite(initial_beaker_g) or initial_beaker_g<0:
            raise ValueError('Initial beaker mass must be finite and nonnegative')
        if initial_beaker_g:
            self._receive('beaker',self.take_from('bottle_A',initial_beaker_g/self.cfg['density_g_ml']['A']))
        self.initial_beaker_g=float(initial_beaker_g)
        self.initial_error_g=abs(initial_beaker_g-self.cfg['targets_g']['A'])
        self.record()

    def log(self, event, **kwargs):
        self.events.append({"time_s": round(self.time, 4), "event": event, **kwargs})

    def cap(self, bottle, turns):
        """Component input: relative cap rotation; thread contacts not simulated."""
        obj = self.vessels[bottle]
        if not bottle.startswith("bottle_") or not math.isfinite(turns):
            raise ValueError("Invalid cap action")
        obj.cap_turns = max(0., min(self.cfg["cap_required_turns"], obj.cap_turns + turns))
        # Any thread engagement blocks pouring in this reduced model.
        obj.capped = obj.cap_turns > EPS
        self.log("cap_rotation", bottle=bottle, turns=turns, engaged_turns=obj.cap_turns)

    def _receive(self, target, packet):
        if target is None:
            self.spill.add(packet)
            return
        vessel = self.vessels[target]
        if vessel.capped or vessel.broken:
            self.spill.add(packet)
            return
        if target.startswith("bottle_"):
            expected = target[-1]
            if packet.ingredient("B" if expected == "A" else "A") > EPS:
                self.stock_contaminated = True
        if target.startswith("cylinder_"):
            foreign = "B" if target.endswith("A") else "A"
            if packet.ingredient(foreign) > EPS:
                self.measuring_contaminated = True
        accepted = packet.take(max(0., vessel.capacity_ml - vessel.liquid.volume))
        if target=='beaker':
            if self.b_started and accepted.ingredient('A')>EPS:self.mass_order_violation=True
            if accepted.ingredient('B')>EPS:
                if self.a_checkpoint_g is None:self.mass_order_violation=True
                self.b_started=True
        vessel.liquid.add(accepted, measured=target.startswith("cylinder_"))
        self.spill.add(packet)

    def take_from(self,source,volume):
        """Withdraw liquid conservatively; ingredient identity is preserved."""
        if source=='beaker' and self.b_started and volume>EPS:self.mass_order_violation=True
        return self.vessels[source].liquid.take(volume)

    def pour_step(self, source, target, tilt_deg, dt, capture_fraction=1.0):
        """Reduced gravity flow. Routing/capture are explicit inputs, not contacts.

        45 degrees is a provisional onset; flow varies continuously with tilt,
        available volume, and time. Retained film is an exposed calibration
        parameter. The demo uses direct state feedback and is not policy data.
        """
        if not 0 <= capture_fraction <= 1 or not 0 < dt <= .1 or not math.isfinite(tilt_deg):
            raise ValueError("Invalid flow action")
        obj = self.vessels[source]
        obj.tilt_deg = tilt_deg
        rate = 18. * max(0., math.sin(math.radians(min(135., max(45., tilt_deg)) - 45.)))
        film = self.cfg["cylinder_retained_film_ml"] if source.startswith("cylinder_") else 0.
        available = max(0., obj.liquid.volume - film)
        amount = 0. if obj.capped or obj.broken else min(available, rate * dt)
        packet = self.take_from(source,amount)
        self._receive(target, packet.take(amount * capture_fraction))
        self.spill.add(packet)
        self.advance(dt)
        return amount

    def set_tip(self, pipette, target, submerged=False):
        tool = self.pipettes[pipette]
        if tool.disposed:
            raise ValueError("A discarded pipette cannot be reused")
        if target is not None and target not in self.vessels:
            raise ValueError("Unknown target")
        tool.tip_target, tool.tip_submerged = target, submerged
        if submerged and target is not None:
            vessel = self.vessels[target]
            foreign = "B" if tool.dedicated_to == "A" else "A"
            if vessel.liquid.ingredient(foreign) > EPS:
                tool.contamination = True
                # A previously used tip touching stock can contaminate without
                # measurable transfer. This is a categorical contact constraint.
            if target.startswith("bottle_") and (tool.contamination or target[-1] != tool.dedicated_to):
                self.stock_contaminated = True
        self.log("tip_placement", pipette=pipette, target=target, submerged=submerged)

    def bulb_step(self, pipette, compression, dt=.02):
        tool = self.pipettes[pipette]
        before_g=self.vessels['beaker'].liquid.mass(self.cfg['density_g_ml'])
        if tool.disposed:
            raise ValueError("A discarded pipette cannot be reused")
        if not math.isfinite(compression) or not 0 <= compression <= 1 or not 0 < dt <= .1:
            raise ValueError("Invalid bulb command")
        # Max displaced flow 2 mL/s. This is a parameter, not a validated bulb.
        delta = max(-2.*dt/tool.capacity_ml, min(2.*dt/tool.capacity_ml, compression-tool.compression))
        displaced = abs(delta) * tool.capacity_ml
        target = tool.tip_target
        if delta < 0 and tool.tip_submerged and target is not None:
            vessel = self.vessels[target]
            if not vessel.capped and not vessel.broken:
                film = self.cfg["cylinder_unrecoverable_film_ml"] if target.startswith("cylinder_") else 0.
                packet = self.take_from(target,min(displaced, tool.capacity_ml-tool.liquid.volume,
                                                   max(0., vessel.liquid.volume-film)))
                foreign = "B" if tool.dedicated_to == "A" else "A"
                if packet.ingredient(foreign) > EPS:
                    tool.contamination = True
                tool.liquid.add(packet)
        elif delta > 0:
            packet = tool.liquid.take(displaced)
            if target and target.startswith("bottle_") and tool.contamination:
                self.stock_contaminated = True
            self._receive(target, packet)
        tool.compression += delta
        self.credit_pipette_correction(pipette,before_g)
        self.advance(dt)

    def credit_pipette_correction(self,pipette,before_g):
        """Credit only a pipette-caused decrease in the final dose error."""
        target=sum(self.cfg['targets_g'].values())
        after=self.vessels['beaker'].liquid.mass(self.cfg['density_g_ml'])
        improvement=max(0.,abs(before_g-target)-abs(after-target))
        self.pipettes[pipette].correction_g+=improvement

    def discard_pipette(self, pipette):
        """Component disposal event; bin containment/release need robot physics."""
        tool = self.pipettes[pipette]
        tool.tip_target = None
        tool.tip_submerged = False
        tool.disposed = True
        self.log("pipette_discarded", pipette=pipette, destination="trash_bin")

    def balance(self):
        """Rounded net mass; no readout until the supported beaker is still."""
        vessel = self.vessels[self.balance_loaded]
        if abs(vessel.tilt_deg) > 2 or not vessel.supported or vessel.speed_m_s > .005:
            return None
        resolution = self.cfg["balance_resolution_g"]
        net = vessel.liquid.mass(self.cfg["density_g_ml"]) - self.tare_g
        decimals = max(0, math.ceil(-math.log10(resolution)))
        return round(round(net / resolution) * resolution, decimals)

    def cylinder_reading(self,name):
        """Component approximation of a settled meniscus read against marks.

        Half-division estimates are not camera-derived measurements. Native
        validation must show the meniscus and graduations in an actual view.
        """
        if name not in ('cylinder_A','cylinder_B'):raise ValueError('A dedicated cylinder is required')
        vessel=self.vessels[name]
        if abs(vessel.tilt_deg)>2 or not vessel.supported or vessel.speed_m_s>.005:return None
        increment=self.cfg['cylinder_reading_increment_ml']
        return round(round(vessel.liquid.volume/increment)*increment,3)

    def tare(self):
        if self.balance() is None:
            raise ValueError("Cannot tare an unstable vessel")
        self.tare_g = self.vessels[self.balance_loaded].liquid.mass(self.cfg["density_g_ml"])
        self.log("tare", tare_g=self.tare_g)

    def checkpoint_A(self):
        value=self.balance()
        if value is None or self.b_started or self.vessels['beaker'].liquid.ingredient('B')>EPS:
            raise ValueError('A checkpoint requires a stable A-only beaker')
        if abs(self.tare_g)>EPS:raise ValueError('Recipe uses the original empty-beaker tare')
        self.a_checkpoint_g=value
        self.log('A_mass_checkpoint',net_g=value)
        return value

    def advance(self, dt):
        self.time += dt
        self.assert_conservation()
        if self.time - self.trace[-1]["time_s"] >= .095:
            self.record()

    def wait(self, duration):
        for _ in range(round(duration/.02)):
            self.advance(.02)

    def record(self):
        self.trace.append({"time_s": round(self.time, 4), "stage": self.stage,
            "vessels": {k: {**v.liquid.snapshot(), "capped": v.capped,
                            "cap_turns": v.cap_turns, "tilt_deg": v.tilt_deg} for k,v in self.vessels.items()},
            "pipettes": {k: {**v.liquid.snapshot(), "compression": v.compression,
                              "target": v.tip_target, "submerged": v.tip_submerged,
                              "disposed": v.disposed,"correction_g":v.correction_g} for k,v in self.pipettes.items()},
            "spill_ml": self.spill.volume, "inflight_ml": sum(p.volume for p in self.inflight),
            "balance_g": self.balance() if self.cfg.get('scale_in_task',True) else None,
            "a_checkpoint_g":self.a_checkpoint_g,
            "b_added_g":None if self.a_checkpoint_g is None or self.balance() is None else self.balance()-self.a_checkpoint_g,
            "cylinder_readings_ml": {n:self.cylinder_reading(n)for n in self.cfg.get('cylinders',[])}})

    def assert_conservation(self):
        liquids = [v.liquid for v in self.vessels.values()] + [p.liquid for p in self.pipettes.values()] + [self.spill] + self.inflight
        for ingredient, initial in self.initial.items():
            actual = sum(v.ingredient(ingredient) for v in liquids)
            if abs(actual-initial) > 1e-7:
                raise AssertionError(f"Conservation failed for {ingredient}: {actual}")
        if any(amount < -EPS or not math.isfinite(amount) for v in liquids for amount in v.amounts.values()):
            raise AssertionError("Invalid liquid state")

    def score(self):
        beaker = self.vessels["beaker"]
        dose = {}
        for ingredient, target in self.cfg["targets_g"].items():
            delivered=beaker.liquid.ingredient(ingredient)*self.cfg['density_g_ml'][ingredient]
            error = abs(delivered-target)
            lo, hi = self.cfg["full_tolerance_g"], self.cfg["zero_credit_error_g"][ingredient]
            if self.cfg.get('objective')=='correct_existing_dose':
                hi=abs(self.initial_beaker_g-target)
                if hi<=lo:raise ValueError('Correction task must start outside tolerance')
            quality = 1. if error<=lo+EPS else max(0., min(1., (hi-error)/(hi-lo)))
            dose[ingredient] = {"delivered_ml": beaker.liquid.ingredient(ingredient),
                                "delivered_g":delivered,"target_g":target,"error_g": error, "quality": quality}
        violations = []
        if self.spill.mass(self.cfg['density_g_ml']) > self.cfg["max_spill_g"] + EPS:
            violations.append("spill exceeds 0.1 g")
        if self.stock_contaminated:
            violations.append("stock contamination")
        if self.measuring_contaminated:
            violations.append("wrong ingredient in a dedicated cylinder")
        if self.mass_order_violation:
            violations.append('mass checkpoints invalid: B before A checkpoint, A added after B, or mixture removed')
        if abs(self.tare_g)>EPS:violations.append('loaded beaker was re-tared')
        if sum(p.volume for p in self.inflight) > .001:
            violations.append("liquid still in flight")
        if any(v.broken for v in self.vessels.values()):
            violations.append("broken vessel")
        stable = beaker.supported and abs(beaker.tilt_deg) <= 5. and beaker.speed_m_s <= .005
        if not stable:
            violations.append("beaker not resting upright")
        cap_points = return_points = disposal_points = 0.
        active = all(d["quality"] > EPS for d in dose.values())
        if active:
            finished=[]
            for ingredient in self.cfg['targets_g']:
                b=self.vessels['bottle_'+ingredient];tool=self.pipettes['pipette_'+ingredient]
                stable_bottle=b.supported and abs(b.tilt_deg)<=5 and b.speed_m_s<=.005
                finished.append(tool.disposed and tool.correction_g+EPS>=self.cfg.get('minimum_pipette_correction_g',.01)
                                and stable_bottle)
            disposal_points=.1*sum(finished)/len(finished)
        total = .9*sum(d['quality']for d in dose.values())/len(dose)+disposal_points
        if violations:
            total = 0.
        return {"score": round(total, 8), "dose": dose, "cap_points": cap_points,
                "return_points": return_points, "disposal_points": disposal_points, "violations": violations,
                "pipette_correction_g":{k:p.correction_g for k,p in self.pipettes.items()},
                "spill_ml": self.spill.volume,
                "spill_g":self.spill.mass(self.cfg['density_g_ml']),
                "measurement_method":self.cfg.get('measurement_method'),
                "initial_beaker_g":self.initial_beaker_g,
                "a_checkpoint_g":self.a_checkpoint_g,"final_net_g":self.balance(),
                "status": "component-model result; not a robot or Dream-1 evaluation"}
