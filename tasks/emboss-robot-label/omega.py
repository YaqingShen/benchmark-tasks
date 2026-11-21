"""Physical dial and trigger observations for a DYMO Omega S0717930."""
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
MODES = ("emboss", "space", "cut")
OMEGA_CATALOG_GLYPH_HEIGHT_MM = 3.7


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def interpolate(points, angle):
    i = bisect_right(points, (angle, math.inf))
    if i == 0:
        return points[0][1]
    if i == len(points):
        return points[-1][1]
    a, u = points[i-1]
    b, v = points[i]
    return u + (v-u)*(angle-a)/(b-a)


def mode(symbol):
    return {" ": "space", "CUT": "cut"}.get(symbol, "emboss")


class Omega:
    def __init__(self, folder=ROOT):
        folder = Path(folder)
        meta = folder / "omega_measurements.json"
        if not meta.is_file():
            raise FileNotFoundError("Omega hardware data missing. See README.md before running a hardware-matched trial.")
        config = json.loads(meta.read_text())
        if config["model"] != "S0717930":
            raise ValueError("Expected DE/GB Omega S0717930")
        for key in ("dial_evidence", "strip_evidence", "gauge_record"):
            if not (folder / config[key]).is_file():
                raise ValueError(f"Missing evidence file for {key}")
        self.symbols = tuple(config["wheel_ccw_from_A"])
        if (not self.symbols or self.symbols[0] != "A" or len(self.symbols) != len(set(self.symbols))
                or not set(("A","B","O","R","T"," ","CUT")).issubset(self.symbols)):
            raise ValueError("Incomplete or ambiguous dial transcription")
        degrees = tuple(map(float, config["slot_degrees_ccw"]))
        if (len(degrees) != len(self.symbols) or degrees[0] != 0 or degrees[-1] >= 360
                or not all(math.isfinite(x) for x in degrees) or any(b<=a for a,b in zip(degrees,degrees[1:]))):
            raise ValueError("Expected increasing detent angles in [0,360) from A")
        self.angles = tuple(map(math.radians, degrees))
        self.radius = positive(config["wheel_diameter_mm"], "dial diameter") / 2000
        self.feed = positive(config["feed_pitch_mm"], "tape pitch") / 1000
        self.glyph_height = positive(config.get("glyph_height_mm", OMEGA_CATALOG_GLYPH_HEIGHT_MM),
                                     "printed glyph height") / 1000
        self.end = math.radians(positive(config["end_stop_deg"], "end stop"))
        self.complete = {m: math.radians(positive(config["completed_deg"][m], m)) for m in MODES}
        self.release = math.radians(positive(config["released_deg"], "released threshold"))
        if not all(self.release < v <= self.end for v in self.complete.values()):
            raise ValueError("Event angles must lie within the measured stroke")
        with (folder / "omega_force.csv").open() as fh:
            self.curves = self._curves(csv.DictReader(fh))

    def _curves(self, rows):
        traces = {}
        for row in rows:
            key = (row["mode"], row["phase"])
            if key[0] not in MODES or key[1] not in ("press","return"):
                raise ValueError(f"Unknown measurement condition {key}")
            a = math.radians(float(row["angle_deg"]))
            f = float(row["force_N"])
            if not all(math.isfinite(x) and x >= 0 for x in (a,f)):
                raise ValueError("Invalid angle or force")
            arm = positive(row["perpendicular_arm_mm"], "moment arm") / 1000
            traces.setdefault(key, {}).setdefault(row["trial"], []).append((a, f*arm))
        for m in MODES:
            for phase in ("press","return"):
                curves = traces.get((m,phase), {})
                if len(curves) < 3:
                    raise ValueError(f"Need three independent {m}/{phase} traces")
                for points in curves.values():
                    points.sort()
                    if (len(points)<3 or points[0][0]!=0 or points[-1][0]<self.end-1e-8
                            or any(b[0]<=a[0] for a,b in zip(points,points[1:]))):
                        raise ValueError(f"Incomplete trace for {m}/{phase}")
        return traces

    def nearest(self, q):
        offsets = [math.remainder(q+a, math.tau) for a in self.angles]
        i = min(range(len(offsets)), key=lambda j:abs(offsets[j]))
        return self.symbols[i], abs(offsets[i]), q-offsets[i]

    def torque(self, symbol, phase, depth):
        traces = self.curves[mode(symbol),phase]
        return median(interpolate(points, depth) for points in traces.values())

class PrototypeLoad(Omega):
    """Engineering comparison using the original dial and nominal spring law."""
    def __init__(self):
        self.file = ROOT / "omega_manual.json"
        data = json.loads(self.file.read_text())
        if data["hardware_measured"] is not False:
            raise ValueError("The manual profile must remain identified as unmeasured")
        v = data["prototype_assumptions"]
        self.symbols = tuple(v["wheel_ccw_from_A"])
        self.angles = tuple(i*math.tau/len(self.symbols) for i in range(len(self.symbols)))
        self.radius = positive(v["wheel_diameter_mm"],"wheel diameter")/2000
        self.feed = positive(v["feed_pitch_mm"],"feed pitch")/1000
        self.glyph_height = positive(v["glyph_height_mm"],"glyph height")/1000
        self.end, self.release = v["end_stop_rad"], v["release_rad"]
        self.complete = dict.fromkeys(MODES, v["completed_rad"])
        self.spring = positive(v["trigger_stiffness_Nm_per_rad"],"trigger spring")
        self.damping = v["trigger_damping_Nms_per_rad"]

    def torque(self, symbol, phase, depth):
        return self.spring * depth


class OfficeMateScenario(PrototypeLoad):
    """Sensitivity run on the prototype fixture, using the historical force ceilings."""
    tape_width = 0.0127
    moment_arm = 0.042
    full_stroke_force = {"emboss": 8.0, "cut": 9.0}

    def __init__(self):
        super().__init__()
        self.feed = 0.0059
        self.glyph_height = 0.0048  # Size 1 nominal in A-A-211B.

    def torque(self, symbol, phase, depth):
        operation = mode(symbol)
        if operation == "space":
            return super().torque(symbol, phase, depth)
        return self.full_stroke_force[operation] * self.moment_arm * depth / self.end


def dial_moves(profile, q, symbol):
    """Each regrasp ends at a real detent, including on nonuniform wheels."""
    delta = math.remainder(-profile.angles[profile.symbols.index(symbol)]-q, math.tau)
    if abs(delta)<1e-8:
        return []
    sign = math.copysign(1, delta)
    stops = sorted((sign*(-a-q)) % math.tau for a in profile.angles)
    moves = []
    current = 0.
    while abs(delta)-current>1e-8:
        reachable = [x for x in stops if current+1e-8 < x <= min(abs(delta),current+math.radians(46))+1e-8]
        if not reachable:
            raise ValueError("No detent within wrist turn range")
        nxt = reachable[-1]
        moves.append(sign*(nxt-current))
        current = nxt
    return moves
