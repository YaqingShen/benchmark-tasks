"""Geometry in metres. Reference data and calibration limits are in PARAMETERS.md."""
import math
import os

# The same nominal cup ring is used at all three tolerance settings.
VARIANT = os.environ.get("CUP_VARIANT","nominal")
limit = {"small":-1,"nominal":0,"large":1}[VARIANT]
GATE_R = .040
BRIM_D = .080 + limit * .0005
BASE_D = .05355 + limit * .0005
HEIGHT = .09131 + limit * .001
MASS = .00587 * (1 + limit * .03)
R = BRIM_D / 2
FOOT = BASE_D / 2
RIM_TOP = .0015
BOTTOM = RIM_TOP - HEIGHT
PITCH = .0063
WRAP_R = GATE_R + .0035
FILM_GAUGE = .0000381
FILM_DENSITY = 923.
PANEL_MASS = math.tau*WRAP_R*(.062-BOTTOM)*FILM_GAUGE*FILM_DENSITY/12
COUNT = 8
FILL_LINE = .050
MOUNT_DROP=.175
TRAY_TOP = -.070-MOUNT_DROP
GATE = .155-MOUNT_DROP
BEAM_HEIGHT = TRAY_TOP + .045
CYCLE_STEPS = 50
WINDOW_STEPS = 60  # Three seconds at 20 Hz after the dispenser finishes cycling.

def station(box):
    # Single exposed service column with the lid open behind it. The upper ring
    # is open so the wrapped stack can enter vertically.
    case=(.115,.155,.175)
    steel=(.52,.55,.55)
    graphite=(.22,.25,.26)
    mark=(.96,.56,.12)
    receiver=[
        box((.087,0,-.057),(.010,.197,.350),case),
        box((.074,-.104,.039),(.035,.012,.158),case),
        box((0,0,TRAY_TOP-.005-GATE),(.158,.182,.006),graphite),
        box((.067,0,.125),(.006,.106,.098),case),
        box((.077,.094,-.035),(.018,.012,.228),steel)]
    # Raised slats form the landing surface seen by the delivery sensor.
    receiver.extend(box((x*.009,0,TRAY_TOP-.001-GATE),(.007,.11,.002),steel) for x in range(-7,8))
    for a in [(i+.5)*math.pi/4 for i in range(8)]:
        receiver.append(box((.048*math.cos(a),.048*math.sin(a),.002),(.004,.004,.142),steel))
    for a in [i*math.pi/8 for i in range(16)]:
        for z,colour in ((-.006,graphite),(FILL_LINE,mark),(.069,steel)):
            receiver.append(box((.055*math.cos(a),.055*math.sin(a),z),(.003,.022,.003),colour,a))
    for s in (-1,1):
        receiver.append(box((0,s*.069,BEAM_HEIGHT-GATE),(.013,.008,.012),case))
    return receiver

def cam_fraction(step,lower):
    # The leading ledge supports the next rim before the lower ledge retreats.
    # Gradual insertion limits impact on the rolled paper rim.
    start,full,release,home=(12,20,25,33) if lower else (0,10,35,46)
    def ramp(t):
        t=max(0.,min(1.,t))
        return t*t*(3.-2.*t)
    return ramp((step-start)/(full-start)) - ramp((step-release)/(home-release))
