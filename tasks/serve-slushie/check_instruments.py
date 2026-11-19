"""Exercise the instrument model and its USD displays on a headless workstation."""
import math
import numpy as np
from pxr import Usd
from cup_geometry import FILL_POSITION
from fill_model import Dispenser
from instruments import Instruments
from instrument_visual import InstrumentVisual


def main():
    model = Dispenser()
    sensors = Instruments()
    stage = Usd.Stage.CreateInMemory()
    display = InstrumentVisual(stage)
    pose = np.eye(4)
    pose[:3, 3] = FILL_POSITION
    empty = sensors.advance(0., model, True, True, True)
    assert empty['pickup_scale_g'] == 12.0 and empty['probe_c'] == -1.6
    display.draw(empty)
    for _ in range(100):
        model.advance(pose, 1., .05)
    for _ in range(24):
        model.advance(pose, 0., .05)
    assert sensors.advance(.05, model, False, True, True)['pickup_scale_g'] == 0.
    assert sensors.advance(.05, model, True, True, False)['pickup_scale_g'] is None
    loaded = sensors.advance(.05, model, True, True, True)
    assert loaded['pickup_scale_g'] == 327. and loaded['hopper_net_g'] == 630.
    display.draw(loaded)
    pose[0, 3] -= .16
    for i in range(30):
        model.advance(pose, 1. if i < 10 else 0., .05)
    pose[:3, :3] = np.diag([1., -1., -1.])
    model.advance(pose, 0., .05)
    result = sensors.advance(.05, model, False, False, True)
    assert result['tray_retained_g'] == 31.5 and result['other_spill_g'] == 315.
    assert math.isclose(945. - result['hopper_net_g'], 346.5)
    for _ in range(2400):
        reading = sensors.advance(.05, model, False, True, True)
    assert -1.6 < reading['probe_c'] < -1.3
    assert sensors.probe_c < sensors.product_c
    model.emitted = 900.
    assert sensors.advance(.05, model, False, True, True)['probe_c'] is None
    assert Instruments().advance(0., Dispenser(), True, True, True) == empty
    print('Empty, off-platform, hand contact, loaded, two spill locations, thermal response, empty hopper and reset checks passed.')


if __name__ == '__main__':
    main()
