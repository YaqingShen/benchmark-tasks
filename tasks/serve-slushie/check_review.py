"""Bench qualification of the simulated recipe, tap metering and spill collectors."""
import json
from pathlib import Path
import sys
import math
import numpy as np
from cup_geometry import SPOUT_EXIT_Z, DRIP_TRAY_TOP, HEIGHT, FILL_POSITION
from fill_model import Dispenser, HOPPER_ML
from instruments import Instruments, DENSITY_G_ML
from slushie_runtime import opening_from_angle, Runtime


def advance(model, pose, angle, seconds):
    for _ in range(round(seconds * 20)):
        model.advance(pose, opening_from_angle(-angle), .05)


def flow_sweep(pose):
    measurements = []
    for inventory, angles in [(900, (15, 20, 32.5, 45, 57.5, 70)), (550, (45, 70))]:
        for angle in angles:
            target = max(0, (angle - 20) * 1.2)
            reads = []
            for _ in range(3):
                m = Dispenser()
                m.emitted = HOPPER_ML-inventory
                advance(m, pose, angle, .5)
                baseline = m.emitted
                duration = 5 if target == 0 else 2
                advance(m, pose, angle, duration)
                reads.append(round((m.emitted-baseline)/duration, 6))
                assert math.isclose(reads[-1], target, abs_tol=.00001)
            measurements.append(dict(initial_ml=inventory, angle_deg=angle, flow_ml_s=reads))
    return measurements


def gravity_fixture_estimate():
    """Gravity rig with an 8 mm outlet and assumed effective coefficient 0.60."""
    rows = []
    ranges = [(15,(0,0)), (20,(0,0)), (32.5,(12,18)), (45,(25,35)), (57.5,(39,51)), (70,(54,66))]
    for inventory in (900,550):
        head = .200 - (900-inventory)*1e-6/(math.pi*.080**2)
        for angle, (low,high) in ranges:
            area = math.pi*.004**2*opening_from_angle(-angle)
            rate = .60*area*math.sqrt(2*9.81*head)*1e6
            assert low <= rate <= high
            rows.append(dict(inventory_ml=inventory, angle_deg=angle,
                             initial_head_mm=round(head*1000,2), predicted_flow_ml_s=round(rate,3)))
    return dict(model='steady orifice, assumed coefficient 0.60 and area linear in lever opening; ice bridging omitted',
                bore_mm=8, hopper_diameter_mm=160, start_head_mm=200, predictions=rows)


def challenge(where):
    pose = np.eye(4)
    pose[:3, 3] = FILL_POSITION
    model = Dispenser()
    advance(model, pose, 70, 2)
    advance(model, pose, 15, 5)
    assert math.isclose(model.inside,120.,abs_tol=1e-8)
    xy = {'tray':(.385,0.),'pan':(.30,.17),'sleeve':(.30,.17),'outside':(.1,.4),'connector':(.49,-.16)}[where]
    pose[:3,3] = [*xy,1.06]
    pose[:3,:3] = np.diag([1.,-1.,-1.])
    if where=='sleeve':
        model.cover_bounds=[(np.array([*xy,.89])-.005,np.array([*xy,.90])+.005)]
    model.advance(pose, 0, .05)
    rec = Instruments().advance(0.,model,False,True,True)
    key = {'tray':'tray_retained_g','pan':'secondary_pan_g','sleeve':'pad_covers_g','outside':'unrecovered_g','connector':'unrecovered_g'}[where]
    assert rec[key]==126.
    assert sum(rec[k] for k in ['tray_retained_g','secondary_pan_g','pad_covers_g','unrecovered_g'])==126.
    assert model.equipment_exposed == (where == 'connector')
    rec['equipment_stop'] = model.equipment_exposed
    return rec


def qualify():
    pose=np.eye(4); pose[:3,3]=FILL_POSITION
    rt=Runtime(); resets=[]
    for _ in range(3):
        rt.new_material();rt.reset_milestones()
        assert not rt.slush.emitted and not rt.slush.spill and not rt.slush.packets
        assert not rt.slush.cover_bounds and not rt.slush.nozzle_ml
        batch=rt.batch
        assert -1.7 <= batch['serving_c'] <= -1.2 and batch['sugar_mass_fraction']==.14
        resets.append(dict(batch=batch, hopper_g=HOPPER_ML*DENSITY_G_ML,
                           sensor_c=rt.instruments.probe_c, spill_ml=rt.slush.spill))
        advance(rt.slush,pose,70,3)
    m=Dispenser()
    missed=pose.copy();missed[0,3]-=.18
    advance(m,missed,70,2)
    before = m.emitted
    advance(m,missed,15,5)
    assert math.isclose(m.emitted-before,.45,abs_tol=1e-7)
    assert math.isclose(m.emitted,m.tray_spill,abs_tol=1e-7)
    rim_gap=(SPOUT_EXIT_Z-DRIP_TRAY_TOP-HEIGHT)*1000
    assert math.isclose(rim_gap,25.)
    return dict(evidence_type='deterministic simulated qualification', batch_resets=resets,
                nominal_cup_rim_gap_mm=round(rim_gap,2), flow=flow_sweep(pose),
                closed_tap_5s=dict(after_drip_ml=round(m.emitted-before,4),
                                   tray_total_ml=round(m.tray_spill,4)),
                gravity_fixture_estimate=gravity_fixture_estimate(),
                controlled_spills={location:challenge(location) for location in ('tray','pan','sleeve','outside','connector')})


if __name__ == '__main__':
    data = qualify()
    Path(sys.argv[1]).write_text(json.dumps(data, indent=2)+'\n')
    print('Passed three preparation/reset cycles, 24 timed flow samples, five-second drip collection and five spill locations including connector exposure.')
