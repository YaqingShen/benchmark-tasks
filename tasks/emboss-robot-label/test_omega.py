"""Synthetic fixture tests for calculation only."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from omega import OfficeMateScenario, Omega, PrototypeLoad


class TestOmega(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        marks = ['A','B','O','R','T',' ','CUT']
        self.meta = {'model':'S0717930', 'wheel_ccw_from_A':marks,
                     'slot_degrees_ccw':[0,51,102,154,205,257,309],
                     'wheel_diameter_mm':76,'feed_pitch_mm':4,'glyph_height_mm':3.7,
                     'end_stop_deg':15,'released_deg':1,'completed_deg':dict(emboss=12,space=11,cut=14)}
        for key in ('gauge_record','dial_evidence','strip_evidence'):
            self.meta[key]=key+'.txt'
            (self.root/self.meta[key]).write_text('Unit test fixture')
        self.save_meta()
        with (self.root/'omega_force.csv').open('w') as f:
            w=csv.writer(f)
            w.writerow('mode,phase,trial,angle_deg,force_N,perpendicular_arm_mm'.split(','))
            for mode,peak in [('emboss',15),('space',6),('cut',30)]:
                for stage,scale in [('press',1),('return',.3)]:
                    for trial in range(3):
                        for a in [0,5,10,15]:
                            w.writerow((mode,stage,trial,a,peak*scale*a/15,20))
    def save_meta(self):
        (self.root/'omega_measurements.json').write_text(json.dumps(self.meta))
    def test_officemate_scenario(self):
        profile = OfficeMateScenario()
        self.assertAlmostEqual(profile.feed, .0059)
        self.assertAlmostEqual(profile.tape_width, .0127)
        for symbol, force in (("R", 8), ("CUT", 9)):
            self.assertAlmostEqual(profile.torque(symbol, "press", profile.end) / .042, force)
        self.assertEqual(profile.torque(" ", "press", .2), PrototypeLoad().torque(" ", "press", .2))

    def test_physics(self):
        p=Omega(self.root)
        for s,expected in [('R',.30),(' ',.12),('CUT',.60)]:
            self.assertAlmostEqual(p.torque(s,'press',p.end),expected)
            self.assertAlmostEqual(p.torque(s,'return',p.end/2),expected*.3/2)
    def test_catalog_height_and_measured_override(self):
        self.meta.pop('glyph_height_mm')
        self.save_meta()
        self.assertAlmostEqual(Omega(self.root).glyph_height, .0037)
        self.meta['glyph_height_mm'] = 3.62
        self.save_meta()
        self.assertAlmostEqual(Omega(self.root).glyph_height, .00362)

    def test_wrap(self):
        p=Omega(self.root)
        q=math.tau-p.angles[-1]+.001
        sym,err,target=p.nearest(q)
        self.assertEqual(sym,'CUT')
        self.assertAlmostEqual(err,.001)
        self.assertAlmostEqual(target,q-.001)
    def test_missing_evidence(self):
        (self.root/self.meta['gauge_record']).unlink()
        with self.assertRaises(ValueError): Omega(self.root)
    def test_incomplete_force(self):
        p=self.root/'omega_force.csv'
        p.write_text('\n'.join(line for line in p.read_text().splitlines() if not line.startswith('cut,return,2,')))
        with self.assertRaises(ValueError): Omega(self.root)
    def test_reject_duplicate_slot(self):
        self.meta['slot_degrees_ccw'][2]=51
        self.save_meta()
        with self.assertRaises(ValueError): Omega(self.root)
