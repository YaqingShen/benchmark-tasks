import copy
import json
import tempfile
import unittest
from pathlib import Path
from bench_trial import check_record
from grip_trial import check as check_grip

class BenchTests(unittest.TestCase):
    def test_physical_templates_are_incomplete(self):
        for file,tool in (("bench-trial.template.json",check_record),("grip-trial.template.json",check_grip)):
            with self.assertRaises(ValueError):
                tool(json.loads(Path(file).read_text()),Path("."))

    def test_dimensions_faces_grain_and_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            d=json.loads(Path("bench-trial.template.json").read_text())
            d.update(trial_id="software-fixture",wood_source="fixture",plate_surface="PP",skillet_surface="aluminum",
                     grain_direction="across slope",mass_g=120,diameter_mm=130,thickness_mm=21)
            for i,r in enumerate(d["sliding"]):
                name=f"fixture{i}.bin";(p/name).write_bytes(b"synthetic evidence file")
                r.update(first_motion_deg=22,release_frame=0,travel_frame=26,fps=240,onset_video=name,travel_video=name)
            self.assertTrue(check_record(d,p)["matched"])
            d["mass_g"]=112
            self.assertFalse(check_record(d,p)["matched"])
            d["mass_g"]=120
            d["sliding"][0]["first_motion_deg"]=30
            self.assertFalse(check_record(d,p)["matched"])
            with self.assertRaises(ValueError):check_record(dict(d,sliding=d["sliding"][:-1]),p)

    def test_grip_peaks_include_uncertainty_and_entire_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);d=json.loads(Path("grip-trial.template.json").read_text())
            d.update(gripper_model="fixture",pad_sensors="fixture",calibration_id="fixture",skillet_handle="fixture",uncertainty_N=.5)
            for j,r in enumerate(d["trials"]):
                name=f"force{j}.csv";(p/name).write_text("time_s,left_pad_N,right_pad_N\n0,0,0\n.004,35,34\n.008,0,0\n")
                (p/"fixture.bin").write_bytes(b"synthetic evidence file")
                r.update(video="fixture.bin",force_csv=name,turn_start_s=0.,turn_end_s=.004,maximum_turn_deg=176.,handle_drift_mm=1.,rested_and_released=True)
            self.assertTrue(check_grip(d,p)["passes"])
            (p/"force0.csv").write_text("time_s,left_pad_N,right_pad_N\n0,0,0\n.004,35,34\n.008,39.8,0\n")
            self.assertFalse(check_grip(d,p)["passes"])
