"""Explicit, versioned parameters shared by scene, lock, timer and scorer."""
import json
from pathlib import Path
CONFIG = json.loads(Path(__file__).with_name('hardware_config.json').read_text())
MODEL = CONFIG['simulation']
assert CONFIG['schema_version'] == 1
assert CONFIG['hardware_reference']['handle_attached_to'] == 'upper lid'
assert CONFIG['hardware_reference']['separate_front_slider'] is False
assert MODEL['stand_requires_completed_first_turn'] is True
assert 0 < MODEL['stand_full_credit_m'] < MODEL['stand_zero_credit_m']
assert 0 < MODEL['locked_lid_clearance_deg'] < 3
assert 0 < MODEL['open_body_stop_deg'] < MODEL['lock_release_zone_deg']
assert MODEL['lock_release_zone_deg'] == MODEL['home_tolerance_deg']
assert MODEL['bake_duration_s'] >= CONFIG['hardware_reference']['manual_bake_interval_s'][0]
