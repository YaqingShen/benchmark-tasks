# Turn a thick omelet with a dinner plate

Use the plate to turn the whole omelet over. Slide it back into the skillet, browned side up, and set the skillet down.

The left hand lifts the skillet by its handle. The right hand grips a dinner plate by the rim, covers the skillet and turns both together. The plate supports the omelet while the empty skillet returns to the hob. The right hand then tilts the plate to slide the omelet back.

## Setup

| Object | Size and material |
| --- | --- |
| Skillet | 175 mm outside diameter, 31 mm rim height, aluminium with a silicone handle grip. Modeled mass 235 g. |
| Dinner plate | 244 mm diameter, smooth rigid polypropylene, approximately 8 mm thick including the rim and foot. Modeled mass 160 g. |
| Omelet replica | One-piece American basswood disk, 130 ± 1 mm diameter, 21.0 ± 0.5 mm thick and 120 ± 3 g after drying. Both flat faces have a bare P320 finish. A shallow recessed mark identifies the browned face. |
| Hob | Fixed, unheated circular support, 156 mm diameter and 50 mm high. |
| Cooling rack | Fixed wire rack, approximately 120 mm square and 66 mm high. The empty plate overhangs the wires. |

Coordinates are in metres in the calibrated YAM station frame. Center the hob and skillet at (0.280, −0.005) and the rack and plate at (0.460, −0.135). The skillet handle points toward the left arm. Start with both hands empty, the omelet centered in the skillet with its browned face down, and the plate face down on the rack.

Prepare the same plain dry sliding faces for all trials. The [specimen preparation and qualification procedure](bench-protocol.md) gives a material source, machining dimensions, cleaning and drying conditions, grain direction, and acceptance windows for each utensil. The [calibration configuration](bench-config.json) and two blank bench records support matching the apparatus across labs.

The skillet, plate and food each have an independent rigid body with gravity, friction and collision. Finger forces drive the grasps. The browned face is attached to the food from the first frame. The scripted controller uses simulated pose measurements after inversion to locate the plate center relative to the hand. A camera policy receives the station observations. The scene models handling a set omelet. Cooking, bending, tearing and food adhesion are outside its scope.

## Score

Scores are awarded in order. Direct hand contact with the food, letting it hit the table or a fixture, or a prolonged unsupported fall after inversion gives zero.

| Score | What the trial shows |
| --- | --- |
| 0.15 | The plate covers and contacts the rim while the upright pan supports the omelet. |
| 0.35 | The lifted, covered pan and plate pass through a steep tilt together with the omelet between them. |
| 0.60 | Both utensils have turned more than 150° and the inverted omelet rests within the plate rim. |
| 0.75 | The skillet separates from the plate. The plate alone supports the whole omelet for at least 0.2 s. |
| 0.85 | The skillet is upright on the hob. The omelet slides at least 20 mm downhill relative to the tipped plate near the skillet. |
| 1.00 | For the final continuous 1 s the omelet rests in the upright skillet, browned side within 15° of up. The skillet rests on the hob, the plate is clear and both hands are released. |

A change in any final condition immediately removes the last 0.15. Returning the objects to their finished state starts a fresh 20-frame hold. Earlier completed stages are retained unless the trial incurs a zero-score failure. The scorer checks measured hob contact, pan centre within 22 mm of the hob centre, pan tilt below 6°, rest height within 6 mm, food supported by the pan, both pan and food moving below 15 mm/s, and plate centre more than 190 mm away in plan. Fingertips must clear the handle reference by 70 mm and the finite plate volume by 40 mm. Each hand-to-utensil contact normal must be below 0.025 N in the release window.

## Run and verify

The validation machine used an RTX 5090, Python 3.12.16, Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, Inspect Robots 1.0.0 and yam-arena 0.1.0 (checkout 0059d02a1a91bc6834c1d0846022891df0c78730). Media export used Pillow 12.3.0 and imageio-ffmpeg 0.6.0. Use the configured YAM Isaac Lab-Arena environment. The runner loads `~/IsaacLab-Arena/yam.env` or the environment named by `YAM_ENV`. The fixed starting layout uses seed 0 without object jitter. Each name below must be a fresh output directory. The action limit is 1,200 at 20 Hz (60 simulated seconds). The torque/force-limited finger drives and the rigid-body contact solver advance at 240 Hz. Run the profile check in the activated station environment to verify the installed values.

```sh
source ~/IsaacLab-Arena/yam.env
python validate_sim_profile.py
python verify_score.py
bash run.sh scripted01 video 1200
bash run.sh dream01 dream 1200
python audit_episode.py runs/scripted01
python audit_episode.py runs/dream01
python export.py runs/scripted01/adhoc_????????.json demo.mp4 setup.png
python export.py runs/dream01/adhoc_????????.json dream1.mp4
```

The score trace and input hashes are written beside the official run output. `audit_episode.py` checks the last state against the official result and confirms 12 measured physics substeps per control interval after the initial observation. Videos show only the original overhead and two wrist cameras.

## Evidence addressing the review

- **Replica slide.** Two dry faces across two utensil materials were released from rest at 11 inclinations each in a separate Isaac test. At 21° both faces stayed within 2 mm on the plate for 1 s, and at 22° both slid. On the skillet the corresponding angles were 22° and 23°. At 35°, the 15 mm travel times were about 0.110 s and 0.113 s. Use `bash run_slide_test.sh coupons01` followed by `python slide_summary.py runs/coupons01/slide.json`. The physical qualification bands allow for gauge and video resolution. [Simulation slide results](evidence/slide-summary.json)
- **End-of-run score.** A short Isaac regression seeds the earlier milestones and uses real contacts to reach full credit. It then moves the skillet and disk together to a separate support at the same height. The next step gives 0.85, and the official run ends at 0.85. Repeat with `bash run_off_hob_test.sh moved01`. The unchanged original scorer gave 1.0 after an equivalent off-hob counterexample. [End-state result](evidence/off-hob-summary.json)
- **Real-gripper reference.** The YAM installation used its normal 50 N finger-drive limit. The scripted controller changes the commanded closing position to reduce the handle load. A verifier reads both individual pad contact normals on every 1/240 s physics step, including acquisition, inversion, return and release. The setup tolerances and a separate physical test procedure are in the [bench protocol](bench-protocol.md). The manufacturer's motor stall setting, simulated pad contacts and physical pad readings have distinct meanings.

## RTX 5090 results

Scripted demo 1.0 in 1,135 actions. A separate run of the same fixed setup scored 1.0 in 1,135 actions. Dream-1 scored 0.0 in 1,200 actions. The recorded demo first earned full credit on action 1,116. The pan reached 179.8°. The independent repeat used a fresh process, seed 0 and the same starting layout.

| Run | Wall time | Skillet handle pad peaks, left / right | Plate rim pad peaks, left / right |
| --- | --- | --- | --- |
| Scripted demo | 6.7 min | 33.89 / 33.90 N | 23.77 / 27.82 N |
| Scripted repeat | 4.3 min | 33.89 / 33.90 N | 23.77 / 27.82 N |
| Dream-1 | 7.0 min | 0.00 / 0.00 N | 0.00 / 7.38 N |

Each peak is the largest resultant contact normal on one of the two pads of the hand assigned to that utensil, across all recorded physics substeps. The row for Dream-1 describes its own attempt. [Run results and code hash](evidence/episode-results.json)
