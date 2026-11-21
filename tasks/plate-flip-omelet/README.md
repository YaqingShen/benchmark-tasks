# Turn a thick omelet with a dinner plate

Use the plate to turn the whole omelet over. Slide it back into the skillet, browned side up, and set the skillet down.

The left hand lifts the skillet by its handle. The right hand grips a dinner plate by the rim, covers the skillet and turns both together. The plate supports the omelet while the empty skillet returns to the hob. The right hand then tilts the plate to slide the omelet back.

## Setup

| Object | Size and material |
| --- | --- |
| Skillet | 175 mm outside diameter, 31 mm rim height, aluminium with a silicone handle grip. Modeled mass 235 g. |
| Dinner plate | 244 mm diameter, lightweight rigid plastic, approximately 8 mm thick including the rim and foot. Modeled mass 160 g. |
| Omelet | A set Spanish omelet, modeled as a 130 mm diameter, 21 mm thick rigid disk weighing 120 g. One browned face makes turnover visible. A two-sided washable food replica of these dimensions can be used for repeatable cold trials. |
| Hob | Fixed, unheated circular support, 156 mm diameter and 50 mm high. |
| Cooling rack | Fixed wire rack, approximately 120 mm square and 66 mm high. The empty plate overhangs the wires. |

Coordinates are in metres in the calibrated YAM station frame. Center the hob and skillet at (0.280, −0.005) and the rack and plate at (0.460, −0.135). The skillet handle points toward the left arm. Start with both hands empty, the omelet centered in the skillet with its browned face down, and the plate face down on the rack.

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
| 1.00 | The whole omelet settles in the skillet, browned face within 15° of face up. The skillet rests on the hob, the plate is clear, and both hands release the utensils. |

## Run

Use Isaac Lab-Arena, YAM Arena and Inspect Robots. The runner loads the station environment from `~/IsaacLab-Arena/yam.env`. Keep `task.py`, `labkit.py` and `arena_assets_fix.py` together.

```sh
bash run.sh my_scripted_trial video 1200
bash run.sh my_dream_trial dream 1200
```

The limit is 1,200 actions at 20 Hz. Each run writes its own subdirectory under `runs/`. The three cameras are the station overhead and both wrists. The optional command `python verify_score.py` checks scoring counterexamples using constructed poses and contacts.

## Recorded trials

Both policies used seed 0, a fixed layout and a limit of 1,200 actions at 20 Hz. The continuous scripted recording scored 1.0 in 1,137 actions (56.85 s simulated, about 7 min 26 s elapsed). A separate reset also scored 1.0. Dream-1 scored 0.0 at the action limit (60 s simulated, about 3 min 46 s elapsed). The task board hosts the setup image and both three-camera recordings.

In the full-score recording, the plate alone supported the omelet by action 564. The omelet slid back by action 920, and both hands cleared the utensils by action 1,117. The largest finger contact normals sampled at 20 Hz were 40.39 and 40.29 N on the skillet and 22.19 and 26.49 N on the plate. Hardware feasibility depends on handle friction and available grip force.
