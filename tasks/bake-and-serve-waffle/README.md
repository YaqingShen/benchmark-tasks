# Bake and serve a waffle with a rotating maker and kitchen tongs

The robot closes and locks a filled waffle maker, turns it over, waits through a five-minute baking interval, returns it upright, opens it, and serves the waffle with kitchen tongs. The plate and tool holder are loose objects on the worktop. The final state requires the food on the plate, the tongs in their holder, and both hands withdrawn.

Category: **Societal relatability**. The manipulation combines control of a rotating appliance with extraction from a grid and transport using a separately grasped passive tool.

## Run a trial

Use an Isaac Sim / Isaac Lab environment with `inspect-robots` and the `yam_arena` embodiment installed. The PBD fluid needs an NVIDIA GPU. The embodiment uses its CPU data backend with GPU PhysX dynamics so particle positions can be read back. The task does not modify shared installed packages. Dream-1 credentials belong in the operator's environment and must not be added to this folder.

```bash
python run_task.py --policy yam_scripted --output runs/demo
python run_task.py --policy dream --output runs/dream
```

Both policies receive the exact `instruction` in `task.json`. Both start from a normal reset, use seed 0, zero object jitter, 20 Hz control, and the same 8500-step limit. The maximum simulated trial duration is 425 seconds, including 300 seconds of continuous closed-and-locked baking. Wall time depends on the workstation and inference service. Default cameras are 1280×720; `--preview` selects 320×180 without changing the physics or timer. No checkpoint or preview animation is part of the final evaluation.

The scripted controller uses simulated object poses to correct robot joint targets. Dream-1 is evaluated separately; scripted feedback is not a model-policy result. Simulated finger actuator effort is capped at 40 N per finger and checked at reset. This is a configuration limit, not a force measurement.

## Scene and reset

Coordinates are metres in the robot-base frame: x forward, y left, z upward. The worktop is at z 0.747. Values below are root positions; some objects include a local geometry offset.

| Object | Dimensions and material approximation | Mass | Starting position |
|---|---|---|---|
| Rotating maker | Base 330×266 mm; circular grid 180 mm diameter; steel housing, polymer handles, rubber feet | 2.865 kg including lid and slider | (0.400, 0.000, 0.7475) |
| Batter / molded waffle | Nominal 250 mL initial fill; molded waffle 174 mm diameter, approximately 27 mm thick | 262.5 g assumed and retained | In lower grid |
| Dinner plate | Ceramic; 250 mm diameter, 194 mm flat well | 550 g | (0.430, 0.290, 0.7475) |
| Tongs |164 mm overall; steel spring arms, nonmetallic 24×24 mm tips; 38 mm free tip gap | 60 g | (0.300, 0.430, 0.769), yaw 90° |
| Tool holder | Open weighted rectangular cup, approximately 93×40×56 mm |450 g | (0.300, 0.430, 0.747), yaw 90° |

Start with the rotating body upright, lid 45° open, slider forward, plate empty, and tongs open in their holder. Preheating and measuring the initial batter are setup operations. Reset all joint positions, particle positions, scoring history and baking timers between trials. A clean fresh batch replaces consumed material; no previous waffle is left in the appliance.

## Material preparation and equipment references

For a repeatable batter preparation, the [Krusteaz Belgian Waffle recipe](https://www.krusteaz.com/recipes/belgian-waffle/) specifies 1½cups mix,⅔cup water, one egg and 3 tablespoons oil, whisked smooth and rested for 2 minutes. Use the same product and package instructions for every batch; record the batch and measured portion. The simulator's 250 mL and 1050 kg/m³ density are nominal modeling inputs, not measured properties of that recipe. The recipe recommends about⅔cup per 7-inch waffle, so the modeled 250 mL fill must not be assumed appropriate for every real appliance.

The [Hamilton Beach flip-maker manual](https://useandcares.hamiltonbeach.com/files/840204801.pdf) provides a reference for preheating, approximately 250 mL fill and a 5–8 minute bake. It calls for nonmetallic removal tools. Its green light indicates preheat readiness; the task does not claim that it senses food doneness. No unsupported numerical plate temperature is specified.

The scene uses a single rotating chamber with a spring slider. Its geometry, joint limits and mass distribution are defined in `task.py`; the cooking manual above supplies the reference operating sequence. Use the documented model parameters when reproducing the simulated task.

For shared equipment, use fitted, temperature-rated protective finger covers and nonmetallic tong tips; use the appliance's drip tray and a removable worktop catch mat. Keep batter away from joints, wrist cameras and electrical parts. Verify the covered grip on the cold, unloaded appliance first. Stop a physical trial for a loose cover, slipping tool, overflow toward equipment or unexpected resistance. Unplug and cool before cleaning. These are preparation requirements, not claims that a hardware trial has been performed.

## Scoring

The six additive components are 0.20 for the first half-turn, 0.20 for returning home, 0.20 for maintaining the closed/locked condition, 0.10 for keeping the stand in place, 0.10 for reopening, and 0.20 for tong extraction, carry, plate support and tool stow. Numeric thresholds are in `task.json` and `task.py`.

An unlocked/open-lid turn, robot-hand contact with the food or either cooking interior, premature removal of the solid food, or placement without tong-carry history revokes all credit. Legitimate incomplete attempts retain applicable stage credit. The upright opening zone is 7° after the completed return; this same zone is used by the lid guard, avoiding contradictory 5° and 7° conditions. A subsequent turn outside that zone still requires a closed, locked lid.

For a physical trial, use the video and marked start positions to check order, locking, both turns, tool use, spills and final release. A ruler checks stand displacement and plate placement; a still frame checks flatness and full support. The food must lie inside the plate, not remain suspended by the tool. The tongs must be open and stable in their holder. Simulated proximity checks approximate visible contact and support; they are not real force sensors. All objects must rest with both hands clear for at least 1 second before full score is awarded.

```bash
python test_cooking_state.py
python test_home_tolerance.py
python test_tool_serving.py
python test_release.py
python test_scorer.py
```

These tests cover positive and counterexample states. They do not substitute for a continuous native demonstration.

## Fidelity and evidence

The initial batter is native PBD fluid with uncalibrated viscosity and cohesion. A declared material-response rule activates a rigid molded waffle after 300 continuous seconds closed and locked, while the maker is inverted. Contained particles are then hidden; visible escaped particles block the transition. Subsequent food motion comes from normal contact. The model does not predict heat transfer, chemistry, evaporation, internal mass distribution, crust fracture or actual cooking quality.

The passive spring-slide tong joint approximates bending tongs. Food collision geometry simplifies yielding near the lid: some top crust faces are visual-only or inset. The score therefore reports success in the declared simulated geometry.

## Verified native demonstration

The corrected r43 run starts from a fresh reset and completes in 7,945 steps (397.25 s) at 20 Hz, including the full 300 s simulated bake. The scorer and independent replay both report 1.0; source hashes, the action limit and material-transition records pass. Both the food-contact and cooking-interior contact guards remain clear. The final stand displacement is 3.4 mm. Native frames show external-handle opening, tong carry to the side plate, tool stow and final release. This is simulation evidence, not a physical cooking trial.

The demonstration video runs at real simulation time, preserving the full 300 s baking interval. All recordings use the native station cameras without identifying text.

## Dream-1 evaluation

The independent d44 run used `dreamlabs/dream-1-beta`, the exact task instruction, seed 0, the same modeled mechanisms and scorer, and a fresh reset. It ran all 8,500 control steps (425 s) and ended normally at `max_steps`, with no model API error. Its final score was 0.0. The model contacted the upper cooking interior at step 311 and turned the maker with the lid open; these actions permanently revoked credit under the declared rules. It did not complete baking or tong serving. Independent replay matched the recorded score and source hashes passed. The complete three-camera model recording is included separately from the 1.0 scripted demonstration.
