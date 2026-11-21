# Bake and serve a waffle with a rotating maker and kitchen tongs

The robot closes a filled waffle maker by its external handle, turns it over to lock it, waits through a five-minute baking interval, returns it upright, opens it, and serves the waffle with kitchen tongs. The plate and tool holder are loose objects on the worktop. The final state requires the food on the plate, the tongs in their holder, and both hands withdrawn.

Category: **Societal relatability**. The manipulation combines control of a rotating appliance with extraction from a grid and transport using a separately grasped passive tool.

## Run a trial

Use an Isaac Sim / Isaac Lab environment with `inspect-robots` and the `yam_arena` embodiment installed. The PBD fluid needs an NVIDIA GPU. The embodiment uses its CPU data backend with GPU PhysX dynamics so particle positions can be read back. The task does not modify shared installed packages. Dream-1 credentials belong in the operator's environment and must not be added to this folder.

```bash
python run_task.py --policy yam_scripted --output runs/demo
python run_task.py --policy dream --output runs/dream
```

Both policies receive the exact `instruction` in `task.json`. Both start from a normal reset, use seed 0, zero object jitter, 20 Hz control, and the same 8500-step limit. The maximum simulated trial duration is 425 seconds, including 300 seconds of continuous closed-and-locked baking. Wall time depends on the workstation and inference service. Default cameras are 1280×720; `--preview` selects 320×180 without changing the physics or timer. No checkpoint or preview animation is part of the final evaluation.

The scripted controller uses simulated object poses to correct robot joint targets. Dream-1 is evaluated separately; scripted feedback is not a model-policy result. Simulated finger actuator effort is capped at 40 N per finger and checked at reset. This is a configuration limit, not a force measurement.

## Parameter configuration

`hardware_config.json` is the manually prepared configuration used by the scene, rotation lock, scorer and bake timer through `appliance_config.py`. It separates manufacturer-confirmed product information, nominal simulation inputs and unfilled hardware measurements. No physical readings are fabricated. Values in metres, degrees, radians, seconds and kilograms have explicit unit suffixes. The same file is copied into every frozen run source directory.

## Scene and reset

Coordinates are metres in the robot-base frame: x forward, y left, z upward. The worktop is at z 0.747. Values below are root positions; some objects include a local geometry offset.

| Object | Dimensions and material approximation | Mass | Starting position |
|---|---|---|---|
| Rotating maker | Base 405×266 mm; circular grid 180 mm diameter; steel housing, polymer handles, rubber feet | 2.850 kg including lid | (0.400, 0.000, 0.7475) |
| Batter / molded waffle | Nominal 250 mL initial fill; molded waffle 174 mm diameter, approximately 27 mm thick | 262.5 g assumed and retained | In lower grid |
| Dinner plate | Ceramic; 250 mm diameter, 194 mm flat well | 550 g | (0.430, 0.290, 0.7475) |
| Tongs |164 mm overall; steel spring arms, nonmetallic 24×24 mm tips; 38 mm free tip gap | 60 g | (0.300, 0.430, 0.769), yaw 90° |
| Tool holder | Open weighted rectangular cup, approximately 93×40×56 mm |450 g | (0.300, 0.430, 0.747), yaw 90° |

Start with the rotating body upright, lid 45° open, rotation lock released, plate empty, and tongs open in their holder. Preheating and measuring the initial batter are setup operations. Reset all joint positions, particle positions, scoring history and baking timers between trials. A clean fresh batch replaces consumed material; no previous waffle is left in the appliance.

## Material preparation and equipment references

For a repeatable batter preparation, the [Krusteaz Belgian Waffle recipe](https://www.krusteaz.com/recipes/belgian-waffle/) specifies 1½cups mix,⅔cup water, one egg and 3 tablespoons oil, whisked smooth and rested for 2 minutes. Use the same product and package instructions for every batch; record the batch and measured portion. The simulator's 250 mL and 1050 kg/m³ density are nominal modeling inputs, not measured properties of that recipe. The recipe recommends about⅔cup per 7-inch waffle, so the modeled 250 mL fill must not be assumed appropriate for every real appliance.

The [Hamilton Beach flip-maker manual](https://useandcares.hamiltonbeach.com/files/840204801.pdf) provides a reference for preheating, approximately 250 mL fill and a 5–8 minute bake. It calls for nonmetallic removal tools. Its green light indicates preheat readiness; the task does not claim that it senses food doneness. No unsupported numerical plate temperature is specified.

The hardware reference is the [Hamilton Beach 26031G](https://hamiltonbeach.com/durathon-removable-grid-belgian-waffle-maker-26031g), listed by its manufacturer at US$75.99 on 21 November 2026. The task follows its rotation-to-lock, return-to-open sequence. The former separately actuated spring slider is removed from the model and the robot plan. See [HARDWARE.md](HARDWARE.md) for the exact source mapping, materials and declared joint-limit approximation. The model is not manufacturer CAD or a measured digital twin.

For shared equipment, use fitted, temperature-rated protective finger covers and nonmetallic tong tips; use the appliance's drip tray and a removable worktop catch mat. Keep batter away from joints, wrist cameras and electrical parts. Verify the covered grip on the cold, unloaded appliance first. Stop a physical trial for a loose cover, slipping tool, overflow toward equipment or unexpected resistance. Unplug and cool before cleaning. These are preparation requirements, not claims that a hardware trial has been performed.

## Scoring

The six additive components are 0.20 for the first half-turn, 0.20 for returning home, 0.20 for maintaining the closed/locked condition, 0.10 for keeping the stand in place **only after the first turn reaches 173°**, 0.10 for reopening, and 0.20 for tong extraction, carry, plate support and tool stow. Numeric thresholds are in `task.json` and `task.py`. The stability function is in `stand_scoring.py`. It returns zero before the first-turn milestone, full normalized credit at displacement ≤10 mm, linearly reduced credit between 10 and 30 mm, and zero at ≥30 mm. Displacement means the maximum distance of the base from its reset mark during the entire trial; moving it back does not restore lost credit. The weighted contribution is 0.10 times that value. An untouched setup, closing only, or turning to 172.999° and returning earns total zero. Any disqualifying contact or unsafe turn overrides every component and returns total zero.

An open-lid turn, robot-hand contact with the food or either cooking interior, premature removal of the solid food, or placement without tong-carry history revokes all credit. Legitimate incomplete attempts retain applicable stage credit. The upright opening zone is 7° throughout the trial; this same zone is used by the lid guard. A subsequent turn outside that zone still requires a closed, locked lid.

For a physical trial, use the video and marked start positions to check order, locking, both turns, tool use, spills and final release. A ruler checks stand displacement and plate placement; a still frame checks flatness and full support. The food must lie inside the plate, not remain suspended by the tool. The tongs must be open and stable in their holder. Simulated proximity checks approximate visible contact and support; they are not real force sensors. All objects must rest with both hands clear for at least 1 second before full score is awarded.

```bash
python verify_tests.py --output runs/test-results.json
```

These tests cover positive and counterexample states. They do not substitute for a continuous native demonstration.

## Fidelity and evidence

The initial batter is native PBD fluid with uncalibrated viscosity and cohesion. A declared material-response rule activates a rigid molded waffle after 300 continuous seconds closed and locked, while the maker is inverted. Contained particles are then hidden; visible escaped particles block the transition. Subsequent food motion comes from normal contact. The model does not predict heat transfer, chemistry, evaporation, internal mass distribution, crust fracture or actual cooking quality.

The passive spring-slide tong joint approximates bending tongs. Food collision geometry simplifies yielding near the lid: some top crust faces are visual-only or inset. The score therefore reports success in the declared simulated geometry.

## Revision validation

The tested software versions and evaluation settings are recorded in `evidence/runtime.json`.

The revised native scripted demonstration reached **1.0** in 8035 control steps (401.75 seconds), including the full 300-second inverted baking interval. It started from a normal reset and used no checkpoint, time compression or stitched footage. Maximum stand displacement was 0.75 cm. The food was supported by the plate, tongs were stowed, and both hands were clear and still at completion.

Dream-1 (`dreamlabs/dream-1-beta`) reached **0.0** in 8500 steps. Both evaluations used the same physical model, scoring rules, instruction, seed, layout and action limit. Only the scripted controller's waypoint plan changed between its earlier freeze and the final demonstration; that function is not used by Dream-1. Runtime-source equivalence is recorded in `evidence/provenance.json`.

All nine regression test files passed. Independent state replay reproduced every recorded score without mismatches. Source hashes and the scripted material-conversion event were verified. `evidence/` contains the native audits, tests, model metadata and hardware sources. These are simulation results; no physical appliance measurements or cooking trial are claimed.

The new continuous Dream-1 video, full-score scripted video and setup screenshot are uploaded as media `M-DMUFKY`, referenced by `task.json`. They replace the earlier slider-based recordings.
