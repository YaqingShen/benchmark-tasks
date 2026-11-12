# Timed wafer immersion with a manual countdown

The task tests passive-tool handling and timed immersion while the other hand operates a physical timer. The pattern is an elapsed-time surrogate, not a prediction of photoresist chemistry. Category: **Other — cleanroom-inspired manipulation**.

The goal and constraints in `task.json` are the instruction given to Dream-1. The scripted strategy uses the left hand for the pliers; its intermediate waypoints and phase names are not scoring requirements. The timer responds to right-finger presses. The pliers may be held by either hand. This revision passed native validation on Brev, including all 21 independent checks. Dream-1 completed the full evaluation with a score of 0.0.

## Run

Use the prepared [RoboCurve Isaac/YAM environment](https://tasks.robocurve.org/walkthrough), with Isaac Sim 6.1, Isaac Lab-Arena, yam-arena, and its environment Python. The tested policy interface targets inspect-robots 1.0.0, inspect-robots-yam 0.38.0 and inspect-robots-dream 0.1.0. This is an Isaac task; plain Python or MuJoCo is not a substitute. CPU PhysX advances at 2,000 Hz; control is 20 Hz. An NVIDIA GPU renders the observations.

From this folder, install only the task plugin into a local directory:

```sh
uv pip install --python "$(command -v python)" --no-deps --no-build-isolation \
  --target .policy-plugin ./runtime/benchmark
```

Run the scripted demonstration and independent native audit:

```sh
python run_develop.py --bundle runtime --output runs/demo001 \
  --lock /tmp/robocurve-shared-gpu.lock
```

Export the verified poses as the submitted review video. Rendering does not advance physics:

```sh
flock /tmp/robocurve-shared-gpu.lock python render_develop_review.py \
  --bundle runtime --run runs/demo001 --output runs/demo001/render
```

The review export is 960 × 360 at 20 fps, with overview, wafer detail and physical-timer views. It uses the same saved poses and assets as the native trial. The original higher-resolution renderer remains available as `render_develop.py`.

Run Dream-1 with the exact `task.json` instruction:

```sh
python tools/run_policy.py --policy dream --output runs/dream001 \
  --lock /tmp/robocurve-shared-gpu.lock
```

Use the same lock as other users of the machine. Both launchers wait for the shared lock and an idle GPU. Each run needs a new output directory. Dream credentials are supplied through the prepared environment; never commit them. A short `--policy hold --steps 2` run checks transport without claiming a model evaluation.

Inspect Robots saves its log and camera frames under `runs/dream001/inspect`; the native evidence directory is printed and is under `runs/dream001/evidence/`. Export its video with `inspect-robots video PATH_TO_INSPECT_LOG.json`.

This revision's successful 5090 scripted run lasted **119.65 simulation seconds**. Wall time is hardware dependent. On Brev with an L40S, the native simulation, full audit and review export took approximately 90 minutes in total; the completed Dream-1 rollout took 36.3 minutes. GPU queues and model-service outages add waiting time. The action budget is **2,400 actions total at 20 Hz**, including startup holds. The scene is fixed, with seed 0 and no randomized resets. A model trial may stop early on an invalid or completed task.

## Score saved evidence

```sh
python tools/score.py --run runs/demo001 \
  --trace runs/demo001/video_samples.jsonl.gz --output runs/demo001/submission_score.json
python tools/score.py --run PATH_TO_POLICY_EVIDENCE \
  --output PATH_TO_POLICY_EVIDENCE/submission_score.json
python tools/test_score.py
```

The primary **practical rubric v1** uses measured geometry, button travel and time at 20 Hz. The offline scorer also recomputes robot collision geometry from the recorded solids and link poses. It does not use command phases, force magnitudes, hidden success flags or a claim that the model read a number. A physical test can use calibrated video, a ruler and the visible countdown. The 50 ms observation interval gives the reported immersion-time resolution.

| Credit | Observable milestone |
|---|---|
| 0.15 | Pliers lifted at least 30 mm, with the wafer clamped at its flat and lifted at least 10 mm for 0.2 s. |
| 0.15 | Whole wafer immersed vertically, within 5° of vertical, with 2 mm clearance from the beaker walls and floor. |
| 0.15 | Right fingertip operates the timer: Start within 1 s after full immersion; leave full immersion 0–2 s after the 30 s countdown ends; Stop within 1 s after leaving full immersion. |
| 0.25 | One completed, uninterrupted full-immersion interval of 28–32 s inclusive. |
| 0.15 | Whole wafer withdrawn at least 2 mm above the liquid and held by the pliers for 1 s. |
| 0.15 | Wafer released on all three pins within 1 mm and 2° of its initial pose. Jaw pads have at least 1 mm separation or no overlap with the wafer-edge strip. Its pose stays within a 1 mm/2° envelope for 1 s. |

Milestones require the preceding physical prerequisites: immersion must follow a valid pickup, withdrawal must follow immersion, and return must follow withdrawal. A wrong dwell can still earn the completed handling milestones. Drops, lost tool grip, modified/moved fixtures, robot collision, wet hands/handles, repeated immersion, an early timer start, invalid evidence or exceeding the time budget invalidate the trial and give zero. Tool retention is measured at the handle paddles relative to the carrying hand, within 2 mm and 3°. After the wafer is safely returned, the pliers may remain held or be placed back within 2 mm/3° of their initial stand pose and remain there for 1 s.

A button event requires at least 1 mm mechanical depression for 50 ms, followed by release before another press. The actual right fingertip geometry must be at the cap; the rendered countdown must agree with the mechanical event to within one observation interval. The script reads simulator timer state; Dream-1 receives only RGB camera views and the 14 robot joint/gripper values. Neither is credited with demonstrated OCR.

The frozen native controller's stricter force, passive-mechanism and contact audits remain separate engineering diagnostics. Applying the practical rubric to an archived trajectory creates a new, versioned score; it does not alter the earlier native report. No airflow, cleanliness, evaporation or chemical-development efficacy is scored.

## Fixed setup

All distances below are millimetres. The initial tabletop plane is world z = 747.5 mm. Coordinates are world (x, y, z); orientations and collision solids are supplied in `runtime/assets`, `cloud_plan.json` and `full_workflow_plan.json`.

| Object | Size/material and initial arrangement |
|---|---|
| Wafer | Silicon surrogate, 100 diameter × 1 thick, 20 g, 32.5 flat; center approximately (420, −140, 813.2). The exposed-resist appearance is a visual layer. |
| Three pins | Metal, diameter 1 × height 20; equilateral triangle side 55.43. Top centers: (420, −172, 812.5), (447.7128, −124, 812.5), (392.2872, −124, 812.5). |
| Heater/pedestal | Inactive metal cylinders, diameter 150; heights 20 and 25. Heater center (420, −140, 782.5); pedestal center (420, −140, 760). |
| Pliers | Passive mechanical tool, approximately 190 long and 47 wide; plastic/composite handles, metal levers, rubber-like contact pads 12 × 32.5 × 1, total mass 150 g. Lower-body root starts near (530, 34.286, 813). The robot squeezes the handles; no tool actuator or artificial wafer attachment is used. |
| Pliers supports | Metal cylinder diameter 14 × 60.5 at (530, 34.286, 777.75); metal block 16 × 4 × 38.343 at (522, 159.286, 766.672). |
| Beaker | Glass-like stand-in, inner diameter 170, outer diameter 180, height 160, floor 6; base (240, −120, 747.5). Virtual bath surface is 130 above the base. |
| Timer | Plastic base 100 × 70 × 30, mass 250 g; screen 65 × 50 tilted 60° toward the audience. Base (240, −400, 747.5), yaw −115°. Button diameter 20, travel 3, passive return spring; starts at 30.0 s. |

Both YAM parallel grippers start empty. Robot joint poses, pliers spring parameters and detailed registration surfaces are in the supplied source bundle. Contact coefficients and compliance are assigned simulation parameters, not measured material properties. Reset restores the scene, the unprocessed wafer appearance and the 30 s timer; no liquid is consumed.

## Fidelity and evidence

The Brev/L40S scripted run scored **1.0** and completed in **119.65 simulation seconds**. Independent practical-rubric scoring confirms all six milestones and **31.6 s** of full immersion at the 20 Hz observation resolution. The detailed native audit passed all **21 checks**. The earlier 5090 run independently reached 1.0 on the same scene and scoring code.

**Dream-1 (`dreamlabs/dream-1-beta`) scored 0.0.** It completed 2,400 control actions and 120 simulation seconds without an execution error. The independent scorer agrees; the model completed none of the scored milestones. The uploaded model video is this completed evaluation. Earlier interrupted or service-capacity-limited attempts are not presented as final scores.

The task instruction, simulation assets, controller and scoring code are unchanged from the evaluated revision. Only submission metadata, documentation and the separate review renderer were added or updated. Machine-readable results and evaluated-source hashes are in `evidence/`.

The lines-and-cross pattern is GOOD only for the specified immersion duration. Partial wetting, developer chemistry, diffusion, resist kinetics, fluid drag, heating, damage and chemical compatibility are not calibrated. For a physical handling demonstration, use an inert bath and a dummy wafer; this repository is not a chemical-processing recipe. Actual photolithography requires separate process and material validation.

The media ID in `task.json` identifies the completed Dream-1 recording, full-score scripted review video and fixed-setup image. A failed model run is reported honestly; a scripted full-score demonstration establishes feasibility in this simulation, not real-hardware reliability.
