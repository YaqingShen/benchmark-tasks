# Clean a wafer with a blow gun

Wafer cleaning is a necessary step in nanofabrication because surface particles can cause defects in later processing. One arm holds the wafer with passive pliers while the other sweeps a blow gun across its exposed face. The wafer and gun then return to their supports.

**Validation: Dream-1 scores 0.0; the scripted demo scores 1.0. Both results match independent evidence audits. Dream-1 stopped during tool pickup on unexpected robot-environment contact.**

## Environment and installation

Use the prepared [RoboCurve YAM/Isaac environment](https://tasks.robocurve.org/walkthrough) with an NVIDIA GPU. From this task folder, activate its Isaac Python environment and run:

```bash
python tools/check_bundle.py
uv pip install --python "$(command -v python)" --no-deps --no-build-isolation --target .policy-plugin runtime/benchmark
```

Validated native environment: Python 3.12.16, Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, Isaac Lab Arena 0.3.0, yam-arena 0.1.0, PyTorch 2.11.0+cu128 and NumPy 2.5.4. The policy interface targets inspect-robots 1.0.0, inspect-robots-yam 0.38.0 and inspect-robots-dream 0.1.0. Physics runs on CPU; the GPU renders camera observations. A plain Python installation is insufficient.

## Run Dream-1

Configure Dream-1 access through the official walkthrough. Use a new output folder for each trial:

```bash
python tools/run_policy.py --policy dream --steps 2400 --output runs/dream-001
```

The wrapper reads the exact instruction from `task.json`, records three 320 × 176 RGB cameras, and sends measured 14-dimensional joint states at 20 Hz. It exposes no object poses or scripted phase labels to the policy. The budget is 2,400 control steps (120 simulated seconds); physics runs at 0.5 ms per step. The `development=true` option permits this task's limited qualification scope while retaining source, state, contact, geometry and evidence checks.

The submitted Dream-1 trial used `dreamlabs/dream-1-beta` and the 2,400-step budget. It ended on a measured contact violation after 45 control steps: 32 startup steps and 13 model-action steps. The native trace spans 2.2085 simulated seconds; the 46-frame video lasts 2.3 seconds at 20 fps. The independent audit checks all 4,418 physics samples. The evaluation took 74.6 wall-clock seconds; scoring and video collection finished within 85.4 seconds total. No runtime error was recorded. See `evidence/dream-1-result.json` and `evidence/dream-1-audit.json`.

For a short local interface check, use `--policy hold --steps 20`. It holds the measured starting joint positions and makes no model call; its expected task score is 0.

## Score and record a policy run

Use the timestamped evidence directory and the Inspect JSON log printed by the run:

```bash
python tools/score.py policy runs/dream-001/evidence/TRIAL --output runs/dream-001/independent-score.json
inspect-robots video runs/dream-001/inspect/LOG.json
```

The independent scorer reads every physics sample, verifies the trace hash, and recomputes handling, robot geometry and particle conditions. A valid unsuccessful trial scores 0. Missing or inconsistent evidence also scores 0 and is marked incomplete. Existing audit files are preserved.

## Reproduce a scripted demonstration

```bash
python tools/run_demo.py --output runs/scripted-001
python tools/score.py native runs/scripted-001 --output runs/scripted-001/primary-score.json --native-audit-output runs/scripted-001/independent_check.json
```

This runs the native scripted controller in a fresh scene and independently checks its full physics trace. Scoring then applies the same particle acceptance and binary score used by the policy interface. To replay a verified native run's issued joint targets through Inspect Robots, use the `independent_check.json` saved by the command above:

```bash
python tools/run_policy.py --policy scripted --native-run runs/scripted-001 --output runs/scripted-policy-001
```

This re-executes joint targets through physics; it does not replay object poses. In contrast, the submitted reference video renders measured states from its audited native run. Its source archive and provenance are in `reference/` and `evidence/`.

One verified native run took about 83 minutes plus 21 minutes for independent checking on the available cloud host. Camera and model runs have additional rendering/inference cost. Reserve storage for a large compressed physics trace. Two nominal revision036 native runs completed in 116.35 and 116.45 simulated seconds; this is not a broad repeatability claim.

## Scoring and reset

The primary score is binary: 1 only when all required handling and particle conditions pass; otherwise 0. The five-stage handling progress value is diagnostic and is not the primary score.

| Condition | Requirement |
| --- | --- |
| Ordered pickup | Pliers lift ≥30 mm; then wafer lift ≥10 mm with ≥5 mm pin clearance; then gun lift ≥20 mm. Pickup/lift holds last 1 s. |
| Exposed-face sweep | Every eligible 2 mm grid point gets ≥50 ms within a 5 mm radius footprint, at 50–110 mm nozzle distance and 30–60° incidence from the wafer normal. Exclude the outer 1 mm rim and 15 mm clamp-shadow strip. |
| Particle removal | All twelve receive ≥1e-8 N·s jet impulse and remain clear for ≥1 s before the lifted wafer first recontacts its pins. No particle may detach before air starts. |
| Wafer return | All three pins support it; pose within 1 mm and 2° of its start; pliers retreat ≥20 mm after release. |
| Gun return | Gun rests on its stand within 2 mm and 2° of its start; right hand retreats ≥40 mm; trigger compression <1 mm. Keep the pliers held. |
| Final stability | Holds last 1 s; pose excursion ≤0.5 mm and 0.5°. Fixed fixtures stay within 0.1 mm and 0.05° of their starts. |
| Invalid trials | Drops, premature release, forbidden contacts, altered geometry, missing particles, unintended external forces or incomplete evidence produce 0. |

The simulation checks measured contacts and spring/force limits to reject unsupported or unphysical outcomes. Grip contact threshold is 0.005 N, unintended contact threshold 0.01 N, and intended supports must carry at least 80% of weight. Support loss beyond 0.1 s is a drop. Exact tolerances are in `runtime/benchmark/wafer_task_score.py`; jet and adhesion assumptions are in `runtime/wafer_dust.py`.

Particle layout seed is 0: radius 40 mm, minimum spacing 8 mm and x ≥ −34 mm in wafer coordinates. Both reference runs use this fixed layout. Start a fresh process for each trial; scene reset occurs before scoring, with both hands empty and tools on their stands. No in-episode object reset is allowed.

## Fidelity and assets

The intended materials are silicon for the wafer, plastic for pliers and gun, rubber for the pads, and metal for support pins and stands. The visible 1 mm particles are surrogates with assigned drag and adhesion. Contact parameters are not calibrated to material grades. The simulation does not predict nanoscale cleanliness, chemical cleaning, drying, heating, wafer fracture or hardware reliability. The heater-shaped fixture is inactive; hose forces are omitted.

Original code and generated task assets are MIT licensed. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for the separately installed robot models and software dependencies.
