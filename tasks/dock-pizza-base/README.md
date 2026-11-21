# Dock a thin pizza base

Use the pin roller to dock the thin pizza base evenly. Return the roller to its rest.

The native dual-YAM station has a fixed board, a 152 mm pizza base and a free handheld docker. Its 60 mm roller has 32 pins on a passive hinge. The right arm operates the roller. The left wrist gives a second view. The script sends arm waypoints only. No command sets the roller angle, tool pose or hole locations.

## Contact model

The 4.5 mm sheet is a stationary visual surface above a rigid backing. Measured pin poses crossing 2.2 mm into the surface leave individual persistent imprints. Each pin must exit the surface before it can leave another. Imprints closer than 2.5 mm are merged. The backing supports the tool and friction drives its rotation. This is a contact-depth proxy for dough penetration. Viscoelastic resistance, tearing and baking are outside the prototype.

Coverage is the fraction of sample points within 12 mm of an imprint. Points are spaced 3 mm apart within a 71 mm radius, leaving a 5 mm edge allowance. Full credit requires 96% coverage and a stable return of the released tool. Otherwise the score reports the coverage fraction, capped at 0.99. These tolerances are prototype assumptions. The real HENDI 515037 docker provides the 60 mm width and 32-pin reference; the layout, grip geometry and dough dimensions are approximations.

## Run and score

Use an Isaac Lab Arena installation with the dual-YAM embodiment, Inspect Robots and the Dream-1 policy available. Activate its Python environment or set `YAM_ENV` to its activation script. The exporter uses NumPy, Pillow and FFmpeg with libx264. Run these commands from this task directory.

```bash
bash run.sh demo-0
DOCK_POLICY=dream bash run.sh dream-0
python media.py runs/demo-0
python media.py runs/dream-0
python audit.py runs/demo-0
python audit.py runs/dream-0
python controls.py
```

Both policies use seed 0, the same instruction, zero object jitter and a 570-step limit at 20 Hz (28.5 s simulated time). Give each run a fresh directory name. Inspect Robots writes the task score to the final JSON in the run's `capture` directory. The exporter writes a three-camera MP4 and a summary in that run's directory. The audit reconstructs coverage from the recorded pin entries.

| Recorded run | Holes | Coverage | Score |
| --- | ---: | ---: | ---: |
| Dream-1 | 0 | 0% | 0.0 |
| Scripted demonstration | 87 | 100% | 1.0 |
| Scripted elevated first pass, 200 steps | 0 | 0% | 0.0 |

In the separate geometry controls, repeating the same lane three times gives the same coverage as one pass (57.8%). Locking the roller while tracing three lanes gives 3.87%.

[HENDI product sheet](https://hendi.xcdn.nl/large/-/en_8711369515037_productsheet.pdf)

## Reproducing the physical check

Place a fixed overhead camera over the workboard and include a ruler in the dough plane. Record the unmarked dough, and take a second frame after the roller is clear. Scale the photograph with the ruler and mark the centres of visible punctures. Overlay the same 71 mm scoring circle and compute coverage using the same 12 mm neighbourhood as the simulator. Photograph the returned tool and the open, withdrawn gripper separately if they obscure the dough. This is a proposed bench check and has not been run on real dough.

To repeat the negative control, run `DOCK_Z=.018 DOCK_STEPS=200 bash run.sh hover-test` in the simulation environment. Compare the hole count with the rolling trial after the first traverse.
