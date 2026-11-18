# Remove a ceramic wall hook with a recessed fastener

A painted wall panel carries a ceramic knob hook held by one centre screw and a nylon washer. One hand supports the hook while the other aligns and withdraws a push-to-start powered screwdriver. The hook, screw and washer must end in the padded tray, and the driver must return to its stand.

`task.json` defines the fixed layout, exact instruction, numerical scoring rules, run limit, and simulation limitations. These are simulation design dimensions, not measurements of a purchased physical specimen. The executable geometry in `task.py` is the source of the dimensions.

## Run

Use the Isaac Lab-Arena environment with `yam-isaac-lab-arena` and Inspect Robots installed. From this task directory:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=352 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --seed 0 --store-frames --log-dir logs/dream
```

For the full scripted demonstration, replace `--policy dream` with `--policy yam_scripted` and use `--log-dir logs/demo`. The layout is fixed: no object jitter. The run seed is 0; the harness derives the scene seed. The limit is 2,400 control steps, or 120 seconds at 20 Hz. Allow approximately 5–10 minutes of wall time per recorded trial on an RTX 5090, including simulator startup; load and cached assets affect this.

Read the `task_score` result and termination reason in the resulting evaluation JSON. Run completion status `success` by itself does not mean that the task received full credit. Use `inspect-robots video logs/demo/adhoc_*.json --out videos` with the actual evaluation filename to export recorded camera views.

## Score

Credit is sequential and evaluated from the current state. The thread must first be disengaged (0.25). The hook must then be at least 20 mm from its mount without triggering the damage proxy (0.5). A released, stationary hook on the tray liner earns 0.75. A released, stationary screw and washer inside the tray, plus the driver resting across both stand cradles, earns 1.0. Screw and washer may stay inside the hook. Full-score conditions must persist for one second for successful termination. Current-state credit may decrease if the final arrangement deteriorates.

The complete numerical definitions are in `task.json`: whole collision bounds must fit the tray footprint; the hook must be within 4 mm of the liner surface; rest requires body-origin and collision-surface speeds below 10 mm/s, including rotational motion; robot gripper and finger bounds must be at least 5 mm clear. Driver scoring checks its axis and position at both cradle centres, not just its origin.

The damage flag is deliberately a conservative operational proxy: downward hook speed above 0.6264 m/s, even while grasped, or a collision lower bound below 4 mm above the table outside the tray footprint permanently caps the score at 0.25. It does not measure free fall, impact force, or actual ceramic fracture. A physical comparison would use a calibrated side video and position measurements; physical feasibility and fracture resistance have not been tested.

## Implementation and limits

- `task.py` contains the scene, coded thread, score and scripted motion. Hook, washer and driver move through native rigid-body contact. There are no grasp attachments. The scripted demo uses the same environment and scorer as Dream-1.
- The screw is constrained to the wall axis while threaded. A correctly aligned driver tip and backward motion release it after 10 mm, capped at 2.8 mm/s. After release it is a free body. Rotation, torque and thread contact are omitted; this tests coordination and alignment, not realistic torque-controlled unscrewing.
- The ceramic does not fracture; the foam is a rigid liner. The hook's collision shape uses overlapping boxes and the washer uses a square ring. These approximations are disclosed in `task.json`.
- `scoring_geometry.py` computes primitive bounds and conservative hand-clearance bounds using native robot link poses and installed URDF collision meshes. It bounds collision-surface speeds using primitive bounding-box vertices, accounting for both translation and rotation. It only reads simulator state. Bounds are a release proxy, not contact-force measurement.
- `labkit.py` builds multi-shape bodies and installs task hooks. `arena_assets_fix.py` redirects library assets to the production asset bucket; this scene uses no downloaded library objects.
- Optional `HOOK_DEBUG=1` emits mechanism progress. `HOOK_AUDIT_PATH=/absolute/path/states.jsonl` records native states and scoring predicates every ten steps for audit, without changing actions or physics. Keep these local paths and logs out of blind-review media.

## Evaluation evidence

Both recorded trials used the same instruction and final task/scorer revision, with seed 0 and no layout jitter:

| Policy | Score | Steps | Termination |
| --- | ---: | ---: | --- |
| Dream-1 (`dreamlabs/dream-1-beta`) | 0.0 | 2,400 | max_steps |
| Scripted demo (`yam_scripted`) | 1.0 | 718 | success |

The demo completes in 35.9 seconds of simulated time and satisfies the one-second success hold. The model recording covers the complete 120-second limit. The setup and camera recordings share the same fixed scene.

Tested with Inspect Robots 1.0.0 and environment `isaaclab-arena-0.3.1-yam-bimanual-station-calibrated-task-task` on an RTX 5090. `validation.json` records the evaluated source hashes and results. Run `python check_scoring.py` in this directory for the ten regression checks covering valid completion, suspended/protruding objects, hand contact, spinning, incorrect driver placement, damage and an engaged thread.
