# Serve a two-thirds cup of slush

A worker serving slush at a concession counter has to stop the flow at the requested portion and hand off an open cup without spilling. The robot places a clear cup on the drip tray, operates the spring-return lever and carries the drink back to the pickup stand. The scored outcome is a released, upright cup holding 60–73% of its 479 mL capacity (about 287–350 mL), with less than 5% of the dispensed volume spilled.

The scripted policy served 327.3 mL (68.3% of the cup capacity) with zero simulated spill. It released the upright cup and met all six conditions for one second. The score reached 1.00 at step 1,539 (76.95 s). Dream-1 earned 0.10 for lifting the cup and reached its 2,400-step limit with the tap closed. Both clean recordings use seed 0 and the same instruction, layout and scorer.

## Reproduce a physical trial

The [physical trial protocol](physical-trial.md) specifies one 14% sucrose recipe, a serving range of −1.7 to −1.2 °C, and the reset between trials. It identifies a $162.99 retail dispenser and gives an approximately $95–125 gravity-feed build for labs with a freezer. Before using either rig, check the tap travel, flow at two closed and four flowing angles and the 25 mm clearance above the cup rim. These are qualification targets for a physical build. The scores above and in `validation.json` refer to the simulated task.

For each trial, measure source mass loss and cup mass gain, and weigh the tray and grate, secondary pan, covers and wipes separately. The protocol gives the volume conversion, a mass-balance example, the procedure for a borderline result and how to protect shared equipment from sticky spills.

## Reproduce the evaluations

Use the calibrated YAM station for Isaac Lab-Arena 0.3.1 and Inspect Robots 1.0.0, with a configured Dream-1 account for the model run. From this task directory in the configured environment, run the following commands. Use a fresh output directory for each replay.

```bash
python check_fill.py
SLUSH_TARGET=.66 SLUSH_GRASP_HEIGHT=.060 SLUSH_SQUEEZE=.016 python run_replay.py --policy replay --max-steps 2400 --output logs/demo
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=352 -E device=cpu \
  --instruction "$instruction" --scorer task_score --max-steps 2400 \
  --seed 0 --store-frames --log-dir logs/dream
```

Both policies use the same scene, seed, fixed starting positions and score. `replay` is a scripted policy. It reads simulated state to plan motions and sends ordinary robot joint and gripper commands. Dream-1 uses the policy observations. The station uses 1 kHz rigid-body physics and 20 control steps per second. The cameras use GPU rendering.

## Dispensing and scoring

`fill_model.py` meters discharge at 20 Hz. Each batch falls vertically from the spout. The model credits it to the cup when its trajectory crosses the opening inside the rim. Batches that miss the cup count as spilled when they reach the drip tray. The retained volume is capped at the volume below the lowest rim. The renderer solves for the horizontal plane that encloses the same retained volume in the tapered cavity. The simulation treats the contents as a freely levelling liquid. The cup rigid body receives the empty cup mass plus retained slush mass. Its inertia tensor is held fixed. The stream model omits transient slosh, impact forces and crystal-scale flow.

The scoring geometry uses native cup pose and velocity and the installed robot hand meshes. The serving platform is underneath the required final position. Both robot hands must be clear while the cup remains upright at rest for one second. Review `results.metrics.task_score` and the termination reason in the evaluation JSON. The overhead and two wrist cameras are recorded through Inspect Robots.

The recorded evaluations took 171.9 s for the script and 304.8 s for Dream-1 on an RTX 5090. The run limit and hold time use simulated time. The initial observation contributes one extra frame to each video. The final states and video hashes are recorded in `validation.json`.

## Simulated instruments

The pickup platform acts as the scale pan. Its green display reads grams, including the 12 g cup and its contents at an assumed bulk density of 1.05 g/mL. The unloaded stand reads zero. Dashes indicate hand contact or motion while the cup is on the stand. A probe in the hopper has an amber display in °C. Both instruments update during the ordinary scripted and policy runs and reset with the episode. Each step records the readings under `simulated_instruments`.

The probe starts at the batch temperature calculated by `prepare_batch`. The simplified preparation model starts with the prechilled mix at 3 °C, removes heat at an assumed 75 W and checks the serving band at ten-minute mixing intervals. Its ideal-solution freezing approximation does not resolve crystal size or mixing efficiency. The effective equilibrium is −1.3 °C, with a 240 s time constant and 3 s probe lag. These are assumed parameters. An empty hopper gives an unavailable temperature. Additional mass channels convert remaining inventory, tray catch and other spills to grams. The tray receives missed vertical discharge and overflow whose projected trajectory intersects its rectangular footprint. Pickup readings use the position, motion and hand-clearance checks, without simulating a load-cell contact force. The inventory and spill channels share the volume model and cannot independently validate its accuracy.

Run `python check_instruments.py` in the configured simulation environment to check unloaded and loaded readings, support conditions, spill allocation, probe response and reset.

## Simulated review checks

Run `python check_review.py review-qualification.json` for three preparation/reset cycles, timed flow samples at two inventories, post-closure drips and spill challenges above the tray, secondary pan, sleeve and unprotected table. The homogeneous dispenser uses the same angle-to-flow law at both inventories until depleted. An assumed 0.45 mL remains in the nozzle and drains at 0.90 mL/s after closing. At termination, a copy of the material model advances five seconds with the final cup pose and tap angle held fixed. This collection window is outside the policy's step allowance.

Both policies use conformal, zero-thickness simulated tip covers with assumed static/dynamic friction of 0.95/0.80. A shallow catch pan extends under the transport path. Vertical spill projections estimate material reaching the pan or fingertip bounds. Adhesion, seepage beneath a cover and splashes on nearby electronics require physical evaluation. The original recipe, price estimate and cleaning procedure remain physical replication instructions.

## Equipment reproduction

[equipment-reproduction.md](equipment-reproduction.md) identifies purchasable parts, minimum-order costs, assembly datums, lever transmission and hardware calibration. [physical-trial.md](physical-trial.md) specifies the preparation route for each equipment option and the trial procedure.

## Parameter sensitivity

Five additional scripted trials varied starting inventory, discharge rate and gripper friction. All achieved 1.00 with zero modeled spill. [robustness.md](robustness.md) gives each condition, limitations and reproduction commands.
