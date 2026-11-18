# Serve a two-thirds cup of slush

A worker serving slush at a concession counter has to stop the flow at the requested portion and hand off an open cup without spilling. The robot places a clear cup on the drip tray, operates the spring-return lever and carries the drink back to the pickup stand. The scored outcome is a released, upright cup holding 60–73% of its 479 mL capacity (about 287–350 mL), with less than 5% of the dispensed volume spilled.

In the scripted trial, the robot served 348.4 mL with zero simulated spill. It released the cup and met all six final conditions for one second. The trial ended successfully at step 1,518 (75.9 s). Dream-1 earned 0.10 for lifting the cup and reached the 2,400-step limit without opening the tap. Both recordings use seed 0 and the same instruction, scene and scorer.

## Reproduce the evaluations

Use the calibrated YAM station for Isaac Lab-Arena 0.3.1 and Inspect Robots 1.0.0, with a configured Dream-1 account for the model run. From this task directory in the configured environment, run the following commands. Use a fresh output directory for each replay.

```bash
python check_fill.py
python run_replay.py --policy replay --max-steps 2400 --output logs/demo
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=352 -E device=cpu \
  --instruction "$instruction" --scorer task_score --max-steps 2400 \
  --seed 0 --store-frames --log-dir logs/dream
```

Both policies use the same scene, seed, fixed starting positions and score. `replay` is a scripted policy. It reads simulated state to plan motions and sends ordinary robot joint and gripper commands. Dream-1 uses the policy observations. The station uses 1 kHz rigid-body physics and 20 control steps per second. The cameras use GPU rendering.

## Dispensing and scoring

`fill_model.py` meters discharge at 20 Hz. Each batch falls vertically from the spout. The model credits it to the cup when its trajectory crosses the opening inside the rim. Batches that miss the cup count as spilled after falling below the workspace. The retained volume is capped at the volume below the lowest rim. The renderer solves for the horizontal plane that encloses the same retained volume in the tapered cavity. The simulation treats the contents as a freely levelling liquid. It omits transient slosh, material reaction forces, crystallization and the weight added to the robot's load.

The scoring geometry uses native cup pose and velocity and the installed robot hand meshes. The serving platform is underneath the required final position. Both robot hands must be clear while the cup remains upright at rest for one second. Review `results.metrics.task_score` and the termination reason in the evaluation JSON. The overhead and two wrist cameras are recorded through Inspect Robots.

On the validation RTX 5090 the scripted trial took 2.8 minutes and the model trial took 4.9 minutes. See `validation.json` for the recorded scores and final state.
