# Isaac parameter sensitivity

Five scripted trials used the same seed 0, controller, cup, scoring and 2,400-step limit. Each changed the inputs shown below. All finished with score 1.00, zero modeled spill and all six end conditions held for one second. The full-open rate scales the existing linear discharge law. Static and dynamic friction refer to the simulated gripper-cover material.

| Case | Starting inventory (mL) | Full-open flow (mL/s) | Static / dynamic friction | Steps | Served (mL) |
|---|---:|---:|---|---:|---:|
| low_inventory | 550 | 60 | 0.95 / 0.80 | 1539 | 327.325 |
| slow_flow | 900 | 54 | 0.95 / 0.80 | 1551 | 326.972 |
| fast_flow | 900 | 66 | 0.95 / 0.80 | 1529 | 327.179 |
| lower_friction | 900 | 60 | 0.50 / 0.40 | 1539 | 327.365 |
| combined | 550 | 66 | 0.50 / 0.40 | 1529 | 327.191 |

All five traces entered and remained within 5 degrees of the closed mark 0.35 s after the finger-opening command. The tested flow endpoints come from the qualification band. Reduced friction is an assumed sensitivity condition. The mass-balance residual was zero within the shared volume model.

## Reproduce

In the configured Isaac / Inspect Robots environment, run each case with a fresh output folder.

```bash
python run_variant.py low_inventory --output logs/low_inventory
python run_variant.py slow_flow --output logs/slow_flow
python run_variant.py fast_flow --output logs/fast_flow
python run_variant.py lower_friction --output logs/lower_friction
python run_variant.py combined --output logs/combined
```

The runner creates a separate task source copy, changes the inventory, flow and cover friction, and runs the complete controller. It also adjusts the empty-hopper status threshold to the selected inventory. It fixes target fill, grasp height and squeeze to the nominal demo settings. Use the configured workstation GPU queue when sharing compute. Each output contains native evaluation logs, frames and task state. Export the native log with `inspect-robots video <evaluation.json> --out <video-folder>`. To inspect the source changes before running, add `--prepare-only` with a separate fresh output directory.

[evidence/robustness.json](evidence/robustness.json) contains the full numerical summary and hashes of the five recordings. validation.json links this supplemental evidence to the nominal demo and Dream-1 results.

## Scope

Each condition was run once with seed 0. The results show performance in these five selected configurations. Initial inventory limits available volume, while pressure-head effects remain outside the current discharge law. Friction is an assumed material parameter. Ice bridging, empirical wet-pad adhesion and the assembled fixture's flow curve require additional modeling or calibration.

The script observes simulated cup volume. These trials evaluate the scripted controller. Dream-1 retains its separately reported nominal score of 0.10.
