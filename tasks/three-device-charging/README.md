# Charge a phone, watch and earbuds case on a three-in-one stand

Place three everyday devices on a powered wireless charging stand and leave them charging after release. The phone and watch need rear-pad alignment; the watch has a flexible, contact-bearing strap. The closed earbuds case rests in a shallow recess without magnetic retention. Charging feedback appears on the devices themselves.

`task.json` contains the evaluated instruction, setup and scoring rules. There is no required device order or prescribed hand assignment.

## Run

Use Linux, an NVIDIA RTX GPU and the native Isaac YAM environment. The evaluated environment used Python 3.12.16, Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, Isaac Lab-Arena 0.3.0, Inspect Robots 1.0.0, inspect-robots-yam 0.38.0 and inspect-robots-dream 0.1.0. The YAM environment is [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02a1a91bc6834c1d0846022891df0c78730`. Follow its installation instructions and activate the resulting Isaac environment. Dream-1 also needs the official Dream client and a valid account credential configured according to that project's instructions. No credential belongs in this task folder.

Run these commands from this task directory in the activated environment. Each output path must be new.

```bash
python run_seeded_demo.py --seed 190145710 --output runs/demo
python run_policy.py --policy dream --steps 6000 --output runs/dream
```

An optional interface check holds the measured joint positions; it is not a model evaluation:

```bash
python run_policy.py --policy hold --steps 24 --output runs/hold
```

The runners reserve `/tmp/robocurve-shared-gpu.lock` and require at least 16 GB free GPU memory. They do not terminate another process to obtain the GPU. Cameras use GPU rendering; native physics runs on the CPU at 1 kHz. Control is 20 Hz.

The scripted demonstration takes about 88 simulated seconds, around 8 minutes on the tested RTX 5090 and 13 minutes on the tested Brev L40S, including startup and recording. Dream-1 receives up to 6,000 actions, or 300 simulated seconds; wall time additionally depends on remote inference and retries. The submitted scene uses world seed `190145710`. The pinned Inspect SDK derives this from evaluation seed `17`, which is used by `run_policy.py`. The explicit-seed demo command above reproduces that same initial layout. Earlier development demonstrations used world seed `17`; they are supplementary checks, not the submitted demo. Dream-1 completed its 6,000-action budget in 50.2 minutes on the Brev L40S.

## Read the result

The demo writes `result.json` in its output directory. Dream-1 writes the task result to `evidence/trial-000/result.json`, the official Inspect evaluation under `inspect/`, and execution status to `process-result.json`. An execution status of `success` means the evaluation completed; the numeric task score and task success are separate fields.

Each trial also writes the original three-camera video (`three-camera.mp4`), a clean top-camera setup image (`setup-top.png`), metadata and a per-step `trace.jsonl`. Videos combine top, left-wrist and right-wrist views without text overlays or added cameras. The submitted screenshot uses only the top camera.

The scripted controller uses privileged object poses to plan joint targets. Dream-1 sees only the three original RGB views and 14 measured robot joint values. It receives no object poses, charging states, contact forces or scripted waypoints. Neither runner writes object poses or creates grasp attachment joints after reset.

## Scoring

For each device, let `d` be its current uninterrupted time charging with no gripper support, once its on-device charging cue has appeared in a camera view. Its score contribution is `min(d / 5, 1) / 3`. Losing charging or touching the device again resets that duration. Watch-band contacts count as support. At the end, sum the three contributions.

The stand must remain within 5 mm translation and 5 degrees rotation of its settled starting pose throughout the trial. Exceeding either limit makes the total score zero. Full success requires all three devices to be charging and released together for at least 5 seconds. The case LED is an 8-second startup indication; its turning off does not stop charging. The 40 settling steps before Dream-1 control are not policy actions.

In a physical version, use device charging status and video to check charging and release, a timer for the hold, and a marked reference with a ruler and angle comparison for stand movement. Simulation contact thresholds implement release checks; they are not additional instruments required in the physical task.

## Fixed setup

World coordinates are metres: x points away from the robot bases, y toward the robot's left and z upward. Tabletop z is 0.7475. Robot bases are at `(0, +0.305, 0.75)` and `(0, -0.305, 0.75)`. Both grippers start open and clear of the objects. The seeded object centres before settling are:

| Object | x | y | Yaw, degrees |
|---|---:|---:|---:|
| Phone | 0.321116 | 0.157922 | 0.091 |
| Closed earbuds case | 0.303837 | -0.039022 | -26.077 |
| Watch, straps unfastened | 0.371138 | -0.196456 | -3.399 |
| Charging stand | 0.510000 | 0.000000 | 0.000 |

The device face normals initially point upward. The watch straps start straight and flat. The stand is free, with power assumed connected. Initial screens are black and the case LED is off. A nominal 30% battery state is recorded; battery chemistry and depletion are outside the simulation.

Object sizes and materials are listed in `task.json`. The phone and watch heads are 100 mm apart. The case fits in a 78 mm-wide, 2 mm-deep base recess. Its collision geometry is a lower floor surrounded by 96 convex rim sectors, so a single convex hull does not fill the recess. Camera poses and intrinsics come from the pinned YAM station environment; all three RGB inputs are 1280 x 720.

## Physical model and limits

The phone and watch use finite paired attraction forces and alignment torques, with opposite forces on the movable stand. Peak attraction is 8 N for the phone and 2 N for the watch, attenuated by distance and alignment. There is no attraction on the earbuds case. These values are provisional, not measured magnet curves.

Charging requires enabled power, a rear-to-pad gap from -0.6 to +0.8 mm, lateral alignment within 3 mm for phone/watch or 6 mm for the case, normal error below 6 degrees, stand contact above 0.04 N and relative speed below 0.015 m/s for 0.3 seconds. These conditions drive a reduced charging-state model and the rendered indicators. They do not simulate electromagnetic energy transfer or prove compatibility with a particular commercial charger.

The watch has 16 colliding strap segments with 32 bend/twist joints. Strap stiffness, damping and stand mass are not calibrated to hardware. All task-object contacts use static friction 0.65, dynamic friction 0.5 and zero restitution. Glass, metal, plastic and silicone are represented by simplified rigid geometry and common contact parameters. The stand is inspired by a Belkin WIZ023, with deliberately wider head spacing; it is not an exact product replica. The case is closed and rigid, with the earbuds represented in its combined mass.

Cue visibility uses privileged projection to locate the indicator, then checks actual camera RGB for its colour. This establishes rendered visibility; it does not establish that the policy recognized or interpreted the cue. Phone/watch icons and the temporary case LED are generated from the same reduced charging state, rather than an independent electrical sensor.

## Source and evidence

`charging_world.py` builds the scene, records camera evidence and computes release/stand scoring. `charging_physics.py` implements magnetic forces, alignment/contact charging conditions and on-device indicators. `charging_benchmark.py` is the Inspect Robots interface. `scripted_charging.py` supplies the physical full-score demonstration. `build_assets.py` is the procedural asset source.

`source_manifest.json` freezes the exact revision used for both demos and Dream-1. Its SHA-256 is `8ca71b2b27548128092940d15fc85c1d3f6ac74d1fe57bb8559d7ffeecb5b6eb`. `run_policy.py` and `run_seeded_demo.py` check every listed file before evaluation. The latter is a thin entry point that calls the unchanged physical demo with an explicit world seed; it does not change the task physics or controller. Do not regenerate or edit these files when reproducing the recorded revision. Historical development/preview labels in `task_spec.json` and `assets/manifest.json` describe their original creation stage; the run metadata and results document the implemented and evaluated revision.

`validation-evidence.json` contains source-tagged results from the demonstrations, Dream-1 and component controls. The earlier RTX 5090 and Brev L40S demonstrations each reached 1.0 at world seed 17. The submitted Brev demo reached **1.0** at the same world seed as Dream-1, in 1746 recorded steps (87.3 simulated seconds, including initial settling). Its maximum stand movement was 2.790 mm and 2.642 degrees. Power-disabled and reversed-device controls produced no charging; disabling magnets removed magnetic retention. A short official SDK hold check completed separately.

Dream-1 (`dreamlabs/dream-1-beta`, real inference, no dry run) scored **0.0** after all 6,000 actions, with no devices charging. Inspect reported one completed trial, zero errored trials and termination at the action budget. The official client delivered 125 action windows; 124 needed a retry after the remote session expired. The physical trial continued without resetting the scene or using a scripted fallback. The preserved evaluation log records those retries. This result is one model trial, not a success-rate estimate.

## Asset sources and licensing

The phone, watch, case, stand and on-device indicator meshes/textures are procedurally created in this task; no commercial CAD model, product photograph or downloaded device mesh is included. Product references for dimensions and appearance are recorded in `assets/manifest.json`.

The external YAM robot model is supplied by the pinned YAM environment, derived from I2RT and covered by its MIT `LICENSE.i2rt`. Robot assets are not copied into this task. The YAM environment itself states that it has no project-wide license yet; it remains an external dependency, not redistributed task code. Isaac Lab is BSD-3-Clause, Isaac Lab-Arena is Apache-2.0, and Isaac Sim is distributed under NVIDIA's own terms. Install and retain those upstream packages and notices separately.
