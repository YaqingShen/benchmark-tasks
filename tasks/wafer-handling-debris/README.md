# Handle a wafer and remove visible debris

**Media uploaded: `M-GWAGQE`.** The camera-enabled Brev run through Isaac Lab-Arena and Inspect Robots scored **1.0**: all twelve pieces removed, all five components passed, 102.55 simulated seconds / 2,051 actions plus the initial observation, layout seed 17. It closed without an execution error. Independent re-scoring matched the measured result and verified the source and geometry bindings. The public copy was audited again after path-only redaction. The clean demonstration and matching setup image are complete. Dream-1 completed a separate trial with **0.0** after a robot-collision violation at 78.5 simulated seconds (1,570 actions). Both complete videos and the matching setup image are uploaded under media ID `M-GWAGQE`.

The pick opens when squeezed and closes when the squeeze is eased. The robot must use this passive tool to hold the wafer at its straight flat while the other hand operates the gun, then put both objects back on their supports.

## Task instruction

Visible debris is scattered on this wafer. Use the wafer pick at its straight flat to lift it from the three supports. Hold it steady and blow off every piece with the gun. Put the wafer back on the supports, return the gun to its stand, and release both.

## Setup

| Object | Geometry and nominal material assumptions |
|---|---|
| Wafer | Oxide-coated silicon surrogate, diameter 100 mm, thickness 1 mm, flat 32.5 mm, mass 20 g |
| Three pins | Metal, diameter 1 mm, height 20 mm; equilateral triangle side 55.43 mm |
| Heater / pedestal | Inactive metal fixtures, diameter 150 mm; heights 20 / 25 mm |
| Spring-closed pick | Three dynamic bodies, total 85 g; polymer body, rigid pads representing elastomer contacts; 32.5 mm pads; fixed handle 70 × 26 × 6 mm, moving handle 54 × 26 × 4 mm |
| Pick supports | Metal blocks, 8 × 40 × 59 mm and 8 × 40 × 53 mm |
| Blow gun / stand | Polymer body 80 g, trigger 4 g, handle width 30 mm, trigger travel 4 mm; metal stand 60 × 60 × 10.5 mm |
| Twelve debris pieces | Three grains (0.4–1.0 mm long), three fibers (1–2 mm long, 80–160 μm diameter), three flakes (0.5–1.2 mm long, 30–70 μm thick), three wear fragments (0.45–0.9 mm long) |
| Robots / table | Two YAM arms with parallel grippers, starting empty, on the packaged Isaac Lab-Arena station table |

The debris is surrogate geometry. Assumed densities are respectively 2400, 1200, 1050 and 2700 kg/m³; masses follow convex volumes. Seed 17 sets sizes, yaw and non-grid positions. Whole shapes start on the wafer without overlap, outside the grip band.

`edge_native/scene.xml`, native assets, the adapter and bundled plan specify exact geometry, transforms and initial robot configuration. Run metadata records measured initial geometry and source hashes; the setup image must match. The pick is an inspired design, not a certified commercial replica.

## Score

`edge_native/wafer_debris_score.py` is authoritative. For a valid trial:

| Component | Credit | Conditions |
|---|---:|---|
| Pick up and lift wafer | 0.20 | Pick rises ≥30 mm; retained wafer rises ≥10 mm and clears all pins by ≥5 mm; qualifying pickup lasts ≥0.2 s |
| Operate second tool | 0.10 | Gun rises ≥20 mm and is retained by the other hand; trigger compression ≥2.2 mm; nozzle axis hits the wafer, 50–110 mm along the axis from its surface and 30–60° from its normal |
| Remove visible debris | 0.40 × N/12 | Each complete measured convex shape projects outside the wafer boundary by ≥1 mm for ≥1 s and remains clear; a fiber still hanging over the edge does not count |
| Return wafer | 0.15 | All three pins support it; position within 1 mm and orientation within 2° of the initial pose; pads clear by ≥1 mm or withdraw entirely; pick retreats ≥20 mm; stable for ≥1 s |
| Return gun | 0.15 | Supported on its stand, within 2 mm and 2° of its initial pose, trigger compression <1 mm, both hands outside the 20 mm grip zone and carrying hand retreated ≥40 mm; stable for ≥1 s |

During blowing, wafer tilt is ≤10°, speed ≤20 mm/s and angular speed ≤5°/s. Debris first moves/departs during valid aimed blowing in the current or preceding 20 Hz sample, while the wafer is held and before return. Indirect debris-to-debris dislodging is allowed; counting/video establish the outcome, not individual airflow causality. Early loss or dumping without valid blowing invalidates the trial; re-entry revokes removal credit.

Pick-pad face contact is restricted to nominal lower/upper bands of 5/3 mm from the straight flat, with a declared 0.1 mm geometric/model allowance (effective boundaries 5.1/3.1 mm). This is an engineering allowance, not certified metrology. Tool retention limits are 2 mm/3° for the pick and 1.5 mm/2° for the gun relative to the carrying hand. Support interruptions lasting 0.1 s invalidate the trial. Return stability allows at most 0.5 mm/0.5° excursion over 1 s; support-plane gaps must be within 0.5 mm. Fixtures must remain within 0.1 mm/0.05° of their initial poses.

Drops, out-of-band pick-pad contact, detected robot self/inter-arm collisions, changed fixtures, altered/missing objects, missing/reordered observations or exceeding 120 s make the overall score zero. Otherwise partial credit is retained for completed components. Keep the pick held after wafer release; returning it is optional, but if released onto its support it must settle for 1 s. All five components and all twelve pieces are required for 1.0.

Physical checks use counted debris and calibrated overhead/side video, a scale and timestamps. Primary scoring needs no contact-force sensors, jet-impulse measurement or exposure grid. Internal contact diagnostics are separate. The 20 Hz audit does not prove continuous collision avoidance or hardware reliability.

## Run and score

This package contains the exact `edge_native/`, `baseline/` and `tools/` source used by the successful Brev run. Historical helper modules and assets bound into that run are retained unchanged; use the entry points below. The run used an NVIDIA L40S (driver 565.57.01), Python 3.12.16 and Isaac Sim `6.1.0-rc.26+release.49347.2d230af4.gl`. Installed distributions were Isaac Lab-Arena 0.3.0, Isaac Lab 17.0.2, NumPy 2.5.4, SciPy 1.17.0 and Torch 2.11.0+cu128. Source commits were Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, Isaac Lab `91a0906f93802eb859192dcb0bca879f891e92d6` and YAM `0059d02a1a91bc6834c1d0846022891df0c78730`.

Inspect dependencies are `inspect-robots==1.0.0`, `inspect-robots-yam==0.38.0` and the supplied task plugin 0.6.1. The provided Brev environment supplies `inspect-robots-dream==0.1.0` and `dreamlabs-client==1.0.14`. The successful scripted Inspect run took 1,927.19 wall-clock seconds (32 min 7 s); this is one measured runtime, not a guarantee.

From the package root, with the Isaac environment active and an unused output directory:

```sh
python edge_native/run_edge_workflow.py --baseline baseline --output runs/demo17 --stage full --seed 17 --device cpu
python tools/score_edge.py runs/demo17 --source-dir edge_native --baseline-dir baseline --output runs/demo17/independent_check.json
```

Reset in a fresh process. The script commands robot joints, not object poses. Preserve metadata, commands, observations, trial report and independent check from the same closed run. CPU PhysX runs at 2000 Hz, control at 20 Hz; cameras need the GPU. The 120 s limit is simulated time.

In the provided Brev launchable, activate its Python 3.12 Isaac Lab-Arena environment first. Keep its Dream adapter, client and service configuration; this package does not redistribute them or promise that the Dream adapter is available on PyPI. Install the readable task plugin and pinned Inspect interfaces in an isolated overlay:

```sh
uv pip install --python "$(command -v python)" --no-deps --no-build-isolation \
  --target .policy-plugin ./baseline/benchmark \
  inspect-robots==1.0.0 inspect-robots-yam==0.38.0
export PYTHONPATH="$PWD/.policy-plugin${PYTHONPATH:+:$PYTHONPATH}"
```

This assumes the provided simulator environment already supplies NumPy, SciPy, Pillow, the Dream client and its dependencies. No simulator/torch upgrade is requested. Run the model with a new evidence and log directory:

```sh
inspect-robots run --no-prompt --embodiment wafer_blow_arena \
  -E bundle="$PWD" -E layout_seed=17 -E evidence_dir="$PWD/runs/model17/evidence" \
  --policy dream --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer wafer_blow_score --max-steps 2400 --seed 17 \
  --log-dir runs/model17/inspect --store-frames
```

The adapter/service must be configured in the provided environment. Set `layout_seed=17` explicitly: Inspect’s separately derived trial seed is not the debris seed. Policies receive three 320 × 176 RGB views and fourteen measured joints, and send fourteen joint/gripper targets. Private object poses/scoring evidence are not observations.

To execute recorded robot targets from a verified run through that same interface, replace `--policy dream` with `--policy wafer_blow_scripted -P run="$PWD/demo" -P candidate_bundle="$PWD"`, and use a new evidence/log directory for that execution. The bridge verifies candidate/seed compatibility and executes fresh physics; check its score separately.

## Included demonstration evidence

`demo/` contains the selected successful Brev run: 2,051 recorded actions, 2,052 measured observations, metadata, the unchanged native trial report, the closed Inspect result and a fresh independent check of this public copy. These are the inputs for the recorded-action policy and offline scorer. The compressed observations bind robot targets to measured actuator states and let the score be checked again.

One native metadata path and two Inspect sidecar paths were replaced with relative labels; `demo/redaction.json` records the original hashes and exact fields. Geometry, commands, observations, scores and the native trial report are unchanged. Raw camera-frame and Inspect record sidecars are not included in this compact package; video media are supplied separately. The original private evidence was preserved. To verify without starting a simulator:

```sh
python tools/score_edge.py demo --source-dir edge_native --baseline-dir baseline
```

A recorded-action policy executes these joint targets in fresh physics; it does not move objects by replaying their recorded poses. A video rendered from measured poses is only a view of the saved run.

## Dependencies and notices

The runtime loads YAM robots and the station from installed `yam_arena`; native YAM USD, URDF and STL files are not bundled. The audit metadata does include 630 cooked YAM convex collision hulls for independent scoring. The installed asset source, `yam-isaac-lab-arena` commit `0059d02a1a91bc6834c1d0846022891df0c78730`, carries `yam_arena/assets/yam/LICENSE.i2rt`: MIT, copyright I2RT Robotics. Its exact notice is preserved in `THIRD_PARTY_NOTICES.txt` and applies to that derived geometry.

Isaac Sim, Isaac Lab-Arena, Inspect Robots and Dream remain external dependencies; their distributions are not included. No blanket license for the task-specific source is inferred from those dependencies. The destination repository’s README, AGENTS and example task specify the submission workflow, but no repository-wide source license or task-license field was present at the checked revision.

## Video views

The clean demonstration is `wafer_mixed_debris_brev.mp4`: H.264, 1920 × 1080, 20 Hz, 2,052 frames and 102.60 s including the initial observation. Its three standard panels are the actual native top, left-wrist and right-wrist RGB recordings from the successful Brev Inspect run. The fourth panel is a wafer-inspection close-up rendered from measured poses in that same cloud trace with physics disabled. It improves visibility and is not an additional physics trial or model observation. The synchronized video contains the whole run, without HUD text, speed changes or removed segments.

`setup_brev.png` shows the matching initial layout. `replay_brev.json` binds the movie to its trace, assets and camera streams and records the rendering/composition checks. `dream1_three_cameras.mp4` contains the actual Dream trial’s three standard RGB views: top above the left/right wrists. It is H.264, 640 × 528, 20 Hz, 1,571 frames and 78.55 s including reset, with no HUD text, dropped frames, speed changes or removed failure segments. Dream receives only these three standard views, not the demonstration’s extra inspection view. Media are uploaded separately from the public task source.

## Dream-1 result

Dream-1 scored **0.0** in its separate camera-enabled Brev trial. The run ended at 78.5 simulated seconds / 1,570 actions with `invalid_task_state`; the recorded invalidation was `robot_collision`, a detected right-gripper/right-link2 self-collision. It made 33 completed Dream inference requests with dry-run mode disabled. The action rate/limit guard modified 481 of the 1,570 commands; the original model outputs and applied commands are preserved in the original evidence archive. Inspect and the native worker closed cleanly, and independent scoring matched the result with the physical source bindings verified. This is a scored task failure, not an evaluation or shutdown exception. The model video includes all 1,571 stored frames, including reset and the final failed state. `model/inspect.json` contains the closed result and request transcript with session identifiers and private paths redacted; `model/validation.json` retains the original validation unchanged. `model/redaction.json` records the exact changed fields and original/public hashes.

## Scope and limitations

Physics uses rigid contact, passive springs, assumed friction, breakable adhesion and approximate directed drag. No calibrated airflow, moisture, film removal, nanoscale cleanliness, chemistry, heating or wafer damage is modeled. The grip band is unseeded; whole-surface cleanliness is not claimed. Particles remain simulated after removal.

The media upload is complete (`M-GWAGQE`). One successful camera-enabled Brev run is not a reliability estimate or evidence of an 80% success rate. The packaged scores and media refer to the verified runs above.
