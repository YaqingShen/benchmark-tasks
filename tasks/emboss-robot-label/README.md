# Manual embossing label: ROBOT — Isaac prototype

Requested word: `Robot`. The consumer embosser reference has uppercase letters, so this task's exact output is `ROBOT`.

The robot turns a letter wheel, squeezes a spring-return handle, releases fully between letters, advances two spaces, and selects the cut position. A visible bench clamp stabilizes the housing. Tape is loaded initially. Loading a cassette, peeling backing and sticking the label are outside this trial.

Operating-sequence reference: [DYMO Omega pictorial instructions](https://download.dymo.com/dymo/user-guides/Embossers/Omega_InstructLeaflet_GRAY.pdf), panels 5–15. This scene uses a declared prototype geometry, 49-slot wheel order, return spring and bench fixture. The simulated CUT position uses an X glyph. Dimensions and model parameters are listed below.

## Physical and abstract behavior

- Both robot arms are driven by their ordinary joint actions. The wheel and trigger move through PhysX contacts. The gripper actuator effort is capped at 40 N per finger.
- The wheel has a nearest-detent spring. Its spring target depends only on its observed angle, never the desired letter. Robot-to-wheel friction must rotate it.
- The trigger is a revolute hinge with a return spring and mechanical end stops. No controller writes task-joint positions.
- Embossing, tape feed and shearing are abstract material responses. They are triggered by measured wheel alignment and a full trigger stroke. Glyphs shown on the output strip come from these events, not from the requested word. The cut strip remains on an output support; free-strip bending and detachment physics are not implemented in this first prototype.
- Results are from native Isaac simulation.

## Scoring

`mechanism.py` receives only measured wheel/trigger angles. It awards 0.60 for the correct prefix, 0.15 for exact text, 0.15 for one correctly cut label with the two-space trailing margin, and 0.10 for complete release after cutting. A wrong extra character, more than one label, squeezing between detents, or rotating the wheel during a squeeze invalidates full success. Holding a squeeze cannot generate repeated characters. Partial strokes do not print. Desired action indices and elapsed demo time do not contribute points.

`test_mechanism.py` tests normal and negative scoring cases. The complete native demonstration separately verifies robot feasibility in this scene.

Starting a second strip after the correct cut revokes the score, even if the first label is correct. The release test uses a correctly cut label and verifies that holding the trigger or returning it for only 0.10 s cannot pass; a full 0.15 s return is required. No observation pose, elapsed demonstration time or camera framing contributes points.

## Prototype setup and reset

The simulated fixture has the following dimensions. The wheel is 76 mm in diameter and 7 mm thick, with 49 detents. Its axis is 135 mm above the worktop. The visible bench clamp has a 120 x 130 mm footprint; the fixed housing is centred at x=0.300 m, y=0 in the station frame. The output tape is 9 mm wide and 0.6 mm thick. Glyph spacing is 4 mm. The output support is 52 x 13 mm. The trigger rotates through 0.30 rad and has a 0.85 N m/rad return spring; these values are the declared simulation spring parameters.

Reset the wheel to its zero detent (A), let the trigger fully return, and load a fresh blank strip. Remove any previously cut strip. The clamp, machine and support remain in the declared starting positions. Clear the event history and score. The simulated tape is unlimited; a real trial needs enough tape for five glyphs, two spaces, and the leading and cutting margins. The present fixture immobilizes the housing so that the task tests wheel manipulation and complete squeeze/release cycles; it does not test holding a free handheld embosser.

For a physical outcome check, photograph the cut strip and inspect the trial video: exactly one label must read ROBOT, all five letters must be legible and intact, the trailing margin must remain, and the trigger must return after every press. Simulator joint thresholds are operational proxies for the stroke endpoints and alignment. 

## Reproduction

Use the workstation's installed `yam_arena` and `inspect-robots` environment. `preflight.py` checks the continuous waypoint plan with the installed robot's IK. Native run:

`python run_task.py --policy yam_scripted --output runs/demo` sets a 120 s action budget from reset (the scripted plan ends earlier). For the separate model evaluation, use `python run_task.py --policy dream --output runs/dream`. The runner uses seed 0, 20 Hz, the same task instruction and scorer, and 1280 x 720 cameras unless `--preview` is supplied. Dream-1 is explicitly `dreamlabs/dream-1-beta`; it receives camera observations and ordinary robot actions, not the scripted waypoint policy. The native replay result is not a physical device test.

## Verified results

The completed native 720p r06 trial returned 1.0 in 1,622 steps (81.1 s), under its original 1,750-step cap. Dream-1 and the published reproduction runner use a larger 2,400-step cap. The successful scripted plan finishes within both caps; its 1,622-step video is complete, not a truncated 2,400-step run. All eight events and releases matched replay; the final native wrist view contains the entire strip. The cut strip remains supported by the fixture after the cut event; free-strip handling is outside this trial.

The independent d07 Dream-1 run used `dreamlabs/dream-1-beta` and the same instruction, geometry and scorer. It exhausted all 2,400 steps (120 s) with score 0.0 and no valid embossing event. The native log records `max_steps`, no sample error, and the Dream-1 checkpoint. Robot motion is visible in the recording. Replay and source hashes passed. This is a single model trial, not an estimate of model success probability.
