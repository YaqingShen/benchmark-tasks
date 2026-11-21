# Apply a flexible phone screen protector

Fit a 140 × 64 mm flexible protector to a 150 × 72 mm flat glass phone dummy. The left arm draws the returned liner along a guide while the right arm presses the exposed film with an application card. Both arms start clear of the work.

The underlying hand procedure follows a commercial PET protector installation. Photodon describes aligning the protector, exposing about 25 mm of adhesive on a small device, then advancing the squeegee as more backing is withdrawn. The benchmark uses a guide and paddle to make this stage repeatable with two grippers. The checkpoint and bubble tolerances are benchmark criteria. [Photodon installation guide, steps 6–8](https://www.photodon.com/2025-Screen-Protector-Instructions.pdf)

## Reset the physical station

Fix the 12 mm thick dummy to a level table. Its long axis is the application direction. Centre a 140 × 64 mm target on the glass. Use printed registration marks beside the phone, a card with a 66 mm straight edge, a backed PET protector with a pull extension, and a clean glass surface. Record the protector and liner material, supplier and lot for each series. Use a fresh set for every trial.

The leading edge travels 9 mm in the registration slot. A stop and detent retain the aligned edge. Start at the near end of this travel. Fold the liner back under the protector to expose 28 mm at the starting edge. Put the far liner ear on the 13 mm support on the side opposite the pull tab. The return flap runs below the remaining laminate. Its side extension moves a rigid upright paddle on a longitudinal slide alongside the phone. Leave the paddle side clear for its full 255 mm stroke. A hand pushing this paddle draws the attached liner. The other hand can grasp the application card from its insert. In the simulator, `laminate.py` and `task.py` specify dimensions and reset poses in metres.

### Bench fixture and materials

The following materials are the proposed physical equivalents. Use the near-left target corner as (0, 0) and measure along the 140 mm application direction. The short-end and side margins of the dummy are 5 and 4 mm. A smooth acrylic surround supports the parts flush with the glass. Its side and end pieces measure 199 × 51 × 12 mm and 25 × 73 × 12 mm. The card has a 66 × 46 × 4 mm rigid plastic body and a 66 × 4 mm elastomer contact lip, 7 mm across the sweep direction. Its rest uses two 5 × 20 × 18 mm blocks with an 8 mm clear gap. At reset the card centre is 51 mm along the application axis and 145 mm to the side of the target centreline, opposite the paddle. The card remains grasped as it lifts clear.

Use a clear PET protector and silicone-coated PET release liner. Attach the liner's returned flap to a rigid plastic paddle by a side extension. The paddle is 7 × 40 × 56 mm, its centre 84 mm from the target centreline. Use a low-friction linear guide with at least 255 mm of usable travel. The simulated guide rail is 320 × 5 × 8 mm. The two leading-edge slots and a detent allow 9 mm of travel. The 32 × 4 mm wing support rises 13 mm above the glass. Select the actual slider, paddle and pull extension together so the removed backing clears the phone and the moving extension clears the support. Record the build and the laminate lot before a physical trial. This geometry has been exercised in simulation.

Print `inspection-ruler.svg` at 100% without page scaling. Verify the 50 mm check segment with a separate ruler. Place the printed strip beside the glass at the glass height with its zero aligned to the target’s starting edge. The three marked windows locate the first-pass observations. Use a side view with minimal parallax and a scale in the same vertical plane as the visible fold and blade edge.

## Physics and measurement

The protector and its backing each have ten dynamic rigid sections. Adjacent sections bend about hinges across the sheet. Every section has a collision shape and loads can pass between the two sheets, glass and card. A force-limited interface spring at the fold represents release. The other interface ties keep the covered area together until the fold reaches it. The front is exposed by measured separation of the physical layers. Pressing an exposed strip to the phone can tack it only when both tool-to-film and film-to-glass contacts carry load. The world-space contact pose defines the tack. The grippers and paddle move through contact. The liner and film poses are integrated by the physics engine.

The strips approximate bending along one axis. Their collision thicknesses are 0.65 and 0.48 mm for numerical stability. A two-dimensional venting model marks air regions enclosed by loaded contact. Its visible outlines are a proxy for air pockets. The model does not resolve dust, microscopic adhesive flow or creases across the width. See [resistance data and gauge procedure](calibration.md) for the release-force reference and the limits of the comparison.

## Run the revision

Use the configured YAM station environment with Isaac Lab, YAM Arena and Inspect Robots. The runner uses seed 0 with object jitter disabled, records source hashes and permits 465 control actions at 20 Hz. Each run name must be new. On the RTX 5090 the scripted run used 20.35 simulated seconds and took 512 seconds of wall time. Dream-1 used 23.25 simulated seconds and took 584 seconds of wall time. Wall time includes recording and varies with machine load. The validation environment used inspect-robots 1.0.0, yam-arena 0.1.0, isaaclab 17.0.2, NumPy 2.3.1, SciPy 1.17.0 and Shapely 2.1.2.

```sh
bash run.sh scripted01 video
bash run.sh policy01 dream
python export.py runs/scripted01/adhoc_????????.json demo.mp4 setup.png
python export.py runs/policy01/adhoc_????????.json dream.mp4
```

Both policies use the same task, embodiment, observation cameras, instruction and score. Each video combines the original overhead and two wrist cameras. The trace records measured poses, solver contact loads, interface releases and score components. The previous submission's scores and media belong to its earlier physics.

## Score a physical trial

Both computers read the same `bench-config.json`. Use an overhead recording, a low-angle recording, an independent ruler and a pair of final illuminated photos. The photographic score covers corner alignment, flat target coverage, liner removal, three video checkpoints and visible bubbles. The card’s lower edge must remain clear for the final finish points. See the [photo and video protocol](photo-rubric.md) for the numerical thresholds and partial credit.

Copy `bench-trial.template.json` alongside the physical trial files. Enter the measured observations and run `python bench.py trial.json`. Run `python -m unittest test_inspection` to exercise the shared scoring rules with synthetic inputs. For standalone scoring, create a Python 3.11 or newer virtual environment and install `python -m pip install -r requirements-inspection.txt`. For simulation, activate the configured YAM environment. Real camera files and force traces must be collected on the bench.

For provenance, run `python audit_run.py runs/scripted01`. It checks the official score against the final physical trace and lists the first qualifying frame at each checkpoint. Contact magnitudes in the trace are reported in newtons. Two diagnostic cameras save overhead and side stills outside the three policy observation streams. For physical comparison, retain the original two-angle photographs and gauge files alongside their completed real-trial record.

Two scoring controls can be repeated from this folder on the configured 5090. Give each script a new output directory. `bash evidence/check-no-draw.sh control-no-draw` holds the lined film in place while the card sweeps. `bash evidence/check-visible-pocket.sh control-bubble` inserts a known perimeter seal around a 5 × 4 mm air pocket and runs the same physical demonstration. The second case is a deliberately seeded defect for checking visual detection and scoring.

Independently measure the rendered finish with `python bench.py --photo-only evidence/inspection-photo.json`. Measure the seeded defect in the separate control image with `python bench.py --photo-only evidence/seeded-pocket-photo.json`. Each annotation records the SHA-256 of its source photograph. For the three motion checkpoints, see the exact-step inspection images and the physical trace. The original video has a reset frame at index 0. Trace step 0 corresponds to video frame 1.

To reproduce the opposite-side diagnostic view, run `bash evidence/check-far-view.sh opposite-view` in the YAM environment. It changes the pose of the inspection camera only. It is a separate replay. The original run’s sensor images are retained, including occluded frames. See [checkpoint evidence](evidence/checkpoints/README.md) for the comparison.

To isolate the effect of interface resistance on the same commanded demonstration, run `bash evidence/check-release-strength.sh strength-control`. The copy changes the per-axis release limit from 0.25 to 1.0 N. The nominal trial, its task score and its videos use 0.25 N.
