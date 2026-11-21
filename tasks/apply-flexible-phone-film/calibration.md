# Resistance reference and bench check

Mitsubishi Polyester Film lists an initial release value of 10 gram-force per 25 mm of width for HOSTAPHAN RN SI63O, measured in an unsupported T-peel against tesa 7475 at 12 inches/minute (5.08 mm/s). Scaled to 64 mm width, this is approximately 0.251 N. It is a reference for a low-release silicone-coated film. It is not a measurement of a particular phone kit. A replacement kit, peel angle and speed can change the result.

[Manufacturer data sheet, page 1](https://www.m-petfilm.de/wp-content/uploads/RN_SI63Oe.pdf)

For a real comparison, clamp the protector's start to a glass dummy. Pull the returned liner using a 0–2 N force gauge in line with its travel. Record actual strip width, test angle and pull speed. Exclude the initial breakaway and record the steady force over the middle 80 mm on three fresh samples. Inspect the force trace for stick-slip. Report the three medians and range. Recheck the simulation under the same geometry before claiming quantitative agreement.

The simulator uses effective rigid section masses of 1.2 g per protector section and 0.9 g per ordinary liner section. The returned liner section, including its paddle, is 4 g. These masses and the enlarged collision thicknesses are numerical approximations. They affect inertia and the measured paddle load.

The simulator applies a 145 N/m spring in each translational direction at the exposed interface. The force cap is 0.25 N per axis. The local opening criterion is 0.25/145 m, about 1.72 mm. The inferred value at first detected separation is a sampled spring estimate and may exceed the threshold between observations. It is not a force-gauge measurement. Solver contact loads on the card and phone are logged separately in newtons. Only the fold can release the next bonded strip, while intact layers behind it transmit shear.

The trace records the in-line component of the gripper's normal load on the draw paddle, independently of the interface spring formula. Run `python load_report.py runs/scripted01/trace.jsonl`. The report integrates contact force against positive travel over the predeclared 30–110 mm interval. It reports forward work divided by distance, median sampled load, pull speed and sample rate. Normal loads include effects of the fixture, inertia and contact. They do not isolate adhesive force. These 20 Hz samples can miss short peaks. Compare the release-force scale and the complete task load separately, using the stated test geometries and speeds.

Other low-release PET grades are available. BFI lists a 3 gf/25 mm silicone-coated PET grade (IS-2503), equivalent by width scaling to about 0.075 N for 64 mm. That figure and the 0.251 N HOSTAPHAN reference concern different materials and tests. The simulation's measured paddle load, speed and filter are reported alongside the chosen material scale. A physical test of the actual kit remains a separate measurement. [BFI silicone release film](https://en.bficop.com/32)
