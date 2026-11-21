# Hardware reference and reproducibility

## Obtainable appliance

Use the **Hamilton Beach Durathon Removable-Grid Belgian Waffle Maker, model 26031G**, with its supplied drip tray. On 21 November 2026, the [manufacturer's product page](https://hamiltonbeach.com/durathon-removable-grid-belgian-waffle-maker-26031g) displayed US$75.99 and an Add to Cart control. Price and availability may change. This identifies a purchasable appliance rather than a custom appliance that a team must fabricate.

The same product page links [Use and Care Guide 840204801](https://useandcares.hamiltonbeach.com/files/840204801.pdf). Printed page 4 specifies closing the filled lid, turning the handle clockwise to flip and lock within 10 seconds, then turning counterclockwise before opening. There is no separately operated front spring slider in this cooking sequence. The grid-release buttons on printed page 3 release removable cooking plates for cleaning; they are not lid locks and must not be operated during a trial.

The manufacturer's materials include stainless housing, ceramic nonstick grids and a cool-touch handle. Its listed overall dimensions are 9.5 × 16 × 10.5 inches (height × width × depth). Follow that appliance's instructions for mains supply, preheating and cooling; do not modify its wiring, heater, lock or removable-grid retention.

## Changes from the reviewed simulation

The red slider, slider joint, spring, ramp and robot pull/release motions are removed. The long turning handle is attached to the upper lid, as shown in the [manufacturer photo](https://hamiltonbeach.com/media/products/26031-02.jpg). The scene includes a front support cradle and rear support. One arm steadies the base while the other closes the lid by that handle. Rotation engages the lid lock. Returning upright releases it. The robot first lifts from the free end of the same handle, then transfers to a proximal grip once there is room below the handle. This avoids inserting a gripper between the closed handle and its lower support. The chamber still rotates clockwise to the inverted position and counterclockwise to upright. The robot uses nonmetallic-tipped tongs to serve onto the side plate.

The operating sequence follows the identified appliance. Internal lock geometry, hinge torque, inertia, friction and exact contact meshes are **not manufacturer measurements**. This is a functional simulation, not a manufacturer CAD model. In particular, no claim is made that the published manual specifies a particular cam, spring, force or clearance.

## Explicit lock approximation

`rotation_lock.py` implements a passive constraint using Isaac joint limits:

| Observed configuration | Permitted movement |
|---|---|
| Upright, lid more than 0.2° open | Rotation is constrained to 0–3° inside the 7° scoring home zone; lid may open to its 100° stop. |
| Upright, lid at most 0.2° open | Rotation can proceed to the 180° stop. |
| Beyond 7° rotation, lid at most 0.2° open | Lid travel is limited to 0.2° while the chamber can rotate. |
| Returned within 7° of upright | Lid-opening constraint releases. |

The 0.2° lid clearance, 3° open-body stop and 7° release-zone values are disclosed simulation tolerances. The update changes constraint limits only; it never writes appliance joint positions, velocities or food poses. A state with an already-open lid outside the opening zone is left observable, and the scorer rejects it. The guard cannot repair a violation and restore credit. The native tests must establish that the solver enforces these limits during actual robot motion; synthetic rule tests alone are insufficient evidence of physical behavior.

## Layout and reset

Use the positions and modeled dimensions in README for simulation. For a physical trial, place the unmodified appliance upright on a level, heat-resistant worktop with the turning handle toward the robot. Mark the base outline and rotation axis in the overhead view. Place the empty 250 mm plate to the robot's left and the open silicone-tipped kitchen tongs in the weighted holder beyond it. Keep the supplied drip tray installed and all cords outside the robot's sweep. The product's overall dimensions and the simplified simulation base dimensions refer to different envelopes; do not use the modeled base size as a hardware mounting template.

Preheating and measuring the portion are setup operations. Use the manufacturer's fill guidance and a repeatable batter batch as described in README. Keep the folding handle extended for the entire trial; storage folding is outside the task. Start upright with the upper lid at 45°, plate empty and tongs open in the holder. A removable setup support may hold the lid while arranging the scene, but must be removed before the trial. Verify that the actual lid remains open at the chosen starting angle; do not represent this model's nominal 1.20 N·m friction as a measured appliance property. If the lid does not remain at 45° on the actual unit, record that difference before claiming a matched hardware trial.

Use insulated gripper covers suitable for the appliance, and keep robot fingers on exterior handles or the stand. Neither gripper may contact food or the cooking interiors. Use the supplied tray and a removable catch mat. Allow cooling and unplug before cleaning or removing grids. Between trials, restore the marked positions, remove food and batter residue, use a fresh portion, and reset scoring history and the bake clock.

## Evidence boundary

The product and manual establish an obtainable appliance and its rotation-lock operating sequence. The revised simulator tests establish only the stated model behavior. Baking remains an explicit 300-second closed-lid state transition; it does not predict heat transfer or prove food doneness. Actual units may require the longer part of the manufacturer's stated cooking interval. No physical cooking trial or measured lock-force validation is claimed.
