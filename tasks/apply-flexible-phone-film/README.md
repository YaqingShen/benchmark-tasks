# Apply a flexible phone screen protector

Align the protector with the phone. Pull its release liner while dragging the squeegee behind the peel front. Cover the screen edge to edge, then lift the card clear.

## Setup

Use a clean flat dummy phone measuring 150 × 72 × 12 mm, a 140 × 64 mm dry-apply PET protector with a folded liner and pull sticker, and a 106 × 48 mm plastic squeegee with a rubber blade. The phone can be secured to a nonslip mat. The card rests on its edge against a small packaging insert. Both grippers start empty; the protector starts shifted 22 mm lengthwise and 12 mm sideways. A fresh, unbonded protector is used each trial.

## Simulation and score

PhysX simulates the grasps on the folded pull tab and card and measures the blade's contact with the phone. A reduced film model uses the measured tab motion to unfold the liner; its 28 × 8 surface cells can bond only after exposure and contact with the loaded blade. The simulated sheet follows the folded tab before the first edge is secured. Pale cells are unpressed and dark cells have bonded. These colors aid inspection. Adhesion, film bending, and visible pockets are represented approximately. The task measures application and coverage. The model does not predict individual air bubbles or dust.

To start the bond, all four protector corners must be within 3 mm of their intended positions, with the card across the leading edge. An uncovered gap wider than 25 mm between the peel front and a newly pressed region loses progressive-application credit for that region. Human scoring can use a marked photograph of the margins, a 25 mm ruler beside the phone and the video of the sweep.

After the first bond, award 0.15 times the proportion of liner removed, 0.45 times the bonded area fraction, and 0.30 times the area fraction pressed progressively. Award 1.0 when all cells are progressively bonded, the liner is removed and the unloaded card is at least 15 mm above the phone for half a second. With zero bond, award zero. Run the material regression checks with `python -m unittest test_material`.

## Run

Activate the calibrated YAM station environment with Isaac Lab-Arena, Inspect Robots and the YAM embodiment. Give each run a unique tag. The standard observation streams are the overhead camera and the two wrist cameras.

```sh
bash run.sh demo01 video
bash run.sh dream01 dream
```

The shared physical scene and action limit apply to both policies. Use a fresh protector for real trials. Different protector kits use different sequences. A commercial example showing gradual backing removal and use of an application card is [Photodon's dry-install guide](https://www.photodon.com/Install-Instructions.pdf).

For reset, place the phone lengthwise away from the operator. The far and near short bezels are 5 mm beyond the intended film ends. The side bezels are 4 mm wide. Put the backed protector 22 mm toward the operator and 12 mm to their left of its target. Leave the pull sticker at the left corner. The squeegee starts to the operator's right of the phone. The source defines every shape and initial position in metres.

The script uses 20 controls per second and up to 465 actions. The recorded reference trial succeeded in 420 actions (21.0 simulated seconds). Its H.264 recording includes the initial observation and one frame per action. The 5090 used Isaac Lab-Arena revision `c7edf3a9d5d8276565208f72422f8f8a0806d6cf` and YAM Arena revision `0059d02a1a91bc6834c1d0846022891df0c78730`.

For a run recorded with frames, `python export.py runs/<tag>/<log>.json media/demo.mp4 media/setup.png` checks contiguous frame numbers and combines all three cameras. Run the export command from the project directory.

Dream-1 (`dreamlabs/dream-1-beta`) reached the 465-action limit in 23.25 simulated seconds, with a final score of 0.0. It displaced the squeegee slightly and never aligned and bonded the protector. Its video contains 466 three-camera frames. Its control interface, objects, instruction and scoring were the same as the reference demonstration.
