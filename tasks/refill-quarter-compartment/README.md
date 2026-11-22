# Refill a cash drawer with a roll of quarters

The middle compartment is empty and the register needs quarters for making change. One closed roll adds exactly $10 if all 40 quarters reach that compartment. The U.S. Bureau of Labor Statistics lists giving change and counting the register among cashiers’ duties. A retail register procedure describes opening a coin roll to replenish change. [BLS cashier profile](https://www.bls.gov/ooh/sales/cashiers.htm) · [Retail register procedure](https://fromagination.com/wp-content/uploads/2018/09/How-to-Administer-the-Cash-Register-at-Fromagination.07.2015.pdf)

## Setup and reset

Use 40 U.S. quarters and one fresh crimp-ended paper quarter wrapper. Each coin is 24.26 mm in diameter, 1.75 mm thick and 5.670 g. Count the coins before loading and crimp both ends. The coin load is 226.8 g. Use wrappers from the same lot, with the same stock, dimensions and end closure. The simulated roll is 75.6 mm long with a 26 mm bore. [U.S. Mint quarter specifications](https://www.usmint.gov/learn/coins-and-medals/circulating-coins/quarter)

Fix a three-compartment plastic till to a level table. Each compartment measures 108 × 116 mm inside. The floor sits 4 mm above the table and the rim sits 41 mm above it. A blunt steel roll breaker, 43 × 3 × 31 mm, projects inward from the far wall of the middle compartment. Its top sits 51.5 mm above the table. A cash drawer with a built-in roll breaker is described in [US4958797A](https://patents.google.com/patent/US4958797A/en).

In station coordinates, center the middle compartment at (352, 0) mm, the steel strip at (389, 0) mm, and the roll at (224, 0) mm with its axis parallel to Y. Clear all three compartments. Start both grippers empty at the ready pose. Replace the wrapper and restore the full count between trials.

## Score a physical trial

Use an overhead view of all three compartments and the counter, plus a side view of the rim. Put a millimeter scale beside the drawer. At the end, count the loose quarters resting wholly inside the middle compartment. From the video, exclude coins that escaped before the wrapper opened or came to rest on the counter or in a neighboring compartment during the trial. A coin that passes over a divider without coming to rest can still count. For overlapping coins, photograph the final state, then count the qualifying contents by hand.

Each qualifying quarter earns 0.0225. A coin counts as settled when its center displacement plus rim displacement due to tilt stays within 1 mm across the final half-second. If all 40 qualify, award the final 0.10 when both empty wrapper halves rest on the counter outside the drawer with both hands at least 50 mm clear of each half’s center for ten consecutive 20 Hz samples. Thus 38 qualifying quarters earn 0.855, all 40 with paper still in hand earn 0.90, and all 40 with the paper parked earn 1.0.

## Simulation and limits

The two paper halves have closed outer ends, open inner mouths and a joint that can break under load. While confined, the coin stacks slide along their sleeve axes under gravity, wrist acceleration and friction. As each quarter clears the mouth it retains its pose and velocity and becomes a freely colliding rigid body. The remaining coins contribute mass and inertia to the sleeve. The model simplifies paper tearing to a single central break and omits individual coin rotation while confined. The scripted policy uses sleeve pose feedback. Dream-1 receives the station observations.

The simulated seam breaks at 65 N joint force or 0.145 N·m bending moment. In the bending fixture, a 70 g load at 80 mm held (0.055 N·m) and a 225 g load at 80 mm broke the seam (0.177 N·m). A locked reference held the heavier load. These are simulated checks. For a physical trial, measure the force needed to open a sample of the actual wrapper lot over the same blunt edge, record the lever arm and report the measured break moment. [Fixture observations](evidence/seam-loads.json)

## Run and verify

Use the bimanual YAM Arena environment for Inspect Robots with Isaac Sim, NumPy and SciPy. Video export also uses Pillow and imageio-ffmpeg. Point `YAM_ENV` at the environment activation script if it differs from `~/IsaacLab-Arena/yam.env`. Each run requires a fresh output name. Both policies use seed 0, a fixed layout and 900 control steps at 20 Hz.

```bash
python -m unittest test_packed test_paper_bounds
bash run.sh reference video 900
python qualify_run.py runs/reference
bash run.sh model dream 900
python verify_recordings.py . reference model validated
```

The verification script checks the policy names, Dream-1 checkpoint, run settings, source hashes and scores. It exports the original three synchronized camera streams and a setup image.

## Recorded trials

With seed 0 on a single RTX 5090, the scripted reference took 30.3 minutes of wall time for 890 actions and scored 1.0. Dream-1 took 24.7 minutes for 900 actions and scored 0.0.
