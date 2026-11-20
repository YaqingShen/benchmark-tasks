# Halve eight tomatoes in one pass

Halve all eight cherry tomatoes in one pass. Leave all sixteen halves in the tray and both tools on their rests.

### Run a trial

Use the native Isaac YAM station with two parallel grippers and its overhead, left-wrist and right-wrist cameras. The controller receives the three RGB views and the station's normal proprioception. The scripted policy uses known fixture coordinates. Dream-1 is evaluated separately from observations. No scripted waypoints or fruit poses are sent to Dream-1.

Activate the YAM Arena environment. From this task directory, run the block below once per policy. Set `POLICY=dream` for Dream-1 or `POLICY=yam_scripted` for the demonstration. Dream-1 needs access to the configured policy service.

```bash
POLICY=dream
RUN="$(mktemp -d)"
export TOMATO_RUN_DIR="$RUN"
inspect-robots run --embodiment yam_arena --policy "$POLICY" \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=360 --seed 0 \
  --instruction 'Halve all eight cherry tomatoes in one pass. Leave all sixteen halves in the tray and both tools on their rests.' \
  --scorer task_score --max-steps 750 --store-frames --log-dir "$RUN/capture"
```

The fixed initial positions are in `fruit.json`. Both policies use seed 0, zero object jitter, one trial and the same 750-action budget. The native simulation runs at 240 Hz with 20 control actions per second. The maximum simulated time is 37.5 s. See `task.json` for the recorded scores. Run all 16 controls with `python -m unittest -v test_cut_checks test_end_state` from this directory in the YAM environment. Seven exercise geometric cutting and nine call the actual end-state scoring callback with controlled scene readings.

Tested stack: Isaac Sim 6.1, Inspect Robots 1.0.0, YAM Arena 0.1.0, NumPy 2.3.1 and SciPy 1.17.0. The tested YAM repository revision is `0059d02a1a91bc6834c1d0846022891df0c78730` and the IsaacLab-Arena revision is `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.

### Reproduce the worktop

Suggested physical materials are given below. The simulator uses rigid collision geometry and appearance proxies. Positions are in millimeters. The task frame points right across the table (+x), away from the arms (+y) and up (+z). The center of the lower lid is at (71, 0), 300 mm forward of the line between the arm bases. The base midpoint is at (71, −300). Heights are above the worktop. Each row in `fruit.json` is an individual center in **meters** in that same frame.

| Item | Size, material and starting position |
| --- | --- |
| Workstation | Rigid worktop and two YAM arms with parallel grippers, bases 610 mm apart. Standard station ready pose. |
| Lower lid and platform | Rigid food-grade plastic tray, 170 × 154 mm, 4 mm floor, 3 mm thick perimeter rims rising 10 mm. Tray surface at z = 80. Fixed plastic pedestal, 168 × 152 × 72 mm. |
| Tomatoes | Eight whole cherry tomatoes, each 28 mm across and 12 g in the model. Centers at z = 94, distributed as in the setup image. |
| Upper lid | Clear rigid plastic plate, 160 × 144 × 6 mm, 115 g including its grip. Grip 19 × 44 × 34 mm, 48 mm toward the left end. Its underside starts at (−117, 12, 110), on a 92 × 85 × 110 mm rigid plastic stand. |
| Knife | Simulated steel blade, 177 × 22 × 1.4 mm, joined to a plastic handle, 92 × 20 × 18 mm. Combined mass 85 g. The midpoint of the cutting edge starts at (71, −110, 94), along x. The handle points right. |
| Handle rests | Two rigid plastic blocks, 13 × 28 × 85 mm, centered at (174, −121) and (238, −121). |

### Score the result

Score the largest batch cut during one continuous pass. **C** is the number of distinct tomatoes that pass cut through near their centers. **H** counts only those tomatoes’ halves which are entirely in the tray and resting at the end. If two passes tie on C, choose the one with more retained halves. The score is **0.0875 C + 0.0125 H**, plus **0.10** when all eight are cut together, every half is retained, both tools are back on their rests and both grippers have withdrawn.

| Outcome | Score |
| --- | ---: |
| No cut, tools returned | 0.0000 |
| Eight passes with one tomato per pass | 0.1125 |
| Two passes with four tomatoes per pass | 0.4500 |
| All eight in one pass, one half lost | 0.8875 |
| All eight in one pass, tools still held | 0.9000 |
| All eight in one pass, complete finish | 1.0000 |

For a physical trial, film the tray from above and from the side with a millimeter scale and marks for the two resting tools. Count separate pieces at the end. If two halves touch, gently separate them for inspection **after** freezing the trial score. An attached skin bridge counts as incomplete. A 28 mm tomato cut within 3.5 mm of its center has two pieces approximately 10.5–17.5 mm thick at their thickest points. Equal weight is unnecessary. Use the recording to distinguish one pass from separate attempts and verify tool return. Preparing and clearing the tray are outside the timed interval.

The simulator uses the following observable proxies:

- Each qualifying edge traversal starts on one side of a fruit and exits the other. The blade plane passes within 3.5 mm of the center, its usable span covers the chord, and the blade advances at least 14 mm. Moving fruit past a stationary knife earns zero.
- The lid is over the tray, within 15 mm horizontally and 8 mm vertically of its working position. The fruit center is below the lid footprint. The lid and blade are each within 12° of horizontal. Either gripper can be within 24 mm of the lid grip while the other is within 25 mm of the handle. These distances approximate tool engagement.
- A pass ends on disengagement, a retraction greater than 10 mm from maximum forward travel, or an edge-direction change greater than 15°. Each fruit counts once. Only cuts belonging to the selected pass contribute to the score.
- The entire mesh of a retained half lies in the tray envelope. The check allows 2 mm at the rim, 1.5 mm at the floor and a 36 mm height above the floor. Linear speed must be below 12 mm/s. The tool origins must return within 12 mm and 12° of their resets, with speed below 12 mm/s. Both grippers must be more than 40 mm from both grip points. The completion conditions must persist for eight checks (0.4 s) before the station's success hold begins.

### Why this setup is used

Cooks sometimes stabilize several cherry tomatoes between shallow lids when preparing a salad. The robot must control a shared lid while the blade reaches multiple independently moving pieces. The timer starts after loading, so the run measures the cutting operation alone. A cooking demonstration shows the human counterpart. [Oklahoma Gardening, May 18, 2024](https://www.pbs.org/video/oklahoma-gardening-may-18-2024-rih3s1/)

### What the model measures

A completed geometric traversal exchanges a sphere for two six-gram hemispheres in the same location, with the source body's velocity. Inactive shapes are stored below the worktop. Only halves linked to an actual cut can earn points. Independent contact lets halves roll, settle or leave the tray. Blade–fruit collision forces are omitted, so this benchmark measures bimanual positioning and batch containment. It cannot assess blade sharpness, realistic slicing forces or damage to soft produce.

One fixed layout is used for reproduction. The native scores establish feasibility on that layout. Physical transfer, labor savings and success across other fruit sizes remain unmeasured.

### Recorded trials

| Policy | Score | Actions | Simulated time | Measured wall time |
| --- | ---: | ---: | ---: | ---: |
| Dream-1 | 0.000 | 750 | 37.50 s | 216.9 s |
| Scripted demonstration | 1.000 | 467 | 23.35 s | 114.1 s |

Wall time includes simulation and inference in the evaluation process. Queue waiting time is separate.
