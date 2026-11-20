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

The fixed initial positions are in `fruit.json`. Both policies use seed 0, zero object jitter, one trial and the same 750-action budget. The native simulation runs at 240 Hz with 20 control actions per second. The maximum simulated time is 37.5 s. See `task.json` for the recorded scores. Run the controls with `python -m unittest -v test_produce_contact test_cut_checks test_end_state` from this directory in the YAM environment. These cover fruit contact, geometric cutting and end-state scoring.

Tested stack: Isaac Sim 6.1, Inspect Robots 1.0.0, YAM Arena 0.1.0, NumPy 2.3.1 and SciPy 1.17.0. The tested YAM repository revision is `0059d02a1a91bc6834c1d0846022891df0c78730` and the IsaacLab-Arena revision is `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.

### Reproduce the worktop

Suggested physical materials are given below. The simulator uses rigid collision geometry and appearance proxies. Positions are in millimeters. The task frame points right across the table (+x), away from the arms (+y) and up (+z). The center of the lower lid is at (71, 0), 300 mm forward of the line between the arm bases. The base midpoint is at (71, −300). Heights are above the worktop. Each row in `fruit.json` is an individual center in **meters** in that same frame.

| Item | Size, material and starting position |
| --- | --- |
| Workstation | Rigid worktop and two YAM arms with parallel grippers, bases 610 mm apart. Standard station ready pose. |
| Lower lid and platform | Rigid food-grade plastic tray, 170 × 154 mm, 4 mm floor, 3 mm thick perimeter rims rising 10 mm. Tray surface at z = 80. A washable insert locates each fruit with four short plastic ribs, each 16 × 3 × 8 mm, centered 14 mm to either side of its nominal center on each horizontal axis. Their tops lie 6 mm below the nominal cutting plane. Fixed plastic pedestal, 168 × 152 × 72 mm. |
| Tomatoes | Eight whole cherry tomatoes, each 28 mm across and 12 g in the model. Centers at z = 94, distributed as in the setup image. |
| Upper lid | Clear rigid plastic plate, 160 × 144 × 6 mm, 115 g including its grip. Grip 19 × 44 × 34 mm, 48 mm toward the left end. Its underside starts at (−117, 12, 110), on a 92 × 85 × 110 mm rigid plastic stand. |
| Knife | Simulated steel blade, 177 × 22 × 1.4 mm, joined to a plastic handle, 92 × 20 × 18 mm. Combined mass 85 g. The midpoint of the cutting edge starts at (71, −110, 94), along x. The handle points right. |
| Handle rests | Two rigid plastic blocks, 13 × 28 × 85 mm, centered at (174, −121) and (238, −121). |

### Prepare and reset a physical trial

Use eight fresh round red cherry tomatoes of the same cultivar and purchase lot per trial. Select uniformly red, intact, unwrinkled fruits without cracks, bruises or leaking juice. Measure two perpendicular equatorial diameters with light contact. Accept 27–29 mm in both directions. Record the lot and dimensions. The simulation's nominal mass is 12 g. Record physical masses without squeezing the fruit.

Rinse the fruit, remove loose stems without tearing the skin, and blot dry. Equilibrate fruit to 20–22 °C. Check the internal temperature of a separate same-lot fruit kept beside the test fruit. Use a fresh set for every attempt. Seat one fruit stem-scar up in each of the eight ribbed positions. Use the fruit.json coordinates when fabricating the insert. Photograph the loaded tray before starting.

The spill tray and pad covers are physical-rig accessories and are omitted from the rendering. Fit securely retained washable food-contact covers to the gripper pads and put a removable spill tray beneath the fixture. Confirm the covers allow normal gripping and leave joints clear. Grip the tool by its handle and the lid by its raised grip. With an inert blade replica, check clearance over the complete approach, cutting stroke and return, including the covered pads. Record the minimum clearance from the covered pads and the margin used for tracking error before authorizing a physical run. Stop a trial if a cover slips, juice reaches the rig or the blade enters a pad's clearance zone.

After each trial, park and secure the knife and disable robot motion. Remove the fruit and covers. Clean the knife, lid, ribbed insert, spill tray and covers under the site's approved food-contact cleaning and sanitation procedure. Pay particular attention to the rib corners and the inside of each cover. Allow the parts to dry. With the covers off, inspect both pads, the finger joints and the worktop. Use the manufacturer's approved cleaning method for exposed robot surfaces.

Before the next trial, check for visible juice, seeds and pulp. Wipe the dry pad faces and the corners of the insert with a fresh white wipe. If it picks up tomato residue or moisture, repeat cleaning and drying, then recheck. Fit clean, dry, undamaged covers. If juice has entered a joint or another inaccessible area, keep the rig out of use until the equipment owner has inspected and cleared it. This reset check detects visible residue. Microbial cleanliness has not been measured.


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

- Each qualifying edge traversal starts on one side of a fruit and exits the other. The entry and exit surfaces follow the chord at the measured height of each observation. The blade plane passes within 3.5 mm of the center, its usable span covers the chord, and the blade advances at least 14 mm. Moving fruit past a stationary knife earns zero.
- The lid is over the tray, within 15 mm horizontally and 8 mm vertically of its working position. The fruit center is below the lid footprint. The lid and blade are each within 12° of horizontal. Either gripper can be within 24 mm of the lid grip while the other is within 25 mm of the handle. These distances approximate tool engagement.
- A pass ends on disengagement, a retraction greater than 10 mm from maximum forward travel, or an edge-direction change greater than 15°. Each fruit counts once. Only cuts belonging to the selected pass contribute to the score.
- The entire mesh of a retained half lies in the tray envelope. The check allows 2 mm at the rim, 1.5 mm at the floor and a 36 mm height above the floor. Linear speed must be below 12 mm/s. The tool origins must return within 12 mm and 12° of their resets, with speed below 12 mm/s. Both grippers must be more than 40 mm from both grip points. The completion conditions must persist for eight checks (0.4 s) before the station's success hold begins.

### Why this setup is used

Cooks sometimes stabilize several cherry tomatoes between shallow lids when preparing a salad. The robot must control a shared lid while the blade reaches multiple independently moving pieces. The timer starts after loading, so the run measures the cutting operation alone. A cooking demonstration shows the human counterpart. [Oklahoma Gardening, May 18, 2024](https://www.pbs.org/video/oklahoma-gardening-may-18-2024-rih3s1/)

### What the model measures

The intact fruit, the kitchen tool and the two arms move as physical bodies. The synthetic fruit contact applies equal and opposite loads to the tool and fruit during physics integration, including the moment about each body's center. Load rises through a 2.5 mm skin region, then decays toward a flesh drag level. The nominal single-fruit skin peak is 1.6 N. A cherry-tomato slice initiation test reported roughly 1.35 and 1.94 N for two fresh test edges. The 0.35 N flesh level and transition distances are modeling assumptions. Transfer to the food and apparatus used in a physical trial requires a bench comparison. [Slicing fixture and results, Figures 4–5](https://cdn.imagearchive.com/kitchenknifeforums/data/attach/273/273117-The-Effect-Of-Edge-Aggression-Between-Steels.pdf)

Run `python single_fruit_fixture.py` in the same environment to reproduce the restrained-fruit PhysX control. In this control, the same 0.40 N drive and 0.50 s interval moved a free 85 g test carriage 24.52 mm in the blank trial and 1.89 mm against the modeled fruit. A 1.80 N drive traversed the fruit. This checks the dynamics of the force proxy. The values are simulation results.

A cut also requires at least 3 mm of relative draw along the blade and 4 mJ of positive contact work during the traversal. These simulator checks qualify a split. The observable physical score uses completed cuts, retained halves and returned tools. Once cut, a fruit yields two rigid halves. The model omits juice, crushing, tearing and blade drag on already-separated pieces. The physical reset protocol specifies splash protection and cleaning for the real rig.

### Recorded trials

| Policy | Score | Actions | Simulated time | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Scripted demonstration | 1.000 | 466 | 23.30 s | 117.5 s |
| Independent scripted repeat | 1.000 | 466 | 23.30 s | 234.3 s |
| Dream-1 | 0.000 | 750 | 37.50 s | 320.2 s |

Both scripted trials used seed 0. All eight fruits split in one pass by action 242. All sixteen halves remained in the tray and both tools were returned and released. The peak resultant contact load along the direction of travel was 3.45 N. Up to three intact fruits were in contact simultaneously. Dream-1 reached its action limit with zero cuts. These results apply to the fixed layout in simulation. No physical robot trial was performed.
