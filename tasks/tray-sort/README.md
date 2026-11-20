# Separate Takeout Scraps from Packaging

At a dining hall waste station, a diner empties leftover food into the compost bin and discards the packaging separately. A loose paper liner complicates the pour. Fries can catch in its folds, and tipping the box can send the liner into the compost along with the food.

## Reproduce a trial

Use Isaac Sim 6.1, Isaac Lab 3.0, Isaac Lab Arena 0.3.1 and Inspect Robots 1.0.0
with the configured bimanual YAM station (`yam_arena`). Work in this folder
and run the following commands from the simulator's Python environment.

```sh
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=352 --seed 0 \
  --instruction "$instruction" --scorer task_score --max-steps 3000 \
  --store-frames --log-dir logs/scripted
```

Set the policy to `dream` and use a separate log directory for a Dream-1 trial.
Use the default scene layout (variant 0) and seed 0 for both policies. The
maximum is 3,000 actions at 20 Hz, or 150 simulated seconds.
Keep the optional `TRAY_SORT_*` environment variables unset. The lid, box,
liner and food move through physical contact and gravity in the recordings.

Read `results.metrics.task_score` in the evaluation JSON. Run
`python test_scorer.py` for the CPU-only checks. The scripted policy uses the
simulator's measured box pose to set its final grasp. The Dream-1 policy runs
independently with the task instruction and the YAM camera observations.

## Recorded trials

The scripted demo scored 1.0 at step 1857 (92.85 simulated
seconds, about 14.0 minutes of wall time on an RTX 5090).
The recording shows the box and liner fully in the landfill bin after the
gripper withdraws. The food mass in compost is 112 g. The tabletop and
floor are clear of food at the end.

The Dream-1 recording is the original 3,000-step model trial from the same
scene and instruction. The container stayed closed and all food remained
inside it. That recorded final state scores 0.0 with the revised rule.

## Check the result

At the end of a physical trial, photograph both catch bins and the tabletop.
Open the box to check for food inside, and check both chutes and the floor.
Weigh the food in compost, the food in landfill, and any spills separately
with a kitchen scale. Count food left in either chute as spilled.
For the simulated pieces, the equivalent masses come from counting fries
at 4 g each and nuggets at 16 g each. The packaging checks are the
destination of each item and whether any food remains in the box. Packaging
points require at most 11.2 g of food in landfill. The liner
may go to landfill inside the empty box.

| Final outcome | Score |
| --- | ---: |
| All food in compost and both packaging items in landfill, with an empty box | 1.000 |
| One 4 g fry on the table, with all other items sorted correctly | 0.948 |
| All food and the empty box sorted correctly, with the liner left on the table | 0.800 |
| Food and liner in compost, with only the empty box in landfill | 0.150 |
| All contents and packaging in compost | 0.000 |
| Entire box with all food and liner in landfill | 0.000 |
| Food and liner sorted, but the box lid remains above the rim | 0.850 |
| One 4 g fry caught in the chute above the compost bin, with all other items sorted correctly | 0.948 |

The simulator counts food whose centers are between the floor and the
opening of a catch bin and within its walls. It checks each section of the liner
and both parts of the box against the catch bin. Packaging lodged in the
chute above the bin does not earn landfill credit. If any packaging enters
compost, only correctly deposited landfill items can earn points.
The full-credit result must hold for four seconds before the simulator ends.

## Reset the physical setup

Use the dimensions in `task.json`. Put the opening centers at
(x, y) = (32, +31) cm for landfill and (32, −31) cm for compost, where x
points forward and y points left from the robot frame origin. A laminate
or plywood deck with two rectangular cutouts over removable plastic bins
works as the waste station. Use a 24 × 17 cm sheet of deli paper. Center
a roughly 18.6 × 12 cm flat area on the box floor and fold the outer
2.5–2.7 cm against the walls. Fold any excess at the corners inward and
leave the sheet loose.

For repeatable physical trials, use dry reusable food props. Print the
fries as 65 × 9 × 9 mm bars and the nuggets as rounded 46 × 34 × 17 mm
pieces in matte PLA. Adjust infill to target 4 g per fry and 16 g per
nugget. Use the nominal mass per piece in the score when the props
vary in weight. The placement table gives
the nominal food placement for variant 0 before settling. Columns x and y
are offsets from the box center, z is the food center height above the
paper, and yaw is measured from the forward axis toward the left.
Use the height groups as layers when setting up real food. Then close the
lid and leave the box free to move.

| Piece | x (mm) | y (mm) | z (mm) | Yaw (°) |
| --- | ---: | ---: | ---: | ---: |
| Nugget 1 | -26.6 | 2.0 | 8.5 | 72.9 |
| Nugget 2 | 31.4 | -34.5 | 8.5 | 85.8 |
| Nugget 3 | -23.2 | -54.5 | 8.5 | 33.6 |
| Fry 1 | 12.5 | 27.5 | 4.5 | 85.8 |
| Fry 2 | -30.9 | 51.7 | 4.5 | 59.9 |
| Fry 3 | 30.9 | 34.8 | 4.5 | 75.6 |
| Fry 4 | 17.0 | -76.1 | 4.5 | 11.2 |
| Fry 5 | -21.2 | 36.2 | 14.0 | 171.0 |
| Fry 6 | 40.5 | 47.3 | 14.0 | 75.4 |
| Fry 7 | -7.2 | 62.7 | 14.0 | 2.1 |
| Fry 8 | 7.4 | -9.4 | 14.0 | 82.3 |
| Fry 9 | 6.4 | 75.2 | 14.0 | 178.9 |
| Fry 10 | -0.6 | 58.7 | 23.5 | 31.4 |
| Fry 11 | 6.3 | -40.4 | 23.5 | 14.1 |
| Fry 12 | 13.1 | 11.3 | 23.5 | 156.7 |
| Fry 13 | -19.3 | -4.7 | 23.5 | 168.7 |
| Fry 14 | -34.9 | -43.0 | 23.5 | 107.8 |
| Fry 15 | 20.9 | -58.8 | 23.5 | 178.0 |
| Fry 16 | 29.3 | 47.7 | 23.5 | 67.3 |

## Model

The container is modeled on a Dart 205HT1 foam clamshell. The task geometry
is built from primitive shapes. The two sign PNGs use original icons and
rendered DejaVu Sans Bold text under the [DejaVu font license](https://dejavu-fonts.github.io/License.html).
The YAM rig and station are provided by
the simulator environment. The recordings use the overhead camera and
the two wrist cameras. The model limitations for the liner, latch and
food are listed in `task.json`.
