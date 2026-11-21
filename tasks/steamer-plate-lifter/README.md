# Lift a loaded plate out of a steamer

Use the plate lifter to take the loaded plate out of the steamer. Keep all three chicken drumsticks on the plate. Rest the plate on the cork trivet before releasing the rim, then lift the tool clear.

A plate resting inside a taller pot gives the fingers little room to reach beneath its rim. A spring-open lifter fits into the gap. Squeezing its handles brings a single hook and an opposing fork against the plate rim. The robot must keep them engaged during the transfer, then disengage them without picking up the plate a second time.

## Setup and reset

Use an open, room-temperature stainless steel steamer, a ceramic plate, a cork trivet, a spring-open metal plate lifter, and three washable rigid drumstick replicas. For repeatable loads, use sealed plastic replicas with internal ballast to match the listed masses. Heating and serving actual food are outside the task. Each replica moves independently and has a broad meat end and a narrow bone end.

| Item | Specification |
| --- | --- |
| Steamer | 200 mm nominal diameter, 60 mm rim height, central support 28 mm above the table |
| Plate | 156 mm outside diameter, 215 g, overhanging rim |
| Trivet | 110 mm diameter, 12 mm thick |
| Plate lifter | 80 g, about 185 × 62 × 110 mm open, with three metal hook tips on two crossed jaws and coated handles |
| Three drumsticks | Each 90 × 36 × 30 mm, with nominal masses of 55, 58 and 60 g |

In the calibrated YAM station's table coordinates, center the pot at (355, 95) mm and the trivet at (196, −58) mm. Lay the lifter open at (174, 217) mm with handles up. Center the plate on the pot support. Arrange the three replicas side by side with centers 37 mm apart, reversing the middle piece. Both hands start empty. Consult the setup photograph for orientation.

## Score a trial

Score 0 until the first milestone. Award partial credit for the furthest completed transfer stage. Full credit requires the final hold. Check the three food pieces throughout the transfer. The pot rim gives a visible clearance reference, and the circular trivet gives a placement reference.

| Score | Observable milestone |
| --- | --- |
| 0.15 | The lifter holds the plate on opposite sides of the rim and raises it more than 9 mm from the rack. |
| 0.40 | The lowest edge of the loaded plate clears the pot rim by more than 4 mm while still over the pot. |
| 0.65 | The lifter carries the loaded plate over the trivet, with its center within 30 mm of the trivet center and its bottom more than 14 mm above the trivet. |
| 1.00 | The trivet supports the plate before the lifter lets go. The plate then rests within 18 mm of the trivet center, within 4 mm of its surface and within 4° of level for one second. The tool is clear of the plate and all three drumsticks remain aboard. |

At the finish each hook is more than 20 mm beyond the plate edge or more than 20 mm above the rim. Neither jaw may touch the plate, and the plate must move slower than 15 mm/s. A dropped drumstick, direct finger contact with the plate, or a tilt of 90° or more after pickup voids the trial. Losing the opposing grip for more than 0.2 s while the plate is in midair also voids the trial. Releasing the plate above the trivet caps credit at the last completed milestone even if it lands level. The simulator checks jaw contact during pickup and contact with the upper surface of the trivet before release. A human reviewer can check the grip, clearance, landing and release from video with a ruler or angle reference. No force magnitude is graded.

## Run

Load the calibrated YAM station environment with Isaac Lab-Arena, the YAM Arena embodiment and Inspect Robots installed. Keep both Python helpers beside the task file. Use a fresh run tag for each trial.

```sh
bash run.sh demo01 video
bash run.sh dream01 dream
```

Both commands use the same scene, instruction, 20 Hz control and 720-action limit. The demonstrations and policy trials use a top camera and the two wrist cameras.

Both evidence runs used seed 0, fixed object positions, the calibrated station, Inspect Robots 1.0.0 and Isaac Lab-Arena 0.3.1. The YAM Arena checkout was `0059d02a1a91bc6834c1d0846022891df0c78730`, and the IsaacLab-Arena checkout was `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`. The recorded Dream policy checkpoint was `dreamlabs/dream-1-beta`. For remote inference, make sure the endpoint is available before starting the policy run.

| Run | Score | Actions | Simulated time | Wall time on RTX 5090 |
| --- | ---: | ---: | ---: | ---: |
| Scripted demonstration | 1.0 | 654 | 32.70 s | 234 s |
| Dream-1 | 0.0 | 720 | 36.00 s | 182 s |

The demonstration ends with all three drumsticks still on the level plate and the lifter clear. Dream-1 reached the action limit with the plate still inside the steamer. Each video shows one continuous trial with the top camera above the two wrist views.
