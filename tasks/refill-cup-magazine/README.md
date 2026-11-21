# Refill a hot-drink cup dispenser

Load all eight paper cups into the hot-drink dispenser without touching the cups with the grippers. Leave the wrapper outside the magazine so one service request dispenses one cup.

Operators load the wrapped stack so they can guide the cups without touching them. The Cafforia manual (p. 21) describes opening the sleeve at the bottom, inserting the wrapped cups and removing the sleeve. It warns that overfilling can jam the turret motor. CPI’s service video shows a technician loading a sleeve at 1:06, the fill-height check at 1:21, and the cup-drop test at 1:50. [Cafforia operator guide](https://rc.cranems.com/Uploads/AP_%20Cafforia_942_943_944_Operator%27s_Guide.pdf#page=29) · [Load the sleeve](https://www.youtube.com/watch?v=Pr2m796q1Ig&t=66s) · [Check the fill height](https://www.youtube.com/watch?v=Pr2m796q1Ig&t=81s) · [Test the cup drop](https://www.youtube.com/watch?v=Pr2m796q1Ig&t=110s)

KM Plast supplies the 8 oz vending cup used for the model dimensions in bags of 50. The UK vending census reports an average of 272 drinks per freestanding hot-drink machine per week in 2025 (p. 20). At one supplied cup per drink, that volume uses 272 ÷ 50 = 5.44 sleeves per week. An operator can load several sleeves in one visit. Actual demand depends on the site and the use of personal cups. [Cup supplier](https://kmplast.com/public/unprinted-1) · [UK vending census](https://www.the-ava.com/downloads/2025AVACensusandMarketReport.pdf)

For comparison, Cafforia lists total machine capacities of about 600 cups at 12 oz and 358 at 20 oz (p. 1). At the same illustrative demand, a full machine would hold about 15 or 9 days of cups respectively. These estimates combine manufacturer capacities with a fleet average. The benchmark uses one shortened 8 oz stack and an accessible single column. [Cafforia capacity table](https://rc.cranems.com/Uploads/AP_%20Cafforia_942_943_944_Operator%27s_Guide.pdf#page=9)

## Run and score

Use the calibrated YAM station at commit `0059d02a1a91bc6834c1d0846022891df0c78730`. The tested environment used Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, Isaac Lab Arena 0.3.1, yam-arena 0.1.0, Inspect Robots 1.0.0 and inspect-robots-dream 0.1.0. Install the YAM environment and configure access to Dream-1 following the [station instructions](https://github.com/kairosiann/yam-isaac-lab-arena/tree/0059d02a1a91bc6834c1d0846022891df0c78730). NumPy, SciPy, Pillow and FFmpeg are used. Run these commands from this task folder with a fresh run name each time.

```bash
source ~/IsaacLab-Arena/yam.env
CUP_WIDTH=1920 CUP_HEIGHT=1080 bash run.sh demo
python validate_result.py runs/demo --full
python movie.py runs/demo
CUP_POLICY=dream CUP_WIDTH=640 CUP_HEIGHT=360 bash run.sh dream
python validate_result.py runs/dream
python movie.py runs/dream
```

The reference controller defaults are recorded in `run.sh`. The policy and scorer both use seed 0, a fixed layout, 20 Hz control and a limit of 1,200 actions. Isaac advances at 240 Hz. The official result is `results.metrics.task_score` in the completed Inspect Robots report under `runs/<name>/capture/`. The three-camera movie plays in simulation time at 20 frames per second.

| Run | Policy | Score | Control steps | Wall time on RTX 5090 |
|---|---|---:|---:|---:|
| Demo | Scripted YAM joint targets | 1.0 | 678 | 9.0 min |
| Model | Dream-1 (`dreamlabs/dream-1-beta`) | 0.0 | 1200 | 13.3 min |

## Physical setup and reset

Use one accessible service column with the cover already open. The origin is the midpoint of the arm base line projected onto the tabletop. Positive x points forward and positive y points left. Positions below are in millimetres. Photograph the start from above to reproduce the layout.

| Item | Specification and starting position |
|---|---|
| Worktop and arms | Rigid 1,100 × 1,300 × 60 mm station worktop, with two calibrated YAM arms and parallel grippers in their standard ready pose. The 206 × 230 mm opening is centred at (220, 0) above the delivery tray. |
| Loading dock | Rigid plastic square, 110 × 110 mm and 10.2 mm high, centred at (98, 28). |
| Paper cups | Eight identical PE-lined paper vending cups, open side up. Each is 91.31 mm tall, with an 80.00 mm rim, 53.55 mm base and 5.87 g empty mass. Nest them at a 6.3 mm pitch. The bottom cup stands on the dock and the stack is about 135 mm tall. |
| Sleeve | Clear LDPE film, open at the bottom. The simulated sleeve is 87 mm across and about 152 mm tall to its top sheet, with a 137 × 24 mm folded seal extending along y. Start with all eight cups inside. |
| Dispenser | A dry, fixed service fixture with metal guides and separator ledges in a rigid housing. The guide rods are 142 mm long on a 48 mm radius. The separator sits 20 mm below tabletop height and the orange fill line sits 50 mm above it. Start with the lower ledges supporting cups and the upper ledges withdrawn. |
| Delivery station | A 158 × 182 mm slatted metal tray beneath the column. Its landing surface is 225 mm below the separator. The drop sensor crosses the path 45 mm above the tray. Start empty. |

For a physical reset, remove every cup from the column and tray. Use eight matching, undamaged cups from one lot with an intact top seal and the sleeve bottom already open. Set the column to its supported loading position and reset the separator. Use the dry cup-test function with the heater and water delivery inactive. Keep the test cups out of service after a robot trial.

## Score

0 if a gripper pad touches exposed paper or the eight cups never seat upright below the fill line. 0.25 once the wrapped stack reaches the column and all eight cups seat. 0.5 when the complete wrapper is clear and released for 0.5 s with the stack still seated. One dry service cycle then runs automatically. 1.0 if exactly one cup stands on the tray and seven stay upright in the magazine for 0.5 s after the cycle. The dispensed rim must be at least 20 mm below the lowest retained cup base, with the wrapper still clear and released.

The milestones are cumulative unless a gripper pad directly contacts a cup, which sets the score to zero. In a physical trial, use overhead and side views to check contact, the fill line and wrapper clearance. Count the cups on the tray and in the magazine after one dry test. The model uses native contact measurements as a proxy for visible direct pad contact. Its 20 Hz samples cannot certify food hygiene.

For numerical comparison, the wrapper and all eight cup centres must first reach within 18 mm of the magazine axis. A seated stack has all eight centres within 11 mm of the magazine axis. Each rim height is within 4 mm of its nesting position and the highest point is below the fill mark. The upright threshold for all cup checks is within 11.5° of vertical. The sleeve must clear the 56.5 mm outer radius of the top ring in overhead projection and have no measured pad contact for ten consecutive samples before the service cycle. The two pairs of physical ledges complete one cycle in 2.5 s. The beam must detect the dropping cup during the cycle or within 3 s afterward. Its base must stand within 3 mm of tray height and its centre within 26 mm of the column axis. All seven retained centres must be within 13 mm of the axis, with rims below the fill mark. The delivered rim must be at least 20 mm below the lowest retained cup base so the cup can be taken out. The result must persist for ten samples after the cycle. In the recorded full-score trial, the measured takeout gap was 44.6 mm at the end of the trial.

The recorded sleeve and cup displacements and a physical comparison procedure are in [Cup and dispenser parameters](PARAMETERS.md).

## Fidelity, assets and sources

The full-size cups are separate rigid bodies. They nest by contact and drop under gravity. The deeper collection bay lets the dispensed cup clear the stack. The sleeve uses four hinged panels and a folded seal. The single service column uses two pairs of moving ledges and a virtual drop sensor. Paper crushing, film wrinkling, heat, drinks, cabinet doors and turret indexing are outside this trial. The scripted reference tracks the simulated seal pose. Dream-1 receives the standard station observations. The eight-cup pack and shortened column make one refill observable in a single trial. The nesting pitch, friction, film hinge stiffness, base support profile and workcell clearances are simulation assumptions. The example does not establish compatibility with a particular commercial machine. The reference controller reads seal pose to guide ordinary gripper actions. Object motion and separation use contacts, gravity and the machine's actuated ledges.

All cup, package, dock, guide, tray and separator geometry is generated by the included scene code. The installed YAM station supplies its arm models and original three cameras. I2RT Robotics licenses the arm assets under the [MIT license](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/yam_arena/assets/yam/LICENSE.i2rt). The simulation uses [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena) under Apache 2.0 and the installed NVIDIA Isaac Sim runtime under its [software terms](https://docs.isaacsim.omniverse.nvidia.com/latest/common/legal.html). No external object meshes or manufacturer artwork are distributed in the task folder.

The loading procedure follows the Cafforia operator guide, p. 21. The cup dimensions and mass come from the KM Plast 8 oz vending cup specification. [Operator guide](https://rc.cranems.com/Uploads/AP_%20Cafforia_942_943_944_Operator%27s_Guide.pdf) · [Cup specification](https://kmplast.com/public/uploads/attachments/67a73712650ba.pdf)
