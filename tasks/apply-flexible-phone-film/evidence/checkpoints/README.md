# First-pass inspection

The native side stills were requested by the scoring callback on the first loaded pass. A trace step is zero-based. The three-camera video starts with reset at frame 0, so its corresponding frame number is one greater. A blank capture is omitted. Sidecar text files retain the actual trace step at capture.

| Station | Trace step | Video frame | Gap from simulated physical poses | What the original inspection image shows |
| --- | ---: | ---: | ---: | --- |
| 35 mm | 213 | 214 | 18.26 mm | The orange lip and the turnaround in the blue liner are distinguishable in the [side image](station-35mm-side.png). |
| 70 mm | 254 | 255 | 19.76 mm | The returned liner doubles back above the glass ahead of the lip in the [side image](station-70mm-side.png). |
| 105 mm | 295 | 296 | 24.42 mm | The support blocks the fold in the [side image](station-105mm-side.png). The arm partly obscures the [overhead image](station-105mm.png). |

A direct check of the first two side images places the centres of the card lip at x ≈ 360 and 574 px and the liner turnarounds at x ≈ 475 and 700 px. The visible 150 mm long edge of the dummy spans about 980 px in this view. The approximate projected gaps are 18 and 19 mm. The oblique depth of a picked edge changes the pixel scale slightly. The unrounded geometry values and the optical readings are reported separately.

The real-trial protocol uses rulers, synchronized recordings and the first loaded frame. It assigns zero for any checkpoint whose fold cannot be identified or whose gap cannot be established. A later frame of the same motion does not substitute for this frame.

## Opposite-side replay

A second 407-action run used one fixed camera above the opposite long edge of the phone. Its scored outcome was 1.0. The positions of every film section, liner section and both blade-edge endpoints matched the original trace at all three first passes. Only the diagnostic camera pose differed in the run manifests. The [position comparison](pose-comparison.json) lists the differences in millimetres. The three original policy cameras stayed fixed.

| Station | Opposite-side native image | Visible evidence |
| --- | --- | --- |
| 35 mm | [Frame 214](station-35mm-opposite.png) | The blue liner turns back just beside the paddle, ahead of the orange lip. |
| 70 mm | [Frame 255](station-70mm-opposite.png) | The lower blue return turns beneath the raised part of the sheet. |
| 105 mm | [Frame 296](station-105mm-opposite.png) | The wide blue liner ear and the orange card lip are both visible from this side. |

At the last station the near and far blue corners protrude beyond the film. The near lip centre is visible. At the far end the card covers part of its orange lip, leaving its outer edge visible. A hand reading of those pixels gives a centreline gap of about **24.5 mm**. The pose measurement is **24.42 mm**. Small changes in the picked edge matter at this margin.

The following calculation documents that reading from the diagnostic photograph. It does not use the trace. The image is 1200 × 720, the focal length is 4200 pixels, and the camera centre is at x = 70 mm, y = 680 mm relative to the target. With c = sqrt(1 − 0.24²), a pixel (u,v) in the lateral plane y maps to x = 70 − ((u−599.5)/4200) × (680−y)/(c−0.24×(v−359.5)/4200). Y is measured from the phone centreline towards the camera. The liner ear and card span 78 and 66 mm, respectively. The lip is 7 mm thick along the direction of travel.

| Landmark | Pixel (u,v) | Nominal lateral plane | Reconstructed x |
| --- | --- | ---: | ---: |
| Near end of liner fold | (247.5, 426) | +39 mm | 125.56 mm |
| Far end of liner fold | (289, 312) | −39 mm | 124.60 mm |
| Near centre of contacting lip | (410, 445) | +33 mm | 100.22 mm |
| Far outer lip edge | (442, 350) | −33 mm | 97.53 mm |

Add half the 7 mm lip thickness to the far outer edge to estimate its centre. Average near and far readings for each object, then subtract the lip centreline from the fold centreline. This gives 24.45 mm before rounding. The use of nominal widths and hand-picked edges limits the accuracy. For a real trial, use the physical reference rulers, observe the same first pass and retain the recorded views.
