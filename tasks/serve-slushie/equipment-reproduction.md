# Dispenser reproduction

The task includes a simulated dispensing station and a proposed gravity fixture for physical replication. The fixture exposes the same cup, spout and lever interfaces. The rendered cabinet and refrigerated hopper are visual context. The physical fixture uses prepared slush and an insulated hopper. The commercial VEVOR machine provides a second equipment option subject to the same interface qualification.

## Parts and cost

Supplier prices checked on 19 November 2026. One fixture uses one of each part. The lid listing requires a minimum order of 12. Prices exclude delivery and tax. Verify the delivered fitting dimensions before drilling.

| Component | Source | Listed price |
|---|---|---:|
| 2 qt polypropylene container, Choice 176RNDTR2 | [Container](https://www.webstaurantstore.com/choice-2-qt-translucent-round-polypropylene-food-storage-container-with-red-gradations/176RNDTR2.html) | US$2.79 |
| Matching 2/4 qt lid, Choice 176RNDTRL2 | [Lid](https://www.webstaurantstore.com/choice-2-and-4-qt-translucent-round-polypropylene-food-storage-container-lid/176RNDTRL2.html) | US$1.19 |
| Fermentap WL301 weldless kit, 1/2-inch full-port ball valve and hose barb, 304 stainless fittings | [Valve kit and installation documents](https://morebeer.com/products/stainless-weldless-ball-valve) | US$38.49 |

The per-fixture allocation is US$42.47. Buying one container, twelve lids at US$1.19 each and one valve kit costs US$55.56 before shipping and tax. Allow US$12–22 for compatible wet-path fittings, hose and insulation, US$8–18 for the lever, pivot, cable, stops and spring, and US$15–25 for the clamped frame, tray and stand. Including the lid minimum order gives US$90.56–120.56, budgeted as approximately US$95–125. The allowance assumes workshop fabrication and reused frame stock. Fabrication labour, tools, freezer and measurement equipment are additional. The selected components have individual supplier specifications; their assembled performance with slush requires qualification.

## Assembly datums

All coordinates below are millimetres in the simulation's horizontal station frame. Heights use the tabletop as zero. The tabletop has world z = 747.5 mm.

| Feature | x | y | Height above table |
|---|---:|---:|---:|
| Lever pivot | 370 | 0 | 302.5 |
| Spout tip | 385 | 0 | 247.5 |
| Tray support surface under cup | 385 | 0 | 112.5 |
| Pickup support centre | 300 | 200 | 140 |

Use a 110 mm tall cup with 93 mm rim and 58 mm base. Spout-to-tray distance is 135 mm and upright cup-rim clearance is 25 mm. Set interfaces within 2 mm and operating angles within 2°. Size the hopper shelf to the actual container base and clamp it independently of the lever bracket. Keep the hopper and support uprights behind the spout, clear of the arm and cup paths shown in the setup image.

The valve supplier specifies a 1-inch mounting hole for its weldless kit. Confirm a flat sealing land and full gasket contact on the selected container before drilling; reinforce the wall externally and support the valve body on its own bracket. Follow the supplied seal stack and installation drawing. Route the hopper outlet through the full-port valve, a downward elbow and a short removable nozzle. Match the supplied thread standard on all purchased adapters. Use the shortest practical wet path, inspect the actual minimum bore and record its dimensions. The 8/10/12 mm inserts are starting calibration options.

Vent the lid and insulate the reservoir. Prime the entire wet path. Mark 900 mL available inventory using the measured batch density and scale. Adjust the shelf until the measured liquid surface is 200 mm above the spout. Recheck after loading. For a 160 mm effective cylindrical bore, a 350 mL draw lowers the surface about 17.4 mm. Measure the actual container's level change because its walls may taper. Record both 900 mL and 550 mL levels.

## Lever fabrication and adjustment

Use a 150 mm metal strip with an M5 shoulder-bolt pivot in two rigid brackets. Fit a 24 mm diameter grip whose centre is 125 mm from the pivot. Align its long axis with the lever as shown in the USD asset. Set the released grip centreline 15° toward the robot from vertical and the commanded full-open position at 70°. The Isaac asset allows travel to 80°; the demonstrated policy commands 70°. Set physical working stops at 15° and 70°.

Attach an input sector with a 30 mm cable-centre radius. Set 2.618 mm of lost motion between 15° and 20°. From 20° to 70°, the input takes up 26.18 mm of cable. A valve sector with a 16.667 mm cable-centre radius converts this to 90° of valve travel. Clamp the sector concentrically to the valve stem or its existing arm, retain the valve stops and tension the cable with an adjustable anchor. Close the valve before setting the released stop. Check full opening before tightening the 70° stop.

Fit a return spring on the valve side so cable tension closes the valve and restores the lever. Adjust its anchor and preload while checking the release time and the effort at the grip. The simulation's spring and damping settings describe an assumed joint model. Spring selection and valve friction remain assembly-dependent. Test release from every calibration angle and repeat full-open release three times. A clamp-on slotted sector permits intermediate flow adjustment after measuring valve angle versus flow.

A constant cable ratio defines end travel. The valve's intermediate flow depends on its port geometry and the slush batch. Measure the four flowing angles listed in physical-trial.md, adjust the sector or cam take-up, then repeat both inventory levels. Retain the resulting angle-to-valve-travel table, dimensions and raw readings with the fixture revision. Complete this calibration before describing a physical rig as equivalent to the simulation.

## Commercial machine preparation

The VEVOR R1-RXR02(W)-N7-3-01 manual specifies liquid ingredients at 20 ± 5 °C initially. It describes a SLUSH program, a COOL holding state and rotating the handle to dispense. Follow its marked fill limits. The separate gravity-fixture recipe uses prechilling and external freezing. Record the chosen equipment and preparation route in every run. Qualify geometry, return and flow on the selected assembly. [Exact-model manual, pages 3 and 13–14](https://m.media-amazon.com/images/I/A18gGyGi04L.pdf)

## Reproduction evidence

The repository contains the scene USDs, runtime, scoring, controller commands and seed. README.md specifies the environment entry points. validation.json records the submitted simulation results and source hashes. The complete Isaac rerun reached 1.00 in 1,539 control steps, with 327.3 mL served and zero modeled spill. Its trace reached the closed-angle tolerance 0.35 s after the finger-opening command. This single complete run establishes the simulated sequence.

The simulated discharge law is linear in normalized lever opening, capped at 60 mL/s. Its homogeneous fluid model omits ice bridging and valve-specific rheology. Geometry and simulated flow checks can be reproduced on the workstation. The proposed hardware's flow curve, spring choice and crystal transport retain their qualification status until measured. Physical-trial.md specifies the recipe, flow bands, repeated trials, mass balance, protection and reset.
