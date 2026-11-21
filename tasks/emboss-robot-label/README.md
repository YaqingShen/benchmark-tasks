# Emboss ROBOT and cut the strip

Turn the wheel to each letter, press and release for each character in ROBOT, feed two spaces, then cut the label and release the handle.

## Reproduce the setup

Use a DYMO Office-Mate II 1540-00 with its factory horizontal character wheel, cutter and DYMO 15809 black half-inch plastic tape. A soft-jaw bench vise holds the fixed housing. Level the dial, set its center 135 mm above the bench at (300, 0) mm, and point the reading index along station +X. Shim under the fixed housing as needed. Leave room for both fingers at the wheel, the entire moving handle and the emerging label. Cycle the trigger and turn the wheel once to check clearance. Load blank tape, set A at the index and release the lever before every trial. The mount height is a fixture setting and physical fit still needs confirmation. [Seattle City Light material standard, section 6](https://web8.seattle.gov/city-light-engineering-standards/Home/GetPDF?libId=BrowseAll&searchForfileName=7658.51.pdf)

## What the recorded simulation implements

Both recordings use the same 76 mm, 49-stop prototype dial. Its symbol order is in omega_manual.json. The script physically indexes the wheel and presses the trigger. Five letters and two blank feeds advance the tape, and one squeeze at the scissors setting separates the displayed strip. Marking, feed and cutting are discrete material responses. The exact retail detent layout and cutter geometry remain unverified.

The modeled feed is 5.9 mm per completed advance. Two photos of one documented Office-Mate II strip yielded 5.85–5.91 mm per feed. The scale comes from four mounting holes in the maker’s faceplate drawing; the D-to-D and E-to-E intervals span 21 and 18 feeds. The maker did not specify the stock width for this specimen. Run `python evidence/officemate_photo_pitch.py` with NumPy to reproduce the calculation. The simulation uses 12.7 mm tape and a 4.8 mm nominal glyph height. [Specimen and photos](https://markus-wobisch.blogspot.com/2018/12/building-bttf-flux-capacitor.html) · [Faceplate CAD](https://github.com/mawob/BackToTheFuture/blob/master/FluxCapacitor/LaserCutter/flux-front-plus.dxf)

The historical GSA description A-A-211B, §2.12, specifies operating-handle ceilings of 8 N for embossing and 9 N for cutting and scoring. It covers 12.7 and 19.0 mm tape and was canceled in 1999. We tested those two loads as full-stroke spring equivalents. With a 42 mm perpendicular moment arm and 0.30 rad travel, the corresponding torsional stiffnesses are 1.12 and 1.26 N·m/rad. Blank feed retains the prototype stiffness of 0.85 N·m/rad. The press event occurs at 0.255 rad. The simulated values exclude inertia, damping and end-stop reaction. The specification gives neither a contact location nor measured peaks for this retail unit. [GSA specification and cancellation record](https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=44636)

## Run and score

In the Isaac Lab-Arena / Inspect Robots YAM environment described in environment.json, run from this directory. The explicit scenario flag selects the profile used in both submitted videos.

```sh
python -m unittest test_omega
EMBOSS_OFFICEMATE_SCENARIO=1 python test_mechanism.py
python run_task.py --officemate-scenario --policy yam_scripted --camera-width 640 --output runs/demo
python run_task.py --officemate-scenario --policy dream --camera-width 640 --output runs/dream
```

Use a fresh output directory and seed 0. Dream-1 requires configured model access. The limit is 2,400 actions at 20 Hz. The demo scored 1.0 in 81.1 s. Dream-1 scored 0.0 in 120 s. The three views show the same continuous trial. Review the strip, cut count and final release against the video. task.json gives the scoring rules. The evidence directory contains the two replay audits and the scored Python source hashes.
