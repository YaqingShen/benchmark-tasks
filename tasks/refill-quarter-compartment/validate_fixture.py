"""Validate the gravity-loaded wrapper against its matched intact control."""
import json
import math
import sys
from pathlib import Path

def check(rows):
    if len(rows) < 51:
        raise ValueError("Incomplete torque fixture recording")
    for row in rows:
        for name in ("light", "locked"):
            if row[name]["broken"] or not 0 <= row[name]["gap_m"] < .001:
                raise ValueError(f"{name} control opened")
        light, heavy, locked = (row[name] for name in ("light", "heavy", "locked"))
        if not (light["moment_Nm"] < light["limit_Nm"] == heavy["limit_Nm"] < heavy["moment_Nm"]):
            raise ValueError("Loads do not bracket the configured rupture moment")
        if not math.isclose(heavy["moment_Nm"], locked["moment_Nm"]):
            raise ValueError("Reference and rupture loads differ")
    if not all(row["heavy"]["broken"] and row["heavy"]["gap_m"] > .010 for row in rows[-5:]):
        raise ValueError("The loaded test joint failed to separate")

if __name__ == "__main__":
    check(json.loads(Path(sys.argv[1]).read_text()))
    print("Low load retained, high load ruptured, matched locked control retained.")
