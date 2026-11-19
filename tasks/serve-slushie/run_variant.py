"""Reproduce a sensitivity trial in an isolated source copy."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CASES = {
    "low_inventory": (550, 60, .95, .8),
    "slow_flow": (900, 54, .95, .8),
    "fast_flow": (900, 66, .95, .8),
    "lower_friction": (900, 60, .5, .4),
    "combined": (550, 66, .5, .4),
}


def prepare(case, output):
    inventory, flow, static, dynamic = CASES[case]
    output.mkdir(parents=True, exist_ok=False)
    task = output / "task"
    task.mkdir()
    for source in HERE.glob("*.py"):
        shutil.copy2(source, task / source.name)
    shutil.copytree(HERE / "assets", task / "assets")
    shutil.copy2(HERE / "task.json", task / "task.json")
    changes = {
        "fill_model.py": [
            ("HOPPER_ML = 900.0", f"HOPPER_ML = {inventory}.0"),
            ("60*opening*dt", f"{flow}*opening*dt"),
        ],
        "slushie_runtime.py": [
            ("self.slush.emitted >= 900.", f"self.slush.emitted >= {inventory}."),
        ],
        "trial_setup.py": [
            ("CreateStaticFrictionAttr(.95)", f"CreateStaticFrictionAttr({static})"),
            ("CreateDynamicFrictionAttr(.80)", f"CreateDynamicFrictionAttr({dynamic})"),
        ],
    }
    for filename, replacements in changes.items():
        path = task / filename
        text = path.read_text()
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError(f"Source changed: inspect {filename} before reproducing this trial")
            text = text.replace(old, new)
        path.write_text(text)
    (output / "parameters.json").write_text(json.dumps({
        "case": case, "inventory_ml": inventory, "full_open_ml_s": flow,
        "static_friction": static, "dynamic_friction": dynamic,
        "seed": 0, "max_steps": 2400,
    }, indent=2) + "\n")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    task = prepare(args.case, output)
    if args.prepare_only:
        print(task)
        return
    env = os.environ.copy()
    env.update(SLUSH_TARGET=".66", SLUSH_GRASP_HEIGHT=".060", SLUSH_SQUEEZE=".016")
    subprocess.run([
        sys.executable, str(task / "run_replay.py"),
        "--output", str(output / "run"), "--max-steps", "2400",
    ], cwd=task, env=env, check=True)


if __name__ == "__main__":
    main()
