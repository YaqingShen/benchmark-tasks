"""Run the frozen physical demonstration at an explicit world seed."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous evidence; use a new output path")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source mismatch: " + name)
    lock = open("/tmp/robocurve-shared-gpu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    if min(map(float, free.splitlines())) < 16000:
        raise RuntimeError("Insufficient free VRAM; other jobs untouched")
    from charging_world import ChargingWorld
    from scripted_charging import demonstrate
    world = None
    try:
        world = ChargingWorld(root, args.output, seed=args.seed)
        demonstrate(world)
        print(json.dumps(world.score()), flush=True)
    finally:
        if world is not None:
            world.close()


if __name__ == "__main__":
    main()
