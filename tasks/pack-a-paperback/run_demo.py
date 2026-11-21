"""Run the full scripted book-packing trial from its fixed initial state."""
import argparse
import json

import numpy as np
from pathlib import Path

from pack_world import PackWorld


class DemoWorld(PackWorld):
    def step(self, action, label=""):
        if self.steps >= 9000:
            raise RuntimeError("9,000-step trial budget exhausted")
        return super().step(action, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    world = DemoWorld(args.output, probe="pack_start", resolution=640)
    try:
        import pack_motion

        world.metadata.update(robot_demonstration=True, diagnostic_only=False,
                              scope="Continuous book packing from the fixed initial scene")
        (world.output / "metadata.json").write_text(json.dumps(world.metadata, indent=2))
        pack_motion.run(world, "square_book")
        with np.load(Path(__file__).with_name("closure_targets.npz"), allow_pickle=False) as teaching:
            actions, phases, labels = (teaching[k] for k in ("actions", "phases", "labels"))
        for action, phase in zip(actions, phases, strict=True):
            row = world.step(action, str(labels[phase]))
            if max(sum(f) for f in row["mean_panel_contact_force_n"].values()) > 15.:
                raise RuntimeError(f"Contact guard stopped closure at step {world.steps}")
            if max(max(f) for f in row["peak_panel_contact_force_n"].values()) > 40.:
                raise RuntimeError(f"Peak contact exceeded 40 N at step {world.steps}")
        print("Final score", world.last["score"], "at step", world.steps, flush=True)
        if not world.last["success"] or world.steps > 9000:
            raise RuntimeError("Full score was not reached within 9,000 steps")
    finally:
        world.close()


if __name__ == "__main__":
    main()
