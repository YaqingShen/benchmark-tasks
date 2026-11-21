"""Evaluate Dream-1 on the book-packing scene with the shared end-state scorer."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from inspect_robots import Observation, Scene, StepResult, Task, eval as evaluate
from inspect_robots.cli import _build_guardrails
from inspect_robots_dream.policy import DreamPolicy
from yam_arena.inspect_embodiment import YamArenaEmbodiment
from yam_arena.scoring import task_score

from pack_world import PackWorld

RECORDED_INSTRUCTION = ("Pack the paperback in the open box. Put the foam pad in first, square the "
               "book on top, press it down, fold the flaps and tuck the lid closed. "
               "Withdraw both hands and leave the box closed.")


class BookStation(YamArenaEmbodiment):
    def __init__(self, output):
        super().__init__(scene="station", control_hz=20, cam_width=640, cam_height=360,
                         object_jitter=0, device="cpu")
        self.info = replace(self.info, environment_id="book-package-full41",
                            supported_target_kinds=frozenset())
        self.output = output
        self.world = None

    def observe(self):
        images = self.world.images()
        time_s = self.world.steps / self.info.control_hz
        return Observation(images=images, state={"joint_pos": self.world.joints()},
                           instruction=self._instruction, image_times=dict.fromkeys(images, time_s),
                           state_time=time_s)

    def reset(self, scene, *, seed=None):
        self._instruction = scene.instruction
        self.world = PackWorld(self.output / "world", probe="pack_start", resolution=640)
        for _ in range(3):
            self.world.base.sim.render()
        return self.observe()

    def step(self, action):
        target = np.clip(action.data, self._low, self._high)
        row = self.world.step(target, "dream")
        done = bool(row["success"])
        return StepResult(observation=self.observe(), terminated=done,
                          termination_reason="success" if done else None,
                          info={"success": done, "score": float(row["score"])})

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recorded-instruction", action="store_true",
                        help="Use the exact instruction from the archived Dream-1 trial.")
    args = parser.parse_args()
    instruction = (RECORDED_INSTRUCTION if args.recorded_instruction else
                   json.loads(Path(__file__).with_name("task.json").read_text())["instruction"])
    args.output.mkdir(exist_ok=False)
    body = BookStation(args.output)
    policy = DreamPolicy()
    guard, names, warnings = _build_guardrails(body.info.action_space, None, body)
    print("Inspect guards:", names, warnings, flush=True)
    task = Task(name="book-package", scenes=[Scene("pack_start", instruction)],
                scorer=task_score(), max_steps=9000)
    try:
        log, = evaluate(task, policy, body, guard=guard, log_dir=str(args.output / "inspect"),
                        seed=0, store_frames=True)
        print("DREAM_RESULT", log.status, log.results.metrics, log.error, flush=True)
        if log.status != "success" or log.error:
            raise RuntimeError("Inspect did not complete the trial")
    finally:
        (args.output / "policy-transcript.json").write_text(json.dumps(policy.transcript()))
        try:
            policy.close()
        finally:
            body.close()


if __name__ == "__main__":
    main()

