"""Ordered, physically observed transfer from the clean hook to the right gripper."""
from dataclasses import dataclass
from math import radians
from contact_rules import owner

@dataclass
class Transfer:
    raised: bool = False
    captured: bool = False
    withdrawn: bool = False
    parked: bool = False
    pinch_frames: int = 0
    capture_frames: int = 0
    independent_grip: bool = False

    def observe(self, pairs, angle, material_open, tool_on_rest):
        ring_fingers=set()
        left_tool=hook_ring=right_tool=False
        for p in pairs:
            if p['max_force_n']<=.005:
                continue
            for actor,other in ((p['actor0'],p['actor1']),(p['actor1'],p['actor0'])):
                tip=actor.rsplit('/',1)[-1]
                if owner(actor)=='RightArm' and tip in ('tip_left','tip_right') and '/seal/pullring' in other:
                    ring_fingers.add(tip)
                if owner(other)=='hook':
                    left_tool |= owner(actor)=='LeftArm'
                    right_tool |= owner(actor)=='RightArm'
                    hook_ring |= '/seal/pullring' in actor
        pinched=len(ring_fingers)==2
        intact=not material_open
        old_raised,old_captured,old_withdrawn=self.raised,self.captured,self.withdrawn
        self.raised |= bool(intact and left_tool and hook_ring and abs(angle)>=radians(30))
        self.pinch_frames=self.pinch_frames+1 if old_raised and intact and pinched else 0
        self.independent_grip=bool(pinched and not right_tool)
        supported_capture=old_raised and intact and self.independent_grip and left_tool and hook_ring
        self.capture_frames=self.capture_frames+1 if supported_capture else 0
        self.captured |= self.capture_frames>=2
        self.withdrawn |= old_captured and intact and self.independent_grip and left_tool and not hook_ring
        self.parked |= bool(old_withdrawn and intact and pinched and tool_on_rest and not (left_tool or right_tool or hook_ring))
        return pinched
