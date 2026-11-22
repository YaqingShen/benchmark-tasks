"""Score visible wheel readings and the unsupported opening of the lid."""
from mechanism import CODE, UNLATCH_MM, indexed

class LockScore:
    def __init__(self):
        self.released = False
        self.invalid = False
        self.clear_frames = 0
        self.initialized = False

    def update(self, angles, travel_mm, gap_mm, lid_contact_n):
        matches = sum(indexed(a, d) for a, d in zip(angles, CODE))
        if not self.initialized:
            self.invalid = gap_mm > 2 or travel_mm > 2
            self.initialized = True
        if matches == 3 and travel_mm >= UNLATCH_MM:
            self.released = True
        if gap_mm > 5 and (not self.released or lid_contact_n > .25):
            self.invalid = True
        open_and_clear = self.released and gap_mm >= 20 and travel_mm <= 1 and lid_contact_n <= .25
        self.clear_frames = self.clear_frames + 1 if open_and_clear else 0
        if self.invalid:
            return 0.
        if self.clear_frames >= 11 and matches == 3:
            return 1.
        return .15 * matches
