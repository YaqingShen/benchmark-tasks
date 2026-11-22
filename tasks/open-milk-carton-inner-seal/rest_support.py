"""Track verified support on both posts while the native solver puts the hook to sleep."""
import numpy as np


class HandleSupport:
    def __init__(self):
        self.reference=None

    def observe(self, pairs, corners_in_rest):
        posts=set()
        for pair in pairs:
            if pair['max_force_n']<=.005:
                continue
            for a,b in ((0,1),(1,0)):
                handle=pair[f'collider{a}']
                post=pair[f'collider{b}']
                if handle.endswith('/hook/body/box_0') and '/tool_mat/body/' in post:
                    posts.add(post.rsplit('/',1)[-1])
        corners=np.asarray(corners_in_rest)
        if {'box_1','box_2'}<=posts:
            self.reference=corners.copy()
        elif self.reference is not None and np.max(np.linalg.norm(corners-self.reference,axis=1))>.0006:
            self.reference=None
        return self.reference is not None
