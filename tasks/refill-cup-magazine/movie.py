"""Encode the three original cameras on a canvas matching their native resolutions."""
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from PIL import Image

run=Path(sys.argv[1])
groups=[]
for cam in ("top_cam","left_cam","right_cam"):
    frames={int(re.search(r"_(\d+)\.npy$",path.name)[1]):path
            for path in (run/"capture/frames").rglob(f"*{cam}_*.npy")}
    if len(sys.argv)>2:
        frames={i:p for i,p in frames.items() if i<=int(sys.argv[2])}
    groups.append(frames)
indices=sorted(groups[0])
assert indices and all(sorted(g)==indices for g in groups)
assert indices==list(range(indices[0],indices[-1]+1))
def read(path):
    return Image.fromarray(np.load(path)[...,:3].astype("uint8"))
w,h=read(groups[0][indices[0]]).size
canvas_h=(h+h//2+1)//2*2
dest=run/("cup-transfer.mp4" if len(sys.argv)==2 else "cup-grasp.mp4")
with subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
    "-s",f"{w}x{canvas_h}","-r","20","-i","-","-c:v","libx264","-crf","16" if w>=1280 else "19",
    "-pix_fmt","yuv420p","-movflags","+faststart",str(dest)],stdin=subprocess.PIPE) as proc:
    for i in indices:
        frame=Image.new("RGB",(w,canvas_h))
        frame.paste(read(groups[0][i]),(0,0))
        for j in (1,2):
            im=read(groups[j][i]).resize((w//2,h//2),Image.Resampling.LANCZOS)
            frame.paste(im,((j-1)*w//2,h))
        proc.stdin.write(frame.tobytes())
        if i==indices[min(105,len(indices)-1)]:frame.save(run/"lift.png")
    proc.stdin.close()
    if proc.wait(): raise RuntimeError("ffmpeg failed")
print(len(indices),dest,flush=True)
