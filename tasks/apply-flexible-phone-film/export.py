"""Export the three synchronized observation cameras from one Inspect Robots run."""
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

log = Path(sys.argv[1])
target = Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
record = json.loads(log.read_text())
stats = record['stats']
fps = record['eval']['embodiment_info']['control_hz']
frames = Path(stats['frames_dir'])
by_cam = {name: {} for name in ('top', 'left', 'right')}
for f in frames.rglob('*.npy'):
    key = re.search(r'(top|left|right)_cam_(\d+)\.npy$',f.name)
    if key:
        camera, index = key.group(1), int(key.group(2))
        if index in by_cam[camera]:
            raise ValueError(f'Duplicate {camera} frame {index}')
        by_cam[camera][index] = f
indexes = sorted(by_cam['top'])
if not indexes or not all(sorted(stream)==indexes for stream in by_cam.values()):
    raise ValueError('Three synchronized camera streams are required')
if indexes != list(range(indexes[0],indexes[-1]+1)) or len(indexes) not in (stats['total_steps'],stats['total_steps']+1):
    raise ValueError(f'Video frame count {len(indexes)} versus {stats["total_steps"]} steps')
ff = imageio_ffmpeg.get_ffmpeg_exe()
temp = target.with_name(target.stem+'.encoding.mp4')
cmd = [ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x540',
       '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','19',
       '-movflags','+faststart',str(temp)]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for idx in indexes:
        views=[]
        for name in ('top','left','right'):
            rgb=np.load(by_cam[name][idx], allow_pickle=False)
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]<3:
                raise ValueError(f'Unexpected RGB buffer {rgb.shape} {rgb.dtype}')
            img=Image.fromarray(rgb[:,:,:3])
            if idx==indexes[min(5,len(indexes)-1)] and name=='top' and len(sys.argv)>3:
                img.save(sys.argv[3])
            views.append(np.asarray(img.resize((640,360) if name=='top' else (320,180),Image.Resampling.BILINEAR)))
        proc.stdin.write(np.vstack((views[0],np.hstack(views[1:]))).tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('Video encoder failed')
    if temp.stat().st_size > 50_000_000: raise ValueError('Video exceeds submission file size')
    temp.replace(target)
finally:
    if proc.poll() is None: proc.kill();proc.wait()
    if temp.exists(): temp.unlink()
print(target,len(indexes),'frames',fps,'fps',target.stat().st_size,'bytes')
