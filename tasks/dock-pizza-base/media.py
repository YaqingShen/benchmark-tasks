"""Export the native camera triptych, setup and final hole map from one rollout."""
from pathlib import Path
import json, subprocess, sys, re
import numpy as np
from PIL import Image
r=Path(sys.argv[1])
groups=[]
for cam in ('top_cam','left_cam','right_cam'):
    groups.append({int(re.search(r'_(\d+)\.npy$',p.name)[1]):p
                   for p in (r/'capture/frames').rglob('*'+cam+'_*.npy')})
idx=sorted(groups[0])
assert idx and all(sorted(g)==idx for g in groups)
def read(p):
    return Image.fromarray(np.load(p)[...,:3].astype('uint8'))
w,h=read(groups[0][idx[0]]).size
out=(w, (h+h//2+1)//2*2)
with subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',
    f'{out[0]}x{out[1]}','-r','20','-i','-','-c:v','libx264','-pix_fmt','yuv420p',
    '-crf','17','-movflags','+faststart',str(r/'demo.mp4')],stdin=subprocess.PIPE) as proc:
    for i in idx:
        frame=Image.new('RGB',out)
        top=read(groups[0][i])
        frame.paste(top)
        for j in (1,2):
            frame.paste(read(groups[j][i]).resize((w//2,h//2),Image.Resampling.LANCZOS),((j-1)*w//2,h))
        proc.stdin.write(frame.tobytes())
        if i==idx[0]: top.save(r/'setup.png')
        if i==idx[-1]: frame.save(r/'final.png')
    proc.stdin.close()
    assert proc.wait()==0
p=next(p for p in (r/'capture').glob('adhoc_*.json') if '.live.' not in p.name)
doc=json.loads(p.read_text())
trace=[json.loads(l) for l in (r/'trace.jsonl').read_text().splitlines()]
(r/'summary.json').write_text(json.dumps(dict(status=doc['status'],metrics=doc['results']['metrics'],
    frames=len(idx),holes=max(t['holes'] for t in trace),coverage=max(t['coverage'] for t in trace),
    returned=any(t['returned'] for t in trace)),indent=2))
