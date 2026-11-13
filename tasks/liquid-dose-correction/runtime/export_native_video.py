"""Export only the three original camera streams, without overlays."""
import argparse,json,shutil,subprocess
from pathlib import Path

CAMERAS=('top_cam','left_cam','right_cam')

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:raise SystemExit('ffmpeg and ffprobe are required')
    meta=json.loads((a.run/'camera-recording.json').read_text())
    if tuple(meta['cameras'])!=CAMERAS or meta['overlays'] or meta['custom_cameras']:
        raise SystemExit('Expected only the original three cameras, without overlays')
    fps=meta['fps'];counts=[]
    for camera in CAMERAS:
        frames=sorted((a.run/'camera_frames'/camera).glob('frame_*.png'))
        if not frames or [f.name for f in frames]!=[f'frame_{i:05d}.png'for i in range(len(frames))]:
            raise SystemExit('Missing or nonconsecutive camera frames: '+camera)
        counts.append(len(frames))
    if len(set(counts))!=1:raise SystemExit('Camera streams differ in length')
    if a.output.exists():raise SystemExit('Use a new media output directory')
    a.output.mkdir(parents=True)
    streams=[]
    for camera in CAMERAS:
        dest=a.output/(camera+'.mp4');streams.append(dest)
        subprocess.run([ffmpeg,'-v','error','-framerate',str(fps),'-i',str(a.run/'camera_frames'/camera/'frame_%05d.png'),
            '-c:v','libx264','-pix_fmt','yuv420p','-crf','21','-movflags','+faststart',str(dest)],check=True)
    cmd=[ffmpeg,'-v','error']
    for stream in streams:cmd+=['-i',str(stream)]
    # Keep every camera's full field of view. Only resize the wrist images
    # proportionally to fit side by side under the overview.
    graph='[1:v]scale=iw/2:ih/2[l];[2:v]scale=iw/2:ih/2[r];[l][r]hstack=inputs=2[w];[0:v][w]vstack=inputs=2[out]'
    combined=a.output/'three-cameras.mp4'
    subprocess.run(cmd+['-filter_complex',graph,'-map','[out]','-c:v','libx264','-pix_fmt','yuv420p',
        '-crf','21','-movflags','+faststart',str(combined)],check=True)
    reports={}
    for f in streams+[combined]:
        info=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(f)]))
        video=[s for s in info['streams']if s['codec_type']=='video']
        if len(video)!=1 or video[0]['codec_name']!='h264' or video[0]['pix_fmt']!='yuv420p':raise RuntimeError('Unexpected video format')
        if f.stat().st_size>50_000_000:raise RuntimeError('Video exceeds the 50 MB upload limit')
        reports[f.name]={'bytes':f.stat().st_size,'duration_s':float(info['format']['duration']),
                        'width':video[0]['width'],'height':video[0]['height']}
    shutil.copyfile(a.run/'setup.png',a.output/'setup.png')
    (a.output/'provenance.json').write_text(json.dumps({**meta,'frame_count':counts[0],
        'video_files':reports,'cover_source':'unmodified native setup.png'},indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
