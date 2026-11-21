import unittest
import numpy as np
from inspection import PHONE,TARGET,from_photo
from pockets import AirInterface

class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.photo=dict(phone_corners_px=PHONE*5+40,film_corners_px=TARGET*5+40,
            film_outline_px=TARGET*5+40,video_gaps_mm=[10.,12.,15.],
            first_press=True,bubble_outlines_px=[],liner_clear=True,card_clear=True)
    def result(self):return from_photo(self.photo)
    def test_complete(self):
        self.assertAlmostEqual(self.result()['score'],1.)
    def test_missing_pass(self):
        self.photo['video_gaps_mm'][1]=None
        self.assertAlmostEqual(self.result()['score'],1.-.25/3,places=5)
    def test_large_bubble(self):
        self.photo['bubble_outlines_px']=[np.array([(0,0),(3,0),(3,1),(0,1)])*5+40]
        self.assertEqual(self.result()['points']['finish'],0.)
    def test_many_small_bubbles(self):
        self.photo['bubble_outlines_px']=[(np.array([(0,0),(1,0),(1,1),(0,1)])+[v*3,0])*5+40 for v in range(10)]
        self.assertLess(self.result()['max_bubble_mm'],2)
        self.assertEqual(self.result()['points']['finish'],0.)
    def test_untouched_starter(self):
        self.photo['first_press']=False
        self.photo['video_gaps_mm']=[None]*3
        self.photo['liner_clear']=False
        self.photo['film_corners_px']+=np.array([30,0])
        self.assertEqual(self.result()['score'],0.)

    def test_misaligned(self):
        self.photo['film_corners_px']+=np.array([0,20])
        self.photo['film_outline_px']=self.photo['film_corners_px']
        self.assertEqual(self.result()['points']['alignment'],0.)
    def test_open_vent_and_closed_pocket(self):
        air=AirInterface()
        ring=np.zeros_like(air.sealed)
        ring[40:48,30]=ring[40:48,37]=True
        ring[40,30:38]=True
        self.assertFalse(air.touch(ring))
        ring[47,30:38]=True
        self.assertTrue(air.touch(ring))
        self.assertEqual(len(air.outlines),1)
        self.photo['bubble_outlines_px']=[5*p+40 for p in air.outlines]
        self.assertGreater(self.result()['max_bubble_mm'],2)

    def test_coverage_uses_union(self):
        half = np.array([(0,0),(70,0),(70,64),(0,64)])*5+40
        self.photo['flat_outlines_px']=[half,half]
        self.assertAlmostEqual(self.result()['coverage'],.5)
        self.assertEqual(self.result()['points']['finish'],0.)

    def test_liner_on_phone_or_card_down(self):
        for key in ('liner_clear','card_clear'):
            self.photo[key]=False
            self.assertEqual(self.result()['points']['finish'],0.)
            self.photo[key]=True

    def test_bench_record(self):
        from bench import measure_trial
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            # Fixture names identify this as a software test only.
            names=('overhead_video','side_video','final_photo_left_light','final_photo_right_light')
            from PIL import Image
            import imageio_ffmpeg
            import subprocess
            names_files={k:k+('.mp4' if 'video' in k else '.png') for k in names}
            for key,filename in names_files.items():
                if 'video' in key:
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-f','lavfi','-i',
                                    'color=c=blue:s=64x64:r=20:d=3','-c:v','libx264',str(folder/filename)],check=True)
                else:
                    Image.new('RGB',(1000,700),'white').save(folder/filename)
            d=dict(self.photo,**names_files,task='TK-VM7GXE',trial_id='synthetic-test',
                   first_press_frame=0,photo_width_px=1000,photo_height_px=700,video_hz=20,ruler_check_px=[[40,40],[290,40]],first_pass=[
                    dict(frame=j*20,blade_mm=v,fold_mm=v+12,card_on_exposed_film=True,fold_visible=True)
                    for j,v in enumerate((35,70,105))],clear_frames=10,minimum_card_clearance_mm=15)
            result=measure_trial(d,folder)
            self.assertEqual(result['photographic_score']['score'],1.)
            self.assertEqual(result['force_trials'],[])
            d['film_corners_px']=None
            partial=measure_trial(d,folder)['photographic_score']
            self.assertEqual(partial['score'],.8)
            self.assertIsNone(partial['corner_error_mm'])
            d['film_corners_px']=self.photo['film_corners_px']
            d['first_press_frame']=41
            with self.assertRaises(ValueError):measure_trial(d,folder)
            d['first_press_frame']=0
            d['first_pass'][-1]['frame']=1000
            with self.assertRaises(ValueError):measure_trial(d,folder)
            d['first_pass'][-1]['frame']=40
            d['ruler_check_px']=[[40,40],[300,40]]
            with self.assertRaises(ValueError):measure_trial(d,folder)

if __name__=='__main__':unittest.main()

