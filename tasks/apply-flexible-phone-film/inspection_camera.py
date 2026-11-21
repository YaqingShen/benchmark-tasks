"""An additional native Isaac RTX sensor for inspection stills only."""
from pathlib import Path
import numpy as np


def register_native_camera(labkit, x, y, glass_z):
    """Register after the Kit app has started, before the sensor scene is built."""
    def attach(module):
        cls = module.BimanualYamEmbodiment
        if getattr(cls, '_film_inspection_camera', False):
            return
        original = cls._camera_cfg

        def with_inspection(emb, height, width, overview):
            ordinary = original(emb, height, width, overview)
            cfg = module.CameraCfg(
                prim_path='{ENV_REGEX_NS}/FilmInspectionCam', height=720, width=1200,
                data_types=['rgb'], renderer_cfg=module.RENDERER,
                spawn=module.sim_utils.PinholeCameraCfg(focal_length=36.,
                      horizontal_aperture=24., clipping_range=(.02, 2.)),
                offset=module.CameraCfg.OffsetCfg(pos=(x, y, glass_z+.288),
                      rot=(0.,0.,0.,1.), convention='opengl'))
            side = module.CameraCfg(
                prim_path='{ENV_REGEX_NS}/FilmSideInspectionCam', height=720, width=1200,
                data_types=['rgb'], renderer_cfg=module.RENDERER,
                spawn=module.sim_utils.PinholeCameraCfg(focal_length=84.,
                      horizontal_aperture=24., clipping_range=(.02, 2.)),
                offset=module.CameraCfg.OffsetCfg(pos=(x, y-.68, glass_z+.014),
                      rot=(2**-.5,0.,0.,2**-.5), convention='opengl'))
            return module.make_configclass('FilmInspectionCameras',
                      [('film_inspection_cam', module.CameraCfg, cfg), ('film_side_cam', module.CameraCfg, side)], bases=(type(ordinary),))()

        cls._camera_cfg = with_inspection
        cls._film_inspection_camera = True

    labkit._TARGETS['yam_arena.embodiment'] = attach
    labkit.install()


class InspectionCamera:
    def __init__(self, scene, folder):
        self.sensor = scene['film_inspection_cam']
        self.side = scene['film_side_cam']
        self.folder = Path(folder)

    def save(self, step, name='inspection-overhead', side=False):
        from PIL import Image

        raw = (self.side if side else self.sensor).data.output['rgb']
        data = getattr(raw, 'torch', raw)
        if hasattr(data, 'detach'):
            data = data.detach().cpu()
        if hasattr(data, 'numpy'):
            data = data.numpy()
        rgb = np.asarray(data)[0, :, :, :3]
        if rgb.shape != (720, 1200, 3) or np.std(rgb.astype(float), axis=(0,1)).max() < 3:
            return False
        Image.fromarray(rgb.astype(np.uint8)).save(self.folder / (name+'.png'))
        (self.folder / (name+'-frame.txt')).write_text(f'{step}\n')
        return True
