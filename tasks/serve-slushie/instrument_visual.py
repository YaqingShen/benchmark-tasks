"""Top-facing USD readouts on the pickup balance and the hopper probe head."""
from pxr import UsdGeom

SEGMENTS = {
    '0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc',
    '5': 'afgcd', '6': 'afgcde', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg',
    '-': 'g', ' ': '', 'C': 'adef', 'g': 'ghijklm',
}
BARS = {
    'a': (0, 1, .78, .17), 'b': (.4, .5, .15, .82),
    'c': (.4, -.5, .15, .82), 'd': (0, -1, .78, .17),
    'e': (-.4, -.5, .15, .82), 'f': (-.4, .5, .15, .82),
    'g': (0, 0, .78, .17),
    'h': (-.4, -.32, .15, .53), 'i': (.4, -.32, .15, .53),
    'j': (0, -.64, .78, .15), 'k': (.4, -.91, .15, .47),
    'l': (0, -1.17, .78, .15), 'm': (-.32, -1.07, .14, .16),
}
ROOT = '/World/SlushInstruments/'


def box(stage, name, centre, size, color):
    cube = UsdGeom.Cube.Define(stage, ROOT + name)
    cube.CreateSizeAttr(1.)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(centre)
    xform.SetScale(size)
    cube.CreateDisplayColorAttr([color])


class Display:
    def __init__(self, stage, name, centre, unit, color):
        self.centre = centre
        self.unit = unit
        self.previous = None
        x, y, z = centre
        box(stage, name + 'Housing', (x, y, z), (.083, .033, .013), (.09, .10, .12))
        self.mesh = UsdGeom.Mesh.Define(stage, ROOT + name + 'Digits')
        self.mesh.CreateSubdivisionSchemeAttr('none')
        self.mesh.CreateDisplayColorAttr([color])
        self.mesh.CreateDoubleSidedAttr(True)

    def draw(self, value):
        label = ('----' if value is None else f'{value:.1f}').rjust(5) + ('°' if self.unit == 'C' else ' ') + self.unit
        if self.previous == label:
            return
        self.previous = label
        x, y, z = self.centre
        cursor = x - .033
        points = []
        for char in label:
            if char == '.':
                shapes = [(-.18, -.90, .27, .24)]
            elif char == '°':
                shapes = [(0,.97,.33,.10), (0,.64,.33,.10), (-.16,.80,.10,.28), (.16,.80,.10,.28)]
            else:
                shapes = [BARS[k] for k in SEGMENTS[char]]
            for dx, dy, w, h in shapes:
                cx, cy = cursor + dx * .008, y + dy * .009
                for sx, sy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
                    points.append((cx + sx*w*.004, cy + sy*h*.0045, z + .0067))
            cursor += .0065 if char == '.' else .010
        self.mesh.GetPointsAttr().Set(points)
        self.mesh.GetFaceVertexCountsAttr().Set([4] * (len(points) // 4))
        self.mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))


class InstrumentVisual:
    def __init__(self, stage):
        UsdGeom.Xform.Define(stage, ROOT.rstrip('/'))
        # The original platform is the weighing pan. The display sits on its base.
        self.scale = Display(stage, 'PickupScale', (.30, .155, .766), 'g', (.30, .98, .46))
        stem = UsdGeom.Cylinder.Define(stage, ROOT + 'ProbeStem')
        stem.CreateRadiusAttr(.002)
        stem.CreateHeightAttr(.128)
        stem.CreateAxisAttr('Z')
        stem.CreateDisplayColorAttr([(.68, .71, .72)])
        UsdGeom.XformCommonAPI(stem).SetTranslate((.705, .064, 1.169))
        self.temperature = Display(stage, 'Probe', (.705, .064, 1.239), 'C', (1., .64, .12))

    def draw(self, readings):
        self.scale.draw(readings['pickup_scale_g'])
        self.temperature.draw(readings['probe_c'])
