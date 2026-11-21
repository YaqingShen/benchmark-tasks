"""Recover print pitch from Markus Wobisch's two photographs and faceplate drawing."""
import numpy as np

# Hole centers in flux-front-plus.dxf, in inches. Order: far left/right, near left/right.
plate = np.array([[-4.65625, 5.65625], [4.65625, 5.65625],
                  [-4.65625, -5.65625], [4.65625, -5.65625]]) * 25.4

# Native image pixels, origin upper left. Identical letters on the opaque faceplate.
# Printed text: DISCONNECT CAPACITOR DRIVE. D-to-D spans 21 feeds, E-to-E spans 18.
photos = {
    "20180506_222418.jpg": (
        [[349.37, 32.46], [931.09, 31.27], [138.79, 583.85], [1166.79, 576.33]],
        [[[462, 21], [762, 21]], [[562, 21], [819, 20]]]),
    "IMGP6219ed-sm.JPG": (
        [[138.11, 308.75], [1070.17, 308.93], [132.05, 1439.70], [1066.41, 1446.22]],
        [[[316, 264], [800, 265]], [[477, 264], [892, 265]]]),
}


def pitches(holes, pairs):
    rows = []
    for (x, y), (u, v) in zip(holes, plate):
        rows.extend([[x, y, 1, 0, 0, 0, -x*u, -y*u],
                     [0, 0, 0, x, y, 1, -x*v, -y*v]])
    transform = np.append(np.linalg.solve(rows, plate.ravel()), 1).reshape(3, 3)
    results = []
    for pixels, feeds in zip(pairs, (21, 18)):
        points = np.column_stack((pixels, [1, 1])) @ transform.T
        xy = points[:, :2] / points[:, 2, None]
        results.append(np.linalg.norm(xy[1] - xy[0]) / feeds)
    return results


if __name__ == "__main__":
    for name, (holes, pairs) in photos.items():
        print(name, "D-D and E-E (mm/feed):", *(f"{x:.3f}" for x in pitches(holes, pairs)))
