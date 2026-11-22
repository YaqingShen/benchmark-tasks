"""Short trapezoidal roof panels between the flat carton sheet and round neck."""
import math
import numpy as np


def flange(mesh_type, rotation, center, mouth_radius, color):
    # Existing roof strips overlap the outside edge of this 48 x 54 mm patch.
    # The inner polygon seats within the neck wall, leaving its bore unobstructed.
    half_x, half_y = .024, .027
    step = 2 * math.pi / 32
    corners = [math.atan2(y, x) % (2 * math.pi) for x in (-half_x, half_x)
               for y in (-half_y, half_y)]
    angles = sorted(set([i * step for i in range(32)] + corners))
    triangles = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                          [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                          [2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    panels = []
    for start, end in zip(angles, angles[1:] + [2 * math.pi]):
        ends = []
        for angle in (start, end):
            direction = np.array([math.cos(angle), math.sin(angle)])
            radius = min(half_x / max(abs(direction[0]), 1e-12),
                         half_y / max(abs(direction[1]), 1e-12))
            ends.append((direction * (mouth_radius + .0005), direction * radius))
        outline = [ends[0][0], ends[0][1], ends[1][1], ends[1][0]]
        vertices = np.array([[x, y, z] for z in (-.00075,.00075) for x,y in outline])
        panels.append(mesh_type(vertices @ rotation.T + center, triangles, color, 'hull'))
    return panels
