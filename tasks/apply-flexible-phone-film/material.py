"""Reduced peel-and-laminate model. Distances are metres in screen coordinates."""
import math
import numpy as np

LENGTH = 0.140
WIDTH = 0.064
ROWS = 28
COLS = 8
DX = LENGTH / ROWS
X = (np.arange(ROWS) + .5) * DX
Y = (np.arange(COLS) + .5) * WIDTH / COLS - WIDTH / 2
TAB_X = .026
TAB_Y = .051


class Laminate:
    def __init__(self):
        self.origin = np.array([-.022, .012])
        self.yaw = 0.
        self.held_edge = False
        self.peeled = 0.
        self.bond = np.zeros((ROWS, COLS), dtype=bool)
        self.started = False
        self.progressive = np.zeros_like(self.bond)

    def aligned(self):
        cs, sn = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array(((cs,-sn),(sn,cs)))
        corners = np.array([(x,y) for x in (0,LENGTH) for y in (-WIDTH/2,WIDTH/2)])
        return np.max(np.linalg.norm(self.origin + corners @ rot.T-corners,axis=1)) <= .003

    def observe(self, tab, angle, pinched, card_edge, loaded):
        """tab = measured folded liner end; card_edge = two endpoints of the loaded blade."""
        if self.held_edge and not self.started and not loaded:
            self.held_edge = False
            self.peeled = 0.
        cs, sn = math.cos(self.yaw), math.sin(self.yaw)
        if not self.held_edge:
            self.yaw = angle
            cs, sn = math.cos(angle), math.sin(angle)
            self.origin = tab[:2] - [cs*TAB_X-sn*TAB_Y, sn*TAB_X+cs*TAB_Y]
        edge = (card_edge[:,:2]-self.origin) @ np.array(((cs,-sn),(sn,cs)))
        middle = edge.mean(axis=0)
        full_width = edge[:,1].min() < Y[0] and edge[:,1].max() > Y[-1]
        if (not self.held_edge and loaded and pinched and full_width and self.aligned() and tab[2] <= .025
                and abs(middle[0]) <= .009 and abs(edge[1,0]-edge[0,0]) < .012):
            self.held_edge = True
        if not self.held_edge:
            return
        local = (tab[:2]-self.origin) @ np.array(((cs,-sn),(sn,cs)))
        if pinched and tab[2] > .004 and abs(local[1]-TAB_Y)<.020:
            # For a folded-back liner, peeled length is the distance from its front to its free end.
            a, h = local[0], max(0.,tab[2])
            if a+TAB_X > .002:
                peel = (a*a+h*h-TAB_X*TAB_X)/(2*(a+TAB_X))
                self.peeled = max(self.peeled,min(LENGTH,max(0.,peel)))
        if loaded and abs(edge[1,1]-edge[0,1]) > .010:
            fraction = (Y-edge[0,1])/(edge[1,1]-edge[0,1])
            press_x = edge[0,0]+fraction*(edge[1,0]-edge[0,0])
            height = card_edge[0,2] + fraction*(card_edge[1,2]-card_edge[0,2])
            covered = (fraction >= 0) & (fraction <= 1) & (height <= .0018) & (height >= -.0012)
            exposed = X <= self.peeled
            touch = exposed[:,None] & covered[None,:] & (np.abs(X[:,None]-press_x[None,:]) <= .004)
            self.progressive |= touch & ~self.bond & ((self.peeled-X[:,None]) <= .025)
            self.bond |= touch
        self.started = bool(self.bond.any())

    @property
    def coverage(self):
        return float(self.bond.mean())
