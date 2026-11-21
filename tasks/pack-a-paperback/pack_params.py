"""pack_grok002 (by grok): contents, staging and SLOW-REBOUND foam parameters. Pure Python.

Design change vs pack_grok001: the contents start CLEARLY overstuffed (9 mm above the closed-lid
underside instead of 2.7 mm) and the pad is a slow-rebound ("memory foam") lumped model: two rigid
plates + one prismatic joint with a weak preloaded spring and a very heavy damper. After a press the
pad creeps back up over ~15-25 s, which leaves a few-second window in which the lid can be closed and
the front tuck inserted. No deformables (CPU physics), no fixtures, the pad never acts on box/lid/robot.

Frames (unchanged from pack_grok002)
- box frame: mailer base body frame (u = box length 220 mm, v = depth 160 mm, front wall at v=-0.08,
  back wall + lid hinge at v=+0.08, z up from base mid-plane).
- world: the mailer base sits at (0.35, 0, support_top+0.00065), turned -90 deg about z,
  so world = (0.35 + v, -u, z). The front wall faces the robots (world x = 0.27).
"""
import math

BOX_ORIGIN_XY = (.35, 0.)
BASE_TOP_Z = .0006               # base upper face, box frame
LID_UNDERSIDE_Z = .0494          # closed lid lower face, box frame (lid mid-plane at 0.050)
WALL_TOP_Z = .0500
INTERIOR_U = .1040               # inner face of the rolled inner side panels
INTERIOR_V = .0794               # inner faces of front / back walls
G = 9.81

# Mass-market paperback (unchanged), one rigid block.
BOOK = dict(size=(.178, .108, .025), mass=.25, friction=(.45, .35), cover_rgb=(.10, .22, .45), page_rgb=(.93, .90, .80))

# Design overstuff: free (uncompressed) stack top above the closed-lid underside.
OVERSTUFF_M = .0090
# Slow-rebound pad.  free height chosen so that BASE_TOP + pad + book = LID_UNDERSIDE + OVERSTUFF.
_PAD_H = round(LID_UNDERSIDE_Z-BASE_TOP_Z-BOOK['size'][2]+OVERSTUFF_M, 5)     # 32.8 mm
PAD = dict(size=(.160, .136, _PAD_H), base_thickness=.004, mass=.025, friction=(.60, .50), rgb=(.93, .91, .78),
           travel=.0120,                 # bottoms out 3.0 mm below flush (hard stop = honest foam densification)
           stiffness_N_m=60.,            # weak elastic part
           preload_N=2.75,               # > book weight 2.45 N: the book alone does not sink the pad
           damping_N_s_m=1200.,          # very heavy viscous part -> slow rebound (time constant c/k = 20 s)
           max_force_N=60.)              # drive force cap; above ~60 N the foam "yields" (only reached by very fast pressing)

# pad_top collider starts this far above the pad_top body origin (stroke gap, see pack_assets.build_pad)
PAD_TOP_GAP = PAD['travel']+.0005

# Lid/tuck closing tolerance (box frame): the largest stack-top excess over LID_UNDERSIDE_Z for which
# (pack_grok002 v2: the first estimate 0.9 mm wrongly used the book FRONT edge as the contact point)
# the lid can still lie on the book AND the front tuck passes pack_check's tuck_inside_front_wall
# without intersecting the front wall.  Computed by check_close_feasibility.py (geometry, not tuned):
CLOSE_TOL_M = 0.               # rigid lid hinged at the back-wall top + back-justified book => ~0.01 mm (close-feasibility.json); use flush

# In-box target (box frame): pad back-justified against the back wall (1 mm gap), book back-justified on the pad.
PAD_TARGET_V = INTERIOR_V-.001-PAD['size'][1]/2
BOOK_TARGET_V = INTERIOR_V-.001-BOOK['size'][1]/2

# Book is staged SKEWED on its pedestal (yaw about z, world frame) so a straight two-end pick lands it
# skewed in the box; the robot must square it (book_square: |yaw| <= 3 deg vs box) before/while pressing.
BOOK_STAGING_YAW_DEG = 10.
BOOK_SQUARE_TOL_DEG = 3.

STAGING = {
    'foam_pad': dict(center_xy=(.258, .220), pedestal_center_xy=(.260, .220), pedestal=(.080, .100), grasp='left hand, front long edge, approach pitched 23 deg down, yawed 23 deg inward'),
    'book': dict(center_xy=(.155, 0.), pedestal_center_xy=(.155, 0.), pedestal=(.100, .100), yaw_deg=BOOK_STAGING_YAW_DEG, grasp='both hands, one per end (book yawed 10 deg)'),
}

def book_weight_N():
    return BOOK['mass']*G

PAD_TOP_MASS_FRACTION = .7      # pack_assets: pad_top carries 70 % of the pad mass

def supported_weight_N():
    """Weight the foam spring carries: book + the pad top plate (found when matching PhysX to the ODE)."""
    return (BOOK['mass']+PAD_TOP_MASS_FRACTION*PAD['mass'])*G

def free_stack_top_m():
    return BASE_TOP_Z+PAD['size'][2]+BOOK['size'][2]

def overstuff_m():
    """Height by which the uncompressed stack exceeds the closed-lid underside."""
    return free_stack_top_m()-LID_UNDERSIDE_Z

def needed_compression_m():
    """Pad compression needed before the lid can close with the tuck inside the front wall."""
    return overstuff_m()-CLOSE_TOL_M

def pad_up_force_N(compression_m):
    """Static upward spring force on the book (positive = pushes book up) minus book weight."""
    return PAD['preload_N']+PAD['stiffness_N_m']*compression_m-supported_weight_N()

def lid_push_at_flush_N():
    return pad_up_force_N(overstuff_m())

def rebound_ode(c0, t_end=60., dt=1e-3, m=None):
    """Analytic reference (first-order, inertia negligible): c*dx/dt = -(F0-W + k x), x = compression.
    Returns (t, x) sampled at dt.  Used only as a cross-check of the PhysX run."""
    P = PAD; k, c = P['stiffness_N_m'], P['damping_N_s_m']; f = P['preload_N']-supported_weight_N()
    ts, xs = [], []; x = c0; t = 0.
    while t <= t_end:
        ts.append(t); xs.append(x)
        x = max(0., x-(f+k*x)/c*dt); t += dt
    return ts, xs

def analytic_cross_time(c0, c1):
    """Time for compression to fall from c0 to c1 under c*xdot = -(f + k x):  t = (c/k) ln((f+k c0)/(f+k c1))."""
    P = PAD; k, c = P['stiffness_N_m'], P['damping_N_s_m']; f = P['preload_N']-supported_weight_N()
    return c/k*math.log((f+k*c0)/(f+k*c1))

def summary():
    P = PAD
    return {'overstuff_mm': round(overstuff_m()*1e3, 2),
            'free_stack_top_mm': round(free_stack_top_m()*1e3, 2),
            'pad_free_height_mm': round(P['size'][2]*1e3, 2),
            'pad_travel_mm': round(P['travel']*1e3, 2),
            'bottomed_stack_top_vs_lid_mm': round((overstuff_m()-P['travel'])*1e3, 2),
            'close_tol_mm': round(CLOSE_TOL_M*1e3, 2),
            'needed_compression_mm': round(needed_compression_m()*1e3, 2),
            'lid_push_at_flush_N': round(lid_push_at_flush_N(), 3),
            'book_weight_N': round(book_weight_N(), 3),
            'pad_preload_margin_N (over book + pad top)': round(P['preload_N']-supported_weight_N(), 3),
            'rebound_time_constant_s': round(P['damping_N_s_m']/P['stiffness_N_m'], 2),
            'analytic_hold_window_from_bottom_s': round(analytic_cross_time(P['travel'], needed_compression_m()), 2),
            'front_gap_book_mm': round((INTERIOR_V+BOOK_TARGET_V-BOOK['size'][1]/2)*1e3, 1),
            'front_gap_pad_mm': round((INTERIOR_V+PAD_TARGET_V-PAD['size'][1]/2)*1e3, 1),
            'side_gap_pad_mm': round((INTERIOR_U-PAD['size'][0]/2)*1e3, 1),
            'side_gap_book_mm': round((INTERIOR_U-BOOK['size'][0]/2)*1e3, 1)}

if __name__ == '__main__':
    import json; print(json.dumps(summary(), indent=2))
