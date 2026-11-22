"""Dimensions in metres for a compact three-wheel frame suitcase latch."""
import math
import numpy as np

CODE = (4, 7, 2)
START = (3, 8, 1)
STEP = math.tau / 10
X = (-.012, 0., .012)
Z = .020
TAB_Y = -.0185
TAB_HALF_LENGTH = .001
LATCH_Y = -.0177
LATCH_HALF_LENGTH = .0024
# Edge-to-edge slider travel at which a seated tab loses support.
UNLATCH_MM = 1000*(TAB_Y+TAB_HALF_LENGTH-(LATCH_Y-LATCH_HALF_LENGTH))
DECK_HEIGHT = .044
SWITCH_X = -.042
SWITCH_Y = .004
HINGE_Y = -.207
FRONT_Y = -.0545
LID_REST = .43

def gap_mm(angle):
    return 1000*(FRONT_Y-HINGE_Y)*math.sin(angle)

R = .009
CAM = .0063
DARK = (.16, .17, .18)
METAL = (.55, .56, .57)
WHITE = (.95, .95, .90)
SEGMENTS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
STROKES = dict(a=(0,1,1,0),g=(0,0,1,0),d=(0,-1,1,0),b=(.50,.50,0,1),c=(.50,-.50,0,1),f=(-.50,.50,0,1),e=(-.50,-.50,0,1))

def digit(angle):
    return round(angle / STEP) % 10

def indexed(angle, desired):
    return abs(math.remainder(angle - desired * STEP, math.tau)) <= math.radians(7)

def cam_sector(mesh, x, angle):
    points = []
    for dx in (-.001, .001):
        for r, offset in ((.002,-.5), (CAM,-.5), (CAM,.5), (.002,.5)):
            a = angle + offset * STEP
            points.append((x+dx, -r*math.cos(a), Z-r*math.sin(a)))
    triangles = [(0,1,2),(0,2,3),(4,6,5),(4,7,6)]
    for i in range(4):
        j=(i+1)%4
        triangles.extend(((i,4+i,4+j),(i,4+j,j)))
    return mesh(np.array(points),np.array(triangles),METAL,'hull')

def markings(mesh, x, initial):
    points=[];triangles=[]
    for n,segments in enumerate(SEGMENTS):
        angle=(initial-n)*STEP
        for seg in segments:
            u,v,horiz,vert=STROKES[seg]
            cx, cy = u*.0023, v*.00165
            hx,hy=(.00125,.0002) if horiz else (.0002,.00086)
            start=len(points)
            for dx,dy in ((-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)):
                a=angle-(dy+cy)/R
                points.append((x+cx+dx,-(R+.0002)*math.sin(a),Z+(R+.0002)*math.cos(a)))
            triangles.extend(((start,start+1,start+2),(start,start+2,start+3)))
    return mesh(np.array(points),np.array(triangles),WHITE,'none')

def mechanism(kit, name, pos, start=START):
    pos=(pos[0],pos[1],pos[2]+DECK_HEIGHT)
    joints=[]
    for i,(x,n,initial) in enumerate(zip(X,CODE,start)):
        cams=[cam_sector(kit.Mesh,x+.0047,(initial+k)*STEP) for k in range(10) if k!=(-n)%10]
        surfaces=[((x,0,Z),R,.0068,DARK,'X')]
        for k in range(10):
            a=(k+.5)*STEP
            surfaces.append(((x,-R*math.sin(a),Z+R*math.cos(a)),.0006,.0069,METAL,'X'))
        joints.append(kit.Joint(f'wheel{i}',kind='hinge',axis='X',anchor=(x,0,Z),initial=0.,
            lower=-100.,upper=100.,target=0.,stiffness=.011,damping=.00014,mass=.006,
            cylinders=surfaces,meshes=(*cams,markings(kit.Mesh,x,initial))))
    # Slider fingers face the three gate disks. A solid sector stops the same
    # bar that retains the two lid keepers. An aligned notch lets the bar advance.
    # Keep the connecting bridge below the moving tabs. Only its three
    # raised gate fingers and the two short retaining arms enter the latch plane.
    slider=[((0,-.018,.0116),(.049,.002,.0024),METAL),
            ((SWITCH_X,SWITCH_Y,.027),(.011,.010,.010),DARK),
            ((SWITCH_X,SWITCH_Y,.018),(.006,.002,.013),METAL),
            ((SWITCH_X,-.007,.0116),(.006,.024,.0024),METAL),
            ((-.032,-.018,.0116),(.022,.002,.0024),METAL)]
    for x in X:
        slider.extend([((x+.0047,-.0130,Z),(.0014,.0115,.0008),METAL),
                       ((x+.0047,-.018,.0158),(.0014,.0016,.0084),METAL)])
    for x in (-.022,.022):
        slider.extend([((x,LATCH_Y,.021),(.003,2*LATCH_HALF_LENGTH,.002),METAL),
                       ((x,-.0162,.0158),(.002,.0014,.010),METAL),
                       ((x,-.017,.0116),(.002,.003,.0024),METAL)])
    joints.append(kit.Joint('button',axis='Y',anchor=(0,-.021,Z),lower=0.,upper=.0057,
        mass=.014,stiffness=150.,damping=.9,boxes=slider))
    shell=(.16,.33,.42)
    rim=(.48,.53,.54)
    # One rigid lid carries the keepers on both sides of the slide.
    # Its rear hinge is preloaded even when the code is wrong.
    lid=[((0,-.130,.014),(.254,.156,.005),shell),
         ((0,-.0545,.017),(.251,.004,.003),rim)]
    # Shallow molded ribs stiffen the shell and outline the moving lid in camera views.
    for x in (-.094,-.063,-.032,0,.032,.063,.094):
        lid.append(((x,-.134,.0175),(.009,.106,.002),(.20,.38,.48)))
    for x in (-.022,.022):
        lid.extend([((x,TAB_Y,.0184),(.003,2*TAB_HALF_LENGTH,.002),METAL),
                    ((x,-.0365,.0152),(.002,.038,.003),METAL),
                    ((x,-.0188,.0170),(.002,.0018,.002),METAL)])
    joints.append(kit.Joint('lid',kind='hinge',axis='X',anchor=(0,HINGE_Y,.014),
        lower=0.,upper=.48,target=LID_REST,stiffness=.82,damping=.060,mass=.17,boxes=lid))
    base=[((0,0,.005),(.072,.080,.010),shell),
          ((0,.015,.014),(.055,.009,.008),DARK),
          ((-.020,0,Z+.006),(.002,.0012,.002),WHITE),
          ((0,-.092,-.042),(.270,.243,.004),shell),
          ((0,-.011,.002),(.270,.082,.004),shell),
          ((0,-.211,-.016),(.270,.004,.052),shell),
          ((0,.027,-.016),(.270,.004,.052),shell),
          ((0,-.211,.007),(.268,.004,.010),rim),
          ((0,-.052,.006),(.260,.003,.009),rim)]
    for x in (-.132,.132):
        base.extend([((x,-.092,-.016),(.004,.241,.052),shell),
                     ((x,-.131,.007),(.004,.160,.010),rim)])
    for x in (-.043,.043):
        base.append(((x,.038,-.016),(.010,.023,.011),DARK))
    base.append(((0,.048,-.016),(.095,.010,.012),DARK))
    # The handle and case bottom stay in the cradle while the lid opens.
    cover=(.22,.24,.26)
    base.append(((0,.013,.022),(.055,.006,.009),cover))
    for a,b in ((-.027,-.0159),(-.0081,-.0039),(.0039,.0081),(.0159,.027)):
        base.append((((a+b)/2,0,.0239),(b-a,.0165,.0054),cover))
    for x in (-.018,.018):
        base.append(((x,-.011,.0253),(.018,.005,.0024),cover))
    return kit.part(name,pos,fixed=True,self_collide=True,joints=tuple(joints),boxes=base,friction=.65)
