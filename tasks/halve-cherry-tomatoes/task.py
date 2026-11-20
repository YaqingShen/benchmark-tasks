"""Eight whole cherry tomatoes sit on a shallow tray between the arms. A loose lid and a kitchen knife rest next to it."""
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import labkit
from yam_arena.demo import Waypoint
from yam_arena.task_file import Sphere, TABLE_TOP
SCENE="station"
TOP=TABLE_TOP[SCENE]
def world(p):
    x,y,z=p
    return (.30+y,.071-x,TOP+z)
def part(name,center,boxes,mass=None):
    return labkit.part(name,pos=world(center),yaw=-math.pi/2,boxes=boxes,**({'mass':mass} if mass else {'fixed':True}))
cream=(.91,.86,.72)
sage=(.38,.58,.54)
red=(.83,.16,.09)
clear=(.92,.97,.97)
labkit.GLASS_COLORS.add(clear)
fruit_layout=json.loads((ROOT/'fruit.json').read_text())
# Low locating ribs are fixed to the food-contact insert. Their tops stay below the edge.
# Four short ribs form each square seat and leave its corners open for cleaning.
seats=[]
for x,y,z in fruit_layout:
    for sign in (-1,1):
        seats.append(((x-.071,y+sign*.014,.004),(.016,.003,.008),sage))
        seats.append(((x-.071+sign*.014,y,.004),(.003,.016,.008),sage))
OBJECTS=[
    part('lower_lid',(.071,0,.080),[((0,0,-.042),(.168,.152,.072),(.24,.27,.27)),((0,0,-.002),(.17,.154,.004),cream)]+seats+
         [((0,y,.005),(.17,.003,.010),cream) for y in (-.077,.077)]+[((x,0,.005),(.003,.154,.010),cream) for x in (-.085,.085)]),
    part('lid_dock',(-.117,.012,.055),[((0,0,0),(.092,.085,.11),sage)]),
    part('upper_lid',(-.117,.012,.11),[((0,0,.003),(.16,.144,.006),clear),((-.048,0,.023),(.019,.044,.034),sage)],.115),
    part('handle_rests',(.071,-.121,.0425),[((x,0,0),(.013,.028,.085),sage) for x in (.103,.167)]),
    part('knife',(.071,-.11,.094),[((0,-.011,0),(.177,.022,.0014),(.74,.77,.78)),((.134,-.011,0),(.092,.020,.018),(.17,.18,.18))],.085)]
for i,p in enumerate(fruit_layout):
    OBJECTS.append(Sphere('tomato_'+str(i),pos=world(p),radius=.014,color=red,mass=.012))
import slicing
slicing.install_filter()
knife=OBJECTS.pop(4)
for idx in range(16):
    OBJECTS.append(labkit.part(f'half_{idx}',pos=(.12+(idx%8)*.036,-.10+(idx//8)*.045,.05),
                               meshes=slicing.segments(1 if idx%2 else -1),mass=.006))
OBJECTS.append(knife)
trial=slicing.Trial(world)
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)
def score(objects):return trial.credit
def success(objects):return trial.final
def demo():
    grip=(-.165,.012,.139); handle=(.205,-.121,.094)
    left=[(1.6,(*grip[:2],.194),-.028),(1.1,grip,-.028),(.9,grip,0),(.8,(*grip[:2],.182),0),
          (1.6,(.023,0,.182),0),(1.2,(.023,0,.132),0),(7,(.023,0,.132),0),(1,(.023,0,.182),0),
          (1.5,(*grip[:2],.182),0),(1,grip,0),(.7,grip,-.028),(1,(*grip[:2],.206),-.028)]
    right=[(1.6,(*handle[:2],.154),-.028),(1.1,handle,-.028),(.9,handle,0),(.9,(*handle[:2],.134),0),
           (1.,(.205,-.105,.134),0),(1.2,(.205,-.105,.0975),0),(1.,(.205,-.105,.0975),0),
           *[(5.1/16,(.205 + .007*math.sin(k*math.pi/2),-.105+.210*k/16,.0975+.003*min(1.,max(0.,(k-5)/2))),0) for k in range(1,17)],(.8,(.205,.105,.1005),0),(1.,(.205,.105,.154),0),
           (3.3,(.205,.105,.154),0),(1.8,(*handle[:2],.154),0),(1.1,handle,0),(.7,handle,-.028),(.9,(*handle[:2],.166),-.028)]
    return {side:[Waypoint(dt,world(p),finger,'',yaw) for dt,p,finger in pts] for side,pts,yaw in [('left',left,0.),('right',right,math.pi/2)]}
