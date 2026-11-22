"""Pure classification of native contact evidence; never controls physics.
Supports only explicit collider identities. The removed cap is reserved on its clean rest; the guard forbids robot/tool
contact with it. It does not claim microbiological cleanliness.
"""
from collections import defaultdict

def owner(path):
    tail=path.split('/env_0/')[-1]
    return tail.split('/')[0]

def classify(pairs,food_colliders=()):
    graph=defaultdict(set);failures=set();robot_contacts=[];forceful=[];opening_load=False
    protected=set(food_colliders)
    supporting_fingers=set(); working_fingers=set(); fin_contacts=defaultdict(set)
    for p in pairs:
        if p['max_force_n']<=1e-5:continue
        a,b=owner(p['actor0']),owner(p['actor1'])
        ca,cb=p['collider0'],p['collider1']
        graph[a].add(b);graph[b].add(a)
        if p['max_force_n']>1. and ('/seal/pullring' in p['actor0'] or '/seal/pullring' in p['actor1']):
            if 'hook' in {a,b} or 'LeftArm' in {a,b} or 'RightArm' in {a,b}:opening_load=True
        arms={'LeftArm','RightArm'}
        for arm, other, actor, other_shape in ((a,b,p['actor0'],cb),(b,a,p['actor1'],ca)):
            if arm not in arms or actor.rsplit('/',1)[-1] not in {'tip_left','tip_right'}:
                continue
            if other == 'carton' and other_shape.rsplit('/',1)[-1] in {'box_1','box_2','box_3','box_4'}:
                supporting_fingers.add(arm)
            # A compressively loaded fingertip on an exterior sloped paperboard
            # panel can oppose the pull. The plastic neck is excluded.
            if other=='carton' and other_shape.rsplit('/',1)[-1] in {'box_6','box_7','box_8','box_9','box_10'} and p['max_force_n']>=1.:
                supporting_fingers.add(arm)
            if other == 'carton' and other_shape.rsplit('/',1)[-1]=='box_5':
                fin_contacts[arm].add(actor.rsplit('/',1)[-1])
                if len(fin_contacts[arm])==2:
                    supporting_fingers.add(arm)
            if other in {'seal','hook'}:
                working_fingers.add(arm)
        if 'cap' in {a,b} and ({a,b} & (arms|{'hook','seal'})):
            failures.add('reserved_cap_contact')
        if a in arms or b in arms:robot_contacts.append((a,b))
        if {a,b}==arms and p['max_force_n']>1.:
            failures.add('arm_arm_collision');forceful.append(p)
        # The narrow neck is reserved for the closures and clean opening tool.
        for o,other,col in [(a,b,ca),(b,a,cb)]:
            if o=='carton' and other in arms|{'table'} and '/mesh_' in col:
                idx=int(col.rsplit('mesh_',1)[1])
                if 1<=idx<=32:failures.add('gripper_mouth_contact' if other in arms else 'worktop_mouth_contact')
            if o=='hook' and other=='table' and col.rsplit('/',1)[-1] != 'box_0':
                failures.add('tool_tip_worktop_contact')
            if col in protected and other in arms|{'table','hook','discard','seal'}:
                failures.add('protected_interior_contact')
    # The seal must touch the discard tray directly after the hook is parked.
    held=any(a in {'LeftArm','RightArm'} and b in {'cap','seal','hook','carton'} or b in {'LeftArm','RightArm'} and a in {'cap','seal','hook','carton'} for a,b in robot_contacts)
    return {'violations':sorted(failures),'hands_released':not held,
            'seal_supported_by_tray':'discard' in graph['seal'],
            'cap_supported_by_rest':'cap_rest' in graph['cap'],
            'carton_supported_by_table':'table' in graph['carton'],
            'arm_collision_pairs':forceful,
            'opening_under_load':opening_load,
            'bimanual_support_active':any(s != w for s in supporting_fingers for w in working_fingers),
            'interior_guard_available':bool(protected)}

def centered_top_grip(pairs, to_local, peak):
    """Identify both loaded left fingertips and their carton-local contact points."""
    fingers = {}
    right_holds = False
    for pair in pairs:
        force = pair['max_force_n']
        if force <= 1e-5:
            continue
        for actor,other,shape in ((pair['actor0'],pair['actor1'],pair['collider1']),
                                  (pair['actor1'],pair['actor0'],pair['collider0'])):
            tip = actor.rsplit('/',1)[-1]
            if tip not in ('tip_left','tip_right'):
                continue
            if owner(actor)=='RightArm' and owner(other) in ('hook','seal'):
                right_holds = True
            if owner(actor)!='LeftArm' or owner(other)!='carton' or force<.1:
                continue
            if shape.rsplit('/',1)[-1] not in ('box_5','box_6','box_7','box_8','box_9','box_10'):
                continue
            for point in pair['last_points']:
                x,y,z=map(float,to_local(point))
                if abs(y)<=.012 and .0007<=abs(x)<=.018 and peak-.016<=z<=peak+.009:
                    if tip not in fingers or force>fingers[tip]['force_n']:
                        fingers[tip]={'xyz_m':[x,y,z],'force_n':force}
    opposite=(len(fingers)==2 and fingers['tip_left']['xyz_m'][0]*fingers['tip_right']['xyz_m'][0]<0)
    return bool(opposite and right_holds), fingers
