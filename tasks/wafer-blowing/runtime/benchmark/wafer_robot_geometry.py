"""Measured robot collision validity; no dynamics or commanded-pose substitution."""
from collections import defaultdict
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import zlib
import xml.etree.ElementTree as ET
import numpy as np
from collision_geometry import GeometryError, prepare_mesh, transform_mesh, convex_screen


VERSION = "measured_robot_convex_geometry_v1"
TOLERANCE_M = 1e-7
LINK_NAMES = ('base','gripper','link1','link2','link3','link4','link5','tip_left','tip_right')
JOINT_PAIRS = (('base','link1'),('link1','link2'),('link2','link3'),('link3','link4'),
               ('link4','link5'),('link5','gripper'),('gripper','tip_left'),('gripper','tip_right'))
REQUIRED_EVIDENCE = 'all_physics_steps_cooked_robot_geometry_v1'
LOCAL_EVIDENCE = 'all_physics_steps_measured_robot_joint_fk_v1'
NATIVE_OFFLINE_EVIDENCE = 'all_physics_steps_measured_robot_link_poses_v1'


def _file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def local_robot_asset_digest(scene_path):
    """Bind external collider bytes at initialization; the XML hash covers inline geometry."""
    scene_path=Path(scene_path);root=ET.parse(scene_path).getroot()
    if root.findall('.//include'):raise ValueError('The frozen local scene must have no external XML includes.')
    compiler=root.find('compiler');meshdir=''if compiler is None else compiler.get('meshdir','')
    meshes=[x for x in root.findall('./asset/mesh')if x.get('file')]
    if len(meshes)!=315 or len({m.get('name')for m in meshes})!=315:raise ValueError('Expected315 uniquely named external robot mesh assets.')
    digest=hashlib.sha256()
    for mesh in sorted(meshes,key=lambda x:x.get('name','')):
        path=Path(mesh.get('file'))
        if not path.is_absolute():path=scene_path.parent/meshdir/path
        digest.update(mesh.get('name').encode()+b'\0'+bytes.fromhex(_file_sha(path)))
    return digest.hexdigest()


def compact_result(result):
    return {k:result[k]for k in ('status','complete','pair','pieces','reason')if k in result}


def apply_validity(task,issue):
    result=copy.deepcopy(task)
    result['evidence_complete']=not issue or issue['status']=='COLLISION'
    result['robot_geometry']={'required_evidence':REQUIRED_EVIDENCE,'first_issue':copy.deepcopy(issue)}
    if issue:
        result['score']=0.;result['success']=False
        if issue['status']=='COLLISION':
            result['invalid_reasons'].append('Measured robot self/inter-arm geometry intersects.')
        else:
            result['evidence_errors']=['Robot collision evidence is missing, unsupported or numerically unresolved.']
    return result


def first_issue(issue,row,result=None):
    if issue is not None:return issue
    result=row.get('robot_geometry_check')if result is None else result
    if not isinstance(result,dict)or result.get('status')not in ('CLEAR','COLLISION','UNSUPPORTED','UNRESOLVED'):
        result={'status':'UNSUPPORTED','complete':False,'reason':'Missing robot geometry result.'}
    if result.get('status')in ('CLEAR','COLLISION','UNRESOLVED')and result.get('complete')is not True:
        result={'status':'UNSUPPORTED','complete':False,'reason':'Incomplete robot geometry result.'}
    if result.get('status')=='CLEAR'and result.get('complete')is True:return None
    return dict(compact_result(result),time_s=float(row['time_s']))


def _rotations(poses):
    p=np.asarray(poses,dtype=float)
    if p.ndim!=2 or p.shape[1]!=7 or not np.isfinite(p).all():raise GeometryError("Invalid measured robot link poses.")
    q=p[:,3:];norm=np.linalg.norm(q,axis=1)
    if np.any(abs(norm-1)>1e-4):raise GeometryError("Non-unit measured robot orientation.")
    x,y,z,w=(q/norm[:,None]).T
    r=np.empty((len(p),3,3))
    r[:,0,0]=1-2*(y*y+z*z);r[:,0,1]=2*(x*y-z*w);r[:,0,2]=2*(x*z+y*w)
    r[:,1,0]=2*(x*y+z*w);r[:,1,1]=1-2*(x*x+z*z);r[:,1,2]=2*(y*z-x*w)
    r[:,2,0]=2*(x*z-y*w);r[:,2,1]=2*(y*z+x*w);r[:,2,2]=1-2*(x*x+y*y)
    return p[:,:3],r


def native_joint_neighbors(root):
    from pxr import Usd,UsdPhysics
    neighbors=set();anchors=0;edges=0
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim)
        targets=[rel.GetTargets()for rel in (joint.GetBody0Rel(),joint.GetBody1Rel())]
        if joint.GetJointEnabledAttr().Get()is not True or any(len(paths)!=1 for paths in targets):
            raise GeometryError('Disabled or incomplete native robot joint: '+str(prim.GetPath()))
        paths=[paths[0]for paths in targets]
        if not all(path.HasPrefix(root.GetPath())for path in paths):
            raise GeometryError('Native robot joint leaves the robot root.')
        bodies=[root.GetStage().GetPrimAtPath(path)for path in paths]
        rigid=[body.HasAPI(UsdPhysics.RigidBodyAPI)for body in bodies]
        if all(rigid):
            neighbors.add(frozenset(path.name for path in paths));edges+=1
        elif (prim.IsA(UsdPhysics.FixedJoint)and root.GetPath()in paths
              and not root.HasAPI(UsdPhysics.RigidBodyAPI)):
            other=1-paths.index(root.GetPath())
            if paths[other].name!='base'or not rigid[other]:
                raise GeometryError('Native fixed anchor does not connect to the rigid base.')
            anchors+=1
        else:raise GeometryError('Unsupported native robot joint endpoints.')
    if anchors!=1 or edges!=len(JOINT_PAIRS)or neighbors!={frozenset(p)for p in JOINT_PAIRS}:
        raise GeometryError('Native robot joint topology differs after fixed-base validation.')
    return neighbors


class RobotGeometryGuard:
    def __init__(self,groups,neighbors,*,source,collider_counts):
        self.keys=tuple(sorted(groups));self.meshes=[tuple(groups[k])for k in self.keys]
        if set(self.keys)!={(side,name)for side in ('left','right')for name in LINK_NAMES}or collider_counts!={"left":315,"right":315}:
            raise GeometryError("Both complete 315-collider robot inventories are required.")
        if {frozenset(p)for p in neighbors}!={frozenset(p)for p in JOINT_PAIRS}:
            raise GeometryError('Robot direct-joint exclusions differ from the checked YAM topology.')
        if any(not meshes for meshes in self.meshes):raise GeometryError("Empty robot collision link.")
        self.centers=[];self.halves=[];self.piece_centers=[];self.piece_halves=[]
        for meshes in self.meshes:
            bounds=np.array([m.bounds for m in meshes]);lo=bounds[:,0].min(0);hi=bounds[:,1].max(0)
            self.centers.append((lo+hi)/2);self.halves.append((hi-lo)/2)
            self.piece_centers.append(bounds.mean(axis=1));self.piece_halves.append((bounds[:,1]-bounds[:,0])/2)
        self.centers=np.array(self.centers);self.halves=np.array(self.halves)
        excluded={frozenset(pair)for pair in neighbors}|{frozenset(("tip_left","tip_right"))}
        self.pairs=np.array([(i,j)for i,(sa,a)in enumerate(self.keys)for j,(sb,b)in enumerate(self.keys)
            if j>i and(sa!=sb or frozenset((a,b))not in excluded)],dtype=int)
        self.axes={}
        self.interiors=[[mesh.vertices.mean(0)for mesh in meshes]for meshes in self.meshes]
        self.planes=[[(mesh.vertices@mesh.normals.T).min(0)for mesh in meshes]for meshes in self.meshes]
        self.ceilings=[[(mesh.vertices@mesh.normals.T).max(0)for mesh in meshes]for meshes in self.meshes]
        self.metadata=dict(version=VERSION,source=source,collider_counts=collider_counts,
            convex_piece_count=sum(map(len,self.meshes)),link_keys=[list(k)for k in self.keys],
            same_arm_excluded_pairs=sorted(sorted(p)for p in excluded),numerical_tolerance_m=TOLERANCE_M,
            scope="All measured link poses; same-link, direct-joint neighbors and finger-carriage sibling overlap excluded. No continuous-time claim.")

    def check(self,poses):
        counts=dict(candidate_link_pairs=0,candidate_piece_pairs=0,sat_tests=0,certificate_tests=0)
        try:
            if set(poses)!=set(self.keys):raise GeometryError("Measured robot link inventory differs.")
            positions,rotations=_rotations([poses[k]for k in self.keys])
            center=np.einsum('nij,nj->ni',rotations,self.centers)+positions
            extent=np.einsum('nij,nj->ni',abs(rotations),self.halves);lo=center-extent;hi=center+extent
            ia,ib=self.pairs.T;gaps=np.maximum(lo[ib]-hi[ia],lo[ia]-hi[ib]).max(1)
            candidates=self.pairs[gaps<=TOLERANCE_M];counts['candidate_link_pairs']=len(candidates)
            if len(candidates):
                ia,ib=candidates.T;ra=rotations[ia];rb=rotations[ib]
                axes=np.concatenate((ra.transpose(0,2,1),rb.transpose(0,2,1),
                    np.cross(ra.transpose(0,2,1)[:,:,None],rb.transpose(0,2,1)[:,None]).reshape(-1,9,3)),axis=1)
                length=np.linalg.norm(axes,axis=2);axes=axes/np.maximum(length[:,:,None],1e-12)
                span_a=np.einsum('nki,ni->nk',abs(np.einsum('nki,nij->nkj',axes,ra)),self.halves[ia])
                span_b=np.einsum('nki,ni->nk',abs(np.einsum('nki,nij->nkj',axes,rb)),self.halves[ib])
                separation=abs(np.einsum('nki,ni->nk',axes,center[ib]-center[ia]))-span_a-span_b
                candidates=candidates[~np.any((separation>TOLERANCE_M)&(length>1e-12),axis=1)]
            counts['obb_candidate_link_pairs']=len(candidates)
            bounds={};world={}
            for a,b in candidates:
                for i in (a,b):
                    if i not in bounds:
                        c=self.piece_centers[i]@rotations[i].T+positions[i]
                        h=self.piece_halves[i]@abs(rotations[i]).T;bounds[i]=(c-h,c+h)
                al,ah=bounds[a];bl,bh=bounds[b]
                gap=np.maximum(bl[None]-ah[:,None],al[:,None]-bh[None]).max(2)
                piece_pairs=np.argwhere(gap<=TOLERANCE_M);counts['candidate_piece_pairs']+=len(piece_pairs)
                for ia,ib in piece_pairs:
                    pair=(int(a),int(ia),int(b),int(ib));certificate=self.axes.get(pair)
                    if certificate is not None:
                        local_axis,low,high=certificate;axis=rotations[a]@local_axis
                        projection=self.meshes[b][ib].vertices@(rotations[b].T@axis)+(positions[b]-positions[a])@axis
                        counts['certificate_tests']+=1
                        if max(float(projection.min())-high,low-float(projection.max()))>TOLERANCE_M:continue
                    for aa,ii,bb,jj in ((a,ia,b,ib),(b,ib,a,ia)):
                        point=rotations[bb].T@(rotations[aa]@self.interiors[aa][ii]+positions[aa]-positions[bb])
                        projection=point@self.meshes[bb][jj].normals.T
                        if np.all(projection>self.planes[bb][jj]+TOLERANCE_M)and np.all(projection<self.ceilings[bb][jj]-TOLERANCE_M):
                            return dict(status='COLLISION',complete=True,pair=[list(self.keys[a]),list(self.keys[b])],
                                pieces=[self.meshes[a][ia].name,self.meshes[b][ib].name],reason='Strict interior-point intersection',**counts)
                    for link,piece in ((a,ia),(b,ib)):
                        key=(int(link),int(piece))
                        if key not in world:
                            pose=np.eye(4);pose[:3,:3]=rotations[link];pose[:3,3]=positions[link]
                            world[key]=transform_mesh(self.meshes[link][piece],pose)
                    result=convex_screen(world[(a,ia)],world[(b,ib)],tolerance=TOLERANCE_M)
                    counts['sat_tests']+=1
                    if result.status=='separated':
                        axis=rotations[a].T@np.asarray(result.axis);projection=self.meshes[a][ia].vertices@axis
                        self.axes[pair]=(axis,float(projection.min()),float(projection.max()))
                    if result.status in ('overlap','touching'):
                        return dict(status='COLLISION'if result.status=='overlap'else'UNRESOLVED',complete=True,
                            pair=[list(self.keys[a]),list(self.keys[b])],
                            pieces=[self.meshes[a][ia].name,self.meshes[b][ib].name],
                            axis_gap_m=result.axis_gap_m,reason=result.status,**counts)
            return dict(status='CLEAR',complete=True,**counts)
        except (GeometryError,ValueError,KeyError,IndexError,TypeError)as exc:
            return dict(status='UNSUPPORTED',complete=False,reason=str(exc),**counts)

    def evidence(self):
        value=[dict(key=list(key),pieces=[dict(name=m.name,vertices=m.vertices.tolist(),faces=m.triangles.tolist())for m in meshes])
               for key,meshes in zip(self.keys,self.meshes)]
        packed=zlib.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),1)
        return dict(self.metadata,required_evidence=REQUIRED_EVIDENCE,encoding='zlib_json_base64_v1',
            geometry_sha256=hashlib.sha256(packed).hexdigest(),geometry=base64.b64encode(packed).decode('ascii'))

    @classmethod
    def from_evidence(cls,evidence):
        if evidence.get('version')!=VERSION or evidence.get('required_evidence')!=REQUIRED_EVIDENCE or evidence.get('encoding')!='zlib_json_base64_v1':
            raise GeometryError('Unsupported or missing robot geometry evidence version.')
        encoded=evidence['geometry']
        if not isinstance(encoded,str)or len(encoded)>24*1024*1024:raise GeometryError('Robot geometry evidence exceeds size bound.')
        packed=base64.b64decode(encoded,validate=True)
        if hashlib.sha256(packed).hexdigest()!=evidence['geometry_sha256']:raise GeometryError('Robot geometry evidence hash differs.')
        decoder=zlib.decompressobj();raw=decoder.decompress(packed,128*1024*1024)
        if decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:raise GeometryError('Invalid or oversized compressed geometry.')
        value=json.loads(raw);groups={}
        if not isinstance(value,list)or len(value)!=18:raise GeometryError('Incomplete robot link inventory.')
        for item in value:
            key=tuple(item['key'])
            if key in groups or not 1<=len(item['pieces'])<=315*32:raise GeometryError('Invalid robot geometry inventory.')
            groups[key]=[prepare_mesh(p['vertices'],p['faces'],name=p['name'])for p in item['pieces']]
        result=cls(groups,JOINT_PAIRS,source=evidence['source'],collider_counts=evidence['collider_counts'])
        if any(evidence.get(k)!=v for k,v in result.metadata.items()):raise GeometryError('Robot geometry metadata differs from its complete solids.')
        return result

    @classmethod
    def from_native(cls,roots):
        from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools
        from scipy.spatial import ConvexHull
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        cooking=get_physx_cooking_interface();groups=defaultdict(list);counts={};digest=hashlib.sha256()
        def convexes(prim):
            received=[]
            cooking.request_convex_collision_representation(
                stage_id=UsdUtils.StageCache.Get().GetId(prim.GetStage()).ToLongInt(),
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),run_asynchronously=False,
                on_result=lambda result,values:received.append((result,values)))
            if len(received)!=1 or received[0][0]!=PhysxCollisionRepresentationResult.RESULT_VALID:
                raise GeometryError('Native synchronous cooked-hull callback did not return valid geometry: '+str(prim.GetPath()))
            values=received[0][1]
            if values is None or not 1<=len(values)<=32:raise GeometryError('Native cooked-hull inventory is incomplete.')
            return values
        for side,root in roots.items():
            neighbors=native_joint_neighbors(root)
            colliders=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies())if p.HasAPI(UsdPhysics.CollisionAPI)]
            counts[side]=len(colliders)
            for prim in colliders:
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()is not True:
                    raise GeometryError("Disabled source robot collider.")
                body=prim
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
                if not body or not body.GetPath().HasPrefix(root.GetPath()):raise GeometryError("Collider has no named robot body.")
                transform=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())*
                    UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse(),dtype=float).T
                if not np.isfinite(transform).all()or abs(np.linalg.det(transform[:3,:3]))<1e-12:
                    raise GeometryError("Invalid cooked-hull-to-link transform.")
                for index,cooked in enumerate(convexes(prim)):
                    if not 4<=len(cooked.polygons)<=20000 or not 4<=len(cooked.vertices)<=20000:raise GeometryError("Invalid cooked robot hull.")
                    vertices=np.array([(v.x,v.y,v.z)for v in cooked.vertices],dtype=float)
                    indices=cooked.indices;polygons=cooked.polygons
                    faces=[list(indices[p.index_base:p.index_base+p.num_vertices])for p in polygons]
                    if any(len(face)<3 or any(not 0<=i<len(vertices)for i in face)for face in faces):
                        raise GeometryError('Invalid native cooked polygon indices.')
                    name=str(prim.GetPath().MakeRelativePath(root.GetPath()))+'#'+str(index)
                    # The support hull uses every original cooked vertex without jitter or dilation.
                    hull=ConvexHull(vertices)
                    mesh=prepare_mesh(vertices@transform[:3,:3].T+transform[:3,3],hull.simplices,name=name)
                    groups[(side,body.GetName())].append(mesh)
                    digest.update(side.encode()+name.encode()+mesh.vertices.tobytes()+mesh.triangles.tobytes())
        return cls(groups,JOINT_PAIRS,source={'kind':'PhysX cooked convex vertex hulls','sha256':digest.hexdigest(),
            'triangulation':'ConvexHull of unmodified cooked vertices; no jitter, dilation or dynamics changes',
            'api':'request_convex_collision_representation'},collider_counts=counts)

    @classmethod
    def from_mujoco(cls,model):
        """Offline check of recorded states; never enables or advances physics."""
        import mujoco
        from scipy.spatial import ConvexHull
        groups=defaultdict(list);counts={'left':0,'right':0};cache={};neighbors=set();digest=hashlib.sha256()
        for g in range(model.ngeom):
            name=model.geom(g).name
            side='right'if name.startswith('right_robot_')else'left'if name.startswith('robot_')else None
            if side is None:continue
            counts[side]+=1;body=int(model.geom_bodyid[g]);link=model.body(body).name.removeprefix('right_')
            parent=model.body(int(model.body_parentid[body])).name.removeprefix('right_')
            if parent!='world':neighbors.add(tuple(sorted((link,parent))))
            if int(model.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_MESH):raise GeometryError("Expected imported robot mesh.")
            mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
            vertices=np.asarray(model.mesh_vert[start:start+n],dtype=float)
            key=vertices.tobytes()
            if key not in cache:
                hull=ConvexHull(vertices);cache[key]=prepare_mesh(vertices,hull.simplices,name=model.mesh(mid).name)
            pose=np.eye(4);r=np.empty(9);mujoco.mju_quat2Mat(r,model.geom_quat[g]);pose[:3,:3]=r.reshape(3,3);pose[:3,3]=model.geom_pos[g]
            mesh=transform_mesh(cache[key],pose);groups[(side,link)].append(mesh)
            digest.update(side.encode()+name.encode()+mesh.vertices.tobytes()+mesh.triangles.tobytes())
        return cls(groups,neighbors,source={'kind':'Compiled MuJoCo mesh convex hulls','sha256':digest.hexdigest()},collider_counts=counts)


def verify_benchmark_trial(directory):
    """Stream the full measured trace; old traces without required geometry fail closed."""
    import importlib
    from wafer_task_score import WaferTaskScorer
    from wafer_dust import CohesiveDustScore
    from wafer_policy_score import combine_score
    directory=Path(directory);metadata=json.loads((directory/'metadata.json').read_text())
    report=json.loads((directory/'task_result.json').read_text())
    for name in ('wafer_benchmark','wafer_benchmark_backend','wafer_task_score','wafer_robot_geometry','collision_geometry','wafer_policy_score'):
        module=importlib.import_module(name)
        if metadata.get('benchmark_sources',{}).get(name+'.py')!=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest():
            raise ValueError('Independent benchmark implementation differs: '+name)
    evidence=metadata['robot_geometry']
    if evidence.get('source',{}).get('kind')!='PhysX cooked convex vertex hulls':raise ValueError('Official native cooked robot vertex geometry is required.')
    guard=RobotGeometryGuard.from_evidence(evidence);monitor=WaferTaskScorer(metadata,sample_dt_s=metadata['physics_dt_s'])
    particles=CohesiveDustScore(metadata)
    dt=float(metadata['physics_dt_s']);issue=None;count=0
    if not np.isfinite(dt)or dt<=0:raise ValueError('Invalid physical sample cadence.')
    digest=hashlib.sha256()
    with (directory/'samples.jsonl.gz').open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=report.get('trace_sha256'):raise ValueError('The recorded trace hash differs.')
    with gzip.open(directory/'samples.jsonl.gz','rt')as stream:
        for line in stream:
            row=json.loads(line)
            if not np.isfinite(row['time_s'])or abs(row['time_s']-count*dt)>1e-8:raise ValueError('Missing or reordered physics sample.')
            poses=row.get('robot_link_poses_world')
            if not isinstance(poses,list)or len(poses)!=len(guard.keys):
                result={'status':'UNSUPPORTED','complete':False,'reason':'Missing measured robot link inventory.'}
            else:result=guard.check(dict(zip(guard.keys,poses)))
            measured=compact_result(result)
            if row.get('robot_geometry_check')!=measured:raise ValueError('Independent robot geometry result differs at sample '+str(count))
            issue=first_issue(issue,row,measured)
            if count==0:monitor.reset(row,report=False)
            else:monitor.update(row,report=False)
            particles.update(row)
            count+=1
    if not count or report.get('physics_sample_count')!=count or abs(report.get('duration_s',-1)-(count-1)*dt)>1e-8:
        raise ValueError('Final trace count or duration differs.')
    if report.get('post_start_pose_write_count')!=0 or metadata.get('benchmark_trial',{}).get('post_start_pose_write_count')!=0:
        raise ValueError('Post-start state replacement is forbidden.')
    task=combine_score(apply_validity(monitor.result(),issue),particles.result())
    if task!=report.get('task'):raise ValueError('Independent measured task result differs.')
    complete=task['evidence_complete']and report.get('runtime_error')is None
    return dict(result='INDEPENDENT_BENCHMARK_EVIDENCE_MATCH'if complete else'INCOMPLETE',matches_report=True,
        passed=complete and task['success']and task['score']==1.,task_score=task,physics_samples=count,
        trace_sha256=digest.hexdigest(),evidence=dict(complete=complete,robot_geometry_version=VERSION))


class _MeasuredRobotFK:
    def __init__(self,model):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=mujoco.MjData(model)
        self.keys=tuple((s,n)for s in ('left','right')for n in LINK_NAMES)
        self.bodies=[model.body(('right_'if s=='right'else'')+n).id for s,n in self.keys]
        self.fields=[]
        for side,prefix in (('left',''),('right','right_')):
            for kind,indices,length in (('arm',range(1,7),6),('finger',(7,8),2)):
                names=[prefix+'joint'+str(i)for i in indices];joints=[model.joint(n).id for n in names]
                expected=mujoco.mjtJoint.mjJNT_HINGE if kind=='arm'else mujoco.mjtJoint.mjJNT_SLIDE
                if any(model.jnt_type[j]!=expected for j in joints):raise ValueError('Unexpected measured robot joint type.')
                field=('measured_'if side=='left'else'measured_right_')+('arm_q_rad'if kind=='arm'else'finger_q_m')
                self.fields.append((field,np.array([model.jnt_qposadr[j]for j in joints],dtype=int),length))
            base=model.body(prefix+'base').id
            if model.body_parentid[base]!=0 or model.body_jntnum[base]or model.body_mocapid[base]>=0:
                raise ValueError('Recorded robot bases must be fixed to the world.')

    def measured(self,row):
        for field,indices,length in self.fields:
            values=np.asarray(row[field])
            if values.shape!=(length,)or values.dtype.kind not in 'fiu'or not np.isfinite(values).all():
                raise ValueError('Missing or invalid actual robot state: '+field)
            self.data.qpos[indices]=values
        self.mj.mj_kinematics(self.model,self.data)
        poses=np.c_[self.data.xpos[self.bodies],self.data.xquat[self.bodies][:,[1,2,3,0]]]
        measured_positions,measured_rotations=_rotations([row['measured_hand_pose_world'],row['right_hand_pose_world']])
        hand_ids=[self.model.body(n).id for n in ('gripper','right_gripper')]
        position_error=float(np.max(abs(measured_positions-self.data.xpos[hand_ids])))
        rotation_error=float(np.max(abs(measured_rotations-self.data.xmat[hand_ids].reshape(2,3,3))))
        if max(position_error,rotation_error)>1e-9:raise ValueError('Measured robot joints and recorded hand poses disagree.')
        return dict(zip(self.keys,poses.tolist())),position_error,rotation_error


def audit_local_robot_geometry(directory,*,expected_trace_sha256=None):
    """All actual joint samples through FK and convex geometry; no dynamics advance."""
    import mujoco
    import collision_geometry
    directory=Path(directory);count=0;first=None;incomplete=None;collisions=0;position_peak=rotation_peak=replay_position_peak=replay_rotation_peak=0.
    result=dict(version=LOCAL_EVIDENCE,result='INCOMPLETE',passed=False,evidence_complete=False,physics_samples=0,
        audit_sources={'wafer_robot_geometry.py':_file_sha(__file__),'collision_geometry.py':_file_sha(collision_geometry.__file__)},
        scope='Every recorded physics-step actual arm/finger joint state; fixed-base FK and complete convex robot geometry. No commanded targets, interpolation or dynamics stepping.')
    try:
        metadata=json.loads((directory/'metadata.json').read_text());report=json.loads((directory/'report.json').read_text())
        for name,digest in result['audit_sources'].items():
            if metadata.get('full_workflow_sources',{}).get('benchmark/'+name)!=digest:
                raise ValueError('Missing or changed startup audit source: '+name)
        scene=directory/'scene.xml';scene_hash=_file_sha(scene);result['scene_sha256']=scene_hash
        if metadata.get('model_xml_sha256')!=scene_hash:raise ValueError('Recorded scene XML hash differs.')
        assets=local_robot_asset_digest(scene);result['robot_collision_assets_sha256']=assets
        if metadata.get('robot_collision_assets_sha256')!=assets:raise ValueError('Missing or changed startup robot collision asset digest.')
        if metadata.get('engine')!='MuJoCo'or metadata.get('mujoco_version')!=mujoco.__version__:
            raise ValueError('The recorded MuJoCo engine/version differs from the FK audit runtime.')
        model=mujoco.MjModel.from_xml_path(str(scene));fk=_MeasuredRobotFK(model);guard=RobotGeometryGuard.from_mujoco(model)
        result['geometry']=dict(guard.metadata)
        dt=float(metadata['physics_dt_s'])
        if dt<=0 or not np.isfinite(dt):raise ValueError('Invalid recorded physics cadence.')
        stride=round(.05/dt)
        if abs(model.opt.timestep-dt)>1e-12 or stride<1 or abs(stride*dt-.05)>1e-12:
            raise ValueError('Recorded physics/control cadence differs from the compiled scene.')
        result['physics_dt_s']=dt;result['trace_sha256']=_file_sha(directory/'samples.jsonl.gz')
        if expected_trace_sha256 is not None and result['trace_sha256']!=expected_trace_sha256:raise ValueError('Recorded physics trace hash differs from the bound audit.')
        result['replay_sha256']=_file_sha(directory/'replay_states.npz');replay_data=mujoco.MjData(model);replay_count=0
        with np.load(directory/'replay_states.npz',allow_pickle=False)as replay:
            times=replay['time_s'];states=replay['qpos']
            if times.ndim!=1 or states.shape!=(len(times),model.nq)or not np.isfinite(times).all()or not np.isfinite(states).all():
                raise ValueError('Incomplete actual replay states.')
            with gzip.open(directory/'samples.jsonl.gz','rt')as stream:
                for line in stream:
                    row=json.loads(line);t=float(row['time_s'])
                    if not np.isfinite(t)or abs(t-count*dt)>1e-8:raise ValueError('Missing, duplicate or reordered physics samples.')
                    poses,pe,re=fk.measured(row);position_peak=max(position_peak,pe);rotation_peak=max(rotation_peak,re)
                    if count%stride==0:
                        if replay_count>=len(times)or abs(float(times[replay_count])-t)>1e-8:raise ValueError('Measured replay timestamps differ.')
                        replay_data.qpos[:]=states[replay_count]
                        for _,indices,_ in fk.fields:
                            if not np.allclose(replay_data.qpos[indices],fk.data.qpos[indices],atol=1e-12,rtol=0):raise ValueError('Actual replay and raw robot joint states differ.')
                        mujoco.mj_kinematics(model,replay_data)
                        pe=float(np.max(abs(replay_data.xpos[fk.bodies]-fk.data.xpos[fk.bodies])))
                        re=float(np.max(abs(replay_data.xmat[fk.bodies]-fk.data.xmat[fk.bodies])))
                        replay_position_peak=max(replay_position_peak,pe);replay_rotation_peak=max(replay_rotation_peak,re)
                        if max(pe,re)>1e-9:raise ValueError('Full actual replay and measured-joint FK differ.')
                        replay_count+=1
                    checked=guard.check(poses);first=first_issue(first,row,checked)
                    if checked['status']=='COLLISION':collisions+=1
                    elif checked['status']!='CLEAR':incomplete=first_issue(incomplete,row,checked)
                    count+=1
            duration=float(report['simulation_seconds'])
            if not count or not np.isfinite(duration)or abs((count-1)*dt-duration)>1e-8 or replay_count!=len(times):
                raise ValueError('Incomplete physics/replay duration or final sample count.')
        complete=incomplete is None
        result.update(result='CLEAR_IN_ALL_RECORDED_PHYSICS_SAMPLES'if first is None else'COLLISION'if complete else'INCOMPLETE',
            passed=first is None,evidence_complete=complete,physics_samples=count,replay_samples=replay_count,first_issue=first,
            collision_samples=collisions,first_incomplete_issue=incomplete,
            maximum_recorded_hand_position_error_m=position_peak,maximum_recorded_hand_rotation_matrix_error=rotation_peak,
            maximum_replay_link_position_error_m=replay_position_peak,maximum_replay_link_rotation_matrix_error=replay_rotation_peak,
            source_binding='Recorded scene/startup external-mesh digest and current audit sources; returned trace/replay hashes bind this independent result.')
    except (ValueError,KeyError,IndexError,TypeError,OSError,RuntimeError)as exc:
        result.update(result='INCOMPLETE',passed=False,evidence_complete=False,physics_samples=count,error=type(exc).__name__+': '+str(exc))
    return result


def evaluate_native_robot_geometry(metadata,rows):
    """Evaluate every saved native link pose; usable before or after the trace closes."""
    import collision_geometry
    count=0;issue=None;incomplete=None;collisions=0
    result=dict(version=NATIVE_OFFLINE_EVIDENCE,result='INCOMPLETE',passed=False,evidence_complete=False,physics_samples=0,
        audit_sources={'wafer_robot_geometry.py':_file_sha(__file__),'collision_geometry.py':_file_sha(collision_geometry.__file__)})
    try:
        for name,digest in result['audit_sources'].items():
            if metadata.get('full_workflow_sources',{}).get('benchmark/'+name)!=digest:
                raise ValueError('Missing or changed startup audit source: '+name)
        capture=metadata['robot_geometry_capture'];signatures=capture['geometry_definition_sha256']
        if capture.get('version')!=NATIVE_OFFLINE_EVIDENCE or set(signatures)!={'left','right'}:
            raise ValueError('Missing native robot geometry capture contract.')
        if any(not isinstance(x,str)or len(x)!=64 or any(c not in '0123456789abcdef'for c in x)for x in signatures.values()):
            raise ValueError('Missing native geometry-definition signatures.')
        evidence=metadata['robot_geometry']
        if evidence.get('source',{}).get('kind')!='PhysX cooked convex vertex hulls':raise ValueError('Actual native cooked robot vertex hulls are required.')
        guard=RobotGeometryGuard.from_evidence(evidence);dt=float(metadata['physics_dt_s'])
        if not np.isfinite(dt)or dt<=0:raise ValueError('Invalid native physics cadence.')
        result.update(geometry_sha256=evidence['geometry_sha256'],geometry=dict(guard.metadata),physics_dt_s=dt,
            geometry_definition_sha256=dict(signatures))
        for row in rows:
            t=float(row['time_s'])
            if not np.isfinite(t)or abs(t-count*dt)>1e-8:raise ValueError('Missing, duplicate or reordered native physics sample.')
            poses=row.get('robot_link_poses_world')
            if row.get('robot_geometry_intact')is not True:
                checked=dict(status='UNSUPPORTED',complete=False,reason='Native robot geometry integrity is missing or changed.')
            elif not isinstance(poses,list)or len(poses)!=len(guard.keys):
                checked=dict(status='UNSUPPORTED',complete=False,reason='Incomplete actual native robot link inventory.')
            else:checked=guard.check(dict(zip(guard.keys,poses)))
            issue=first_issue(issue,row,checked)
            if checked['status']=='COLLISION':collisions+=1
            elif checked['status']!='CLEAR':incomplete=first_issue(incomplete,row,checked)
            count+=1
        if count<2:raise ValueError('No complete native physics interval was recorded.')
        complete=incomplete is None
        result.update(result='CLEAR_IN_ALL_RECORDED_PHYSICS_SAMPLES'if issue is None else'COLLISION'if complete else'INCOMPLETE',
            passed=issue is None,evidence_complete=complete,physics_samples=count,duration_s=(count-1)*dt,
            first_issue=issue,collision_samples=collisions,first_incomplete_issue=incomplete)
    except (ValueError,KeyError,IndexError,TypeError,OSError,RuntimeError)as exc:
        result.update(result='INCOMPLETE',passed=False,evidence_complete=False,physics_samples=count,error=type(exc).__name__+': '+str(exc))
    return result


def audit_native_robot_geometry(directory,*,expected_trace_sha256=None):
    """Bind the offline native geometry result to the closed raw physics trace."""
    directory=Path(directory);result=dict(version=NATIVE_OFFLINE_EVIDENCE,result='INCOMPLETE',passed=False,evidence_complete=False)
    try:
        metadata=json.loads((directory/'metadata.json').read_text());report=json.loads((directory/'trial_report.json').read_text())
        digest=_file_sha(directory/'samples.jsonl.gz')
        if expected_trace_sha256 is not None and digest!=expected_trace_sha256:raise ValueError('Native trace hash differs from the bound audit.')
        with gzip.open(directory/'samples.jsonl.gz','rt')as stream:
            result=evaluate_native_robot_geometry(metadata,(json.loads(line)for line in stream))
        result['trace_sha256']=digest
        if result.get('error'):return result
        trace=report['trace']
        if (report.get('recorded_physics_samples')!=result['physics_samples']or trace.get('sample_count')!=result['physics_samples']
            or abs(float(trace.get('start_time_s',-1)))>1e-10
            or abs(float(trace.get('physics_dt_s',-1))-result['physics_dt_s'])>1e-12
            or abs(float(trace.get('end_time_s',-1))-result['duration_s'])>1e-8):
            raise ValueError('Native report count, duration or cadence differs from the raw trace.')
    except (ValueError,KeyError,IndexError,TypeError,OSError,RuntimeError)as exc:
        result.update(result='INCOMPLETE',passed=False,evidence_complete=False,error=type(exc).__name__+': '+str(exc))
    return result
