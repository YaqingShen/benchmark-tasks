"""Select a scripted receiving target using the modeled lip, hook and molded ring."""
import itertools
import xml.etree.ElementTree as ET

import hppfcl as fcl
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from yam_arena.ik import YAM_URDF
from yam_arena.urdf import forward_kinematics


def mesh(components):
    vertices=fcl.StdVec_Vec3s()
    triangles=fcl.StdVec_Triangle()
    for points in components:
        offset=len(vertices)
        for point in points:
            vertices.append(point)
        for face in ConvexHull(points).simplices:
            triangles.append(fcl.Triangle(*(int(i)+offset for i in face)))
    result=fcl.BVHModelOBBRSS()
    result.beginModel(len(triangles),len(vertices))
    result.addSubModel(vertices,triangles)
    result.endModel()
    return result


def frame(rotation,position):
    return fcl.Transform3f(rotation,position)


class ReceivingGeometry:
    def __init__(self,ns):
        self.ns=ns
        corners=np.array(list(itertools.product((-1,1),repeat=3)))
        self.lip=mesh([m.points for m in ns['CARTON'].meshes[1:33]])
        self.wire=mesh([m.points-ns['anchor'] for m in ns['ringjoint'].meshes])
        self.tool=mesh([np.asarray(b[0])+corners*np.asarray(b[1])/2 for b in ns['HOOK'].boxes])
        tree=ET.parse(YAM_URDF)
        poses=forward_kinematics(YAM_URDF,{})
        opened=forward_kinematics(YAM_URDF,dict(joint7=-.01,joint8=-.01))
        self.fingers=[]
        self.slides=[]
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        for side in ('tip_left','tip_right'):
            local=np.linalg.inv(poses['gripper'])@poses[side]
            displaced=np.linalg.inv(opened['gripper'])@opened[side]
            self.slides.append((displaced[:3,3]-local[:3,3])/.01)
            parts=[]
            for col in tree.find(f'link[@name="{side}"]').findall('collision'):
                raw=(YAM_URDF.parent/col.find('geometry/mesh').get('filename')).read_bytes()
                pts=np.unique(np.frombuffer(raw,offset=84,dtype=dtype)['vertices'].reshape(-1,3),axis=0)
                origin=col.find('origin')
                rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                pos=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                parts.append((pts@rotation.T+pos)@local[:3,:3].T+local[:3,3]-[0,0,.134])
            self.fingers.append(mesh(parts))
        self.request=fcl.DistanceRequest()

    def clearance(self,carton,hook,rotation,target,apertures):
        lip_pose=frame(carton.rotation(),carton.pos)
        tool_pose=frame(hook.rotation(),hook.pos)
        result=float('inf')
        for finger,slide,gap in zip(self.fingers,self.slides,apertures):
            pose=frame(rotation,target+rotation@slide*gap)
            for body,where in ((self.lip,lip_pose),(self.tool,tool_pose)):
                result=min(result,fcl.distance(finger,pose,body,where,self.request,fcl.DistanceResult()))
        return result

    def select(self,carton,seal,hook,rotation,spot,inset):
        normal=carton.rotation()@self.ns['N']
        approach=rotation[:,2]
        lip_pose=frame(carton.rotation(),carton.pos)
        tool_pose=frame(hook.rotation(),hook.pos)
        ring_angle=seal.joints['pullring']
        wire_pose=frame(seal.rotation()@Rotation.from_rotvec([0,ring_angle,0]).as_matrix(),
                        seal.to_world(self.ns['anchor']))
        nominal=spot-inset*approach
        best=None
        for ahead,side,lift in itertools.product((0.,.003,.006,.009,.012),(-.004,0.,.004),(.001,.003,.005,.007)):
            target=nominal+ahead*approach+side*rotation[:,0]+lift*normal
            # Reserve distal pad length for grip settling and initial pull.
            if (spot-target)@approach>.008:continue
            clearance=.025
            gap_at_contact=0.
            for travel,gap in ((.022,.017),(.012,.016),(.005,.014),(0.,.013),(0.,.011),(0.,.009),(0.,.007),(0.,.005),(0.,.003)):
                positions=[frame(rotation,target-travel*approach+rotation@slide*gap) for slide in self.slides]
                def gap_to(body,pose):
                    return [fcl.distance(finger,where,body,pose,self.request,fcl.DistanceResult())
                            for finger,where in zip(self.fingers,positions)]
                clearance=min(clearance,*gap_to(self.lip,lip_pose),*gap_to(self.tool,tool_pose))
                if clearance<.001:
                    break
                if not travel and max(gap_to(self.wire,wire_pose))<.0006:
                    # The wire must still intersect both pads after a small slip.
                    wire_r=seal.rotation()@Rotation.from_rotvec([0,ring_angle,0]).as_matrix()
                    for shift in (.003*approach,-.003*normal):
                        shifted=frame(wire_r,seal.to_world(self.ns['anchor'])+shift)
                        if max(gap_to(self.wire,shifted))>.001:break
                    else:
                        gap_at_contact=gap
                        break
            if gap_at_contact<.005:
                continue
            rank=(gap_at_contact>=.007,min(clearance,.004)+.6*gap_at_contact,clearance)
            if best is None or rank>best[0]:
                best=(rank,target,clearance,gap_at_contact)
        if best is None:
            return None
        _,target,clearance,gap=best
        return target,{'predicted_min_clearance_mm':round(1000*clearance,2),
                       'predicted_half_gap_mm':round(1000*gap,2),
                       'wire_in_tcp_mm':np.round((spot-target)@rotation*1000,2).tolist()}
