"""Mass properties of the same clipped liquid volume used for visualization."""
import numpy as np
from native_liquid_visuals import clipped_interior


def liquid_properties(interior,pose,level,mass_kg):
    if mass_kg<=0:return np.zeros(3),np.zeros((3,3))
    points,counts,indices=clipped_interior(interior,pose,level)
    points=np.asarray(points);triangles=[];start=0
    for count in counts:
        face=indices[start:start+count]
        triangles.extend([[face[0],face[i],face[i+1]] for i in range(1,count-1)])
        start+=count
    if not triangles:raise ValueError('Positive liquid mass has no geometric volume')
    t=points[np.asarray(triangles)];a,b,c=t[:,0],t[:,1],t[:,2]
    volumes=np.einsum('ij,ij->i',a,np.cross(b,c))/6
    volume=volumes.sum()
    if volume<=0:raise ValueError('Liquid boundary is not consistently outward')
    sums=t.sum(axis=1)
    centroid=np.einsum('i,ij->j',volumes,sums)/(4*volume)
    second=(np.einsum('n,nvi,nvj->ij',volumes,t,t)+np.einsum('n,ni,nj->ij',volumes,sums,sums))/(20*volume)
    covariance=second-np.outer(centroid,centroid)
    inertia=mass_kg*(np.trace(covariance)*np.eye(3)-covariance)
    if np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Nonpositive liquid inertia')
    return centroid,inertia


def combine_masses(parts):
    """Combine (mass, centre, inertia about centre), all in one body frame."""
    mass=sum(m for m,_,_ in parts)
    if mass<=0:raise ValueError('Total mass must be positive')
    center=sum(m*np.asarray(c) for m,c,_ in parts)/mass
    inertia=np.zeros((3,3))
    for m,c,I in parts:
        offset=np.asarray(c)-center
        inertia+=I+m*(np.dot(offset,offset)*np.eye(3)-np.outer(offset,offset))
    return mass,center,inertia


def principal_frame(inertia):
    """Return a right-handed principal-axis frame and its positive moments."""
    inertia=np.asarray(inertia,dtype=float)
    if inertia.shape!=(3,3) or not np.allclose(inertia,inertia.T,atol=1e-12):
        raise ValueError('Expected symmetric 3 by 3 inertia')
    moments,rotation=np.linalg.eigh(inertia)
    if not np.isfinite(moments).all() or moments.min()<=0:
        raise ValueError('Expected positive principal moments')
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    return moments,rotation


def pipette_liquid_properties(volume_ml,mass_kg):
    """Axial two-section approximation shared with the pneumatic column model.

    The liquid is assigned to the stem rigid body. It fills the narrow tube
    first, followed by a wider 47 mm column. This approximates liquid held in a
    vertically oriented pipette; it does not model sloshing in a tilted bulb.
    """
    from pipette_geometry import STEM_LENGTH_M,STEM_LIQUID_CAPACITY_ML
    if volume_ml<=0 or mass_kg<=0:return np.zeros(3),np.zeros((3,3))
    if volume_ml>5.+1e-9:raise ValueError('Pipette capacity exceeded')
    parts=[];remaining=volume_ml
    for start,length,capacity in [(0.,STEM_LENGTH_M,STEM_LIQUID_CAPACITY_ML),
                                  (STEM_LENGTH_M,.047,5.-STEM_LIQUID_CAPACITY_ML)]:
        volume=min(remaining,capacity)
        if volume<=0:break
        height=length*volume/capacity;radius=np.sqrt(capacity*1e-6/(np.pi*length))
        mass=mass_kg*volume/volume_ml
        inertia=np.diag([mass*(3*radius**2+height**2)/12]*2+[mass*radius**2/2])
        parts.append((mass,np.array([0.,0.,start+height/2]),inertia));remaining-=volume
    _,center,inertia=combine_masses(parts)
    return center,inertia
