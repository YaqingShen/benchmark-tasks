"""Externally checkable scoring for one roll, one marked compartment and its empty wrapper."""
import numpy as np
import geometry as g

def exits_at_open_end(previous, current, direction):
    """Interpolate centre crossing of a paper mouth between observations."""
    a=direction*previous[:,2]-g.L/2
    b=direction*current[:,2]-g.L/2
    crossing=(a<0)&(b>=0)
    f=np.divide(-a,b-a,out=np.zeros_like(a),where=crossing)
    at_plane=previous[:,:2]+f[:,None]*(current[:,:2]-previous[:,:2])
    return crossing&(np.linalg.norm(at_plane,axis=1)<g.BORE)

def in_compartment(pos, axis, table_height):
    """Check the footprint and height of the entire tilted coin inside the rim."""
    axis=np.clip(axis,-1.,1.)
    extent=g.R*np.sqrt(np.maximum(0.,1.-axis**2))+g.T/2*abs(axis)
    return ((abs(pos[:,:2]-g.TARGET)+extent[:,:2]<=g.CELL/2+.0005).all(axis=1)
            &(pos[:,2]+extent[:,2]<=table_height+g.RIM+.0005)
            &(pos[:,2]-extent[:,2]>=table_height+g.FLOOR-.001))

def terminal_score(delivered, rest, paper_on_counter):
    n=int(np.count_nonzero(delivered&rest))
    return .9*n/g.COUNT+.1*float(n==g.COUNT and paper_on_counter)
