"""Read-only geometric diagnostics. No score, motor target, or object-state writes.
All positions are material feature positions, not off-center rigid-body origins.
Opening fraction is a deterministic sampled projected area, not a fluid measurement.
"""
import numpy as np

class GeometryObserver:
 def __init__(self,ns):
  self.ns=ns;self.previous=None
  r=ns['RIM']*np.sqrt((np.arange(64)+.5)/64)
  theta=2*np.pi*(np.arange(128)+.5)/128
  self.samples=np.stack([np.repeat(r,128)*np.tile(np.cos(theta),64),np.repeat(r,128)*np.tile(np.sin(theta),64),np.zeros(8192)],axis=1)

 def read(self,states,dt=.05):
  n=self.ns;c=states['carton'];s=states['seal'];cap=states['cap']
  mouth=c.to_world(n['C']+n['S']@np.array([0,0,n['SEAL_Z']]))
  axis=c.rotation()@n['S'];seal_c=s.to_world(n['C']+n['S']@np.array([0,0,n['SEAL_Z']]))
  seal_axis=s.rotation()@n['S'];local=axis.T@(seal_c-mouth);normal=axis.T@seal_axis[:,2]
  denom=normal[2]
  blocked=np.zeros(len(self.samples),dtype=bool)
  if abs(denom)>1e-8:
   t=(local@normal-self.samples@normal)/denom
   hits=self.samples.copy();hits[:,2]=t
   radial=np.linalg.norm(hits-local,axis=1)
   blocked=(t>=-.030)&(t<=.030)&(radial<=n['DISK_R'])
  cap_c=cap.to_world(n['C']+n['S']@np.array([0,0,.012])) if n['FULL_CAP'] else cap.to_world([0,0,.008])
  speed=None if self.previous is None else float(np.linalg.norm(seal_c-self.previous)/dt)
  self.previous=seal_c.copy()
  theta=2*np.pi*np.arange(96)/96
  rim=seal_c+np.stack([n['DISK_R']*np.cos(theta),n['DISK_R']*np.sin(theta),np.zeros(96)],axis=1)@seal_axis.T
  # This clearance certifies the disk is fully outside the mouth region. It
  # does not report individual PhysX constraint break events.
  clear=bool(np.min(np.linalg.norm(rim-mouth,axis=1))>.030 and (local[2]>.030 or np.linalg.norm(local[:2])>.030))
  return {'mouth_center':mouth.tolist(),'seal_center':seal_c.tolist(),'cap_center':cap_c.tolist(),
   'seal_center_relative_mm':(1000*local).tolist(),'ring_angle_rad':s.joints['pullring'],
   'cap_clear_of_mouth':bool(np.linalg.norm(cap_c-mouth)>.055),
   'seal_clear_of_mouth':clear,'sampled_open_fraction':float(1-blocked.mean()),
   'seal_center_speed_mps':speed,'carton_tilt_deg':c.tilt_deg()}

def ring_inside_tray(ring_offsets, hinge_position, hinge_angle, seal, tray):
 """Check the actual articulated ring mesh against the tray's interior footprint."""
 c, s = np.cos(hinge_angle), np.sin(hinge_angle)
 rotation = np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
 seal_local = ring_offsets @ rotation.T + hinge_position
 world = seal_local @ seal.rotation().T + seal.pos
 tray_local = (world - tray.pos) @ tray.rotation()
 return bool((np.abs(tray_local[:,0]) <= .098).all() and
             (np.abs(tray_local[:,1]) <= .058).all() and
             (tray_local[:,2] >= .003).all())

def penetrates_carton_interior(center, skin, bore_radius, inner_plane):
 """The interior starts at the underside of the modeled spout, below the seal seat."""
 if center[2] < inner_plane and np.linalg.norm(center[:2]) < bore_radius:
  return True
 return bool(((skin[:,2] < inner_plane) &
              (np.linalg.norm(skin[:,:2],axis=1) < bore_radius-.001)).any())
