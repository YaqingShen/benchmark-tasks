"""Bimanual joint and contact control used by the packing demonstrator."""
import numpy as np
from gripper_geometry import finger_vertices


class Driver:
 def __init__(self,w):self.w=w;self.ik=w.ik;self.action=w.joints();self.home=self.action.copy();self.max_error=0.;self.support={};self.outer_handles={}

 def pose(self,side):
  o=0 if side=='left' else 7;return self.ik.tcp_pose(self.w.joints()[o:o+6],self.w.station.BASES[side])

 def solve_command(self,side,p,r,gap=None,label='servo'):
  o=0 if side=='left' else 7;q,ep,er=self.ik.solve(p,self.ik.matrix_to_quat_xyzw(r),self.action[o:o+6],self.w.station.BASES[side],iters=180,damping=.018,tol=3e-5)
  self.max_error=max(self.max_error,ep)
  if ep>.0025 or er>.025:
   candidate,ep2,er2=self.bounded_ik(p,r,self.action[o:o+6],side)
   if ep2<.0025 and er2<.025 and np.max(np.abs(candidate-self.action[o:o+6]))<.25:q,ep,er=candidate,ep2,er2
  if ep>.0025 or er>.025:raise RuntimeError(f'IK {label}: {ep} {er}')
  self.action[o:o+6]=q
  if gap is not None:self.action[o+6]=gap/.0939

 def step(self,label,exclude=None):
  for side,(p,r,gap) in self.support.items():
   if side!=exclude:self.feedback_command(side,p,r,gap,'support '+side)
  return self.w.step(self.action,label)

 def command_pose(self,side,p,r,gap=None,label='servo'):
  self.solve_command(side,p,r,gap,label);return self.step(label,side)

 def move(self,side,p,r,gap=None,seconds=2.,label='move'):
  o=0 if side=='left' else 7;q=self.action[o:o+6];p0,r0=self.ik.tcp_pose(q,self.w.station.BASES[side]);g0=self.action[o+6]*.0939;g1=g0 if gap is None else gap
  print('ROBOT_PHASE',label,flush=True)
  for i in range(round(seconds*20)):
   s=self.ik.smoothstep((i+1)/(seconds*20));self.command_pose(side,p0*(1-s)+np.asarray(p)*s,self.ik.slerp_matrix(r0,r,s),g0*(1-s)+g1*s,label)
  self.w.snapshot(f'{self.w.steps:05d}-'+label.replace(' ','-'))

 def feedback_command(self,side,target,rotation,gap,label):
  o=0 if side=='left' else 7;actual,actual_r=self.pose(side);command,command_r=self.ik.tcp_pose(self.action[o:o+6],self.w.station.BASES[side])
  delta=(np.array(target)-actual)*.35;delta*=min(1.,.001/max(np.linalg.norm(delta),1e-12))
  correction=self.ik.slerp_matrix(np.eye(3),rotation@actual_r.T,.25)
  compensation=float(np.linalg.norm(command-np.array(target)));tracking=float(np.linalg.norm(actual-np.array(target)))
  if compensation>.035 or (compensation>.015 and tracking>.010):raise RuntimeError(f'Contact servo obstruction/compliance limit: side={side}, phase={label}, tracking={tracking}, compensation={compensation}, actual={actual.tolist()}, target={np.asarray(target).tolist()}, command={command.tolist()}')
  return self.solve_command(side,command+delta,correction@command_r,gap,label)

 def feedback_step(self,side,target,rotation,gap,label):
  self.feedback_command(side,target,rotation,gap,label);return self.step(label,side)

 def bounded_ik(self,point,rotation,seed,side):
  from scipy.optimize import least_squares
  def residual(q):
   p,r=self.ik.tcp_pose(q,self.w.station.BASES[side]);return np.r_[p-np.asarray(point),.3*self.ik._rot_error(rotation,r)]
  sol=least_squares(residual,np.clip(seed,self.ik.LOWER+1e-8,self.ik.UPPER-1e-8),bounds=(self.ik.LOWER,self.ik.UPPER),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  error=residual(sol.x);return sol.x,float(np.linalg.norm(error[:3])),float(np.linalg.norm(error[3:])/.3)

 def move_joints_safe(self,side,point,rotation,seconds=3.,label='clearance joint approach'):
  o=0 if side=='left' else 7;start=self.action[o:o+6].copy()
  q,ep,er=self.ik.solve(point,self.ik.matrix_to_quat_xyzw(rotation),self.w.constants.READY,self.w.station.BASES[side],iters=800,damping=.018)
  if ep>.001 or er>.01:q,ep,er=self.bounded_ik(point,rotation,self.w.constants.READY,side)
  if ep>.001 or er>.01:raise RuntimeError(f'Unreachable clearance endpoint: {label} ep={ep} er={er}')
  minimum=min(self.ik.tcp_pose(start*(1-f)+q*f,self.w.station.BASES[side])[0][2] for f in np.linspace(0,1,100))
  if minimum<.85:raise RuntimeError('Clearance joint path approaches table')
  g0=self.action[o+6]
  for i in range(round(seconds*20)):
   f=self.ik.smoothstep((i+1)/(seconds*20));self.action[o:o+6]=start*(1-f)+q*f;self.action[o+6]=g0*(1-f);self.step(label,side)
  self.w.snapshot(f'{self.w.steps:05d}-'+label.replace(' ','-'))

 def hold(self,t,label):
  for _ in range(round(t*20)):self.step(label)

 def hand_force(self,side):
  ids=[0,1] if side=='left' else [2,3]
  return sum(float(self.w.hand_panel_forces(i).sum()) for i in ids)

 def panel_force(self,side,name):
  ids=[0,1] if side=='left' else [2,3];order=list(self.w.manifest['panels']);index=order.index(name)
  return sum(float(self.w.hand_panel_forces(i)[index]) for i in ids)

 def return_home(self,side):
  o=0 if side=='left' else 7;start=self.action[o:o+6].copy();finish=self.home[o:o+6]
  minimum=min(self.ik.tcp_pose(start*(1-f)+finish*f,self.w.station.BASES[side])[0][2] for f in np.linspace(0,1,100))
  if minimum<self.w.support_top+.045:raise RuntimeError('Home return has insufficient cardboard clearance')
  for k in range(50):
   f=self.ik.smoothstep((k+1)/50);self.action[o:o+6]=start*(1-f)+finish*f;self.action[o+6]=.040/.0939;self.step('return '+side+' hand to clearance pose',side)

 def face_tool_pose(self,point,normal,rotation,preload=.0002,tips_only=False):
  vertices=finger_vertices()
  if tips_only:vertices=vertices[vertices[:,2]>.009]
  rotated=vertices@rotation.T;dots=rotated@(-normal);extreme=rotated[dots>dots.max()-.0001].mean(0)
  return np.asarray(point)+normal*(.0006-preload)-extreme

