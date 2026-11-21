"""Display the same film cells, release liner and separation front used by the material model."""
import math
import numpy as np
from material import LENGTH, WIDTH, ROWS, COLS, TAB_Y


class FilmMesh:
    def __init__(self, stage, root):
        from pxr import UsdGeom, Vt
        self.film = UsdGeom.Mesh.Define(stage,root+"/Protector")
        self.liner = UsdGeom.Mesh.Define(stage,root+"/ReleaseLiner")
        for mesh,count in ((self.film,ROWS*COLS),(self.liner,3)):
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]*count))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(range(4*count)))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
        self.film.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        self.liner.CreateDisplayColorAttr([(0.29,0.60,0.86)])
        self.prev = None
        x=np.linspace(0,LENGTH,ROWS+1)
        y=np.linspace(-WIDTH/2,WIDTH/2,COLS+1)
        self.quads=np.array([[(x[i],y[j]),(x[i+1],y[j]),(x[i+1],y[j+1]),(x[i],y[j+1])]
                              for i in range(ROWS) for j in range(COLS)])

    @staticmethod
    def points(mesh,xyz):
        from pxr import Vt
        pts=xyz.reshape((-1,3)).astype('float32')
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([pts.min(0),pts.max(0)])))

    def update(self,m,base,tab):
        from pxr import Vt
        cs,sn=math.cos(m.yaw),math.sin(m.yaw)
        rot=np.array(((cs,-sn),(sn,cs)))
        center=base[:2]+m.origin
        film=np.empty((ROWS*COLS,4,3))
        film[:,:,:2]=self.quads @ rot.T+center
        up=max(.001,float(tab[2]-base[2]-.007))
        if m.held_edge:
            # The unpressed stack bends up to make room for the folded-back liner.
            tail_x=max(.012,np.dot(tab[:2]-center,rot[:,0])-m.peeled)
            ramp=np.clip((self.quads[:,:,0]-m.peeled)/tail_x,0.,1.)
            free_z=.002 + max(0.,tab[2]-base[2])*ramp
        else:
            free_z=up
        film[:,:,2]=base[2]+np.where(m.bond.reshape(-1,1),.0007,free_z)
        self.points(self.film,film)
        colors=np.where(m.bond.reshape(-1,1),np.array([[.07,.15,.19]]),np.array([[.63,.84,.88]]))
        if self.prev is None or not np.array_equal(colors,self.prev):
            self.film.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(colors.astype('float32')))
            self.prev=colors
        front=m.peeled
        tab_local=(tab[:2]-center) @ rot
        back_z=base[2]+max(.001,up-.0004)
        start_z=base[2]+max(.0003,up-.001)
        def segment(x0,x1,w0,w1,y0,y1,z0,z1):
            xy=np.array([[x0,y0-w0],[x1,y1-w1],[x1,y1+w1],[x0,y0+w0]]) @ rot.T+center
            return np.column_stack((xy,[z0,z1,z1,z0]))
        if m.held_edge:
            start_z=base[2]+.0008
            back_z=float(film[-1,:,2].max())-.0005
        tab_center=tab_local[1]-TAB_Y
        bottom=segment(front,LENGTH,WIDTH/2,WIDTH/2,0,0,start_z,back_z)
        if m.peeled>=LENGTH-.0005:
            # The detached liner hangs from the measured paper tab after it leaves the glass.
            back_x=tab_local[0]-LENGTH
            bottom=segment(back_x,back_x+.0001,WIDTH/2,WIDTH/2,0,0,tab[2]-.004,tab[2]-.004)
            free=segment(back_x,tab_local[0],WIDTH/2,WIDTH/2,tab_center,tab_center,tab[2]-.004,tab[2])
        else:
            free=segment(front,tab_local[0],WIDTH/2,WIDTH/2,0,tab_center,start_z+.0002,tab[2])
        sticker_mid=tab_center+(WIDTH/2+TAB_Y)/2
        sticker_half=(TAB_Y-WIDTH/2)/2
        sticker=segment(tab_local[0]-.009,tab_local[0]+.009,sticker_half,sticker_half,
                        sticker_mid,sticker_mid,tab[2],tab[2])
        self.points(self.liner,np.array([bottom,free,sticker]))
