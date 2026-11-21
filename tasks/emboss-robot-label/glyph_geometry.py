"""Small raised/ink glyph geometry, generated from actual observed symbol events."""
from pathlib import Path
import json, math
import numpy as np
FONT=json.loads(Path(__file__).with_name('glyphs.json').read_text())

def glyph(c, center, height, angle=0.):
    rows=FONT[c]; h=len(rows);w=len(rows[0]);pitch=height/h
    ca,sa=math.cos(angle),math.sin(angle)
    pts=[];faces=[]
    for y,row in enumerate(rows):
        for x,on in enumerate(row):
            if not on:continue
            idx=len(pts)
            for dx,dy in ((0,0),(1,0),(1,1),(0,1)):
                xx=(x+dx-w/2)*pitch; yy=(h/2-y-dy)*pitch
                pts.append([center[0]+ca*xx-sa*yy,center[1]+sa*xx+ca*yy,center[2]])
            faces.extend([[idx,idx+2,idx+1],[idx,idx+3,idx+2]])
    return pts,faces

def combined(items):
    pts=[];faces=[]
    for c,center,height,angle in items:
        p,f=glyph(c,center,height,angle);n=len(pts)
        pts.extend(p);faces.extend([[i+n for i in t] for t in f])
    return np.asarray(pts,float).reshape(-1,3),np.asarray(faces,int).reshape(-1,3)
