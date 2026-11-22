import numpy as np
import math

from itertools import combinations
from pathlib import Path

from .transformations import tranPoints
from .loadingObjects import loadObj
from .geometry import rowNorm
from .geometry import normals


def faces20edron(vert):
    distances=np.linalg.norm(vert[:, None, :]-vert[None, :, :], axis=-1)
    edgeLength=np.min(distances[distances>1e-6])
    faces=[]
    for i, j, k in combinations(range(len(vert)), 3):
        if(np.isclose(distances[i,j], edgeLength)and np.isclose(distances[j, k], edgeLength) and np.isclose(distances[k, i], edgeLength)):
            v0=vert[i]
            v1=vert[j]
            v2=vert[k]
            norm=np.cross(v1-v0, v2-v0)
            cen=(v0+v1+v2)/3.0
            if(np.dot(norm, cen)>0):
                faces.append([i, j, k])
            else:
                faces.append([i, k, j])
    return np.array(faces, dtype=np.int64)
  
def midP(i1, i2, mids, vList):
    if(i1<i2):
        key=(i1, i2)
    else:
        key=(i2, i1)
    if key in mids:
        return mids[key]
    p1=vList[i1]
    p2=vList[i2]
    mx=(0.5*(p1[0]+p2[0]))
    my=(0.5*(p1[1]+p2[1]))
    mz=(0.5*(p1[2]+p2[2]))
    apos=math.sqrt(mx*mx + my*my + mz*mz)
    ind=len(vList)
    vList.append([mx/apos, my/apos, mz/apos])
    mids[key]=ind
    return ind
  
def sph20(detail):
    t=((1.0+math.sqrt(5.0))/2)
    v20edrou=np.array([
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]
    ], dtype=np.float64)
    v20edrou=rowNorm(v20edrou)
    faces=faces20edron(v20edrou)
    
    vList=v20edrou.tolist()
    for _ in range(int(detail)):
        midpoints={}
        newFaces=[]
        for a, b, c in faces:
            ab=midP(int(a), int(b), midpoints, vList)
            bc=midP(int(b), int(c), midpoints, vList)
            ca=midP(int(c), int(a), midpoints, vList)
            
            newFaces.append([a, ab, ca])
            newFaces.append([b, bc, ab])
            newFaces.append([c, ca, bc])
            newFaces.append([ab, bc, ca])
        faces=np.asarray(newFaces, dtype=np.int64)
    return np.asarray(vList, dtype=np.float64), faces
  
def trigonopoiisi(geometryD, matrix, baseDirectory, smooth=False):
    tipos=geometryD["type"].lower()
    M=np.asarray(matrix, dtype=np.float64)
    if(tipos=="quad"):
        c=np.asarray(geometryD["corner"], dtype=np.float64)
        e1=np.asarray(geometryD["edge1"], dtype=np.float64)
        e2=np.asarray(geometryD["edge2"], dtype=np.float64)
        #counter-clockwise
        p0=c
        p1=c+e2
        p2=c+e1+e2
        p3=c+e1
        
        lclVert0=np.vstack([p0, p0])
        lclVert1=np.vstack([p3, p2])
        lclVert2=np.vstack([p2, p1])
        v0=tranPoints(lclVert0, M)
        v1=tranPoints(lclVert1, M)
        v2=tranPoints(lclVert2, M)
        n0, n1, n2=normals(v0, v1, v2, smooth=smooth)
        return v0, v1, v2, n0, n1, n2
    elif (tipos=="box"):
        mins=np.asarray(geometryD["min"], dtype=np.float64)
        maxs=np.asarray(geometryD["max"], dtype=np.float64)
        x0, y0, z0=mins
        x1, y1, z1=maxs
        ps=np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1]
        ], dtype=np.float64)
        facesTrigs=np.array([
            [1, 0, 3],
            [1, 3, 2],
            
            [5, 1, 2],
            [5, 2, 6],
            
            [4, 5, 6],
            [4, 6, 7],
            
            [0, 4, 7],
            [0, 7, 3],
            
            [7, 6, 2],
            [7, 2, 3],
            
            [0, 1, 5],
            [0, 5, 4]
        ], dtype=np.int64)
        
        
        v0=tranPoints(ps[facesTrigs[:, 0]], M)
        v1=tranPoints(ps[facesTrigs[:, 1]], M)
        v2=tranPoints(ps[facesTrigs[:, 2]], M)
        n0, n1, n2=normals(v0, v1, v2, smooth=smooth)
        return v0, v1, v2, n0, n1, n2
    elif (tipos=="sphere"):
        cntr=np.asarray(geometryD["centre"], dtype=np.float64)
        th=float(geometryD["radius"])
        detail=int(geometryD.get("detail", 3))
        
        verts, faces=sph20(detail)
        lclVerts=verts*th + cntr
        
        worldV=tranPoints(lclVerts, M)
        worldCntr=tranPoints(cntr.reshape(1, 3), M)
                
        v0=worldV[faces[:, 0]]
        v1=worldV[faces[:, 1]]
        v2=worldV[faces[:, 2]]
        n0=rowNorm(v0-worldCntr)
        n1=rowNorm(v1-worldCntr)
        n2=rowNorm(v2-worldCntr)
        return v0, v1, v2, n0, n1, n2
    elif (tipos=="mesh"):
        if "file" in geometryD:
            fullPath=Path(baseDirectory)/geometryD["file"]
            verts, faces=loadObj(fullPath)
        else:
            verts=np.asarray(geometryD["vertices"], dtype=np.float64)
            faces=geometryD["faces"]
            
            trigFaces=[]
            for poligono in faces:
                for i in range(1, (len(poligono)-1)):
                    trigFaces.append([poligono[0], poligono[i], poligono[i+1]])
            faces=np.asarray(trigFaces, dtype=np.int64).reshape(-1, 3)
            
        if(faces.size==0):
            empty=np.zeros((0, 3), dtype=np.float64)
            return empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy()
        
        v0=tranPoints(verts[faces[:, 0]], M)
        v1=tranPoints(verts[faces[:, 1]], M)
        v2=tranPoints(verts[faces[:, 2]], M)
        n0, n1, n2=normals(v0, v1, v2, smooth=smooth)
        return v0, v1, v2, n0, n1, n2
    else:
        raise ValueError(f"Unsupported geometry type: {tipos!r}")
  