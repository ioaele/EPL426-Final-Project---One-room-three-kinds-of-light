import numpy as np

from . import camera as Cam
from . import materials as Mat

def backfaceCull(v0, v1, v2, materialIDs, materials, camera):
    N=len(v0)
    if(N==0):
        return np.zeros((0,), dtype=bool), 0
    cop=camera[Cam.COP:Cam.COP+3]
    nFace=np.cross(v1-v0, v2-v0)
    eye=cop-v0
    facing=np.sum(nFace*eye, axis=1)>0.0
    alphas=materials[materialIDs, Mat.ALPHA]
    transparent=alphas<1.0
    keep=facing | transparent
    culledAmount=int(N-np.count_nonzero(keep))
    return keep, culledAmount

def addPl(normal, pOnPl, pl):
    n=normal/np.linalg.norm(normal)
    disOffs= -np.dot(n, pOnPl)
    pl.append((n[0], n[1], n[2], disOffs))

def getPlanes(camera, far=1111.0):
    cop=camera[Cam.COP:Cam.COP + 3]
    u=camera[Cam.U:Cam.U + 3]
    v=camera[Cam.V:Cam.V + 3]
    fwd=camera[Cam.FWD:Cam.FWD + 3]
    d=camera[Cam.D]
    umin=camera[Cam.UMIN]
    umax=camera[Cam.UMAX]
    vmin=camera[Cam.VMIN]
    vmax=camera[Cam.VMAX]
    near=camera[Cam.NEAR]
    
    winCntr=cop+d*fwd
    bl=winCntr + umin*u + vmin*v
    br=winCntr + umax*u + vmin*v
    tr=winCntr + umax*u + vmax*v
    tl=winCntr + umin*u + vmax*v
    
    #Near, Far, Left, Right, Top, Bottom planes
    planes=[]
    addPl(fwd, cop+near*fwd, planes)
    addPl(-fwd, cop+far*fwd, planes)
    addPl(np.cross(bl-cop, tl-cop), cop, planes)
    addPl(np.cross(tr-cop, br-cop), cop, planes)
    addPl(np.cross(tl-cop, tr-cop), cop, planes)
    addPl(np.cross(br-cop, bl-cop), cop, planes)
    
    return np.asarray(planes, dtype=np.float64)

def axAlBouBoxInPl(bmin, bmax, pl):
    for i in range(6):
        nx, ny, nz, disOffs=pl[i]
        if(nx>=0.0):
            ix=bmax[0]
        else:
            ix=bmin[0]
        if(ny>=0.0):
            iy=bmax[1]
        else:
            iy=bmin[1]
        if(nz>=0.0):
            iz=bmax[2]
        else:
            iz=bmin[2]
        
        if(nx*ix + ny*iy + nz*iz + disOffs < 0.0):
            return True
        
    return False

def viewVolumeCull(v0, v1, v2, camera, bvh=None, far=1111.0):
    N=len(v0)
    if(N==0):
        return np.zeros((0,), dtype=bool), 0
    planes=getPlanes(camera, far=far)
    
    if bvh is not None:
        nodeMin, nodeMax, left, right, start, count, trigOrder=bvh
        if(len(nodeMin)>0):
            keep=np.zeros(N, dtype=bool)
            stack=[0]
            while stack:
                nodeInd=stack.pop()
                bmin=nodeMin[nodeInd]
                bmax=nodeMax[nodeInd]
                if(axAlBouBoxInPl(bmin, bmax, planes)):
                    continue
                
                if(left[nodeInd]==-1):
                    beg=int(start[nodeInd])
                    cnt=int(count[nodeInd])
                    keep[trigOrder[beg:beg+cnt]]=True
                else:
                    stack.append(int(left[nodeInd]))
                    stack.append(int(right[nodeInd]))  
            culledAmount=int(N-np.count_nonzero(keep))
            return keep, culledAmount
    
    trigMin=np.minimum(np.minimum(v0, v1), v2)
    trigMax=np.maximum(np.maximum(v0, v1), v2)
    keep=np.ones(N, dtype=bool)
    for i in range(6):
        nx, ny, nz, disOffs=planes[i]
        
        ix=np.where(nx>=0.0, trigMax[:, 0], trigMin[:, 0])
        iy=np.where(ny>=0.0, trigMax[:, 1], trigMin[:, 1])
        iz=np.where(nz>=0.0, trigMax[:, 2], trigMin[:, 2])
        
        inPlane=(nx*ix + ny*iy + nz*iz + disOffs)>=0.0
        keep=keep&inPlane
    
    culledAmount=int(N-np.count_nonzero(keep))
    return keep, culledAmount
