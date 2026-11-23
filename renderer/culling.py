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