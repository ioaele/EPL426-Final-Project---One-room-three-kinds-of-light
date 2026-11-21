import numpy as np
import math

def rowNorm(vectors):
    norm=np.linalg.norm(vectors, axis=1, keepdims=True)
    norm=np.where(norm>1e-15, norm, 1.0)
    a=vectors/norm
    return a
    
def normals(v0, v1, v2, smooth=False):
    N=len(v0)
    if(N==0):
        empty=np.zeros((0, 3), dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()
    cross=np.cross(v1-v0, v2-v0)
    norms=rowNorm(cross)
    if not smooth:
        return norms.copy(), norms.copy(), norms.copy()
    trig=np.stack([v0, v1, v2], axis=1)
    fSum={}
    for i in range(N):
        weightedN=cross[i]
        for j in range(3):
            p=trig[i, j]
            x, y, x=p
            #32-tiny
            key=(round(float(x), 7), round(float(y), 7), round(float(z), 7))
            if key in fSum:
                fSum[key]=fSum[key]+weightedN
            else:
                fSum[key]=weightedN.copy()
    
    
    n0=np.zeros((N,3), dtype=np.float64)
    n1=np.zeros((N,3), dtype=np.float64)
    n2=np.zeros((N,3), dtype=np.float64)
    
    for k in range(N):
        p0=trig[k, 0]
        key0=(round(float(p0[0]), 7), round(float(p0[1]), 7), round(float(p0[2]), 7))
        vec0=fSum[key0]
        length0=np.linalg.norm(vec0)
        if(length0>1e-15):
            n0[k]=vec0/length0
        else:
            n0[k]=norms[k]
        
        p1=trig[k, 1]
        key1=(round(float(p1[0]), 7), round(float(p1[1]), 7), round(float(p1[2]), 7))
        vec1=fSum[key1]
        length1=np.linalg.norm(vec1)
        if(length1>1e-15):
            n1[k]=vec1/length1
        else:
            n1[k]=norms[k]
        
        p2=trig[k, 2]
        key2=(round(float(p2[0]), 7), round(float(p2[1]), 7), round(float(p2[2]), 7))
        vec2=fSum[key2]
        length2=np.linalg.norm(vec2)
        if(length2>1e-15):
            n2[k]=vec2/length2
        else:
            n2[k]=norms[k]
    return n0, n1, n2
    
