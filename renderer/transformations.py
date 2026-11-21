import math
import numpy as np

def rotation(axis, angleDeg):
    axis=np.asarray(axis, dtype=np.float64)
    n=np.linalg.norm(axis)
    if(n==0.0):
        return np.eye(4, dtype=np.float64)
    x, y, z=axis/n
    th=math.radians(float(angleDeg))
    cosTh=math.cos(th)
    sinTh=math.sin(th)
    w=1.0-cosTh
    
    M=np.eye(4, dtype=np.float64)
    #Rodrigue's Rotation formula
    M[0, 0]= w*x*x + cosTh
    M[0, 1]= w*x*y - sinTh*z
    M[0, 2]= w*x*z + sinTh*y
    
    M[1, 0]= w*x*y + sinTh*z
    M[1, 1]= w*y*y + cosTh
    M[1, 2]= w*y*z - sinTh*x
    
    M[2, 0]= w*x*z - sinTh*y
    M[2, 1]= w*y*z + sinTh*x
    M[2, 2]= w*z*z + cosTh
    return M

def tranStep(step):
    transformType=str(step[0]).lower
    transformValues=step[1:]
    M=np.eye(4, dtype=np.float64)
    
    if(transformType=="translate"):
        tx, ty, tz=map(float, transformValues[:3])
        M[0,3]=tx
        M[1,3]=ty
        M[2,3]=tz
    elif(transformType=="rotate_y"):
        M=rotation([0.0, 1.0, 0.0], transformValues[0])
    elif(transformType=="rotate_x"):
        M=rotation([1.0, 0.0, 0.0], transformValues[0])
    elif(transformType=="rotate_z"):
        M=rotation([0.0, 0.0, 1.0], transformValues[0])
    elif(transformType=="scale"):
        if(len(transformValues)==1):
            sx=float(transformValues[0])
            sy=float(transformValues[0])
            sz=float(transformValues[0])
            M[0,0]=sx
            M[1,1]=sy
            M[2,2]=sz
    else:
        raise ValueError(f"Unsupported transformation arguments: {transformValues!r}")
    return M

def composeTransforms(steps, parentMatrix=None):
    if parentMatrix is None:
        M=np.eye(4, dtype=np.float64)
    else:
        M=np.asarray(parentMatrix, dtype=np.float64).copy()
    
    if not steps:
        return M
    
    for step in steps:
        M=np.matmul(M, tranStep(step))
    return M

def tranPoints(point, matrix):
    points=np.asarray(point, dtype=np.float64)
    if(points.size==0):
        return points.reshape(0, 3)
    M=np.asarray(matrix, dtype=np.float64)
    tansformedPoints=np.matmul(points, np.transpose(M[:3, :3])) + M[:3, 3]
    return tansformedPoints