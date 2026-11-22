import numpy as np

def loadObj(fpath):
    vert=[]
    faces=[]

    with open(fpath, "r", encoding="utf-8") as f:
        for dirtyln in f:
            ln=dirtyln.strip()
            if not ln or ln.startswith('#'):
                continue
            tokens=ln.split()
            if tokens[0]=="v":
                vert.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tokens[0]=="f":
                polygons=[]
                for tok in tokens[1:]:
                    vIndex=int(tok.split("/")[0])  #if it has texture.
                    if(vIndex>0):
                        polygons.append(vIndex-1)
                    else:
                        polygons.append(len(vert)+vIndex)
                for i in range(1, (len(polygons)-1)):
                    faces.append([polygons[0], polygons[i], polygons[i+1]])

    veFlAr=np.asarray(vert, dtype=np.float64).reshape(-1,3)
    faFlAr=np.asarray(faces, dtype=np.int64).reshape(-1,3)
    return veFlAr, faFlAr