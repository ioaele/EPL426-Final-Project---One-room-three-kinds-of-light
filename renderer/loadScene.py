import json
import numpy as np

from pathlib import Path

from .camera import load_camera, load_scene_settings
from .lights import load_lights
from .materials import MaterialTable

from .transformations import composeTransforms
from .trigonometria import trigonopoiisi


def doMaterial(matEntry, inherMatId, matName, matTable):
    if matEntry is None:
        return inherMatId
    if isinstance(matEntry, str):
        return matName[matEntry]
    return matTable.index_of(matEntry)

def loadScene(scenePath, width=None, height=None):
    sPath=Path(scenePath)
    baseDirectory=sPath.parent
    
    with open(sPath, "r", encoding="utf-8") as fr:
        data=json.load(fr)
    
    settings=load_scene_settings(data)
    if width is not None:
        imgW=int(width)
    else:
        imgW=settings["image"][0]
    if height is not None:
        imgH=int(height)
    else:
        imgH=settings["image"][1]
    camera=load_camera(data["camera"], imgW, imgH)
    matTable=MaterialTable()
    matName={}
    if(isinstance(data.get("materials"), dict)):
        for mKey, mValue in data["materials"].items():
            matName[mKey]=matTable.index_of(mValue)
        
    lightDes=[]
    lightM=[]
    for i in data.get("lights", []):
        lightDes.append(i)
        lightM.append(np.eye(4, dtype=np.float64))
    
    nodes=data.get("nodes", {})
    root=data.get("root")
    if isinstance(nodes, list):
        nodesDict={f"nodeNode{i}": n for i, n in enumerate(nodes)}
        rootKeys=list(nodesDict.keys())
    else:
        nodesDict=nodes
        if root is None:
            rootKeys=list(nodesDict.keys())
        elif isinstance(root, list):
            rootKeys=root
        else:
            rootKeys=[root]
    
    v0L=[]
    v1L=[]
    v2L=[]
    n0L=[]
    n1L=[]
    n2L=[]
    matIdL=[]
    velL=[]
    #####################
    def traverse(nodeKey, parentM, inherMatId, inherVel):
        if isinstance(nodeKey, str):
            node=nodesDict[nodeKey]
        else:
            node=nodeKey
        M=composeTransforms(node.get("transform"), parentM)
        matID=doMaterial(node.get("material"), inherMatId, matName, matTable)
        
        if "velocity" in node:
            lclVel=np.asarray(node["velocity"], dtype=np.float64).reshape(3)
            vel=np.matmul(M[:3, :3], lclVel)
        else:
            vel=inherVel
        
        if "light" in node:
            lightDes.append(node["light"])
            lightM.append(M)
        for i in node.get("lights", []):
            lightDes.append(i)
            lightM.append(M)
        
        if "geometry" in node:
            smooth=bool(node.get("smooth", False))
            tv0, tv1, tv2, tn0, tn1, tn2=trigonopoiisi(node["geometry"], M, baseDirectory, smooth=smooth)
            trAmount=tv0.shape[0]
            if trAmount>0:
                if matID is None:
                    matID=matTable.index_of({})
                v0L.append(tv0)
                v1L.append(tv1)
                v2L.append(tv2)
                n0L.append(tn0)
                n1L.append(tn1)
                n2L.append(tn2)
                matIdL.append(np.full(trAmount, matID, dtype=np.int64))
                velL.append(np.repeat(vel.reshape(1, 3), trAmount, axis=0))
        for i in node.get("children", []):
            traverse(i, M, matID, vel)
    #####################
    iM=np.eye(4, dtype=np.float64)
    zer=np.zeros(3, dtype=np.float64)
    for i in rootKeys:
        traverse(i, iM, None, zer)
    if(len(v0L)>0):
        v0=np.vstack(v0L)
        v1=np.vstack(v1L)
        v2=np.vstack(v2L)
        n0=np.vstack(n0L)
        n1=np.vstack(n1L)
        n2=np.vstack(n2L)
        materialID=np.concatenate(matIdL)
        vel=np.vstack(velL)
    else:
        v0=np.zeros((0, 3), dtype=np.float64)
        v1=np.zeros((0, 3), dtype=np.float64)
        v2=np.zeros((0, 3), dtype=np.float64)
        n0=np.zeros((0, 3), dtype=np.float64)
        n1=np.zeros((0, 3), dtype=np.float64)
        n2=np.zeros((0, 3), dtype=np.float64)
        materialID=np.zeros((0,), dtype=np.int64)
        vel=np.zeros((0, 3), dtype=np.float64)
    materials=matTable.to_array()
    lights=load_lights(lightDes, lightM)
    
    return {"v0": v0,
            "v1": v1,
            "v2": v2,
            "n0": n0,
            "n1": n1,
            "n2": n2,
            "background": settings["background"],
            "ambient": settings["ambient"],
            "max_depth": settings["max_depth"],
            "min_weight": settings["min_weight"],
            "material_id": materialID,
            "velocities": vel,
            "materials": materials,
            "lights": lights,
            "camera": camera,
            "width": imgW,
            "height": imgH,}
    