
import math

import numpy as np

# --- light types ---
POINT = 0
DIRECTIONAL = 1
SPOT = 2
AREA = 3
TYPE_NAMES = {"point": POINT, "directional": DIRECTIONAL, "spot": SPOT, "area": AREA}

# --- column layout of one light row ---
TYPE = 0
INTENSITY = 1    
POSITION = 4    
DIRECTION = 7     
ATTEN = 10      
COS_CUTOFF = 13   
EXPONENT = 14    
CORNER = 15       
EDGE1 = 18       
EDGE2 = 21        
NORMAL = 24       
AREA_ = 27        
LIGHT_SIZE = 28


def _vec(v):
    return np.asarray(v, dtype=np.float64).reshape(3)


def _rgb(v):
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    return np.repeat(a, 3) if a.size == 1 else a.reshape(3)


def _unit(v):
    v = _vec(v)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length direction in a light")
    return v / n


def _intensity(desc):
    for key in ("intensity", "color", "colour", "emission"):
        if key in desc:
            return _rgb(desc[key])
    return np.ones(3)


def light_row(desc):
    kind = desc["type"]
    if kind not in TYPE_NAMES:
        raise ValueError(f"unknown light type {kind!r}")
    row = np.zeros(LIGHT_SIZE, dtype=np.float64)
    row[TYPE] = TYPE_NAMES[kind]
    row[INTENSITY:INTENSITY + 3] = _intensity(desc)
    row[ATTEN:ATTEN + 3] = _vec(desc.get("attenuation", [1.0, 0.0, 0.0]))
    row[COS_CUTOFF] = -1.0  # no cone
    if kind in ("point", "spot"):
        row[POSITION:POSITION + 3] = _vec(desc["position"])
    if kind == "directional":
        row[DIRECTION:DIRECTION + 3] = _unit(desc["direction"])
    if kind == "spot":
        row[DIRECTION:DIRECTION + 3] = _unit(desc["direction"])
        row[COS_CUTOFF] = math.cos(math.radians(float(desc.get("cutoff", 180.0))))
        row[EXPONENT] = float(desc.get("exponent", 0.0))
    if kind == "area":
        c, e1, e2 = _vec(desc["corner"]), _vec(desc["edge1"]), _vec(desc["edge2"])
        cross = np.cross(e1, e2)
        area = np.linalg.norm(cross)
        row[CORNER:CORNER + 3] = c
        row[EDGE1:EDGE1 + 3] = e1
        row[EDGE2:EDGE2 + 3] = e2
        row[NORMAL:NORMAL + 3] = cross / area
        row[AREA_] = area
    return row


def transform_light(desc, matrix):
   
    out = dict(desc)
    M = np.asarray(matrix, dtype=np.float64)

    def pt(p):
        return (M @ np.append(_vec(p), 1.0))[:3]

    def dirv(d):
        return M[:3, :3] @ _vec(d)

    for k in ("position", "corner"):
        if k in desc:
            out[k] = pt(desc[k])
    for k in ("direction", "edge1", "edge2"):
        if k in desc:
            out[k] = dirv(desc[k])
    return out


def load_lights(light_descs, matrices=None):
  
    rows = []
    for i, d in enumerate(light_descs):
        if matrices is not None:
            d = transform_light(d, matrices[i])
        rows.append(light_row(d))
    if not rows:
        return np.zeros((0, LIGHT_SIZE), dtype=np.float64)
    return np.vstack(rows)
