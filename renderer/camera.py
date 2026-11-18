import math

import numpy as np
from numba import njit

# --- column layout ----
COP = 0        
U = 3          
V = 6           
FWD = 9       
D = 12          
UMIN, UMAX, VMIN, VMAX = 13, 14, 15, 16
APERTURE = 17  
FOCUS = 18      
T0, T1 = 19, 20 
NEAR = 21       
CAM_SIZE = 22

SCENE_DEFAULTS = {
    "image": [320, 240],
    "background": [0.0, 0.0, 0.0],
    "ambient": [0.0, 0.0, 0.0],
    "max_depth": 4,
    "min_weight": 0.01,
}


def _get(desc, *names, default=None):
    for n in names:
        if n in desc:
            return desc[n]
    if default is None:
        raise KeyError(f"camera needs one of {names}")
    return default


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def load_camera(desc, width, height):
    
    cop = np.asarray(_get(desc, "vrp", "cop", "eye", "position", "from"), dtype=np.float64)

    if any(k in desc for k in ("lookat", "look_at", "target", "to")):
        fwd = _unit(np.asarray(_get(desc, "lookat", "look_at", "target", "to"), float) - cop)
    else:
        fwd = _unit(_get(desc, "vpn", "n", "direction", "dir"))

    up = np.asarray(_get(desc, "vuv", "up", "vup", default=[0.0, 1.0, 0.0]), dtype=np.float64)
    u = _unit(np.cross(fwd, up))
    v = np.cross(u, fwd)

    d = float(_get(desc, "d", "distance", default=1.0))
    if "window" in desc:
        umin, vmin, umax, vmax = map(float, desc["window"])
    else:
        fov = math.radians(float(_get(desc, "fov", "fovy", default=45.0)))  # vertical
        half_h = d * math.tan(fov / 2.0)
        half_w = half_h * width / height
        umin, umax, vmin, vmax = -half_w, half_w, -half_h, half_h

    cam = np.zeros(CAM_SIZE, dtype=np.float64)
    cam[COP:COP + 3] = cop
    cam[U:U + 3] = u
    cam[V:V + 3] = v
    cam[FWD:FWD + 3] = fwd
    cam[D] = d
    cam[UMIN], cam[UMAX], cam[VMIN], cam[VMAX] = umin, umax, vmin, vmax
    cam[APERTURE] = float(desc.get("aperture", 0.0))
    cam[FOCUS] = float(desc.get("focus", d))
    t0, t1 = desc.get("shutter", [0.0, 0.0])
    cam[T0], cam[T1] = float(t0), float(t1)
    cam[NEAR] = float(desc.get("near", 0.05))
    return cam


def apply_overrides(cam, aperture=None, focus=None, shutter=None):
   
    cam = cam.copy()
    if aperture is not None:
        cam[APERTURE] = aperture
    if focus is not None:
        cam[FOCUS] = focus
    if shutter is not None:
        cam[T0], cam[T1] = shutter
    return cam


def load_scene_settings(scene):
    
    s = {k: scene.get(k, v) for k, v in SCENE_DEFAULTS.items()}
    img = s["image"]
    if isinstance(img, dict):          
        img = [img.get("width", 320), img.get("height", 240)]
    s["image"] = [int(x) for x in img]
    s["background"] = np.asarray(s["background"], dtype=np.float64)
    a = np.asarray(s["ambient"], dtype=np.float64).reshape(-1)
    s["ambient"] = np.repeat(a, 3) if a.size == 1 else a
    s["max_depth"] = int(s["max_depth"])
    s["min_weight"] = float(s["min_weight"])
    return s


# --- rays (Numba) ----------------------------------------------------------------

@njit(cache=True)
def generate_pixel_ray(cam, x, y, width, height):
 
    su = cam[UMIN] + (x / width) * (cam[UMAX] - cam[UMIN])
    sv = cam[VMAX] - (y / height) * (cam[VMAX] - cam[VMIN])  
    o = cam[COP:COP + 3].copy()
    dvec = cam[D] * cam[FWD:FWD + 3] + su * cam[U:U + 3] + sv * cam[V:V + 3]
    dvec /= np.sqrt(np.dot(dvec, dvec))
    return o, dvec


@njit(cache=True)
def depth_of_field_ray(cam, direction, xi1, xi2):
    
    cop = cam[COP:COP + 3]
    fwd = cam[FWD:FWD + 3]
    t = cam[FOCUS] / np.dot(direction, fwd)
    F = cop + t * direction
    r = cam[APERTURE] * np.sqrt(xi1)     
    phi = 2.0 * np.pi * xi2
    o = cop + r * np.cos(phi) * cam[U:U + 3] + r * np.sin(phi) * cam[V:V + 3]
    dvec = F - o
    dvec /= np.sqrt(np.dot(dvec, dvec))
    return o, dvec


@njit(cache=True)
def ray_time(cam, xi):
    
    return cam[T0] + xi * (cam[T1] - cam[T0])


@njit(cache=True)
def sample_position(i, j, a, b, n, jitter):
   
    if jitter:
        ox, oy = np.random.random(), np.random.random()
    else:
        ox, oy = 0.5, 0.5
    return i + (a + ox) / n, j + (b + oy) / n
