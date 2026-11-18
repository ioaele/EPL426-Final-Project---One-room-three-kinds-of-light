"""The camera: loading it, and making camera rays (Appendix B and C).

The camera is stored as one flat float64 array so that Numba code can use it.

Conventions used here
  * u (right), v (up) and fwd (viewing direction) are unit vectors.
  * The image window lies at distance d in front of the COP and spans
    [umin, umax] x [vmin, vmax] in the (u, v) plane.
  * Pixel (i, j): column i, row j, row 0 at the TOP. A position inside the
    image is given as continuous coordinates (x, y); the pixel centre is
    (i + 0.5, j + 0.5).

The scene files use the Assignment 2 camera: vrp is the COP (eye),
vpn the viewing direction, vuv the up vector, and window is
[umin, vmin, umax, vmax] at distance d. Checked against cornell.json.
"""
import math

import numpy as np
from numba import njit

from .vecmath import dot3

# --- column layout -------------------------------------------------------------
COP = 0         # (0, 1, 2)
U = 3           # (3, 4, 5)
V = 6           # (6, 7, 8)
FWD = 9         # (9, 10, 11)
D = 12          # distance from COP to the window
UMIN, UMAX, VMIN, VMAX = 13, 14, 15, 16
APERTURE = 17   # lens radius, 0 = pinhole
FOCUS = 18      # distance of the plane in focus from the COP, along fwd
T0, T1 = 19, 20 # shutter interval
NEAR = 21       # near clipping plane of the rasteriser
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
    """Turn the camera dict of the scene file into a flat array.

    width, height: image size in pixels (needed when the window comes from fov).
    """
    # Assignment 2 camera: vrp = COP (eye), vpn = viewing direction n,
    # vuv = view-up vector, window = [umin, vmin, umax, vmax] at distance d.
    cop = np.asarray(_get(desc, "vrp", "cop", "eye", "position", "from"), dtype=np.float64)

    if any(k in desc for k in ("lookat", "look_at", "target", "to")):
        fwd = _unit(np.asarray(_get(desc, "lookat", "look_at", "target", "to"), float) - cop)
    else:
        # In cornell.json vpn = (0,0,-1) points from the camera INTO the room.
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
    """Apply --aperture, --focus and --shutter from the command line."""
    cam = cam.copy()
    if aperture is not None:
        cam[APERTURE] = aperture
    if focus is not None:
        cam[FOCUS] = focus
    if shutter is not None:
        cam[T0], cam[T1] = shutter
    return cam


def load_scene_settings(scene):
    """Image size, background, ambient, max_depth, min_weight, with defaults."""
    s = {k: scene.get(k, v) for k, v in SCENE_DEFAULTS.items()}
    img = s["image"]
    if isinstance(img, dict):          # the course files: {"width": 320, "height": 320}
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
    """Pinhole ray through image position (x, y) (pixel centre = i+0.5, j+0.5).

    Returns (origin[3], unit direction[3]).
    """
    su = cam[UMIN] + (x / width) * (cam[UMAX] - cam[UMIN])
    sv = cam[VMAX] - (y / height) * (cam[VMAX] - cam[VMIN])  # row 0 at the top
    o = cam[COP:COP + 3].copy()
    dvec = cam[D] * cam[FWD:FWD + 3] + su * cam[U:U + 3] + sv * cam[V:V + 3]
    dvec /= np.sqrt(dot3(dvec, dvec))
    return o, dvec


@njit(cache=True)
def depth_of_field_ray(cam, direction, xi1, xi2):
    """Thin lens: given the pinhole direction, return a new (origin, direction).

    The pinhole ray meets the plane in focus (distance `focus` along fwd) at F.
    The new ray starts at a random point of the lens disc (radius `aperture`,
    in the u-v plane around the COP) and goes towards F.
    xi1, xi2: random numbers in [0, 1).
    """
    cop = cam[COP:COP + 3]
    fwd = cam[FWD:FWD + 3]
    t = cam[FOCUS] / dot3(direction, fwd)
    F = cop + t * direction
    r = cam[APERTURE] * np.sqrt(xi1)        # sqrt -> uniform over the disc
    phi = 2.0 * np.pi * xi2
    o = cop + r * np.cos(phi) * cam[U:U + 3] + r * np.sin(phi) * cam[V:V + 3]
    dvec = F - o
    dvec /= np.sqrt(dot3(dvec, dvec))
    return o, dvec


@njit(cache=True)
def ray_time(cam, xi):
    """Random time in the shutter interval [t0, t1] for motion blur."""
    return cam[T0] + xi * (cam[T1] - cam[T0])


@njit(cache=True)
def sample_position(i, j, a, b, n, jitter):
    """Position of sample (a, b) of an n x n grid inside pixel (i, j).

    Cell centre without jitter, random point in the cell with jitter.
    This is what --spp N means: N x N rays per pixel.
    """
    if jitter:
        ox, oy = np.random.random(), np.random.random()
    else:
        ox, oy = 0.5, 0.5
    return i + (a + ox) / n, j + (b + oy) / n