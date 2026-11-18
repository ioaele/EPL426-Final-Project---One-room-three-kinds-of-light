
import numpy as np
from numba import njit

from . import lights as L
from . import materials as Mt

EPS_PREVIEW = 1e-3  
EPS_RAY = 1e-4       


@njit(cache=True)
def unit(v):
    return v / np.sqrt(np.dot(v, v))


@njit(cache=True)
def material_kd(mats, mi, P):
    
    row = mats[mi]
    if row[Mt.CHECKER] > 0.5:
        s = row[Mt.CHECKER_SIZE]
        k = int(np.floor(P[0] / s)) + int(np.floor(P[2] / s))
        if k % 2 != 0:  # works for negative k too: -1 % 2 == 1 in Python/Numba
            return row[Mt.KD2:Mt.KD2 + 3].copy()
    return row[Mt.KD:Mt.KD + 3].copy()


@njit(cache=True)
def face_forward(N, E):
    if np.dot(N, E) < 0.0:
        return -N
    return N


@njit(cache=True)
def light_direction(lights, li, P):
  
    row = lights[li]
    t = int(row[L.TYPE])
    if t == L.DIRECTIONAL:
        return row[L.DIRECTION:L.DIRECTION + 3].copy(), np.inf, 1.0
    d = row[L.POSITION:L.POSITION + 3] - P
    r = np.sqrt(np.dot(d, d))
    l = d / r
    f = 1.0 / (row[L.ATTEN] + row[L.ATTEN + 1] * r + row[L.ATTEN + 2] * r * r)
    if t == L.SPOT:
        a = row[L.DIRECTION:L.DIRECTION + 3]
        cos_phi = -np.dot(l, a)
        if cos_phi >= row[L.COS_CUTOFF]:
            if row[L.EXPONENT] != 0.0:
                f *= max(cos_phi, 0.0) ** row[L.EXPONENT]
        else:
            f = 0.0
    return l, r, f


@njit(cache=True)
def blinn_phong(kd, ks, m, N, E, l, specular):
    out = np.zeros(3)
    ndotl = np.dot(N, l)
    if ndotl <= 0.0:
        return out
    out += kd * ndotl
    if specular:
        h = unit(l + E)
        ndoth = np.dot(N, h)
        if ndoth > 0.0:
            out += ks * ndoth ** m
    return out


@njit(cache=True)
def shade_hit(P, N, E, mats, mi, lights, vis, Ia,
              use_ambient, use_specular, use_emission):
 
    row = mats[mi]
    kd = material_kd(mats, mi, P)
    ks = row[Mt.KS:Mt.KS + 3]
    m = row[Mt.M]
    c = np.zeros(3)
    if use_ambient:
        c += kd * Ia
    if use_emission:
        c += row[Mt.EMISSION:Mt.EMISSION + 3]
    for k in range(lights.shape[0]):
        if int(lights[k, L.TYPE]) == L.AREA or vis[k] <= 0.0:
            continue
        l, r, f = light_direction(lights, k, P)
        if f <= 0.0:
            continue
        I = lights[k, L.INTENSITY:L.INTENSITY + 3]
        c += vis[k] * f * I * blinn_phong(kd, ks, m, N, E, l, use_specular)
    return c


@njit(cache=True)
def area_sample_point(lights, li, a, b, n, jitter):
    lights[li]
    if jitter:
        s = (a + np.random.random()) / n
        t = (b + np.random.random()) / n
    else:
        s = (a + 0.5) / n
        t = (b + 0.5) / n
    return (row[L.CORNER:L.CORNER + 3] + s * row[L.EDGE1:L.EDGE1 + 3]
            + t * row[L.EDGE2:L.EDGE2 + 3])


@njit(cache=True)
def area_sample_contribution(lights, li, P, N, E, kd, ks, m, y, n):
  
    row = lights[li]
    out = np.zeros(3)
    d = y - P
    r2 = np.dot(d, d)
    w = d / np.sqrt(r2)
    cos_t = np.dot(N, w)
    cos_tl = -np.dot(row[L.NORMAL:L.NORMAL + 3], w)
    if cos_t <= 0.0 or cos_tl <= 0.0:
        return out
    h = unit(w + E)
    ndoth = np.dot(N, h)
    spec = ks * ndoth ** m if ndoth > 0.0 else np.zeros(3)
    G = cos_tl * row[L.AREA_] / (np.pi * r2)
    out = row[L.INTENSITY:L.INTENSITY + 3] * G * (kd * cos_t + spec) / (n * n)
    return out
