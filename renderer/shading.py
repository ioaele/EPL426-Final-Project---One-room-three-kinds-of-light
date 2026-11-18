"""The shading model of Appendix C, shared by the preview and the ray tracer.

    colour = kd*Ia + E_emission
             + sum over lights of  vis * f * I * ( kd (N.l) + ks (N.h)^m )
             + kr * C_reflected + kt * C_refracted

This module computes the LOCAL part (everything except the reflected and
refracted rays, which the ray tracer adds). Visibility is NOT computed here:
the caller passes it in (shadow ray, shadow map, or "always visible"),
because the preview and the ray tracer test it in different ways.

Area lights are handled by `area_sample_point` and `area_sample_contribution`
(soft shadows); `shade_hit` skips them, since the preview ignores area lights
and the ray tracer samples them separately.
"""
import numpy as np
from numba import njit

from . import lights as L
from . import materials as Mt
from .vecmath import dot3

EPS_PREVIEW = 1e-3   # shadow ray offset P + 1e-3 N in the preview
EPS_RAY = 1e-4       # offset for the ray tracer


@njit(cache=True)
def unit(v):
    return v / np.sqrt(dot3(v, v))


@njit(cache=True)
def material_kd(mats, mi, P):
    """kd of material mi at point P, including the checker pattern.

    A point (x, y, z) uses kd2 instead of kd when floor(x/s) + floor(z/s) is odd.
    """
    row = mats[mi]
    if row[Mt.CHECKER] > 0.5:
        s = row[Mt.CHECKER_SIZE]
        k = int(np.floor(P[0] / s)) + int(np.floor(P[2] / s))
        if k % 2 != 0:  # works for negative k too: -1 % 2 == 1 in Python/Numba
            return row[Mt.KD2:Mt.KD2 + 3].copy()
    return row[Mt.KD:Mt.KD + 3].copy()


@njit(cache=True)
def face_forward(N, E):
    """Flip N if it points away from the eye (E = unit vector from P to the eye)."""
    if dot3(N, E) < 0.0:
        return -N
    return N


@njit(cache=True)
def light_direction(lights, li, P):
    """For a point, directional or spot light: (l, r, f).

    l: unit vector from P towards the light; r: distance (inf for directional);
    f: attenuation * spot factor (0 if P is outside the spot cone).
    """
    row = lights[li]
    t = int(row[L.TYPE])
    if t == L.DIRECTIONAL:
        return row[L.DIRECTION:L.DIRECTION + 3].copy(), np.inf, 1.0
    d = row[L.POSITION:L.POSITION + 3] - P
    r = np.sqrt(dot3(d, d))
    l = d / r
    f = 1.0 / (row[L.ATTEN] + row[L.ATTEN + 1] * r + row[L.ATTEN + 2] * r * r)
    if t == L.SPOT:
        a = row[L.DIRECTION:L.DIRECTION + 3]
        cos_phi = -dot3(l, a)
        if cos_phi >= row[L.COS_CUTOFF]:
            if row[L.EXPONENT] != 0.0:
                f *= max(cos_phi, 0.0) ** row[L.EXPONENT]
        else:
            f = 0.0
    return l, r, f


@njit(cache=True)
def blinn_phong(kd, ks, m, N, E, l, specular):
    """kd (N.l) + ks (N.h)^m, with h = unit(l + E). Zero if N.l <= 0."""
    out = np.zeros(3)
    ndotl = dot3(N, l)
    if ndotl <= 0.0:
        return out
    out += kd * ndotl
    if specular:
        h = unit(l + E)
        ndoth = dot3(N, h)
        if ndoth > 0.0:
            out += ks * ndoth ** m
    return out


@njit(cache=True)
def shade_hit(P, N, E, mats, mi, lights, vis, Ia,
              use_ambient, use_specular, use_emission):
    """Local colour at P from the ambient term, emission and all NON-area lights.

    N: unit normal, already flipped towards the eye (face_forward).
    vis[k]: visibility (0..1) of light k, computed by the caller.
    use_ambient : False in distributed mode.
    use_specular: False for transparent fragments in the preview.
    use_emission: the ray tracer passes True only on the front side.
    """
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
    """Sample point y of cell (a, b) of an n x n grid on area light li.

    y = C + s e1 + t e2, at the cell centre or (jitter) a random point in it.
    """
    row = lights[li]
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
    """Light that ONE visible sample point y of area light li sends to P.

        (1/n^2) * E_L * cos(theta_L) * A / (pi r^2) * ( kd cos(theta) + ks (N.h)^m )

    with w = unit(y - P), cos(theta) = N.w > 0, cos(theta_L) = -n_L.w > 0,
    h = unit(w + E). Returns zeros if either cosine is <= 0.
    The caller must check visibility of y first (shadow ray P+1e-4 N -> y).
    """
    row = lights[li]
    out = np.zeros(3)
    d = y - P
    r2 = dot3(d, d)
    w = d / np.sqrt(r2)
    cos_t = dot3(N, w)
    cos_tl = -dot3(row[L.NORMAL:L.NORMAL + 3], w)
    if cos_t <= 0.0 or cos_tl <= 0.0:
        return out
    h = unit(w + E)
    ndoth = dot3(N, h)
    spec = ks * ndoth ** m if ndoth > 0.0 else np.zeros(3)
    G = cos_tl * row[L.AREA_] / (np.pi * r2)
    out = row[L.INTENSITY:L.INTENSITY + 3] * G * (kd * cos_t + spec) / (n * n)
    return out