"""Small vector helpers for Numba code.

We do NOT use np.dot inside @njit functions: Numba implements np.dot with
SciPy's BLAS, and SciPy is not one of the libraries the project allows
(NumPy, Numba, Matplotlib, Pillow). For 3-vectors a hand-written dot product
is just as fast anyway.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def dot3(a, b):
    """Dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def unit3(v):
    """v scaled to length 1."""
    return v / np.sqrt(dot3(v, v))