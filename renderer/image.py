"""Writing and reading images exactly as the handout specifies (Appendix C).

A renderer works with *linear* colour values. To save an image we
  1. clip every value to [0, 1],
  2. apply gamma correction v -> v ** (1 / 2.2),
  3. store floor(255 * v + 0.5) as an 8-bit PNG.
"""
import numpy as np
from PIL import Image

GAMMA = 2.2#value asked in pdf!!!


def linear_to_srgb8(img):
    """Convert a float image (H, W, 3) of linear values to uint8 (H, W, 3)."""
    img = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    return np.floor(255.0 * img ** (1.0 / GAMMA) + 0.5).astype(np.uint8)


def save_png(img, path):
    """Save a linear float image (H, W, 3) as a gamma-corrected 8-bit PNG."""
    Image.fromarray(linear_to_srgb8(img), mode="RGB").save(path)


def load_png(path):
    """Read a PNG and return floats in [0, 1] (H, W, 3), still gamma-encoded.

    This is what the comparison of Appendix C works on: read, divide by 255.
    """
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
