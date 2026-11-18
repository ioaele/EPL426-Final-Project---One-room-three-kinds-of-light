"""Compare two PNG images the way the handout does (Appendix C).

A pixel counts as different when any of its three channels differs by more
than 0.05 (after dividing by 255). Two images match when fewer than 1% of
the pixels are different.

Usage:
    python tools/compare_images.py ours.png reference/whitted.png [--save-diff diff.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from renderer.image import load_png  # noqa: E402

CHANNEL_TOLERANCE = 0.05
MATCH_FRACTION = 0.01


def diff_mask(a, b, tol=CHANNEL_TOLERANCE):
    """Boolean (H, W) mask: True where any channel differs by more than tol."""
    if a.shape != b.shape:
        raise ValueError(f"image sizes differ: {a.shape} vs {b.shape}")
    return np.any(np.abs(a - b) > tol, axis=2)


def compare_images(path_a, path_b, tol=CHANNEL_TOLERANCE):
    """Return (number of different pixels, fraction of different pixels, match?)."""
    mask = diff_mask(load_png(path_a), load_png(path_b), tol)
    n_diff = int(mask.sum())
    frac = n_diff / mask.size
    return n_diff, frac, frac < MATCH_FRACTION


def save_diff_image(path_a, path_b, out_path, tol=CHANNEL_TOLERANCE):
    """Save a black/red image that shows where the two images differ."""
    from PIL import Image
    mask = diff_mask(load_png(path_a), load_png(path_b), tol)
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[mask] = (255, 0, 0)
    Image.fromarray(out, mode="RGB").save(out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=CHANNEL_TOLERANCE)
    p.add_argument("--save-diff", metavar="PNG")
    args = p.parse_args()

    n_diff, frac, ok = compare_images(args.a, args.b, args.tol)
    print(f"{n_diff} pixels differ ({100 * frac:.3f}%) -> {'MATCH' if ok else 'NO MATCH'}")
    if args.save_diff:
        save_diff_image(args.a, args.b, args.save_diff, args.tol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
