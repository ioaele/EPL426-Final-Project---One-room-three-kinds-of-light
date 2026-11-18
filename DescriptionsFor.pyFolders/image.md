# `image.py`

> **Basically** `image.py` saves the renderer's images as PNG files using the
> **clip → gamma 2.2 → 8-bit** rule of Appendix C, so they can be compared with
> the reference images, and reads PNGs back for that comparison.

---

## How the code works IN DETAIL:

### 1. `linear_to_srgb8(img)`: renderer colours → PNG colours

The renderer works out colours as **linear light values**: decimal numbers like
`0.42` that can even go below 0 or above 1. A PNG can only store whole numbers
from **0 to 255**, so this function converts between the two in three steps:

1. **clipping:** every value above 1 becomes 1, and every value below 0 becomes 0.
2. **gamma correction:** raise it to the power `1/2.2`. Screens show dark
   values darker than they really are, so this brightens the darks to make the
   image look right.
3. **converting to 0–255:** `⌊255 · v + 0.5⌋`, which rounds to the nearest
   whole number.

### 2. `save_png(img, path)`: saves an image

Runs `linear_to_srgb8(img)` on the image and saves the result as an
**8-bit RGB PNG** file.