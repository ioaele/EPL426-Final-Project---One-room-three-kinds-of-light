# Reference images

These images were made by our solution with the commands below. The first four must match yours:
fewer than 1% of the pixels may differ, where a pixel differs when any of its three channels differs
by more than 0.05 (on a 0 to 1 scale). Appendix C of the handout explains how images are saved and compared.

| File | Command | Match |
|---|---|---|
| `whitted.png` | `python render.py scenes/cornell.json --mode whitted -o images/whitted.png` | exact |
| `whitted_depth0.png` | `python render.py scenes/cornell.json --mode whitted --max-depth 0 -o images/whitted_depth0.png` | exact |
| `spotlight_rays.png` | `python render.py scenes/spotlight.json --mode preview --shadows rays -o images/spotlight_rays.png` | exact |
| `glass_abuffer.png` | `python render.py scenes/glass_panes.json --mode preview --shadows none --transparency abuffer -o images/glass_abuffer.png` | exact |
| `radiosity.png` | `python render.py scenes/cornell.json --mode radiosity -o images/radiosity.png` | looks like |
| `final.png` | `python render.py scenes/cornell.json --mode final --spp 4 --jitter --light-samples 2 -o images/final.png` | looks like |
| `soft_reference.npy` | `python render.py scenes/cornell.json --mode distributed --spp 32 --jitter --seed 7 -o ...` (1,024 rays per pixel). Linear colour values (before gamma), a NumPy array of shape (320, 320, 3). `soft_reference.png` is the same image as a PNG. | for measuring noise |
