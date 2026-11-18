# `camera.py`

> **In short:** `camera.py` reads the camera from the scene file once, and then
> makes every ray the ray tracer needs: normal camera rays, extra rays inside a
> pixel (antialiasing), lens rays (depth of field) and a time for each ray
> (motion blur).

---

## The idea

The whole camera is stored as **one array of 22 numbers**, because Numba can
only work with arrays, not Python objects or dictionaries.

The scene files use the **Assignment 2 camera**:

| Key in the file | Meaning |
|---|---|
| `vrp` | the eye (COP), where the camera stands |
| `vpn` | the viewing direction, where it looks |
| `vuv` | the up vector |
| `d` | distance from the eye to the image window |
| `window` | the window edges `[umin, vmin, umax, vmax]` |
| `aperture` | radius of the lens (0 = pinhole, everything sharp) |
| `focus` | distance of the plane in focus |
| `shutter` | `[t0, t1]`, how long the shutter stays open |
| `near` | near clipping plane of the rasteriser |

---

## How the code works

### 1. The column names (`COP = 0`, `U = 3`, … `CAM_SIZE = 22`)

These names say where each value sits in the 22-number array:

| Positions | Name | What it is |
|:---:|---|---|
| 0–2 | `COP` | the eye |
| 3–5 | `U` | the camera's "right" direction |
| 6–8 | `V` | the camera's "up" direction |
| 9–11 | `FWD` | where the camera looks |
| 12 | `D` | distance from the eye to the window |
| 13–16 | `UMIN`, `UMAX`, `VMIN`, `VMAX` | the window edges |
| 17 | `APERTURE` | lens radius |
| 18 | `FOCUS` | focus distance |
| 19–20 | `T0`, `T1` | shutter interval |
| 21 | `NEAR` | near plane |

### 2. `SCENE_DEFAULTS`

The values from the PDF for anything the scene file leaves out:

| image | background | ambient | max_depth | min_weight |
|:---:|:---:|:---:|:---:|:---:|
| 320 × 240 | black | 0 | 4 | 0.01 |

### 3. Helpers: `_get` and `_unit`

- `_get(desc, *names, default)` looks for a value under several possible
  names (e.g. `vrp`, `cop` or `eye`) and returns the first one it finds.
- `_unit(v)` makes a vector length 1.

### 4. `load_camera(desc, width, height)`: reads the camera

1. Reads the **eye position** (`vrp`).
2. Reads the **viewing direction** (`vpn`) and makes it length 1, giving `fwd`.
3. Reads the **up vector** (`vuv`).
4. Builds the camera's own axes:
   - `u = fwd × up` points **right**
   - `v = u × fwd` points **up**
5. Reads `d` and the **window**.
6. Reads **aperture, focus, shutter, near**, with the PDF defaults.
7. Packs everything into the 22-number array.

### 5. `apply_overrides(...)`

Replaces aperture, focus or shutter when they're given on the command line
(`--aperture`, `--focus`, `--shutter`).

### 6. `load_scene_settings(scene)`

Reads the image size, background, ambient light, max_depth and min_weight, and
fills in the defaults. It also accepts the size written as
`{"width": 320, "height": 320}`, like our scene files.

---

## The ray functions (`@njit`, compiled by Numba)

The ray tracer calls these **millions of times**, so they run in Numba.

### 7. `generate_pixel_ray(cam, x, y, width, height)`: one camera ray

1. Turns the image position `(x, y)` into a point on the window. `x` goes from
   left to right, and `y` from top to bottom, because **row 0 is at the top**.
2. Finds that point in 3D: `eye + d·fwd + su·u + sv·v`.
3. Returns the ray: it **starts at the eye** and **points at that window
   point**.

The centre of pixel `(i, j)` is at `(i + 0.5, j + 0.5)`.

### 8. `sample_position(i, j, a, b, n, jitter)`: antialiasing

With `--spp N`, every pixel is split into an **N × N grid**, and one ray goes
through each cell:

- without `--jitter`: through the **centre** of the cell
- with `--jitter`: through a **random point** in the cell

### 9. `depth_of_field_ray(cam, direction, xi1, xi2)`: the lens

1. Follows the normal ray until it hits the **plane in focus**. That point
   is **F**.
2. Picks a **random point on the lens**, a disc of radius `aperture` around
   the eye.
3. Makes a new ray from that lens point **towards F**.

Objects at the focus distance stay **sharp** (every lens ray meets at F), and
everything nearer or further becomes **blurry**.

### 10. `ray_time(cam, xi)`: motion blur

Picks a **random moment** while the shutter is open, between `t0` and `t1`. The
moving ball is drawn where it is at that moment, so it smears along its path.