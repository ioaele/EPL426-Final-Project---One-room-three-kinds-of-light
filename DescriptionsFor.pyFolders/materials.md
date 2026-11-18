# `materials.py`

> **basically:** `materials.py` turns the materials of the scene file into **one
> table of numbers** (one row per material, with the PDF's defaults filled in)
> so that Numba code like the shading can read them quickly

---

## The idea

every object in the scene has a **material** which says how it looks color,shininess,if its a mirror a glass glowing etc

---

## How the code works in detail

### 1. The column names (`KD = 0`, `KS = 3`, … `MAT_SIZE = 20`)

Each material row has **20 numbers** These names say which number is which:

| Name | Numbers | What it is |
|---|:---:|---|
| `KD` | 3 | the diffuse colour, the object's normal colour |
| `KS` | 3 | the colour of the shiny highlight |
| `M` | 1 | shininess: higher means a smaller, sharper highlight |
| `KR` | 1 | how much it reflects like a mirror |
| `KT` | 1 | how much light goes through it, like glass |
| `IOR` | 1 | index of refraction: how much glass bends light |
| `ALPHA` | 1 | opacity in the preview (1 = solid) |
| `GLOSS` | 1 | 0 = perfect mirror, above 0 = blurry reflection |
| `EMISSION` | 3 | light the surface gives off itself (the ceiling panel) |
| `CHECKER` | 1 | checkerboard pattern on (1) or off (0) |
| `CHECKER_SIZE` | 1 | size of the checker squares |
| `KD2` | 3 | second colour of the checkerboard |

### 2. `DEFAULTS`

The values from the PDF for anything the scene file leaves out:

| kd | ks | m | kr | kt | ior | alpha | gloss | emission |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.7 (grey) | 0 | 32 | 0 | 0 | 1.5 | 1 | 0 | none |

### 3. `_rgb(value)`: helper

Lets a colour be written either as **one number** (`0.5` means grey) or as
**three numbers** `[r, g, b]` and always returns three

### 4. `material_row(desc)`: one material → one row

Takes a material from the scene file for example:

```json
{"kd": [0.8, 0.1, 0.1], "ks": 0.3}
```

fills in the defaults and returns its **20 numbers** if it has a checker it
sets the checker values too

### 5. `MaterialTable`: collects all the materials

while the scene is loading every object asks the table
*"which row is my material?"* with `index_of(desc)`

- if the same material appears twice they **share one row** so nothing is stored twice
- at the end `to_array()` gives the finished table

### 6. `load_materials(material_descs)`

The function the **scene loader** calls it takes a list of materials and
returns

- **the table** and
- **which row each material got**

each triangle then stores just that row number