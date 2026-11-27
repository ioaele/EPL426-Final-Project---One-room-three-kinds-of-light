# EPL426 Final Project · the project files

This folder has only the files you cannot make yourselves. There is no starter code: you write the
whole renderer, starting from your own code of Assignments 1 and 2. Read the handout first.

- `scenes/`: the scene files of the project and the mesh `teapot.obj`. `terrain.json` loads
  `terrain.obj`, which your fractal generator writes (Chapter 1).
- `reference/`: images made by our solution, which your renderer must match or resemble. See
  `reference/README.md` for the command behind each one.

Copy both folders into your repository.

# EPL426 Final Project — One room, three kinds of light
University of Cyprus · Department of Computer Science  
EPL426 — Computer Graphics

## Team

- Charalambia Kousrarou
- Eleni Ioannou
- Elisavet Tsioli
- Katerina Ioannidou

## Overview

## Repository structure

```text
.
├── README.md
├── render.py # The main file you run to generate an image.
│
├── reference/
│ ├── spotlight_rays.png
│ ├── glass_abuffer.png
│ ├── whitted.png
│ ├── whitted_depth0.png
│ ├── soft_reference.npy
│ ├── radiosity.png
│ └── final.png
|
├── scenes/
│   ├── cornell.json
│   ├── furnace.json
│   ├── glass_panes.json
│   ├── spotlight.json
│   ├── terrain.json
│   ├── terrain.obj            # (Eleni) made by tools/generate_terrain.py
│   └── teapot.obj
│
├── renderer/
│   ├── intersections.py       # (Eleni) ray-triangle, ray-box, brute-force nearest/any hit (one ray or many rays - batch)
│   ├── bvh_median.py          # (Eleni) BVH build, median split
│   ├── bvh_sah.py             # (Eleni) BVH build, surface area heuristic
│   ├── bvh_walk.py            # (Eleni) BVH walk: nearest hit, any hit (one ray or many rays - batch)
│   └── radiosity.py            # (Katerina) splits surfaces into patches,calculates light transfer between them,finds their final colours.
│
├── tools/
│   └── generate_terrain.py    # (Eleni) diamond-square terrain -> scenes/terrain.obj
│
├── tests/
│   ├── test_intersections.py  # (Eleni) ray-triangle, ray-box and brute-force tests
│   ├── test_bvh.py            # (Eleni) BVH build and BVH against brute force (median and SAH)
│   └── test_radiosity.py        # (Katerina) checks that the radiosity functions are calculated correctly.
│
├── benchmarks/
│   ├── check_chapter1.py      # (Eleni) the 3 "It works when" checks of Chapter 1
│   ├── benchmark_bvh.py       # (Eleni) cost of a ray from 2000 to 500000 triangles (report measurement 1)
│   └── benchmark_radiosity.py # (Katerina)
│
├── pytest.ini                 # pytest settings
│
├── output/
│
└── report/
    └── figures/
```

## Installation (only once)

```bash
pip install numpy numba matplotlib pillow pytest
```

## Running the tests

```bash
pytest
```

## Generating the terrain (Eleni)

```bash
python tools/generate_terrain.py 8
```

Writes `project-files/scenes/terrain.obj` 

## Running the renderer

## Report

Each member contributes:

- A description of their modules and design decisions.
- Tests and validation results.
- Benchmark tables or plots.
- Rendered images and known limitations.

## Elisabeth part

**My files:** in the renderer file are image,camera,light,materials,shading and vecmath.py (all of them are .py btw) in tools folder my files are compare_images.py and the test i made in tests folder is test_photograph_basics.py

- To see how they work you need to check ur python version with : 
py --version (if u dont have it installed it)
- then type **ONLY ONCE** :
py -m pip install numpy numba pillow pytest (do not install SciPy)

- now to run the tests do :
py -m pytest tests/test_photograph_basics.py -v

- if the last line says **11 passed** u are all good
- Compare an image with a reference
py tools/compare_images.py project-files/reference/whitted.png project-files/reference/whitted.png
- It should say **MATCH** (fewer than 1% of the pixels differ) add `--save-diff diff.png` at the end to see the different pixels in red LIKE:
py tools/compare_images.py project-files/reference/whitted.png project-files/reference/whitted.png --save-diff diff.png

## Eleni part

**My files:** in the renderer folder: intersections.py, bvh_median.py, bvh_sah.py and bvh_walk.py. In the tools folder: generate_terrain.py. My tests in the tests folder: test_intersections.py and test_bvh.py. In the benchmarks folder: check_chapter1.py and benchmark_bvh.py. I also made pytest.ini (so the tests can find the renderer and tools folders), the terrain file project-files/scenes/terrain.obj and our room project-files/scenes/my_scene.json.

**What they do:**
- intersections.py: tests if a ray hits a triangle or a box, and the brute force (tests the ray against every triangle)
- bvh_median.py and bvh_sah.py: build the BVH tree of boxes (median split or SAH with 12 bins)
- bvh_walk.py: finds the nearest triangle a ray hits (bvh_nearest_hit) or if it hits anything (bvh_any_hit, for shadow rays)
- the batch functions (bvh_nearest_hits_batch, brute_force_nearest_hits_batch): the same for many rays at once, on all the cores
- generate_terrain.py: makes the fractal landscape (with the diamond-square algorithm) and generates terrain.obj
- check_chapter1.py: checks the 3 "It works when" points of Chapter 1 (same hit as brute force, Cornell room speed and image, terrain build time)
- benchmark_bvh.py: the measurement 1 of the report (cost of a ray with brute force, median and sah, from 2000 to 500000 triangles)

Run my tests:
py -m pytest tests/test_intersections.py tests/test_bvh.py -v
   -> It should output **29 passed**

Check Chapter 1 (takes about 1 minute): py benchmarks/check_chapter1.py
   -> For every check it prints the result and what the handout expects (for example: sah: about 5 million rays per second (expected: 2 to 5 million))

The measurement for the report (takes about 1 minute): py benchmarks/benchmark_bvh.py
   -> It prints the results for every terrain level and saves them in output/benchmark_bvh.csv (opens in excel)

Make the terrain again (8 is the level, level L gives 2*4^L triangles): py tools/generate_terrain.py 8
   -> It should output **level 8: 66049 points, 131072 triangles** and it generates project-files/scenes/terrain.obj

If you change intersections.py, delete the numba cache first (numba does not notice changes in functions from another file):
rm renderer/__pycache__/*.nbi renderer/__pycache__/*.nbc


## Katerina Part
**My files** :

In the *renderer* folder: radiosity.py 

My tests files in the *tests* folder: test_radiosity.py 

My benchmark in the *benchmarks* folders: benchmarks_radiosity.py

**What they do**
My module calculates the light that bounces between diffuse surfaces in the scene. Radiosity splits the diffuse surface into small triagles called 'patches' and each patch can receive light, reflect a part of it and send light to other patches. This creates the bounced light that a normal ray tracer doesn't calculate. If the scene has mirrors and glass, they remain in the scene and can block light between the patches but are not turned into patches, because radiosity can't handle specular reflection or refractor.

**Function description**

*radiosity.py*
- triangle_properties(): Calculates and returns the centre, area and unit normal of a single triangle using its three vertices.
- compute_patch_properties():Calculates and returns the centre, area and unit normal of every patch.
- create_patches(): Splits diffuse surfeces into small triagular patches and stores theis material and source triagles IDs.
- compute_visibility(): Checks whether light can travel between two patches without being blocked.
- compute_form_factor_disc(): Calculates the form factor between two patches using the small-disc formula.
- compute_form_factor_rays(): Estimates form factors by shooting random rays from one patch and counting witch patches they hit.
- ray_triangle_intersect(): Finds the distance where a ray hits a triangle.
- solve_radiocity_full_matrix(): Solves radiosity using the full form factor matrix.
- shoot_patch_light(): Sends the unshot light of one patch to all the other patches.
- solve_radiosity_progressive(): repeatedly chooses the patch with the most unshot light ans sends the patches light to the others.
- seperate_direct_light(): Saves the light received directly from the light panel before the diffuse bounces start.
- compute_ambient_term(): Estimates the light that is still unshot and adds it as an ambient term for an unfinished solution.
- subdivide_patches(): Splits selected patches into four smaller triangles.
- compute_patch_vertex_radiosity(): Calculate radiosity at each patch corner.
- render_radiosity(): Displays radiosity using the rasteriser.

**Tests**
Run them with: *py -m pytest tests/test_radiosity.py -v*

*Furnace test*: The Furnace scene ia a closed box where every surface emits *E=1* and reflects *0.5*. Because every patch sees only the inside of the box the form factors of each patch add up to 1. The radiosity equation equals to => *B=1+0.5B* and the expected result equals to *B=2*.
