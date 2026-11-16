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

##Team

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
│   └── teapot.obj
│
├── tests/
│
├── output/
│
└── report/
    └── figures/
```

## Installation

## Running the renderer

## Report

Each member contributes:

- A description of their modules and design decisions.
- Tests and validation results.
- Benchmark tables or plots.
- Rendered images and known limitations.
