"""
Tests for the radiocity module.

Checks triangle properties,patch creation,form factors and the radiosity solvers.
"""

import numpy as np
import pytest

from render.radiosity import ( triangle_properties, compute_patch_properties, create patches, compute_form_factor_disc, compute_form_factor_rays, solve_radiosity_progressive, seperate_direct_light, compute_ambient_term)

#Visibility function that says nothing blocks the rays.
def no_blocker(origin,direction,max_distance):
  return False

#Create a simple closed box from 12 triangles.
def create_box_patches(patch_size=0.5)
  #Box corners.
  min_corner=np.array([0.0,0.0,0.0])
  max_corner=np.array([1.0,1.0,1.0])

  #Create the 12 triangles for the box.
  triangles=[]

  #Bottom face.
  triangles.append(np.array)[
                   [0.0,0.0,0.0],
                   [1.0,0.0,0.0],
                   [1.0,0.0,1.0]
                   ])
  )
  triangles.append(np.array)[
                   [0.0,0.0,0.0],
                   [1.0,0.0,1.0],
                   [0.0,0.0,1.0]
                   ])
  )

  #Top face.
  triangles.append(np.array)[
                   [0.0,1.0,0.0],
                   [1.0,1.0,0.0],
                   [1.0,1.0,1.0]
                   ])
  )
  triangles.append(np.array)[
                   [0.0,1.0,0.0],
                   [1.0,1.0,1.0],
                   [0.0,1.0,1.0]
                   ])
  )
  #Front face.
  triangles.append(np.array)[
                   [0.0,0.0,0.0],
                   [1.0,0.0,0.0],
                   [1.0,1.0,0.0]
                   ])
  )
  triangles.append(np.array)[
                   [0.0,0.0,0.0],
                   [1.0,1.0,0.0],
                   [0.0,1.0,0.0]
                   ])
  )
  #Back face.
  triangles.append(np.array)[
                   [0.0,0.0,1.0],
                   [1.0,0.0,1.0],
                   [1.0,1.0,1.0]
                   ])
  )
  triangles.append(np.array)[
                   [0.0,0.0,1.0],
                   [1.0,1.0,1.0],
                   [0.0,1.0,1.0]
                   ])
  )
  #Left face.
  triangles.append(np.array)[
                   [0.0,0.0,0.0],
                   [0.0,1.0,0.0],
                   [0.0,1.0,1.0]
                   ])
  )
  triangles.append(np.array)[
                   [0.0,0.0,0.0],
                   [0.0,1.0,1.0],
                   [0.0,0.0,1.0]
                   ])
  )
  #Right face.
  triangles.append(np.array)[
                   [1.0,0.0,0.0],
                   [1.0,1.0,0.0],
                   [1.0,1.0,1.0]
                   ])
  )
  triangles.append(np.array)[
                   [1.0,0.0,0.0],
                   [1.0,1.0,1.0],
                   [1.0,0.0,1.0]
                   ])
  )

  #Convert triangles to a numpy array.
  triangles=np.asarray(triangles,dtype=np.float64)
  
  #Every triangle is diffuse and turn into patches.
  material_ids=np.zeros(len(triangles),dtype=np.int64)
  patch_mask=np.ones(len(triangles),dtype=bool)

  #Split the triangles into patches.
  patch_vertices,patch_material_ids,source_triangle_ids=create_patches( triangles,material_ids,patch_mask,patch_size)

  #Return the patch data.
  return patch_vertices,patch_material_ids,source_triangle_ids
