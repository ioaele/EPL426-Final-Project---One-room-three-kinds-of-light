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

#Tests that the triangle properties are correct.
def test_triangle_properties():
  #Create a triangle.
  vertices=np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [0.0,1.0,0.0]
  ])
  #Calculate the properties.
  centre,area,normal=trianlge_properties(vertices)
  #Check the centre.
  assert np.allclose(centre,[1/3,1/3,0.0])
  #Check the area.
  assert np.isclose(area,0.5)
  #Ckeck that the normal is unit length.
  assert np.isclose(np.linalg.norm(normal),1.0)
  #Check that the normal points in the z direction.
  assert np.allclose(normal,[0.0,0.0,1.0])

#Tests that large triangles are splits into smaller patches.
def test_create_patches():
  #Create one large triangle.
  triangles=np.array([
    [0.0,0.0,0.0],
    [2.0,0.0,0.0],
    [0.0,2.0,0.0]
  ])

  #Give it one material and make it a patch.
  material_ids=np.array([0])
  patch_mask=np.array([True])

  #Split in into patches.
  patch_vertices,patch_material_ids,source_triangles_ids=create_patches(
    triangles,
    material_ids,
    patch_mask,
    patch_size=0.5
  )

  #The large triangle must be split.
  assert len(patch_vertices)>1
  #Every patch must keep the same material ID.
  assert np.all(patch_material_ids==0)
  #Every patch must keep the same material ID.
  assert np.all(patch_triangle_ids==0)

  #No patch edge can be longer than the patch size.
  for patch in patch_vertices:
    edge_lengths=np.array([
      np.linalg.norm(patch[1]-patch[0]),
      np.linalg.norm(patch[2]-patch[1]),
      np.linalg.norm(patch[0]-patch[2]),
    ])
    assert np.max(edge_lengths)<=0.5+1e-9

#Tests that the disc form factors are valid.
def test_form_factor_disc_bounds():
  #Create a small closed box.
  patch_vertices,_,_=create_box_patches(patch_size=0.5)
  #Calculate the patch properties.
  centres,areas,normals=compute_patch_properties(patch_vertices)

  #Check every pair of patches.
  for i in range(len(centres)):
    for j in range (len(centres)):
      #A patch does not send light to itself.
      if i==j:
        continue
      #Calculate the form factor.
      form_factor=compute_form_factor_disc(
        centres[i],normals[i],areas[i],
        centres[j],normals[j],areas[j],
        no_blocker
      )
      #The form factor must be valid.
      assert form_factor>=0.0
      assert form_factor<=1.0
