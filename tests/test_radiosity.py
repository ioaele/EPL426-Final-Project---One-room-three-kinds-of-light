"""
Tests for the radiocity module.

Checks triangle properties,patch creation,form factors and the radiosity solvers.
"""

import numpy as np
import pytest

from renderer.radiosity import ( triangle_properties, compute_patch_properties, create_patches, compute_form_factor_disc, compute_form_factor_rays,solve_radiosity_full_matrix, solve_radiosity_progressive, separate_direct_light, compute_ambient_term)

#Visibility function that says nothing blocks the rays.
def no_blocker(origin,direction,max_distance):
  return False

#Create a simple closed box from 12 triangles.
def create_box_patches(patch_size=0.5):
  #Box corners.
  min_corner=np.array([0.0,0.0,0.0])
  max_corner=np.array([1.0,1.0,1.0])

  #Create the 12 triangles for the box.
  triangles=[]

  #Bottom face.
  triangles.append(np.array([
                   [0.0,0.0,0.0],
                   [1.0,0.0,0.0],
                   [1.0,0.0,1.0]
                   ])
  )
  triangles.append(np.array([
                   [0.0,0.0,0.0],
                   [1.0,0.0,1.0],
                   [0.0,0.0,1.0]
                   ])
                  )
  #Top face.
  triangles.append(np.array([
                   [0.0,1.0,0.0],
                   [1.0,1.0,0.0],
                   [1.0,1.0,1.0]
                   ])
                  )
  triangles.append(np.array([
                   [0.0,1.0,0.0],
                   [1.0,1.0,1.0],
                   [0.0,1.0,1.0]
                   ])
                  )  
  #Front face.
  triangles.append(np.array([
                   [0.0,0.0,0.0],
                   [1.0,0.0,0.0],
                   [1.0,1.0,0.0]
                   ])
                  )
  triangles.append(np.array([
                   [0.0,0.0,0.0],
                   [1.0,1.0,0.0],
                   [0.0,1.0,0.0]
                   ])
                  )
  #Back face.
  triangles.append(np.array([
                   [0.0,0.0,1.0],
                   [1.0,0.0,1.0],
                   [1.0,1.0,1.0]
                   ])
                  )  
  triangles.append(np.array([
                   [0.0,0.0,1.0],
                   [1.0,1.0,1.0],
                   [0.0,1.0,1.0]
                   ])
                  )  
  #Left face.
  triangles.append(np.array([
                   [0.0,0.0,0.0],
                   [0.0,1.0,0.0],
                   [0.0,1.0,1.0]
                   ])
                  )  
  triangles.append(np.array([
                   [0.0,0.0,0.0],
                   [0.0,1.0,1.0],
                   [0.0,0.0,1.0]
                   ])
                  )
  #Right face.
  triangles.append(np.array([
                   [1.0,0.0,0.0],
                   [1.0,1.0,0.0],
                   [1.0,1.0,1.0]
                   ])
                  )
  triangles.append(np.array([
                   [1.0,0.0,0.0],
                   [1.0,1.0,1.0],
                   [1.0,0.0,1.0]
                   ])
                  )
  
  #Convert triangles to a numpy array.
  triangles=np.asarray(triangles,dtype=np.float64)

  #Point all box normals inward.
  flip_ids=[0, 1, 6, 7, 8, 9]
  triangles[flip_ids]=triangles[flip_ids][:, [0, 2, 1], :]
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
  centre,area,normal=triangle_properties(vertices)
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
  triangles=np.array([[
    [0.0,0.0,0.0],
    [2.0,0.0,0.0],
    [0.0,2.0,0.0]
  ]])

  #Give it one material and make it a patch.
  material_ids=np.array([0])
  patch_mask=np.array([True])

  #Split in into patches.
  patch_vertices,patch_material_ids,source_triangle_ids=create_patches(
    triangles,
    material_ids,
    patch_mask,
    patch_size=0.5
  )

  #The large triangle must be split.
  assert len(patch_vertices)>1
  #Every patch must keep the same material ID.
  assert np.all(patch_material_ids==0)
  #Every patch must come from the same source triangle.
  assert np.all(source_triangle_ids==0)

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
    for j in range(len(centres)):
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

#Tests that the full matrix and progressive solver agree.
def test_full_vs_progressive():
  #Create 4 patches.
  patch_count=4
  #Every patch emits light.
  emissions=np.ones((patch_count,3))
  #Every patch reflects half of the incoming light.
  reflectances=np.full((patch_count,3),0.5)
  #Give every patch the same area.
  areas=np.ones(patch_count)
  #Create form factors where every row adds up to 1.
  form_factors=np.full((patch_count,patch_count),1.0/(patch_count-1))
  #Remove the diagonal values.
  np.fill_diagonal(form_factors,0.0)

  #Return one row from the test matrix.
  def compute_form_factor_row(patch_index):
    return form_factors[patch_index]
  #Solve with the full matrix.
  full_radiosity=solve_radiosity_full_matrix(emissions,reflectances,areas,form_factors)

  #Solve with the progressive refinement.
  progressive_radiosity=solve_radiosity_progressive(emissions,reflectances,areas,compute_form_factor_row,tolerance=1e-6,max_shots=1000)

  #The two solvers must give the same result or a close result.
  assert np.allclose(full_radiosity,progressive_radiosity,atol=1e-3,rtol=0.0)

  #In this case the radiosity must be around 2.
  assert np.allclose(full_radiosity,2.0,atol=0.05,rtol=0.0)

#Tests that direct light is saved without extra bounces.
def test_separate_direct_light():
  #Create two light patches.
  emissions=np.array([
    [1.0,1.0,1.0],
    [2.0,2.0,2.0]
  ])

  #Each patch reflects half the light.
  reflectances=np.full((2,3),0.5)

  #Give both patches the same area.
  areas=np.ones(2)

  #Each patch sends all its light to the other.
  form_factors=np.array([
    [0.0,1.0],
    [1.0,0.0]
  ])

  #Return one form-factor row.
  def compute_form_factor_row(patch_index):
    return form_factors[patch_index]

  #Calculate direct light.
  direct_radiosity=separate_direct_light(emissions,reflectances,areas,compute_form_factor_row)

  #Expected emission and direct reflection.
  expected=np.array([
    [2.0,2.0,2.0],
    [2.5,2.5,2.5]
  ])

  #Check the result.
  assert np.allclose(direct_radiosity,expected)

#Tests the ambient light for each patch.
def test_compute_ambient_term():
  #Give both patches unshot light.
  unshot_radiosity=np.ones((2,3))

  #Give each patch a different reflectance.
  reflectances=np.array([
    [0.2,0.2,0.2],
    [0.6,0.6,0.6]
  ])

  #Give each patch a different area.
  areas=np.array([1.0,3.0])

  #Calculate the ambient light.
  ambient=compute_ambient_term(unshot_radiosity,reflectances,areas)

  #Expected ambient light for each patch.
  expected=np.array([
    [0.4,0.4,0.4],
    [1.2,1.2,1.2]
  ])

  # Check that each patch has three colour values.
  assert ambient.shape==(2,3)

  # Check the ambient light values.
  assert np.allclose(ambient,expected)

#Test the furnace scene.
def test_furnace():
  #Create a closed box with small patches.
  patch_vertices, _, _ = create_box_patches(patch_size=0.25)
  #Calculate the patch properties.
  centres,areas,normals=compute_patch_properties(patch_vertices)
  #Number of patches.
  patch_count=len(centres)

  #Every patch emits light.
  emissions=np.ones((patch_count,3))
  #Every patch reflects half of the incoming light.
  reflectances=np.full((patch_count,3),0.5)

  #Create the form-factor matrix using the disc method.
  form_factors=np.zeros((patch_count,patch_count))

  #Calculate the form factors.
  for i in range(patch_count):
    for j in range(patch_count):
      #A patch doesn't send light to itself.
      if i==j:
        continue
      form_factors[i,j]=compute_form_factor_disc(centres[i], normals[i], areas[i], centres[j], normals[j], areas[j], no_blocker)

  #Check that the form factors add up to 1 or close to 1.
  row_sums=np.sum(form_factors,axis=1)

  assert np.all(row_sums>0.5)
  assert np.all(row_sums<1.5)

  #Solve radiosity with the full matrix.
  radiosity=solve_radiosity_full_matrix(emissions, reflectances,areas, form_factors)

  #Count patches close to 2.
  close_to_two=np.sum(np.all(np.abs(radiosity-2.0)<=0.1,axis=1))

  #At least 0.8 of the patches are close to 2.
  assert close_to_two>=0.8*patch_count  
  
