'''
Measures how radiosity changes with patch size,form factor method and shots
'''
import time
import numpy as np
import matplotlib.pyplot as plt

from renderer.radiosity import (create_patches,compute_patch_properties,compute_form_factor_disc,compute_form_factor_rays,solve_radiosity_full_matrix,solve_radiosity_progressive)

#Visibility function that says nothing blocks the rays.
def no_blocker(origin,direction,max_distance):
  return False

#Create a simple closed box for the benchmark.
def create_box_triangles():
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
  flip_ids=[0,1,6,7,8,9]
  triangles[flip_ids]=triangles[flip_ids][:,[0,2,1],:]

  #Return the triangles.
  return triangles

#Calculates the form-factor matrix using the disc method.
def calculate_disc_form_factors(patch_vertices,centres,areas,normals):
  #Number of patches.
  patch_count=len(centres)

  #Start with zero form factors.
  form_factors=np.zeros((patch_count,patch_count))

  #Calculate every pair of patches.
  for i in range(patch_count):
    for j in range(patch_count):
      #A patch does not send light to itself.
      if i==j:
        continue

      form_factors[i,j]=compute_form_factor_disc(centres[i],normals[i],areas[i],centres[j],normals[j],areas[j],no_blocker)
      
  #Return the form-factor matrix.
  return form_factors

#Calculates the form-factor matrix using ray shooting.
def calculate_disc_form_factors(patch_vertices,normals,ray_count,nearest_hit,triangle_to_patch):
  #Number of patches.
  patch_count=len(patch_vertices)

  #Start with zero form factors.
  form_factors=np.zeros((patch_count,patch_count))

  #Shoot rays from every patch.
  for i in range(patch_count):
    form_factors[i]=compute_form_factor_rays(i,patch_vertices,normals,ray_count,nearest_hit,triangle_to_patch,seed=i)

  #Return the form-factor matrix.
  return form_factors
