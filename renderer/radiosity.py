"""
Calculates how light bounces between surfaces in the scene.
Splits surfaces into small triangles called patches and calculates how much light passes between them to find their final colours.
"""
import numpy as np

#Calculates and returns the centre, area and unit normal of a single triangle using its three vertices.
def triangle_properties(vertices):
  #Converting the coordinates to a numpy array of floating point numbers.
  vertices=np.asarray(vertices,dtype=np.float64)

  #Extract the 3 vertices (x,y,z).
  v0,v1,v2=vertices

  #Calculate two edged starting from the same vertex, v0.
  edge1=v1-v0
  edge2=v2-v0

  #The cross product gives a vector vertical to the triagle.
  cross=np.cross(edge1,edge2)
  #The length equals twice the triagles area.
  length=np.linalg.norm(cross)

  #Reject near zero area triangles and avoid division by zero.
  if length<=1e-12:
    raise ValueError("Triangle has zero or near zero area")

  #The centroid is the average of the 3 vertices.
  centre=(v0+v1+v2)/3.0
  #The area is half the length of the cross product.
  area=length/2.0

  #Normalize the perpendicular vector to length 1 and its direction depends on the order of thr vertices.
  normal=cross/length

  #Return the 3 calculated properties.
  return centre,area,normal

#Calculates and returns the centre, area and unit normal of each triangular patch provided.
def compute_patch_properties(patch_vertices):
  #Convert the patch vertices to a floating-point array.
  patch_vertices=np.asarray(patch_vertices,dtype=np.float64)

  #Count the patches.
  count=len(patch_vertices)

  #Create arrays to store the results.
  centres=np.zeros((count,3))
  areas=np.zeros(count)
  normals=np.zeros((count,3))

  #Calculate the properties of each triangular patch.
  for i in range(count):
    centres[i],areas[i],normals[i]=triangle_properties(patch_vertices[i])

  #Return the properties of all patches.
  return centres,areas,normals

#Splits diffuse surfeces into small triagular patches and stores theis material and source triagles IDs.
def create_patches(triangles,material_ids,patch_mask,patch_size=0.6):
  #Convert the inputs to NumPy arrays.
  triangles=np.asarray(triangles,dtype=np.float64)
  material_ids=np.asarray(material_ids,dtype=np.int64)
  patch_mask=np.asarray(patch_mask,dtype=bool)

  #Check the triangle shape.
  if triangles.ndim!=3 or triangles.shape[1:]!=(3,3):
    raise ValueError("Triangles must  have shape (T,3,3)")

  #Check that each triangles has a material and mask value.
  count=len(triangles)
  if material_ids.shape!=(count,) or patch_mask.shape!=(count,):
    raise ValueError("Each triangle needs a material ID and a mask value")

  #Check that the coordinates and patch size are valid.
  if not np.all(np.isfinite(triangles)):
    raise ValueError("Triangle coordinates must be finite")
  if not np.isfinite(patch_size) or patch_size<=0:
    raise ValueError("patch_size must be positive and finite")

  #Store the new patches and their IDs.
  patches=[]
  patch_material_ids=[]
  source_triangle_ids=[]

  #Process each original triangle.
  for triangle_id in range(count):
    #Skip surfaces that should not become patches.
    if not patch_mask[triangle_id]:
      continue

    #Check that the triangle has a valid area.
    triangle_properties(triangles[triangle_id])

    #Keep triangles that still need to be checked.
    stack=[triangles[triangle_id]]

    while stack:
      triangle=stack.pop()
      v0,v1,v2=triangle
      #Measure the three edges.
      lengths=np.array([
        np.linalg.norm(v1-v0),
        np.linalg.norm(v2-v1),
        np.linalg.norm(v0-v2)
      ])
      #Keep the triangle if all edges are small enough.
      if np.max(lengths)<=patch_size:
        patches.append(triangle)
        patch_material_ids.append(material_ids[triangle_id])
        source_triangle_ids.append(triangle_id)
        continue
      #Split the longest edge at its midpoint.
      longest=int(np.argmax(lengths))
      a=triangle[longest]
      b=triangle[(longest+1)%3]
      c=triangle[(longest+2)%3]
      midpoint=(a+b)/2.0

      #Keep the same vertex order and normal direction.
      stack.append(np.array([a,midpoint,c]))
      stack.append(np.array([midpoint,b,c]))

  #Return arrays,including correctly shaped empty results.
  return (np.asarray(patches,dtype=np.float64).reshape(-1,3,3), np.asarray(patch_material_ids,dtype=np.int64),np.asarray(source_triangle_ids,dtype=np.int64))

#Checks whether light can travel between 2 patches without being blocked.
def compute_visibility(centre_i,normal_i,centre_j,normal_j,any_hit):
  #Vector from patch i to j.
  direction=centre_j-centre_i
  distance=np.linalg.norm(direction)

  #Check that the patches are apart.
  if distance<=1e-12:
    return False

  #Unit direction from patch i to j.
  direction=direction/distance.

  #Check thet patch i faces towards j.
  if np.dot(normal_i,direction)<=0.0:
    return False
  #Check thet patch j faces towards i.
  if np.dot(normal_j,-direction)<=0.0:
    return False
    
  #Start a bit above i to avoid hitting itself.
  start=centre_i+1e-4*normal_i
  #Check if something blocks the ray.
  blocked=any_hit(start,direction,distance-2e-4)
  
  #The patched are visible.
  return not blocked.
    
#Estimates how much light passes between 2 patches using the small-disc formula.
def compute_from_factor_disc(centre_i,normal_i,area_i,centre_j,normal_j,area_j,any_hit):
  #Vector and disatnce between the patch centres.
  direction=centre_j-centre_i
  distance=np.linalg.norm(direction)

  #Very close or identical patches get no form factor.
  if distance<=1e-12:
    return 0.0

  #Unit direction from patch i to j.
  direction=direction/distance
  #Cosine at patch i.
  cos_i=np.dot(normal_i,direction)
  #Cosine at patch j.
  cos_j=np.dot(normal_j,-direction)

  #Light can't leave or arrive from behind a patch.
  if cos_i<=0.0 or cos_j<=0.0:
    return 0.0

  #Check if the patches can see each other.
  if not compute_visibility(centre_i,normal_i,centre_j,normal_j,any_hit):
    return 0.0
  #Small-disc from factor formula.
  form_factor=area_j*cos_i*cos_j/(np.pi*distance*distance)

  #A form factor can't be larger than 1.
  return min(form_factor,1.0)
  

#Estimates how much light reaches each patch by shooting random rays from source patch.
def compute_from_factor_rays():

#Calculate the light leaving each patch by repeatedly updating the solution using the full form-factor matrix.
def solve_radiosity_full_matrix():

#Sends a patch's unshot light to the other patches and updates their total and unsot light.
def shoot_patch_light():

#Calculates radiosity by repeatedly choosing a patch and sending its unshot light ti the others.
def solve_radiosity_progressive():

#Saves the direct-light contribution seperately before light starts bouncing between diffuse surfaces.
def separate_direct_light():

#Estimates the remaining unshit light as an ambient term for displaying an unfinished solution.
def compute_ambient_term():
  


