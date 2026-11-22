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
  direction=direction/distance

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
  return not blocked
    
#Estimates how much light passes between 2 patches using the small-disc formula.
def compute_form_factor_disc(centre_i,normal_i,area_i,centre_j,normal_j,area_j,any_hit):
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
def compute_form_factor_rays(patch_index,patch_vertices,centres,normals,areas,ray_count,any_hit,seed=0):
  #Number of patches.
  patch_count=len(centres)

  #Start with zero form factors.
  form_factors=np.zeros(patch_count,dtype=np.float64)
  #To avoid division by zero if no rays are requested.
  if ray_count<=0:
    return form_factors

  #Get the source patch.
  triangle=patch_vertices[patch_index]
  normal=normals[patch_index]

  #Two edges of the source patch.
  edge1=triangle[1]-triangle[0]
  edge2=triangle[2]-triangle[0]

  #First direction in the patch plane.
  u=edge1/np.linalg.norm(edge1)
  #Second direction in the patch plane.
  v=np.cross(normal,u)
  v=v/np.linalg.norm(v)

  #Use a fixed seed for repeatable results.
  rng=np.random.default_rng(seed)

  #Shoot rays from random points on the source patch.
  for ray_index in range(ray_count):
    #Random barycentric coordinates.
    r1=rng.random()
    r2=rng.random()
    #Keep the point inside the triangle.
    if r1+r2>1.0:
      r1=1.0-r1
      r2=1.0-r2
    #Random point on the source patch.
    point=triangle[0]+r1*edge1+r2*edge2
    #Random direction in the upper hemisphere.
    cos_theta=rng.random()
    sin_theta=np.sqrt(max(0.0,1.0-cos_theta*cos_theta))
    phi=2.0*np.pi*rng.random()

    #Direction using the patch coordinate system.
    direction = (u *(sin_theta*np.cos(phi))+v*(sin_theta*np.sin(phi))+normal * cos_theta)

    #Make the direction unit length.
    direction=direction/np.linalg.norm(direction)

    #Find the closest patch hit by this ray.
    hit_index=-1
    hit_distance=np.inf

    #Test the ray against every patch.
    for other_index in range(patch_count):
        #Don't test the source patch with itself.
        if other_index==patch_index:
             continue
        #Find where the ray hits this patch.
        hit=ray_triangle_intersect(point,direction,patch_vertices[other_index])
        #Keep the closest hit.
        if hit is not None and hit<hit_distance:
            hit_distance=hit
            hit_index=other_index
    #Ignore rays that dont hit another patch.
    if hit_index==-1:
      continue
    #Start slightly above the patch to avoid self intersection.
    start=point+1e-4*normal
    #Check whether something blocks the ray.
    blocked=any_hit(start,direction,hit_distance-2e-4)
    #Count only visible hits.
    if not blocked:
      form_factors[hit_index]+=1.0
  #Turn hit counts into form factors.
  form_factors=form_factors/ray_count
  
  #Return the form factors from thuis patch.
  return form_factors

#Finds the distance where a ray hits a triangle.
def ray_triangle_intersect(origin,direction,vertices):
  #Triangle corners.
  v0,v1,v2=vertices
  #Two triangle edges.
  edge1=v1-v0
  edge2=v2-v0

  #Vector used in the intesection test.
  h=np.cross(direction,edge2)
  a=np.dot(edge1,h)

  #The ray is parallel to the triangle.
  if abs(a)<1e-12:
    return None

  #Barycentric coordinate u.
  f=1.0/a
  s=origin-v0
  u=f*np.dot(s,h)

  #The hit is outside the triangle.
  if u<0.0 or u>1.0:
    return None

  #Barycentric coordinate v.
  q=np.cross(s,edge1)
  v=f*np.dot(direction,q)

  #The hit is outside the triangle.
  if v<0.0 or u+v>1.0:
    return None

  #Distance along the ray.
  t=f*np.dot(edge2,q)
  #The hit is behind the ray.
  if t<=1e-8:
    return None

  #Return the positive hit distance.
  return t

#Calculate the light leaving each patch by repeatedly updating the solution using the full form-factor matrix.
def solve_radiosity_full_matrix(emissions,reflectances,form_factors,iterations=100,tolerance=1e-4):
  #Number of patches.
  patch_count=len(emissions)
  #Start with only emitted light.
  radiosity=emissions.copy()

  #repeat the radiosity equatoin.
  for iteration in range(iterations):
    #Calculate incoming light for every patch.
    incoming=form_factors@radiosity
    #Add emitted light and reflected incoming light.
    new_radiosity=emissions+reflectances*incoming
    #Find the largest change from the last step
    change=np.max(np.abs(new_radiosity-radiosity))
    #Keep the new result.
    radiosity=new_radiosity
    #Stop when the solution barely changes.
    if change<tolerance:
      break

  #Return the final radiosity values.
  return radiosity

#Sends a patch's unshot light to the other patches and updates their total and unsot light.
def shoot_patch_light(patch_index,unshot_radiosity,total_radiosity,reflectances,areas,form_factors):
  #Light that this pach still needs to send.
  sent_light=unshot_radiosity[patch_index]
  #If there isn't any unshot light.
  if sent_light<=0.0:
    return
  #Send light to every other patch.
  for other_index in range(len(unshot_radiosity)):
    #A patch does not send light to itself.
    if other_index== patch_index:
      continue
    #Area ratio from the shooting patch to the receiving patch.
    area_ratio=areas[patch_index]/areas[other_index]
    #Light received by the other patch.
    received_light=(reflectances[other_index]*form_factors[patch_index,other_index]*area_ratio*sent_light)
    #Add the received light to the other patch.
    total_radiosity[other_index]+=received_light
    #Keep the received light ready to be sent.
    unshot_radiosity[other_index]+=received_light
  #This patch has sent all the unshot light.
  unshot_radiosity[patch_index]=0.0

#Calculates radiosity by repeatedly choosing a patch and sending it's unshot light to the others.
def solve_radiosity_progressive(emissions,reflectances,areas,form_factors,tolerance=0.01,max_shots=1000):
  #Start with only emitted light.
  total_emission=np.sum(emissions*areas[:,None],axis=0)
  #Light that has not been sent yet.
  unshot_radiosity=emissions.copy()
  #Total emitted light in the scene.
  total_emission=np.sum(emissions*areas[:,None],axis=0)

  #Repeat until little light is still unshot.
  for shot in range(max_shots):
    #Find the patch that has the most unsot light.
    patch_index=int(np.argmax(np.sum(unshot_radiosity*areas[:,None],axis=1)))
    #Remaining light that still waits to be sent.
    remaining=np.sum(unshot_radiosity*areas[:,None],axis=0)
    #Break when no light is left or less than the tolerance.
    if np.all(remaining<=tolerance*total_emission):
      break
    #Send light from the chosen patch.
    shoot_patch_light(patch_index,unshot_radiosity,total_radiosity,reflectances,areas,form_factors)

  #Return the final radiosity values.
  return total_radiosity

#Saves the direct-light contribution seperately before light starts bouncing between diffuse surfaces.
def separate_direct_light(emissions,reflectance,areas,form_factors):
  #Start with the emmited light.
  direct_radiosity=emissions.copy()
  #Find the patch with the most emitted light.
  light_patch=int(np.argmax(np.sum(emissions*areas[:,None],axis=1)))
  #Make a copy of the unshot light.
  unshot_radiosity=emission.copy()

  #Send light from the light patch once.
  shoot_patch_light(light_patch,unshot_radiosity,direct_radiosity,reflectances,areas,form_factors)

  #Return the direct light received by every patch.
  return direct_radiocity

#Estimates the remaining unshot light as an ambient term for displaying an unfinished solution.
def compute_ambient_term(unshot_radiosity,reflectances.areas):
  #Total patch area.
  total_area=np.sum(areas)
  #Area weighted average reflectance.
  average_reflectance=np.sum(reflectances*areas)/total_area

  #Total unshot light per color channel.
  remaining_light=np.sum(unshot_radiosity*areas[:,None],axis=1)/total_area

  #Formula for ambient light.
  ambient=(remaining_light*average_reflectance/(1.0-average_reflectance))

  #Return the ambient light color.
  return ambient



  
  
