"""
Calculates how light bounces between surfaces in the scene.
Splits surfaces into small triangles called patches and calculates how much light passes between them to find their final colours.
"""
import numpy as np

#To return centre,area and unit normal of a triangle.
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
  normal=cross/length.

#Return the 3 calculated properties.
return centre,area,normal

  

  
