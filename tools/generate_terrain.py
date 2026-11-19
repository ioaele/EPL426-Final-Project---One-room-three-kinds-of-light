import sys
import numpy as np

#reference: https://janert.me/blog/2022/the-diamond-square-algorithm-for-terrain-generation/

#the 4 neighbours we average over (as offsets from the new point)
DIAMOND=[(-1,-1),(-1,1),(1,1),(1,-1)]  #the 4 corners of a square (diagonal neighbours)
SQUARE=[(-1,0),(0,-1),(1,0),(0,1)]  #up, left, down, right

#STEP 1 MAKE THE GRID
def average(heights,row,col,distance,offsets):
    #average of the neighbours of point (row,col) at the given distance
    #we only use the neighbours that are inside the grid
    #(at the border only 3 exist, so we divide by how many we found)
    grid_size=heights.shape[0]
    total=0.0
    found=0
    for row_offset,col_offset in offsets:
        neighbour_row=row+row_offset*distance
        neighbour_col=col+col_offset*distance
        if 0<=neighbour_row<grid_size and 0<=neighbour_col<grid_size:
            total+=heights[neighbour_row,neighbour_col]
            found+=1
    return total/found


def diamond_square_step(heights,square_size,offset_size,random):
    #one round of the algorithm
    #square_size: the distance from one new point to the next (the size of the squares)
    #offset_size: how big the random offsets are in this round
    grid_size=heights.shape[0]
    half=square_size//2  #the distance from a new point to the neighbours we average

    #DIAMOND: the centre of every square = average of its 4 corners + random offset
    for row in range(half,grid_size,square_size):
        for col in range(half,grid_size,square_size):
            heights[row,col]=average(heights,row,col,half,DIAMOND)+random.uniform(-offset_size,offset_size)

    #SQUARE(rows): the middles of the vertical edges
    for row in range(half,grid_size,square_size):
        for col in range(0,grid_size,square_size):
            heights[row,col]=average(heights,row,col,half,SQUARE)+random.uniform(-offset_size,offset_size)

    #SQUARE(columns): the middles of the horizontal edges
    for row in range(0,grid_size,square_size):
        for col in range(half,grid_size,square_size):
            heights[row,col]=average(heights,row,col,half,SQUARE)+random.uniform(-offset_size,offset_size)


def generate_terrain(level,seed,roughness=0.6):
    #level: level of detail we want
    #seed: so the random numbers are the same every time we run the function
    #roughness: how much smaller the offsets get each round (0..1)
    # smaller = smoother hills (Janert suggests 0.55-0.7)
    #returns the grid of heights, shape (grid_size,grid_size)

    #"every new height is the average of its neighbours plus a
    # random offset, and the size of the offsets shrinks at every
    # level of detail. Level L has (2^L + 1) x (2^L + 1) heights and
    # gives 2*4^L triangles."

    #the empty grid
    grid_size=2**level+1  #the grid size (2^L + 1)
    heights=np.zeros((grid_size,grid_size))  #the heights (initialized to 0)
    random=np.random.default_rng(seed)

    #giving the 4 corners of the grid a random height
    last=grid_size-1  #the index of the last row/column
    heights[0,0]=random.uniform(-1,1)
    heights[0,last]=random.uniform(-1,1)
    heights[last,0]=random.uniform(-1,1)
    heights[last,last]=random.uniform(-1,1)

    square_size=grid_size-1  #the size of the squares (at the start: the whole grid)
    offset_size=1.0  #how big the random offsets are
    while square_size>1:
        diamond_square_step(heights,square_size,offset_size,random)
        square_size=square_size//2  #next round: squares half as big
        offset_size=offset_size*roughness  #"the size of the offsets shrinks at every level of detail."

    return heights

#STEP 2 GRID POINTS TO COORDS
def make_points(heights,size=16.0,height=2.0):
    #every grid point (row,col) becomes a 3D point (x,y,z)
    #y is up (the camera in terrain.json looks down at the ground)
    #size: the width of the landscape. x and z go from -size/2 to size/2 
    #height: how tall the hills are (the heights are about -1..1 so the hills are about -2..2)
    #returns an array (grid_size*grid_size,3) numbered row by row like in make_triangles
    grid_size=heights.shape[0]
    points=np.zeros((grid_size*grid_size,3))
    for row in range(grid_size):
        for col in range(grid_size):
            x=-size/2+size*col/(grid_size-1)  #left to right across the columns
            z=-size/2+size*row/(grid_size-1)  #back to front across the rows
            y=heights[row,col]*height  #the height of the ground at this point
            points[row*grid_size+col]=[x,y,z]  #point number=row*grid_size+col
    return points

#STEP 3 SQUARES TO TRIANGLES
def make_triangles(grid_size):
    #every small square of the grid becomes 2 triangles
    #the points are numbered row by row: point number=row*grid_size+col
    #returns an array (number of triangles,3) with the 3 point numbers of each triangle
    triangles=[]
    for row in range(grid_size-1):  #grid_size-1 squares per side
        for col in range(grid_size-1):
            a=row*grid_size+col  #top left corner of the square
            b=row*grid_size+(col+1)  #top right
            c=(row+1)*grid_size+col  #bottom left
            d=(row+1)*grid_size+(col+1)  #bottom right
            triangles.append([a,c,b])
            triangles.append([b,c,d])
    return np.array(triangles)


#STEP 4 MAKING THE TERRAIN FILE
def write_obj(path,points,triangles):
    #writes the landscape as an .obj file
    #v x y z  -> one line per point
    #f a b c  -> one line per triangle
    #(the .obj counts the points from 1 while python counts from 0, so +1)
    with open(path,"w") as file:
        for x,y,z in points:
            file.write(f"v {x} {y} {z}\n")
        for a,b,c in triangles:
            file.write(f"f {a+1} {b+1} {c+1}\n")


# python tools/generate_terrain.py level_of_detail
if __name__=="__main__":
    level=8 #if not specified
    if len(sys.argv)>1:
        level=int(sys.argv[1])  #the level given in the command line

    heights=generate_terrain(level,seed=0)  #step 1
    points=make_points(heights)  #step 2
    triangles=make_triangles(heights.shape[0])  #step 3
    write_obj("project-files/scenes/terrain.obj",points,triangles)  #step 4
    print(f"level {level}: {len(points)} points, {len(triangles)} triangles -> project-files/scenes/terrain.obj")
