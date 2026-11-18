import numpy as np
from numba import njit
from renderer.intersections import intersect_ray_triangle,intersect_ray_box
LEAF_SIZE = 4   #a node with this many triangles or fewer becomes a leaf


def build_bvh_median(v0, v1, v2):
    #builds the BVH tree with the median method 
    #v0,v1,v2:the corners of all triangles shape (N,3)
    #the tree is stored as arrays one row per node:
    #node_min,node_max:the box of the node (two corners)
    #left,right:the rows of the two children (-1 for a leaf)
    #start,count:for a leaf its triangles are tri_order[start:start+count]
    #tri_order:the triangle numbers
    
    N = v0.shape[0] #how many triangles there are

    #for every triangle finding its own box 
    #(we know it if we know the min and max x,y,z) and its centre 
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)
    centre=(v0 + v1 + v2)/3

    #initializing empty arrays for the nodes (the maximum nodes a tree with n triangle has are up to 2n)
    node_min=np.zeros((2*N,3))
    node_max=np.zeros((2*N,3))
    left=np.full(2*N,-1)
    right=np.full(2*N,-1)
    start=np.zeros(2*N,dtype=np.int64)
    count=np.zeros(2*N,dtype=np.int64)

    tri_order=np.arange(N)   #0,1,2,...,N-1

    #to-do list of nodes to build: (row, start, end)
    #the root is row 0 and has all triangles 
    stack = [(0, 0, N)]
    next_free_row=1   #row 0 is the root so the first free row is 1

    while len(stack)>0:
        row,s,e = stack.pop()
        node_triangles=tri_order[s:e] #getting the triangles of this node

        #the box of this node that contains all the boxes of its triangles:
        #its min corner is the smallest x, y, z of all their min corners
        #its max corner is the largest x, y, z of all their max corners
        node_min[row]=tri_min[node_triangles].min(axis=0)
        node_max[row]=tri_max[node_triangles].max(axis=0)

        #if the node has few triangles then this node is a leaf
        if e-s<=LEAF_SIZE:
            start[row]=s
            count[row]=e-s
            continue

        #splitting the node into two using the MEDIAN  

        #the axis where the centres spread the most (how far apart the centres are along each axis)
        spread=centre[node_triangles].max(axis=0)-centre[node_triangles].min(axis=0)
        axis=np.argmax(spread) #0=x,1=y,2=z

        #sort this node's triangles by their centre on that axis
        order=np.argsort(centre[node_triangles, axis])
        tri_order[s:e]=node_triangles[order]

        #median - the first half goes left and the second half goes right
        mid=(s+e)//2
        
        left[row]=next_free_row  #left child gets the first free row
        right[row]=next_free_row+1  #right child gets the one after it
        next_free_row+=2  #those two are now taken

        #build the children later
        stack.append((left[row],s,mid))
        stack.append((right[row],mid,e))

    #keep only the rows we used
    nodes_used=next_free_row
    return node_min[:nodes_used], node_max[:nodes_used], left[:nodes_used], right[:nodes_used], start[:nodes_used], count[:nodes_used], tri_order



@njit(cache=True)
def bvh_nearest_hit(o,d,v0,v1,v2,tmin,tmax,node_min,node_max,left,right,start,count,tri_order):
    #finds the nearest triangle the ray hits between tmin and tmax, using the BVH tree
    #it goes down only into the boxes the ray hits and checks the triangles only in those 
    #returns (t,triangle number,u,v), or (inf, -1, 0, 0) if nothing is hit (same as brute force)
    #u and v: where on the triangle the hit is (barycentric coordinates)

    
    #initializing the intersection parameteres
    best_t=tmax #there is no hit yet so we look up to tmax
    best_tri = -1
    best_u=0.0
    best_v = 0.0
    
    
    stack=np.empty(64,np.int64) #stack: the rows of nodes still to visit
    stack[0]=0 #first visitor row 0 (root)
    top=1 #how many rows are in the stack 

    while top>0:
        
        top -= 1
        row = stack[top] #pop row


        hit,t_entry=intersect_ray_box(o,d,node_min[row],node_max[row],tmin,best_t) #intersecting the ray with the whole node box
            
        if hit==False: #if there is no hit we continue to the next row in our stack 
            continue

        if left[row] == -1: #if left row is -1 -> leaf node
            
            #now is the same as the brute force method as we check every triange individually
            for leaf_triangle in tri_order[start[row]:start[row]+count[row]]:
                t,u,v = intersect_ray_triangle(o, d, v0[leaf_triangle], v1[leaf_triangle],v2[leaf_triangle],tmin,best_t)
                if t!=np.inf:
                    best_t=t
                    best_tri=leaf_triangle
                    best_u=u
                    best_v=v    
        else: #if the node is not a leaf we add its children to the stack to check them in the next rounds
            stack[top]=left[row]
            top+=1 
            stack[top]=right[row]
            top+=1
           
    if best_tri==-1 :
        return np.inf, -1, 0.0, 0.0
    
    else: 
        return best_t, best_tri, best_u, best_v




@njit(cache=True)
def bvh_any_hit(o,d,v0,v1,v2,tmin,tmax,node_min,node_max,left,right,start,count,tri_order):
    #for shadow rays: True if the ray hits ANY triangle between tmin and tmax
    #same walk as bvh_nearest_hit, but it stops at the first hit

    stack=np.empty(64, np.int64) #stack: the rows of nodes still to visit
    stack[0] = 0  #start with row 0 (the root)
    top=1 #how many rows are in the stack

    while top>0:
        top -= 1
        row = stack[top] #pop row


        hit,t_entry=intersect_ray_box(o,d,node_min[row],node_max[row],tmin,tmax) #intersecting the ray with the whole node box
        if hit == False: #the ray misses the box -> skip this node and everything under it
            continue

        if left[row] == -1: #no children (left = -1): this node is a leaf

            #check every triangle of the leaf, like brute force
            for leaf_triangle in tri_order[start[row]:start[row]+count[row]]:
                t,u,v = intersect_ray_triangle(o,d,v0[leaf_triangle],v1[leaf_triangle],v2[leaf_triangle],tmin,tmax)
                if t != np.inf: #found a hit: something blocks the ray, no need to look further
                    return True

        else: #if the node is not a leaf we add its children to the stack to check them in the next rounds
            stack[top]=left[row]
            top+=1
            stack[top]=right[row]
            top+=1

    return False #the stack is empty and nothing was hit
