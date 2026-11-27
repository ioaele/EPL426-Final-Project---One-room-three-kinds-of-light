import numpy as np
from numba import njit,prange
from renderer.intersections import intersect_ray_triangle,intersect_ray_box


@njit(cache=True)
def bvh_nearest_hit(o,d,v0,v1,v2,tmin,tmax,node_min,node_max,left,right,start,count,tri_order):
    #finds the nearest triangle the ray hits between tmin and tmax, using the BVH tree
    #it goes down only into the boxes the ray hits and checks the triangles only in those 
    #returns (t,triangle number,u,v), or (inf,-1,0,0) if nothing is hit (same as brute force)
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
            #BUT for more optimization we:"At each node, test the ray against
# the boxes of both children, visit the nearer child first, and skip any box that the ray enters only
# beyond the nearest hit found so far. "
            hit_l,t_entry_l=intersect_ray_box(o,d,node_min[left[row]],node_max[left[row]],tmin,best_t) #intersecting the ray with the whole left child node box
            hit_r,t_entry_r=intersect_ray_box(o,d,node_min[right[row]],node_max[right[row]],tmin,best_t) #intersecting the ray with the whole right child node box
            
            if hit_l==False and hit_r==True: 
                stack[top]=right[row]
                top+=1
            elif hit_r==False and hit_l==True:
                stack[top]=left[row]
                top+=1 
            elif hit_r==True and hit_l==True:
                if t_entry_l>t_entry_r:
                    stack[top]=left[row]
                    top+=1
                    stack[top]=right[row]
                    top+=1 
                else:
                    stack[top]=right[row]
                    top+=1
                    stack[top]=left[row]
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

        if left[row] == -1: #no children (left=-1) meaning this node is a leaf

            #check every triangle of the leaf, like brute force
            for leaf_triangle in tri_order[start[row]:start[row]+count[row]]:
                t,u,v = intersect_ray_triangle(o,d,v0[leaf_triangle],v1[leaf_triangle],v2[leaf_triangle],tmin,tmax)
                if t != np.inf: #found a hit: something blocks the ray, no need to look further
                    return True

        else: #if the node is not a leaf we add its children to the stack to check them in the next rounds
            #BUT for more optimization we:"At each node, test the ray against
# the boxes of both children, visit the nearer child first, and skip any box that the ray enters only
# beyond the nearest hit found so far. "
            hit_l,t_entry_l=intersect_ray_box(o,d,node_min[left[row]],node_max[left[row]],tmin,tmax) #intersecting the ray with the whole left child node box
            hit_r,t_entry_r=intersect_ray_box(o,d,node_min[right[row]],node_max[right[row]],tmin,tmax) #intersecting the ray with the whole right child node box

            if hit_l==False and hit_r==True: 
                stack[top]=right[row]
                top+=1
            elif hit_r==False and hit_l==True:
                stack[top]=left[row]
                top+=1 
            elif hit_r==True and hit_l==True:
                if t_entry_l>t_entry_r:
                    stack[top]=left[row]
                    top+=1
                    stack[top]=right[row]
                    top+=1 
                else:
                    stack[top]=right[row]
                    top+=1
                    stack[top]=left[row]
                    top+=1 

    return False #the stack is empty and nothing was hit



@njit(cache=True, parallel=True)
def bvh_nearest_hits_batch(O,D,v0,v1,v2,tmin,tmax,node_min,node_max,left,right,start,count,tri_order):
    #finds the nearest hit for MANY rays at once (not one by one)
    #O,D: the origins and directions of all the rays shape (k,3), one row per ray
    #returns 4 arrays of length k:t,triangle number,u,v (the answer of ray i is at position i)
    k=O.shape[0] #how many rays there are

    #empty arrays for the answers, one place for each ray
    best_t=np.empty(k)
    best_tri=np.empty(k,np.int64) #int because the triangle number is a whole number
    best_u=np.empty(k)
    best_v=np.empty(k)

    #prange instead of range because the rays are shared between the cores of the computer and run at the same time
    #this is safe because every ray is solved on its own and writes only in its own place i    
    for i in prange(k):
        t,tri,u,v=bvh_nearest_hit(O[i],D[i],v0,v1,v2,tmin,tmax,node_min,node_max,left,right,start,count,tri_order) #the same walk as before for ray i
        #saving the answer of ray i
        best_t[i]=t
        best_tri[i]=tri
        best_u[i]=u
        best_v[i]=v
    return best_t, best_tri, best_u, best_v
