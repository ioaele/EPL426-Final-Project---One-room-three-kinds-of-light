import numpy as np
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


