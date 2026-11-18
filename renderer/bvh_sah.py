import numpy as np
LEAF_SIZE = 4   #a node with this many triangles or fewer becomes a leaf


def build_bvh_sah(v0, v1, v2):
    #builds the BVH tree with the sah method 
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

        bins= np.zeros(e-s, dtype=np.int64)  #initializing the bins for the splitting
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

        #splitting the node into two using the SAH  

        #the axis where the centres spread the most (how far apart the centres are along each axis)
        spread=centre[node_triangles].max(axis=0)-centre[node_triangles].min(axis=0)
        axis=np.argmax(spread) #0=x,1=y,2=z

        max_center=np.max(centre[node_triangles, axis])
        min_center=np.min(centre[node_triangles, axis])

        if max_center == min_center: #all the centres are at the same spot:
#the bins cannot separate them (and we'd divide by 0) so we make it a leaf
            start[row]=s
            count[row]=e-s
            continue
        
        #sah parting
        N_BINS=12 #dozen bins=12
        for p in range(e-s):
            #in which bin does the triangle go in
            b=int(N_BINS*(centre[node_triangles[p],axis]-min_center)/(max_center-min_center))
            if b == N_BINS: 
                b=N_BINS-1  #12->11 (0...11 bins
            
            bins[p]=b
            
        #try every cut between the bins: left=0..k right=k+1..N_BINS-1
        best_cost=np.inf
        best_k=-1
            
        for k in range(N_BINS-1):
                
            left_triangles=node_triangles[bins <= k] #triangles in bins 0..k
            right_triangles=node_triangles[bins > k] #triangles in bins k+1..11
                
            if len(left_triangles)==0 or len(right_triangles)==0:
                continue
            
            left_min=tri_min[left_triangles].min(axis=0) #smallest x,y,z of the left triangles
            left_max=tri_max[left_triangles].max(axis=0) #largest x,y,z of the left triangles
            dx,dy,dz=left_max-left_min  #the 3 side lengths of the left box
            s_l=2*(dx*dy + dy*dz + dz*dx)   #its surface area
                
            right_min=tri_min[right_triangles].min(axis=0) 
            right_max=tri_max[right_triangles].max(axis=0) 
            dx,dy,dz=right_max-right_min  
            s_r=2*(dx*dy + dy*dz + dz*dx) 
                
                
            cost=s_l*len(left_triangles)+s_r*len(right_triangles)
            
            if cost<best_cost:
                best_cost=cost
                best_k=k

                #"If even the best
    # split costs more than keeping all the triangles of the node together (SN, with S the area of the
    # node’s own box), make the node a leaf instead."
        dx,dy,dz=node_max[row]-node_min[row]
        s_node=2*(dx*dy + dy*dz + dz*dx) 
                
        if best_k == -1 or best_cost>=s_node*len(node_triangles):   
            start[row]=s
            count[row]=e-s
            continue
            
            
        left_triangles = node_triangles[bins <= best_k]#the triangles going left
        right_triangles = node_triangles[bins > best_k]#the triangles going right
        tri_order[s:e] = np.concatenate((left_triangles, right_triangles))#the left ones first then the right ones
        mid=s+len(left_triangles) #where the left part ends and the right part starts

        
        left[row]=next_free_row  #left child gets the first free row
        right[row]=next_free_row+1  #right child gets the one after it
        next_free_row+=2  #those two are now taken

        #build the children later
        stack.append((left[row],s,mid))
        stack.append((right[row],mid,e))

    #keep only the rows we used
    nodes_used=next_free_row
    return node_min[:nodes_used], node_max[:nodes_used], left[:nodes_used], right[:nodes_used], start[:nodes_used], count[:nodes_used], tri_order


