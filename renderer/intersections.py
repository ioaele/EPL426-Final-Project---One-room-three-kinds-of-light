import numpy as np
from numba import njit,prange

EPS = 1e-12 #values smaller than this are treated as 0 because floating-point rounding 
#means a result that should be exactly 0 usually comes out as a number very close to it 

#helper functions
@njit(cache=True)
def sub3(a, b):
    #a - b for two 3D vectors
    #returns a tuple (like cross3) because creating a new numpy array every time is slow
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

@njit(cache=True)
def dot3(a, b):
    #dot product of two 3D vectors
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def cross3(a, b):
    #cross product of two 3D vectors
    #return np.array([a[1] * b[2] - a[2] * b[1],a[2] * b[0] - a[0] * b[2],a[0] * b[1] - a[1] * b[0]])
    #MORE OPTIMAL WAY: returns a tuple (x, y, z) and not a numpy array because creating a new array every time is slow and
    #a tuple of 3 numbers is almost free in numba 
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)


@njit(cache=True)
def intersect_ray_triangle(o,d,a,b,c,t_min,t_max):
    """Testing 1 ray against 1 triangle
    (1) find where the ray meets the triangle's plane.
    (2) check that this point is inside the triangle by testing it against each of the 3 edges.

    The ray is P(t) = o + t * d.
    Returns (t, u, v). If there is no hit, t is np.inf.
    u and v are the barycentric coordinates of the hit point P:
        P = (1 - u - v) * a + u * b + v * c
    """
    #STEP1

    #normal of the plane
    ab=sub3(b,a)  #the edge a->b (used again below, so we compute it once)
    n=cross3(ab,sub3(c,a))

    #checking n.d=0. if 0 it means that the ray is parallel to the plane
    n_d=dot3(n,d)
    if abs(n_d)<EPS:  #we use < EPS and not == 0 because of rounding errors
        return np.inf, 0.0, 0.0

    #plane equation (in lectures):n.p+d_plane=0 where d_plane is a number
    #corner a is on the plane so n.a+d_plane=0  ->  d_plane = -n.a
    d_plane = -dot3(n, a)

    #we put the ray p = o + t*d into the plane equation: n.(o + t*d) + d_plane = 0
    #solving for t (lecture slides): t = -(d_plane + n.o) / n.d
    t = -(d_plane + dot3(n, o)) / n_d

    #t must be between t_min and t_max 
    #t <= t_min:the hit is behind the ray's start 
    #t >= t_max:the hit is too far beyond the light or behind a closer hit we already have
    if t <= t_min or t >= t_max:
        return np.inf, 0.0, 0.0

    # The point where the ray meets the plane
    P=(o[0]+t*d[0],o[1]+t*d[1],o[2]+t*d[2])  #o+t*d as a tuple (not a new array)

    #STEP2 checking if the intersecting point is inside the triangle


    #For each edge: cross(edge,P - edge start) points the same way as the triangle's normal n 
    #only if P is on the inner side of that edge.
    #dot(...,n) >= 0 -> inside of this edge and dot(...,n)< 0 -> P is outside the triangle.

    #edge a -> b
    e_ab=dot3(cross3(ab,sub3(P,a)),n)
    if e_ab < 0.0:
        return np.inf, 0.0, 0.0      

    #edge b -> c
    e_bc=dot3(cross3(sub3(c,b),sub3(P,b)),n)
    if e_bc < 0.0:
        return np.inf, 0.0, 0.0     

    #edge c -> a
    e_ca=dot3(cross3(sub3(a,c),sub3(P,c)),n)
    if e_ca < 0.0:
        return np.inf, 0.0, 0.0      


    #P splits the triangle into 3 small triangles.
    #Each one's area / the whole area = the weight of the corner OPPOSITE it.
    #e / n.n is exactly that fraction.
    n_n=dot3(n,n)
    u=e_ca / n_n    # weight of b
    v=e_ab / n_n    # weight of c   (weight of a = 1 - u - v)
    return t,u,v



@njit(cache=True)
#findind the nearist hit triangle with brute force 
def brute_force_nearest_hit(o, d, v0, v1, v2, t_min, t_max): #o origin of the ray, d direction of the ray, v0,v1,v2 arrays with each triangles vertices,
    #tmin and tmax
    best_t=t_max #anything farther than this is ignored
    best_tri=-1 #closest hit triangle      
    best_u=0.0
    best_v=0.0

    #looping over every triangle
    for i in range(v0.shape[0]):
        t,u,v=intersect_ray_triangle(o,d,v0[i],v1[i],v2[i],t_min,best_t)
        
        if t!=np.inf : #if it hit it is closer than best_t because the function already rejects t >= best_t (tmax)
            best_tri=i 
            best_t=t
            best_u=u
            best_v=v

    if best_tri==-1 :
        return np.inf,-1,0.0,0.0
    
    return best_t,best_tri,best_u,best_v



@njit(cache=True)
#returns true if there is any hit for the ray with any triangle else returns false
def brute_force_any_hit(o,d,v0,v1,v2,t_min,t_max):
    for i in range(v0.shape[0]):
        t,_,_=intersect_ray_triangle(o,d,v0[i],v1[i],v2[i],t_min,t_max)
        
        if t!=np.inf:
             return True


    return False



@njit(cache=True, parallel=True)
def brute_force_nearest_hits_batch(O,D,v0,v1,v2,t_min,t_max):
    #brute force for MANY rays at once, like bvh_nearest_hits_batch (so we can compare them)
    #O,D: the origins and directions of all the rays, shape (k,3), one row per ray
    #returns 4 arrays of length k: t, triangle number, u, v (the answer of ray i is at position i)
    k=O.shape[0]  #how many rays there are
    best_t=np.empty(k)
    best_tri=np.empty(k,np.int64)
    best_u=np.empty(k)
    best_v=np.empty(k)
    #prange: the rays are shared between the cores and run at the same time
    for i in prange(k):
        t,tri,u,v=brute_force_nearest_hit(O[i],D[i],v0,v1,v2,t_min,t_max)
        best_t[i]=t
        best_tri[i]=tri
        best_u[i]=u
        best_v[i]=v
    return best_t,best_tri,best_u,best_v

@njit(cache=True)
#REFERENCE: https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure//bounding-volume-hierarchy-BVH-part1.html
def intersect_ray_box(o,d,bmin,bmax,t_min,t_max):
#tests if the ray goes through the box [xmin,xmax] x [ymin,ymax] x [zmin,zmax]
#the box is the space shared by 3 slabs which is the space between 2 parallel planes):
#x-slab:xmin<=x<=xmax,y-slab:ymin<=y<=ymax,z-slab:zmin<=z<= zmax
#for each slab we find when the ray enters (t1) and exits (t2) it.
#the ray is inside the box from the LATEST entry (t_enter) to the EARLIEST exit (t_exit).
#if t_enter > t_exit, the ray is never inside all 3 slabs at the same time: it misses.
#returns (True, t_enter) on a hit, (False, inf) on a miss

        #we only look between t_min and t_max so we start with that range and make it smaller
        t_enter=t_min
        t_exit=t_max
        
        for axis in range(3):
            #axis=x,y and z
            #d[axis] = 0: the ray is parallel to the axis-slab, its the axis never changes
            if d[axis] == 0.0:
                if o[axis] < bmin[axis] or o[axis] > bmax[axis]: #ray parallel to the box
                    #outside the slab forever so it can never reach the box
                    return False, np.inf
                
            else: 
                #the ray crosses the 2 planes for x: x = xmin and x = xmax: solve o[0] + t*d[0] = xmin (and = xmax)
                #same with y and z
                t1=(bmin[axis]- o[axis])/d[axis]
                t2=(bmax[axis]-o[axis])/d[axis]
            
                #if the ray moves towards -x,-y or -z it meets xmax first: swap so that t1 is the entry t2 the exit
                if t1 > t2: 
                    temp=t2
                    t2=t1
                    t1=temp
                
                #latest entry so far
                if t1>t_enter:
                    t_enter=t1

                #earliest exit so far
                if t2<t_exit:
                    t_exit=t2
                    
            
                #the entry is after the exit: no moment inside all slabs so far so it misses
                if t_enter > t_exit: 
                    return False,np.inf
                
                
        
        #inside all 3 slabs between t_enter and t_exit: hit. t_enter is where it enters the box
        return True,t_enter
        
        
        