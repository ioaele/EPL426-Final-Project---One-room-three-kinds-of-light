import numpy as np
from numba import njit

EPS = 1e-12 #values smaller than this are treated as 0 because floating-point rounding 
#means a result that should be exactly 0 usually comes out as a number very close to it 

#helper functions
@njit(cache=True)
def dot3(a, b):
    #dot product of two 3D vectors
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def cross3(a, b):
    #cross product of two 3D vectors
    return np.array([a[1] * b[2] - a[2] * b[1],a[2] * b[0] - a[0] * b[2],a[0] * b[1] - a[1] * b[0]])


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
    n = cross3(b - a, c - a)

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
    P=o+t*d

    #STEP2 checking if the intersecting point is inside the triangle


    #For each edge: cross(edge,P - edge start) points the same way as the triangle's normal n 
    #only if P is on the inner side of that edge.
    #dot(...,n) >= 0 -> inside of this edge and dot(...,n)< 0 -> P is outside the triangle.

    #edge a -> b
    e_ab = dot3(cross3(b - a, P - a), n) 
    if e_ab < 0.0:
        return np.inf, 0.0, 0.0      

    #edge b -> c
    e_bc = dot3(cross3(c - b, P - b), n)
    if e_bc < 0.0:
        return np.inf, 0.0, 0.0     

    #edge c -> a
    e_ca = dot3(cross3(a - c, P - c), n)
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

@njit(cache=True)
def intersect_ray_box(o,d,xmin,xmax,ymin,ymax,zmin,zmax,t_min,t_max):
#or def intersect_ray_box(o,d,bmin,bmax,t_min,t_max): we'll see later
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
        
        #X SLAB
        #d[0] = 0: the ray is parallel to the x-slab, its x never changes
        if d[0] == 0.0:
            if o[0] < xmin or o[0] > xmax: #ray parallel to the box
                #outside the slab forever so it can never reach the box
                return False, np.inf


        else: 
            #the ray crosses the 2 planes x = xmin and x = xmax: solve o[0] + t*d[0] = xmin (and = xmax)
            t1=(xmin- o[0])/d[0]
            t2=(xmax-o[0])/d[0]
        
            #if the ray moves towards -x it meets xmax first: swap so that t1 is the entry t2 the exit
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
            
            
        #Y SLAB
        if d[1] == 0.0:
            if o[1] < ymin or o[1] > ymax: #ray parallel to the box
                return False, np.inf
            
        else:
            t1=(ymin- o[1])/d[1]
            t2=(ymax-o[1])/d[1]
        
            if t1 > t2: 
                temp=t2
                t2=t1
                t1=temp
            
            if t1>t_enter:
                t_enter=t1

            if t2<t_exit:
                t_exit=t2
                
        
            if t_enter > t_exit: 
                return False,np.inf
            
            
        #---------- z-slab (same steps as x) ----------
        if d[2] == 0.0:
                if o[2] < zmin or o[2] > zmax: #ray parallel to the box
                    return False, np.inf
                
        else:
            t1=(zmin- o[2])/d[2]
            t2=(zmax-o[2])/d[2]
        
            if t1 > t2: 
                temp=t2
                t2=t1
                t1=temp
            
            if t1>t_enter:
                t_enter=t1

            if t2<t_exit:
                t_exit=t2
                
        
            if t_enter > t_exit: 
                return False,np.inf
        
        #inside all 3 slabs between t_enter and t_exit: hit. t_enter is where it enters the box
        return True,t_enter
        
        
        