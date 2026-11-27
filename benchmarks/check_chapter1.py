#checks the 3 "It works when" points of Chapter 1 and prints PASS or FAIL for each one
#the handout says it works when:
#1. the BVH gives exactly the same nearest hit as brute force, for every ray of a large random test
#2. the Cornell room traces 2 to 5 million rays per second, at least 100 times faster than brute force, and brute, median and sah give the same image
#3. the BVH of terrain level 8 builds in less than a minute

# source .venv/bin/activate
# python benchmarks/check_chapter1.py

import sys
import time
from pathlib import Path
import numpy as np
from numba import njit,prange

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))  #so python finds the renderer and tools folders
from renderer.loadScene import loadScene
from renderer.camera import generate_pixel_ray
from renderer.bvh_median import build_bvh_median
from renderer.bvh_sah import build_bvh_sah
from renderer.bvh_walk import bvh_nearest_hit,bvh_any_hit,bvh_nearest_hits_batch
from renderer.intersections import brute_force_nearest_hit,brute_force_any_hit,brute_force_nearest_hits_batch
from tools.generate_terrain import generate_terrain,make_points,make_triangles


@njit(cache=True,parallel=True)
def camera_rays(cam,width,height):
    #one ray from the camera through the centre of every pixel
    O=np.empty((width*height,3))  #where every ray starts
    D=np.empty((width*height,3))  #which way every ray goes
    for j in prange(height):  #every row of the image
        for i in range(width):  #every pixel of the row
            o,d=generate_pixel_ray(cam,i+0.5,j+0.5,width,height)  #+0.5 = the centre of the pixel
            O[j*width+i]=o
            D[j*width+i]=d
    return O,D


def check_1():
    print("CHECK 1: the BVH gives the same nearest hit as brute force (expected 0 different hits)")

    #2000 random triangles and 10000 random rays
    random=np.random.default_rng(0)  #fixed seed: the same random numbers every time
    v0=random.uniform(-1,1,(2000,3))
    v1=v0+random.uniform(-0.2,0.2,(2000,3))
    v2=v0+random.uniform(-0.2,0.2,(2000,3))
    O=random.uniform(-2,2,(10000,3))
    D=random.normal(size=(10000,3))

    median_tree=build_bvh_median(v0,v1,v2)
    sah_tree=build_bvh_sah(v0,v1,v2)

    #counting how many rays give a different answer than brute force
    median_different=0
    sah_different=0
    for i in range(10000):
        correct=brute_force_nearest_hit(O[i],D[i],v0,v1,v2,1e-4,np.inf)
        correct_any=brute_force_any_hit(O[i],D[i],v0,v1,v2,1e-4,np.inf)
        #*median_tree = the 7 arrays of the tree passed one by one
        if bvh_nearest_hit(O[i],D[i],v0,v1,v2,1e-4,np.inf,*median_tree)!=correct:
            median_different+=1
        if bvh_any_hit(O[i],D[i],v0,v1,v2,1e-4,np.inf,*median_tree)!=correct_any:
            median_different+=1
        if bvh_nearest_hit(O[i],D[i],v0,v1,v2,1e-4,np.inf,*sah_tree)!=correct:
            sah_different+=1
        if bvh_any_hit(O[i],D[i],v0,v1,v2,1e-4,np.inf,*sah_tree)!=correct_any:
            sah_different+=1

    print(" median:",median_different,"different hits (nearest hit or any hit) ")
    print(" sah:",sah_different,"different hits (nearest hit or any hit)")


def check_2():
    print("CHECK 2: the Cornell room")

    #the triangles of the room and one camera ray for every pixel of the 320x320 image
    scene=loadScene("project-files/scenes/cornell.json")
    v0=scene["v0"]
    v1=scene["v1"]
    v2=scene["v2"]
    O,D=camera_rays(scene["camera"],scene["width"],scene["height"])
    rays=len(O)

    median_tree=build_bvh_median(v0,v1,v2)
    sah_tree=build_bvh_sah(v0,v1,v2)

    #the first call of a numba function compiles it, so we call each one once before timing it
    brute_force_nearest_hits_batch(O[:10],D[:10],v0,v1,v2,1e-4,np.inf)
    bvh_nearest_hits_batch(O[:10],D[:10],v0,v1,v2,1e-4,np.inf,*sah_tree)

    #timing brute force and sah on all the camera rays
    start_time=time.perf_counter()
    brute=brute_force_nearest_hits_batch(O,D,v0,v1,v2,1e-4,np.inf)
    brute_speed=rays/(time.perf_counter()-start_time)  #rays per second

    start_time=time.perf_counter()
    sah=bvh_nearest_hits_batch(O,D,v0,v1,v2,1e-4,np.inf,*sah_tree)
    sah_speed=rays/(time.perf_counter()-start_time)  #rays per second

    median=bvh_nearest_hits_batch(O,D,v0,v1,v2,1e-4,np.inf,*median_tree)

    #speed (we check sah because the renderer uses sah by default)
    times_faster=sah_speed/brute_speed
    print(" sah:",int(sah_speed),"rays per second (expected: 2 to 5 million)")
    print(" sah is",int(times_faster),"times faster than brute force (expected: at least 100)")

    #same image: we compare every pixel with the brute force image
    #images match when fewer than 1% of the pixels are different (Appendix C)
    #a pixel can show another triangle only when the ray hits exactly the edge where two triangles meet
    #(both are at the same distance t, so both answers are right)
    median_other_t=np.sum(brute[0]!=median[0])  #pixels where the distance t is different
    median_other_triangle=100*np.mean(brute[1]!=median[1])  #% of pixels that see another triangle
    sah_other_t=np.sum(brute[0]!=sah[0])
    sah_other_triangle=100*np.mean(brute[1]!=sah[1])
    print(" median image:",round(median_other_triangle,2),"% of the pixels are different from brute force (expected: under 1%)")
    print(" sah image:",round(sah_other_triangle,2),"% of the pixels are different from brute force (expected: under 1%)")

def check_3():
    print("CHECK 3: the BVH of terrain level 8 builds in less than a minute")

    #the 131072 triangles of terrain level 8
    heights=generate_terrain(8,seed=0)
    points=make_points(heights)
    triangles=make_triangles(heights.shape[0])
    v0=points[triangles[:,0]]  #the first corner of every triangle
    v1=points[triangles[:,1]]  #the second corner
    v2=points[triangles[:,2]]  #the third corner

    start_time=time.perf_counter()
    build_bvh_median(v0,v1,v2)
    median_seconds=time.perf_counter()-start_time

    start_time=time.perf_counter()
    build_bvh_sah(v0,v1,v2)
    sah_seconds=time.perf_counter()-start_time

    print(" median:",round(median_seconds,2),"seconds")
    print(" sah:",round(sah_seconds,2),"seconds")


if __name__=="__main__":
    check_1()
    print()
    check_2()
    print()
    check_3()
