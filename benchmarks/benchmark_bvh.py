#the measurement of the report: how the cost of a ray grows from 2000 to 500000 triangles,
#with brute force, the median BVH and the SAH BVH
# source .venv/bin/activate
# python3 benchmarks/benchmark_bvh.py

#to get 2000 to 500000 triangles we use our terrain at levels 5 to 9 (level L has 2*4^L triangles)
#the rays go from above the terrain down to random points on the ground

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))  #so python finds the renderer and tools folders
from renderer.bvh_median import build_bvh_median
from renderer.bvh_sah import build_bvh_sah
from renderer.bvh_walk import bvh_nearest_hits_batch
from renderer.intersections import brute_force_nearest_hits_batch
from tools.generate_terrain import generate_terrain,make_points,make_triangles


#200000 rays from 6 units above the terrain to random points on the ground (x and z go from -8 to 8)
random=np.random.default_rng(0)  #fixed seed: the same rays every time
rays=200000
O=np.zeros((rays,3))  #where every ray starts
O[:,0]=random.uniform(-8,8,rays)
O[:,1]=6.0
O[:,2]=random.uniform(-8,8,rays)
ground=np.zeros((rays,3))  #the point on the ground every ray goes to
ground[:,0]=random.uniform(-8,8,rays)
ground[:,2]=random.uniform(-8,8,rays)
D=ground-O  #which way every ray goes

brute_rays=400  #brute force is too slow for all the rays, so it only gets the first 400

results=[]

for level in [5,6,7,8,9]:
    #the triangles of the terrain at this level
    heights=generate_terrain(level,seed=0)
    points=make_points(heights)
    triangles=make_triangles(heights.shape[0])
    v0=points[triangles[:,0]]  #the first corner of every triangle
    v1=points[triangles[:,1]]  #the second corner
    v2=points[triangles[:,2]]  #the third corner

    #building the two trees and timing it
    start_time=time.perf_counter()
    median_tree=build_bvh_median(v0,v1,v2)
    build_median=time.perf_counter()-start_time

    start_time=time.perf_counter()
    sah_tree=build_bvh_sah(v0,v1,v2)
    build_sah=time.perf_counter()-start_time

    #the first call of a numba function compiles it so we call each one once before timing it
    brute_force_nearest_hits_batch(O[:10],D[:10],v0,v1,v2,1e-4,np.inf)
    bvh_nearest_hits_batch(O[:10],D[:10],v0,v1,v2,1e-4,np.inf,*median_tree)

    #time of all the rays divided by the number of rays in microseconds (1 second = 1000000 microseconds)
    start_time=time.perf_counter()
    brute_force_nearest_hits_batch(O[:brute_rays],D[:brute_rays],v0,v1,v2,1e-4,np.inf)
    brute=(time.perf_counter()-start_time)/brute_rays*1000000

    start_time=time.perf_counter()
    bvh_nearest_hits_batch(O,D,v0,v1,v2,1e-4,np.inf,*median_tree)  #*median_tree = the 7 arrays of the tree
    median=(time.perf_counter()-start_time)/rays*1000000

    start_time=time.perf_counter()
    bvh_nearest_hits_batch(O,D,v0,v1,v2,1e-4,np.inf,*sah_tree)
    sah=(time.perf_counter()-start_time)/rays*1000000

    results.append([level,len(v0),brute,median,sah,build_median,build_sah])
    print("LEVEL",level)
    print(" triangles:",len(v0))
    print(" brute force:",round(brute,2),"microseconds per ray")
    print(" median:",round(median,3),"microseconds per ray")
    print(" sah:",round(sah,3),"microseconds per ray")
    print(" median build:",round(build_median,2),"seconds")
    print(" sah build:",round(build_sah,2),"seconds")
    print()

#saving the table for the report (it opens in excel)
Path("output").mkdir(exist_ok=True)
np.savetxt("output/benchmark_bvh.csv",np.array(results),delimiter=",",fmt="%.6g",header="level,triangles,brute_force_us,median_us,sah_us,build_median_s,build_sah_s",comments="")
print("saved in output/benchmark_bvh.csv")
