import time
from tools.generate_terrain import generate_terrain,make_points,make_triangles
from renderer.bvh_median import build_bvh_median
from renderer.bvh_sah import build_bvh_sah


def test_terrain_level_8_builds_in_less_than_a_minute():
    #"the BVH of terrain level 8 builds in less than a minute"
    heights=generate_terrain(8,seed=0)
    points=make_points(heights)
    triangles=make_triangles(heights.shape[0])

    #the 3 corners of every triangle, like the scene loader gives them
    v0=points[triangles[:,0]]
    v1=points[triangles[:,1]]
    v2=points[triangles[:,2]]
    assert len(v0)==131072  #2*4^8 triangles

    for build in [build_bvh_median,build_bvh_sah]:
        start_time=time.perf_counter()  #a stopwatch: the time now
        build(v0,v1,v2)
        seconds=time.perf_counter()-start_time  #how long the build took
        assert seconds<60
