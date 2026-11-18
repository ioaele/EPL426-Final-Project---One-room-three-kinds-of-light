#tests for "It works when
#your BVH gives exactly the same nearest hit as a brute-force search that tests every triangle, for
#every ray of a large random test"
# source .venv/bin/activate
# pytest -v
import numpy as np
from renderer.bvh_median import build_bvh_median,LEAF_SIZE
from renderer.bvh_sah import build_bvh_sah
from renderer.bvh_walk import bvh_nearest_hit,bvh_any_hit
from renderer.intersections import brute_force_nearest_hit,brute_force_any_hit

#every test below runs once with the median tree and once with the sah tree
BUILDS=[build_bvh_median,build_bvh_sah]


def random_triangles(N):
    #N small random triangles inside the cube [-1, 1]
    rng=np.random.default_rng(0)  #fixed seed so the same triangles every run
    v0=rng.uniform(-1,1,(N,3))
    v1=v0+rng.uniform(-0.2,0.2,(N,3))
    v2=v0+rng.uniform(-0.2,0.2,(N,3))
    return v0,v1,v2


#TESTING the builds (median and sah)

def test_every_triangle_is_in_the_tree_once():
    v0,v1,v2=random_triangles(1000)
    for build in BUILDS:
        node_min,node_max,left,right,start,count,tri_order=build(v0,v1,v2)
        assert sorted(tri_order)==list(range(1000))


def test_leaves_are_small():
    #only for the median: sah may keep more triangles in a leaf when splitting does not help
    v0,v1,v2=random_triangles(1000)
    node_min,node_max,left,right,start,count,tri_order=build_bvh_median(v0,v1,v2)
    for row in range(len(left)):
        if left[row]==-1:  #a leaf
            assert count[row]<=LEAF_SIZE


def test_children_are_inside_their_parent():
    v0,v1,v2=random_triangles(1000)
    for build in BUILDS:
        node_min,node_max,left,right,start,count,tri_order=build(v0,v1,v2)
        for row in range(len(left)):
            if left[row]!=-1:  #not a leaf
                for child in (left[row],right[row]):
                    assert np.all(node_min[child]>=node_min[row])
                    assert np.all(node_max[child]<=node_max[row])


def test_one_triangle():
    #only one triangle meaning the tree is just one leaf (the root)
    v0,v1,v2=random_triangles(1)
    for build in BUILDS:
        node_min,node_max,left,right,start,count,tri_order=build(v0,v1,v2)
        assert len(left)==1
        assert left[0]==-1 and count[0]==1


def test_sah_triangles_at_the_same_spot():
    #10 identical triangles: all centres are the same so they cannot be split (no crash, one leaf)
    v0=np.zeros((10,3))
    v1=v0+np.array([1.0,0.0,0.0])
    v2=v0+np.array([0.0,1.0,0.0])
    node_min,node_max,left,right,start,count,tri_order=build_bvh_sah(v0,v1,v2)
    assert len(left)==1
    assert count[0]==10


#TESTING bvh_nearest_hit and bvh_any_hit against brute force
#for many random rays the BVH must give exactly the same answer as brute force
def test_nearest_hit_same_as_brute_force():
    v0,v1,v2=random_triangles(1000)
    for build in BUILDS:
        tree=build(v0,v1,v2)
        rng=np.random.default_rng(1)
        for i in range(1000):
            o=rng.uniform(-2,2,3)
            d=rng.normal(size=3)
            #*tree = the 7 tree arrays (node_min,node_max,left,right,start,count,tri_order) passed one by one
            assert bvh_nearest_hit(o,d,v0,v1,v2,1e-4,np.inf,*tree)==brute_force_nearest_hit(o,d,v0,v1,v2,1e-4,np.inf)


def test_any_hit_same_as_brute_force():
    v0,v1,v2=random_triangles(1000)
    for build in BUILDS:
        tree=build(v0,v1,v2)
        rng=np.random.default_rng(2)
        for i in range(1000):
            o=rng.uniform(-2,2,3)
            d=rng.normal(size=3)
            assert bvh_any_hit(o,d,v0,v1,v2,1e-4,np.inf,*tree)==brute_force_any_hit(o,d,v0,v1,v2,1e-4,np.inf)


def test_short_rays_same_as_brute_force():
    #like shadow rays: only look up to tmax = 0.5
    v0,v1,v2=random_triangles(1000)
    for build in BUILDS:
        tree=build(v0,v1,v2)
        rng=np.random.default_rng(3)
        for i in range(1000):
            o=rng.uniform(-2,2,3)
            d=rng.normal(size=3)
            assert bvh_nearest_hit(o,d,v0,v1,v2,1e-4,0.5,*tree)==brute_force_nearest_hit(o,d,v0,v1,v2,1e-4,0.5)
            assert bvh_any_hit(o,d,v0,v1,v2,1e-4,0.5,*tree)==brute_force_any_hit(o,d,v0,v1,v2,1e-4,0.5)


def test_rays_parallel_to_the_axes():
    #directions with zeros, like (1, 0, 0): tests the d == 0 case of the box test
    v0,v1,v2=random_triangles(1000)
    for build in BUILDS:
        tree=build(v0,v1,v2)
        rng=np.random.default_rng(4)
        directions=[(1,0,0),(0,-1,0),(0,0,1),(1,1,0),(0,1,-1)]
        for i in range(200):
            o=rng.uniform(-1,1,3)
            for d in directions:
                d=np.array(d,dtype=float)
                assert bvh_nearest_hit(o,d,v0,v1,v2,1e-4,np.inf,*tree)==brute_force_nearest_hit(o,d,v0,v1,v2,1e-4,np.inf)
