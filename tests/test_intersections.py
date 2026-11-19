#tests for renderer/intersections.py
# source .venv/bin/activate
# pytest -v
import numpy as np
from renderer.intersections import intersect_ray_triangle,brute_force_nearest_hit,brute_force_any_hit,intersect_ray_box

#a test triangle
A=np.array([0.0,0.0,0.0])
B=np.array([1.0,0.0,0.0])
C=np.array([0.0,1.0,0.0])

DOWN=np.array([0.0,0.0,-1.0])  #a ray looking vertical down


def two_triangles():
    #two parallel triangles
    #both cover the point (0, 0) so a vertical ray through (0, 0) passes through both
    v0=np.array([[-1.0,-1.0,0.0],[-1.0,-1.0,-1.0]])
    v1=np.array([[3.0,-1.0,0.0],[3.0,-1.0,-1.0]])
    v2=np.array([[-1.0,3.0,0.0],[-1.0,3.0,-1.0]])
    return v0,v1,v2

#TESTING intersect_ray_triangle
def test_ray_hits_triangle():
    t,u,v=intersect_ray_triangle(np.array([0.25,0.5,2.0]),DOWN,A,B,C,0.0,np.inf)
    assert np.isclose(t,2.0)
    assert np.isclose(u,0.25) and np.isclose(v,0.5)


def test_barycentrics_give_the_hit_point():
    #the point from the ray (o + t d) must be the same as the point from u, v
    o=np.array([0.3,0.1,5.0])
    d=np.array([0.02,0.05,-1.0])
    t,u,v=intersect_ray_triangle(o,d,A,B,C,0.0,np.inf)
    assert t < np.inf
    assert np.allclose(o + t * d,(1 - u - v) * A + u * B + v * C)


def test_ray_from_below_also_hits():
    #the back side of the triangle counts too
    t,u,v=intersect_ray_triangle(np.array([0.2,0.2,-1.0]),-DOWN,A,B,C,0.0,np.inf)
    assert np.isclose(t,1.0)


def test_ray_misses_outside_triangle():
    #(0.8, 0.8) is on the plane but outside the triangle (u + v > 1)
    t,u,v=intersect_ray_triangle(np.array([0.8,0.8,2.0]),DOWN,A,B,C,0.0,np.inf)
    assert t==np.inf
    #(-0.1, 0.5) is outside on the other side
    t,u,v=intersect_ray_triangle(np.array([-0.1,0.5,2.0]),DOWN,A,B,C,0.0,np.inf)
    assert t==np.inf


def test_parallel_ray_misses():
    #moving sideways, above the triangle: never meets the plane
    t,u,v=intersect_ray_triangle(np.array([0.2,0.2,1.0]),np.array([1.0,0.0,0.0]),A,B,C,0.0,np.inf)
    assert t==np.inf


def test_triangle_behind_the_ray():
    #ray starts above the triangle but goes UP: the triangle is behind it
    t,u,v=intersect_ray_triangle(np.array([0.2,0.2,2.0]),-DOWN,A,B,C,0.0,np.inf)
    assert t==np.inf


def test_hit_beyond_t_max_is_ignored():
    #the hit is at t = 2, but we only look up to t = 1.5
    t,u,v=intersect_ray_triangle(np.array([0.2,0.2,2.0]),DOWN,A,B,C,0.0,1.5)
    assert t==np.inf


def test_hit_before_t_min_is_ignored():
    #the hit is at t = 2, but we only look from t = 2.5 on
    t,u,v=intersect_ray_triangle(np.array([0.2,0.2,2.0]),DOWN,A,B,C,2.5,np.inf)
    assert t==np.inf


def test_direction_does_not_need_length_1():
    #d is 4 times longer, so we reach the triangle at t = 0.5 instead of 2
    t,u,v=intersect_ray_triangle(np.array([0.2,0.2,2.0]),4 * DOWN,A,B,C,0.0,np.inf)
    assert np.isclose(t,0.5)


#TESTING brute_force_nearest_hit

def test_nearest_hit_from_above_is_triangle_0():
    v0,v1,v2=two_triangles()
    t,tri,u,v=brute_force_nearest_hit(np.array([0.0,0.0,5.0]),DOWN,v0,v1,v2,1e-4,np.inf)
    assert tri==0
    assert np.isclose(t,5.0)


def test_nearest_hit_from_below_is_triangle_1():
    v0,v1,v2=two_triangles()
    t,tri,u,v=brute_force_nearest_hit(np.array([0.0,0.0,-5.0]),-DOWN,v0,v1,v2,1e-4,np.inf)
    assert tri==1
    assert np.isclose(t,4.0)


def test_nearest_hit_miss():
    #ray goes up, away from both triangles
    v0,v1,v2=two_triangles()
    t,tri,u,v=brute_force_nearest_hit(np.array([0.0,0.0,5.0]),-DOWN,v0,v1,v2,1e-4,np.inf)
    assert tri==-1
    assert t==np.inf


#TESTING brute_force_any_hit

def test_any_hit_true():
    v0,v1,v2=two_triangles()
    assert brute_force_any_hit(np.array([0.0,0.0,5.0]),DOWN,v0,v1,v2,1e-4,np.inf)


def test_any_hit_stops_at_t_max():
    #like a shadow ray to a light at distance 4: both triangles are farther, so nothing blocks it
    v0,v1,v2=two_triangles()
    assert not brute_force_any_hit(np.array([0.0,0.0,5.0]),DOWN,v0,v1,v2,1e-4,4.0)


def test_any_hit_false_when_pointing_away():
    v0,v1,v2=two_triangles()
    assert not brute_force_any_hit(np.array([0.0,0.0,5.0]),-DOWN,v0,v1,v2,1e-4,np.inf)


#TESTING intersect_ray_box
#the test box: x from 1 to 3, y from 0 to 1, z from 0 to 1

def box(o,d,t_min=0.0,t_max=np.inf):
    #shortcut: test the ray against the test box
    #bmin = the smallest x, y, z of the box, bmax = the largest
    return intersect_ray_box(np.array(o),np.array(d),np.array([1.0,0.0,0.0]),np.array([3.0,1.0,1.0]),t_min,t_max)


def test_box_hit():
    #from x = 0 moving +x with speed 2: reaches x = 1 at t = 0.5
    #(d[1] = d[2] = 0, so this also tests the parallel case inside the y and z slabs)
    hit,t=box([0.0,0.5,0.5],[2.0,0.0,0.0])
    assert hit
    assert np.isclose(t,0.5)


def test_box_pointing_away():
    #the box is in front (+x) but the ray goes -x
    hit,t=box([0.0,0.5,0.5],[-1.0,0.0,0.0])
    assert not hit
    assert t==np.inf


def test_box_parallel_outside():
    #moving +x at height y = 5: parallel to the y-slab and outside it, never reaches the box
    hit,t=box([0.0,5.0,0.5],[1.0,0.0,0.0])
    assert not hit


def test_box_diagonal_miss():
    #passes the x-slab and the y-slab at different times: never inside both at once
    hit,t=box([0.0,5.0,0.5],[1.0,-1.0,0.0])
    assert not hit


def test_box_diagonal_hit():
    #enters the y-slab at t = 0.5 and the x-slab at t = 1: inside the box from t = 1
    hit,t=box([0.0,-0.5,0.5],[1.0,1.0,0.0])
    assert hit
    assert np.isclose(t,1.0)


def test_box_ray_starts_inside():
    #starts inside the box: it is in the box from the start, so t_enter = t_min
    hit,t=box([2.0,0.5,0.5],[0.3,0.2,1.0])
    assert hit
    assert t==0.0


def test_box_beyond_t_max():
    #the box starts at t = 1 but we only look up to t = 0.5
    hit,t=box([0.0,0.5,0.5],[1.0,0.0,0.0],t_max=0.5)
    assert not hit
