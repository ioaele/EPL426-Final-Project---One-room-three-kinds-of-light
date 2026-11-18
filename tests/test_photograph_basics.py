import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from renderer import camera as C        
from renderer import lights as L        
from renderer import materials as Mt      
from renderer import shading as S        
from renderer.image import linear_to_srgb8, load_png, save_png 
from tools.compare_images import compare_images  


# --- image ----

def test_gamma_formula():
    v = np.array([[[0.0, 1.0, 0.5]]])
    out = linear_to_srgb8(v)[0, 0]
    assert out[0] == 0 and out[1] == 255
    assert out[2] == math.floor(255 * 0.5 ** (1 / 2.2) + 0.5)  # 186


def test_clipping():
    out = linear_to_srgb8(np.array([[[-1.0, 2.0, 1.0]]]))[0, 0]
    assert list(out) == [0, 255, 255]


def test_save_and_compare(tmp_path):
    img = np.random.default_rng(0).random((20, 30, 3))
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    save_png(img, a)
    save_png(img, b)
    assert load_png(a).shape == (20, 30, 3)
    n, frac, ok = compare_images(a, b)
    assert n == 0 and ok
    img2 = img.copy()
    img2[:5] = 1.0 - img2[:5]          
    save_png(img2, c)
    n, frac, ok = compare_images(a, c)
    assert not ok and frac > 0.1


# --- materials ---

def test_material_defaults_and_checker():
    mats, ids = Mt.load_materials([{}, {"ks": 0.5, "checker": {"size": 1, "kd2": [0, 0, 0]}}, {}])
    assert ids[0] == ids[2] != ids[1]          
    assert np.allclose(mats[0, Mt.KD:Mt.KD + 3], 0.7)
    assert mats[0, Mt.M] == 32 and mats[0, Mt.IOR] == 1.5 and mats[0, Mt.ALPHA] == 1
    assert np.allclose(mats[ids[1], Mt.KS:Mt.KS + 3], 0.5)
    m = ids[1]
    assert np.allclose(S.material_kd(mats, m, np.array([0.5, 0, 0.5])), 0.7)   
    assert np.allclose(S.material_kd(mats, m, np.array([1.5, 0, 0.5])), 0.0)  
    assert np.allclose(S.material_kd(mats, m, np.array([-0.5, 0, 0.5])), 0.0)  


# --- lights ---

def test_point_attenuation_and_spot_cone():
    lights = L.load_lights([
        {"type": "point", "position": [0, 2, 0], "intensity": 1, "attenuation": [1, 0, 1]},
        {"type": "spot", "position": [0, 2, 0], "direction": [0, -1, 0],
         "cutoff": 30, "exponent": 2},
    ])
    P = np.zeros(3)
    l, r, f = S.light_direction(lights, 0, P)
    assert np.allclose(l, [0, 1, 0]) and r == 2 and math.isclose(f, 1 / 5)
    _, _, f = S.light_direction(lights, 1, P)        
    assert math.isclose(f, 1.0)
    _, _, f = S.light_direction(lights, 1, np.array([2.0, 0, 0]))  
    assert f == 0.0


def test_area_light_centre_matches_formula():
    lights = L.load_lights([{"type": "area", "corner": [-0.5, 2, -0.5],
                             "edge1": [1, 0, 0], "edge2": [0, 0, 1], "intensity": 3}])
    assert np.allclose(lights[0, L.NORMAL:L.NORMAL + 3], [0, -1, 0])
    P, N, E = np.zeros(3), np.array([0.0, 1, 0]), np.array([0.0, 1, 0])
    y = S.area_sample_point(lights, 0, 0, 0, 1, False)
    assert np.allclose(y, [0, 2, 0])
    kd, ks = np.full(3, 0.5), np.zeros(3)
    c = S.area_sample_contribution(lights, 0, P, N, E, kd, ks, 32.0, y, 1)
    expected = 3 * 1 * 1 / (math.pi * 4) * 0.5
    assert np.allclose(c, expected)


# --- shading ---

def test_shade_hit_diffuse_ambient_and_shadow():
    mats, _ = Mt.load_materials([{"kd": 0.5, "emission": [0.1, 0, 0]}])
    lights = L.load_lights([{"type": "directional", "direction": [0, 1, 0], "intensity": 1}])
    P, N, E = np.zeros(3), np.array([0.0, 1, 0]), np.array([0.0, 1, 0])
    Ia = np.full(3, 0.2)
    c = S.shade_hit(P, N, E, mats, 0, lights, np.ones(1), Ia, True, True, True)
    assert np.allclose(c, [0.5 * 0.2 + 0.1 + 0.5, 0.1 + 0.5, 0.1 + 0.5])
    c = S.shade_hit(P, N, E, mats, 0, lights, np.zeros(1), Ia, False, True, False)
    assert np.allclose(c, 0.0)            


# --- camera ----

def test_centre_ray_points_forward_and_row0_is_top():
    cam = C.load_camera({"eye": [0, 0, 5], "lookat": [0, 0, 0], "up": [0, 1, 0],
                         "fov": 60}, 100, 100)
    o, d = C.generate_pixel_ray(cam, 50.0, 50.0, 100, 100)
    assert np.allclose(o, [0, 0, 5]) and np.allclose(d, [0, 0, -1])
    _, d_top = C.generate_pixel_ray(cam, 50.0, 0.5, 100, 100)
    assert d_top[1] > 0                   


CORNELL_CAMERA = {"projection": "perspective", "vrp": [0, 2.75, 6.6], "vpn": [0, 0, -1],
                  "vuv": [0, 1, 0], "d": 1.0, "window": [-0.37, -0.37, 0.37, 0.37],
                  "near": 0.1, "aperture": 0.0, "focus": 9.0, "shutter": [0.0, 0.0]}


def test_cornell_camera_from_assignment2_format():
    cam = C.load_camera(CORNELL_CAMERA, 320, 320)
    o, d = C.generate_pixel_ray(cam, 160.0, 160.0, 320, 320)
    assert np.allclose(o, [0, 2.75, 6.6]) and np.allclose(d, [0, 0, -1])
    _, d_tl = C.generate_pixel_ray(cam, 0.0, 0.0, 320, 320)  
    assert np.allclose(d_tl, np.array([-0.37, 0.37, -1.0]) / np.linalg.norm([-0.37, 0.37, -1.0]))
    assert cam[C.NEAR] == 0.1 and cam[C.FOCUS] == 9.0


def test_scene_settings_image_size_and_defaults():
    s = C.load_scene_settings({"image": {"width": 320, "height": 320}})
    assert s["image"] == [320, 320]
    s = C.load_scene_settings({})
    assert s["image"] == [320, 240] and s["max_depth"] == 4 and s["min_weight"] == 0.01


def test_depth_of_field_ray_hits_focus_point():
    cam = C.load_camera({"eye": [0, 0, 5], "lookat": [0, 0, 0], "fov": 60,
                         "aperture": 0.3, "focus": 5.0}, 64, 64)
    _, d = C.generate_pixel_ray(cam, 10.0, 20.0, 64, 64)
    F = cam[C.COP:C.COP + 3] + d * (5.0 / np.dot(d, cam[C.FWD:C.FWD + 3]))
    o2, d2 = C.depth_of_field_ray(cam, d, 0.7, 0.3)
    t = np.dot(F - o2, d2)
    assert np.allclose(o2 + t * d2, F)              
    assert np.linalg.norm(o2 - cam[C.COP:C.COP + 3]) <= 0.3 + 1e-12
