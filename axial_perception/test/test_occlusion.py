import numpy as np

from axial_perception.core.occlusion import front_mask

W, H = 40, 20


def test_wall_behind_pipe_is_rejected_next_to_it_is_kept():
    pipe_u = np.arange(18, 22)
    wall_u = np.arange(0, W)
    u = np.concatenate([pipe_u, wall_u])
    v = np.full(len(u), 10)
    z = np.concatenate([np.full(len(pipe_u), 2.0), np.full(len(wall_u), 2.3)])
    keep = front_mask(u, v, z, W, H, window=5, tol=0.1)
    assert keep[:len(pipe_u)].all()
    wall_keep = keep[len(pipe_u):]
    assert not wall_keep[16:24].any()
    assert wall_keep[:16].all() and wall_keep[24:].all()


def test_within_tolerance_is_kept():
    keep = front_mask(np.array([5, 6]), np.array([5, 5]), np.array([2.0, 2.08]), W, H, window=3, tol=0.1)
    assert keep.all()


def test_same_pixel_keeps_only_front():
    keep = front_mask(np.array([5, 5]), np.array([5, 5]), np.array([3.0, 1.0]), W, H, window=1, tol=0.1)
    np.testing.assert_array_equal(keep, [False, True])


def test_image_border():
    keep = front_mask(np.array([0, W - 1]), np.array([0, H - 1]), np.array([1.0, 1.0]), W, H, window=5, tol=0.1)
    assert keep.all()
