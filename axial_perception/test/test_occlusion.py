import numpy as np

from axial_perception.core.occlusion import front_mask, surface_mask

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


def _mask_frame(pts, mask_id):
    """Pinhole projection (f = 100, centre 50, 50) into a 100 x 100 mask image."""
    u = (100 * pts[:, 0] / pts[:, 2] + 50).astype(int)
    v = (100 * pts[:, 1] / pts[:, 2] + 50).astype(int)
    img = np.zeros((100, 100), int)
    img[v, u] = mask_id
    return u, v, img


def test_surface_mask_drops_wall_behind_pipe():
    x = np.linspace(-0.5, 0.5, 60)
    pipe = np.c_[x, np.zeros(60), np.full(60, 2.0)]
    wall = np.c_[x[::3] * 1.2, np.full(20, 0.01), np.full(20, 2.6)]
    pts = np.vstack([pipe, wall])
    u, v, img = _mask_frame(pts, 1)
    keep = surface_mask(u, v, pts, img, 0.1, 15, 0.3)
    assert keep[:60].all() and not keep[60:].any()


def test_surface_mask_keeps_both_arms_of_a_receding_elbow():
    t = np.linspace(0, 1, 50)
    across = np.c_[t - 1, np.zeros(50), np.full(50, 2.0)]
    away = np.c_[np.zeros(50), np.zeros(50), 2.0 + 2 * t]
    pts = np.vstack([across, away])
    u, v, img = _mask_frame(pts, 1)
    assert surface_mask(u, v, pts, img, 0.1, 15, 0.3).all()


def test_surface_mask_keeps_a_separate_piece_at_the_same_depth():
    x = np.linspace(-0.4, -0.1, 30)
    pts = np.vstack([np.c_[x, np.zeros(30), np.full(30, 2.0)], np.c_[-x[:20], np.zeros(20), np.full(20, 2.05)]])
    u, v, img = _mask_frame(pts, 1)
    assert surface_mask(u, v, pts, img, 0.1, 15, 0.3).all()


def test_surface_mask_empty():
    assert surface_mask(np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), np.zeros((4, 4), int),
                        0.1, 15, 0.3).shape == (0,)
