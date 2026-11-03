import numpy as np
import pytest

from axial_perception.core.camera import project, quat_to_matrix, transform

K = (400.0, 400.0, 320.0, 240.0)
W, H = 640, 480


def test_quat_identity():
    np.testing.assert_allclose(quat_to_matrix((0, 0, 0, 1)), np.eye(3))


def test_quat_yaw_90_maps_x_to_y():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(quat_to_matrix((0, 0, s, s)) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_quat_random_is_rotation():
    q = np.random.default_rng(0).normal(size=4)
    R = quat_to_matrix(q / np.linalg.norm(q))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_transform():
    s = np.sqrt(0.5)
    out = transform(np.array([[1.0, 0, 0], [0, 0, 1]]), quat_to_matrix((0, 0, s, s)), [1, 2, 3])
    np.testing.assert_allclose(out, [[1, 3, 3], [1, 2, 4]], atol=1e-12)


def test_project_known_pixels():
    pts = np.array([[0.0, 0.0, 2.0], [1.0, 0.5, 2.0]])
    u, v, idx = project(pts, K, W, H, 0.1, 30.0)
    np.testing.assert_array_equal(u, [320, 520])
    np.testing.assert_array_equal(v, [240, 340])
    np.testing.assert_array_equal(idx, [0, 1])


def test_project_drops_behind_out_of_fov_and_out_of_range():
    pts = np.array([
        [0.0, 0.0, -1.0],
        [5.0, 0.0, 1.0],
        [0.0, 0.0, 0.05],
        [0.0, 0.0, 40.0],
        [0.0, 0.0, 5.0],
    ])
    u, v, idx = project(pts, K, W, H, 0.1, 30.0)
    np.testing.assert_array_equal(idx, [4])


def test_project_pixel_edges():
    def at(col):
        return [(col - K[2]) / K[0], 0.0, 1.0]

    pts = np.array([at(-0.4), at(-0.6), at(W - 0.6), at(W - 0.4)])
    u, _, idx = project(pts, K, W, H, 0.1, 30.0)
    np.testing.assert_array_equal(idx, [0, 2])
    np.testing.assert_array_equal(u, [0, W - 1])
