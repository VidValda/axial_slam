import numpy as np
import pytest

from axial_perception.core.segments import METHODS, gap_split, split_clusters

SPLITTERS = [m for m in METHODS if m != "none"]


def polyline(verts, step=0.05, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pts, truth = [], []
    for k, (a, b) in enumerate(zip(verts[:-1], verts[1:])):
        a, b = np.array(a, float), np.array(b, float)
        t = np.arange(0, 1, step / np.linalg.norm(b - a))[:, None]
        pts.append(a + t * (b - a))
        truth.append(np.full(len(t), k))
    pts = np.concatenate(pts)
    return pts + rng.normal(0, noise, pts.shape), np.concatenate(truth)


def purity(labels, truth):
    hit = labels > 0
    return sum(np.bincount(truth[labels == k]).max() for k in np.unique(labels[hit])) / max(hit.sum(), 1)


def split(method, pts, tol=0.05, min_points=5, gap=0.5):
    return METHODS[method](pts, tol, min_points, gap, np.random.default_rng(0))


@pytest.mark.parametrize("method", METHODS)
def test_straight_is_one_line(method):
    pts, _ = polyline([(0, 0, 0), (3, 0, 0)])
    assert set(split(method, pts)) == {1}


@pytest.mark.parametrize("method", METHODS)
def test_thick_half_cylinder_is_one_line(method):
    z, a = np.meshgrid(np.arange(0, 3, 0.05), np.linspace(0, np.pi, 8))
    pts = np.column_stack([0.2 * np.cos(a.ravel()), 0.2 * np.sin(a.ravel()), z.ravel()])
    assert set(split(method, pts, tol=0.3)) == {1}


@pytest.mark.parametrize("method", SPLITTERS)
def test_l_splits_at_corner(method):
    pts, truth = polyline([(0, 0, 0), (2, 0, 0), (2, 0, 1.5)])
    labels = split(method, pts)
    assert labels.max() == 2
    assert purity(labels, truth) > 0.9


@pytest.mark.parametrize("method", SPLITTERS)
def test_u_dip_splits_collinear_pieces(method):
    pts, truth = polyline([(0, 0, 0), (2, 0, 0), (2, 0, -0.35), (4.5, 0, -0.35), (4.5, 0, 0), (6.5, 0, 0)])
    labels = split(method, pts)
    assert labels.max() >= 3
    assert purity(labels, truth) > 0.9


@pytest.mark.parametrize("method", METHODS)
def test_small_cluster_is_unassigned(method):
    pts, _ = polyline([(0, 0, 0), (0.15, 0, 0)])
    assert not split(method, pts).any()


@pytest.mark.parametrize("method", METHODS)
def test_empty(method):
    assert len(split(method, np.empty((0, 3)))) == 0


def test_gap_split_cuts_collinear_pieces():
    pts = np.array([[x, 0, 0] for x in (0, 0.1, 0.2, 2, 2.1, 2.2)], float)
    np.testing.assert_array_equal(gap_split(pts, np.ones(6, int), 0.5, 3), [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(gap_split(pts, np.ones(6, int), 0, 3), np.ones(6))


def test_split_clusters_numbers_lines_per_frame():
    a, _ = polyline([(0, 0, 0), (2, 0, 0), (2, 0, 1.5)])
    b, _ = polyline([(0, 1, 0), (0, 1, 2)])
    pts = np.concatenate([a, b])
    seg = np.r_[np.ones(len(a), int), np.full(len(b), 2)]
    cls = np.r_[np.full(len(a), 4), np.full(len(b), 8)]
    line = split_clusters(pts, cls, seg, "ransac", {4: 0.05, 8: 0.1}, 5, 0.5)
    assert set(line[seg == 1]) == {1, 2}
    assert set(line[seg == 2]) == {3}
