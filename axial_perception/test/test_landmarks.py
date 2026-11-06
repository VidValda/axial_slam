import numpy as np
import pytest

from axial_perception.core.landmarks import LineLandmarks

UP = np.array([0.0, 0.0, 1.0])


def pillar(x, y, z0=0.0, z1=3.0, step=0.02, radius=0.0):
    z = np.arange(z0, z1, step)
    a = np.arange(-1.2, 1.2, step / max(radius, step))
    pts = np.array([(x + radius * np.cos(t), y + radius * np.sin(t), h) for h in z for t in a])
    return pts if radius > 0 else np.column_stack([np.full_like(z, x), np.full_like(z, y), z])


def test_shift_along_axis_keeps_id():
    lm = LineLandmarks(0.3)
    a = lm.assign(8, (1.0, 2.0, 0.5), UP)
    b = lm.assign(8, (1.05, 2.0, 2.5), -UP)
    assert a == b
    assert len(lm.landmarks) == 1


def test_far_line_spawns():
    lm = LineLandmarks(0.3)
    assert lm.assign(8, (0.0, 0.0, 1.0), UP) != lm.assign(8, (1.0, 0.0, 1.0), UP)


def test_other_class_spawns():
    lm = LineLandmarks(0.3)
    assert lm.assign(8, (0.0, 0.0, 1.0), UP) != lm.assign(4, (0.0, 0.0, 1.0), UP)


def test_angle_gate_spawns():
    lm = LineLandmarks(0.3, max_angle=0.2)
    assert lm.assign(8, (0.0, 0.0, 1.0), UP) != lm.assign(8, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def test_force_match_band_does_not_touch_landmark():
    lm = LineLandmarks(0.3, min_anchor_separation=1.0)
    a = lm.assign(8, (0.0, 0.0, 1.5), UP, pillar(0.0, 0.0))
    n = len(lm.landmarks[a].points)
    assert lm.assign(8, (0.6, 0.0, 1.5), UP, pillar(0.6, 0.0)) == a
    np.testing.assert_allclose(lm.landmarks[a].point[:2], (0.0, 0.0), atol=1e-9)
    assert len(lm.landmarks[a].points) == n


def test_match_refits_line_to_all_points():
    lm = LineLandmarks(0.3, voxel=0.0)
    a = lm.assign(8, (0.0, 0.0, 1.5), (0.1, 0.0, 1.0), pillar(0.0, 0.0, 0.0, 1.0))
    lm.assign(8, (0.0, 0.0, 2.0), UP, pillar(0.0, 0.0, 1.0, 3.0))
    np.testing.assert_allclose(lm.landmarks[a].direction, UP, atol=1e-9)


def test_separation_below_threshold_raises():
    with pytest.raises(ValueError):
        LineLandmarks(0.5, min_anchor_separation=0.2)


def test_voxel_keeps_cloud_bounded():
    lm = LineLandmarks(0.3, voxel=0.05)
    pts = pillar(0.0, 0.0)
    for _ in range(10):
        gid = lm.assign(8, pts.mean(axis=0), UP, pts)
    assert len(lm.landmarks[gid].points) == len(lm.voxelize(pts))


def test_set_anchor_reserves_id():
    lm = LineLandmarks(0.3)
    lm.set_anchor(7, (5.0, 0.0, 0.0), UP, class_id=8)
    assert lm.assign(8, (5.1, 0.0, 1.0), UP) == 7
    assert lm.assign(8, (9.0, 0.0, 1.0), UP) == 8


def frame(*clouds):
    xyz = np.concatenate(clouds)
    line = np.concatenate([np.full(len(c), k + 1) for k, c in enumerate(clouds)])
    return xyz, np.full(len(xyz), 8), line


def test_update_maps_lines_of_a_frame():
    lm = LineLandmarks(0.3)
    xyz, cls, line = frame(pillar(0.0, 0.0), pillar(2.0, 0.0))
    first = lm.update(xyz, cls, line)
    assert lm.update(xyz, cls, line) == first
    assert sorted(first.values()) == [1, 2]


def test_update_drops_lines_without_a_direction():
    lm = LineLandmarks(0.3, min_elongation=3.0)
    stub = pillar(0.0, 0.0, 0.0, 0.3, radius=0.2)
    good = pillar(2.0, 0.0, radius=0.15)
    assert lm.update(*frame(stub, good)) == {2: 1}


def test_dense_points_drop_strays():
    lm = LineLandmarks(0.3, voxel=0.03)
    pts = pillar(0.0, 0.0, radius=0.15, step=0.01)
    strays = np.array([[0.4, 0.0, 1.0], [0.0, 0.5, 2.0]])
    gid = lm.assign(8, pts.mean(axis=0), UP, np.concatenate([pts, strays]))
    kept = lm.dense_points(gid, 4)
    assert len(kept) >= len(lm.landmarks[gid].points) - len(strays) - 20
    assert min(np.linalg.norm(kept - s, axis=1).min() for s in strays) > 0.1
