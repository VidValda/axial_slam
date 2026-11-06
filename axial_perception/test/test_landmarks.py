import numpy as np
import pytest

from axial_perception.core.landmarks import LineLandmarks

UP = np.array([0.0, 0.0, 1.0])


def pillar(x, y, z0=0.0, z1=3.0, step=0.02):
    z = np.arange(z0, z1, step)
    return np.column_stack([np.full_like(z, x), np.full_like(z, y), z])


def test_shift_along_axis_keeps_id():
    lm = LineLandmarks(0.3)
    a = lm.assign(8, (1.0, 2.0, 0.5), UP)
    b = lm.assign(8, (1.05, 2.0, 2.5), -UP)
    assert a == b
    assert len(lm.landmarks) == 1


def test_far_line_spawns():
    lm = LineLandmarks(0.3)
    a = lm.assign(8, (0.0, 0.0, 1.0), UP)
    b = lm.assign(8, (1.0, 0.0, 1.0), UP)
    assert a != b


def test_other_class_spawns():
    lm = LineLandmarks(0.3)
    assert lm.assign(8, (0.0, 0.0, 1.0), UP) != lm.assign(4, (0.0, 0.0, 1.0), UP)


def test_angle_gate_spawns():
    lm = LineLandmarks(0.3, max_angle=0.2)
    a = lm.assign(8, (0.0, 0.0, 1.0), UP)
    b = lm.assign(8, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    assert a != b


def test_force_match_band_does_not_move_anchor():
    lm = LineLandmarks(0.3, min_anchor_separation=1.0, alpha=0.5)
    a = lm.assign(8, (0.0, 0.0, 1.0), UP)
    b = lm.assign(8, (0.6, 0.0, 1.0), UP)
    assert a == b
    np.testing.assert_allclose(lm.landmarks[a].point, (0.0, 0.0, 1.0))


def test_ema_moves_only_perpendicular():
    lm = LineLandmarks(0.3, alpha=0.5)
    a = lm.assign(8, (0.0, 0.0, 1.0), UP)
    lm.assign(8, (0.2, 0.0, 2.0), UP)
    np.testing.assert_allclose(lm.landmarks[a].point, (0.1, 0.0, 1.0))


def test_separation_below_threshold_raises():
    with pytest.raises(ValueError):
        LineLandmarks(0.5, min_anchor_separation=0.2)


def test_voxel_keeps_cloud_bounded():
    lm = LineLandmarks(0.3, voxel=0.05)
    pts = pillar(0.0, 0.0)
    for _ in range(10):
        gid = lm.assign(8, pts.mean(axis=0), UP, pts)
    n = len(lm.landmarks[gid].points)
    assert n == len(lm.voxelize(pts))
    assert n <= 3.0 / 0.05 + 1


def test_set_anchor_reserves_id():
    lm = LineLandmarks(0.3)
    lm.set_anchor(7, (5.0, 0.0, 0.0), UP, class_id=8)
    assert lm.assign(8, (5.1, 0.0, 1.0), UP) == 7
    assert lm.assign(8, (9.0, 0.0, 1.0), UP) == 8


def test_update_maps_lines_of_a_frame():
    lm = LineLandmarks(0.3)
    xyz = np.concatenate([pillar(0.0, 0.0), pillar(2.0, 0.0)])
    line = np.repeat([1, 2], len(xyz) // 2)
    first = lm.update(xyz, np.full(len(xyz), 8), line)
    assert lm.update(xyz, np.full(len(xyz), 8), line) == first
    assert sorted(first.values()) == [1, 2]
