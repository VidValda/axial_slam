import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
from line_eval import bundles, evaluate, fit, map_lines, match, slam_lines

PIPE = {"cls": 4, "name": "p1", "a": np.array([0.0, 0.0, 2.4]), "b": np.array([20.0, 0.0, 2.4]), "r": 0.03}
TWIN = {"cls": 4, "name": "p2", "a": np.array([0.0, 0.07, 2.4]), "b": np.array([20.0, 0.07, 2.4]), "r": 0.03}
HIGH = {"cls": 4, "name": "p3", "a": np.array([0.0, 0.0, 2.75]), "b": np.array([20.0, 0.0, 2.75]), "r": 0.03}
PILLAR = {"cls": 8, "name": "c1", "a": np.array([5.0, 2.0, 0.0]), "b": np.array([5.0, 2.0, 3.0]), "r": 0.15}


def sec(cls, a, b, r=0.03):
    return {"cls": cls, "name": f"s{a}", "a": np.array(a, float), "b": np.array(b, float), "r": r}


def line_points(a, b, n=200):
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


def test_bundle_groups_close_parallel_pipes_only():
    g = bundles([PIPE, TWIN, HIGH, PILLAR])
    assert g[0] == g[1] and g[2] != g[0] and g[3] != g[0]


def test_offset_splits_into_z_and_lateral():
    pts = line_points(np.array([2.0, 0.1, 2.5]), np.array([12.0, 0.1, 2.5]))
    c, d, lo, hi = fit(pts)
    i, v, angle = match(c, d, lo, hi, 4, [PIPE, PILLAR])
    assert i == 0 and angle < 1e-6
    assert np.allclose(v, [0.0, 0.1, 0.1], atol=1e-9)
    r = evaluate([(4, c, d, lo, hi)], [PIPE, PILLAR])[4]
    assert math.isclose(r["z_median"], 0.1, abs_tol=1e-9)
    assert math.isclose(r["lateral_median"], 0.1, abs_tol=1e-9)


def test_nearest_of_stacked_pipes_and_ghosts():
    near_high = line_points(np.array([0.0, 0.0, 2.7]), np.array([10.0, 0.0, 2.7]))
    far = line_points(np.array([0.0, 3.0, 1.0]), np.array([10.0, 3.0, 1.0]))
    lines = [(4,) + fit(near_high), (4,) + fit(far)]
    r = evaluate(lines, [PIPE, HIGH])[4]
    assert r["matched"] == 1 and r["ghosts"] == 1
    assert math.isclose(r["z_median"], -0.05, abs_tol=1e-9)


def test_duplicates_and_coverage_count_bundles_once():
    a = line_points(np.array([1.0, 0.0, 2.4]), np.array([8.0, 0.0, 2.4]))
    b = line_points(np.array([11.0, 0.07, 2.4]), np.array([18.0, 0.07, 2.4]))
    traj = line_points(np.array([0.0, -1.0, 1.0]), np.array([20.0, -1.0, 1.0]), 400)
    r = evaluate([(4,) + fit(a), (4,) + fit(b)], [PIPE, TWIN, HIGH], traj)[4]
    assert r["matched"] == 2 and r["duplicates"] == 1
    assert r["near"] == 2 and r["covered"] == 1


def test_pillar_lateral_is_horizontal_offset():
    pts = line_points(np.array([5.2, 2.0, 0.5]), np.array([5.2, 2.0, 2.5]))
    r = evaluate([(8,) + fit(pts)], [PILLAR])[8]
    assert math.isclose(r["lateral_median"], 0.2, abs_tol=1e-9) and math.isnan(r["z_median"])


def test_map_and_slam_lines_from_ids():
    xyz = np.vstack([line_points(np.array([0.0, 0.0, 2.4]), np.array([10.0, 0.0, 2.4]), 50),
                     line_points(np.array([5.0, 2.0, 0.0]), np.array([5.0, 2.0, 3.0]), 50)])
    cls = np.r_[np.full(50, 4), np.full(50, 8)]
    lm = np.r_[np.full(50, 3), np.full(50, 7)]
    lines = map_lines(xyz, cls, lm)
    assert [ln[0] for ln in lines] == [4, 8] and [ln[5] for ln in lines] == [3, 7]
    ext = {ln[5]: ln[1:5] for ln in lines}
    slam = slam_lines(["pipe_3", "pillar_7", "pillar_9"], [np.array([0.0, 0.0, 2.5]), np.array([5.0, 2.1, 0.0])] * 1
                      + [np.zeros(3)], [np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0, 0, 1.0])],
                      ext)
    assert len(slam) == 2
    r = evaluate(slam, [PIPE, PILLAR])
    assert math.isclose(r[4]["z_median"], 0.1, abs_tol=1e-9)
    assert math.isclose(r[8]["lateral_median"], 0.1, abs_tol=1e-9)


def test_slam_line_keeps_the_map_extent_when_its_direction_is_flipped():
    ext = {3: (np.array([2.0, 0.0, 2.4]), np.array([1.0, 0.0, 0.0]), -2.0, 8.0)}
    _, c, d, lo, hi = slam_lines(["pipe_3"], [np.array([0.0, 0.0, 2.5])], [np.array([-1.0, 0.0, 0.0])], ext)[0]
    assert np.allclose(sorted([(c + lo * d)[0], (c + hi * d)[0]]), [0.0, 10.0])


def test_line_over_collinear_sections_matches_each():
    secs = [sec(4, (0, 0, 2), (5, 0, 2)), sec(4, (5, 0, 2), (10, 0, 2)), sec(4, (0, 1, 2.4), (10, 1, 2.4))]
    lines = [(4,) + fit(line_points(np.array([0.5, 0.0, 2.01]), np.array([9.5, 0.0, 2.01])))]
    m = match(*lines[0][1:5], 4, secs)
    assert m[0] in (0, 1) and abs(np.linalg.norm(m[1]) - 0.01) < 1e-3
    r = evaluate(lines, secs, np.array([[5.0, 0.0, 1.0]]))[4]
    assert r["matched"] == 1 and r["ghosts"] == 0


def test_length_covered_counts_the_part_under_lines():
    secs = [sec(4, (0, 0, 2), (10, 0, 2))]
    lines = [(4,) + fit(line_points(np.array([0.0, 0.0, 2.05]), np.array([4.0, 0.0, 2.05])))]
    r = evaluate(lines, secs, np.array([[5.0, 0.0, 1.0]]))[4]
    assert 0.35 < r["length_covered"] < 0.45
