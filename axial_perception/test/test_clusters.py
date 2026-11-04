import numpy as np

from axial_perception.core.clusters import label_points

CLASS = np.array([[0, 4, 4, 8], [0, 4, 4, 1]], np.uint8)
MASK = np.array([[0, 10, 10, 20], [0, 11, 11, 30]], np.int32)


def run(u, v, classes=(4, 8), min_points=1):
    u, v = np.array(u, np.int32), np.array(v, np.int32)
    return label_points(u, v, np.arange(len(u)) + 100, CLASS, MASK, classes, min_points)


def test_keeps_only_axial_classes_and_masks():
    idx, cls, seg = run([0, 1, 3, 3], [0, 0, 0, 1])
    np.testing.assert_array_equal(idx, [101, 102])
    np.testing.assert_array_equal(cls, [4, 8])


def test_segments_follow_masks():
    idx, cls, seg = run([1, 2, 1, 2, 3], [0, 0, 1, 1, 0])
    assert seg[0] == seg[1] != seg[2] == seg[3] != seg[4]
    assert set(seg) == {1, 2, 3}


def test_min_points_drops_small_masks():
    idx, cls, seg = run([1, 2, 1, 3], [0, 0, 1, 0], min_points=2)
    np.testing.assert_array_equal(idx, [100, 101])
    np.testing.assert_array_equal(seg, [1, 1])


def test_empty():
    idx, cls, seg = run([], [])
    assert len(idx) == len(cls) == len(seg) == 0
