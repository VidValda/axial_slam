import numpy as np

from axial_perception.core.scan_buffer import ScanBuffer


def test_empty():
    assert ScanBuffer(3).points().shape == (0, 3)


def test_keeps_last_n_in_order():
    buf = ScanBuffer(2)
    for k in range(4):
        buf.add(np.full((k + 1, 3), k, float))
    assert len(buf) == 2
    np.testing.assert_array_equal(buf.points()[:, 0], [2, 2, 2, 3, 3, 3, 3])
