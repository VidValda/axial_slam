import numpy as np


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform(pts, R, t):
    return pts @ np.asarray(R).T + np.asarray(t)


def project(pts, K, width, height, depth_min, depth_max):
    """Pinhole projection of (N, 3) optical-frame points (x right, y down, z forward), K = (fx, fy, cx, cy).

    Returns pixel columns, rows and the indices of the points that land inside the image.
    """
    fx, fy, cx, cy = K
    idx = np.flatnonzero((pts[:, 2] >= depth_min) & (pts[:, 2] <= depth_max))
    p = pts[idx]
    # pixel centres sit on integer coordinates, so pixel i spans [i - 0.5, i + 0.5)
    u = fx * p[:, 0] / p[:, 2] + cx + 0.5
    v = fy * p[:, 1] / p[:, 2] + cy + 0.5
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside].astype(np.int32), v[inside].astype(np.int32), idx[inside]
