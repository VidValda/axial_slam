import numpy as np


def _window_min(img, window):
    r = window // 2
    pad = np.pad(img, r, constant_values=np.inf)
    rows = np.min([pad[:, k:k + img.shape[1]] for k in range(window)], axis=0)
    return np.min([rows[k:k + img.shape[0]] for k in range(window)], axis=0)


def front_mask(u, v, z, width, height, window, tol):
    """True for points at most tol behind the nearest point in their window x window pixel neighbourhood."""
    zbuf = np.full((height, width), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    return z <= _window_min(zbuf, window)[v, u] + tol
