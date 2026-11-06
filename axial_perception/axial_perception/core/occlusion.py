import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


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


def surface_mask(u, v, xyz, mask_img, link, radius, behind, min_points=10):
    """True for the points on their mask's own surface. Per mask, points are grouped by linking them in 3D within link
    m. The largest group is the surface. Another group of min_points or more is dropped when its points lie (median)
    more than behind m behind the surface points within radius px in the image (background seen through the mask), or
    more than behind m in front of the nearest surface points (seen by the lidar but hidden from the camera)."""
    keep = np.zeros(len(u), bool)
    m = mask_img[v, u]
    for s in np.unique(m):
        i = np.flatnonzero(m == s)
        a, b = cKDTree(xyz[i]).query_pairs(link, output_type="ndarray").T
        lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(i), len(i))), directed=False)[1]
        size = np.bincount(lab)
        top = lab == size.argmax()
        uv = np.c_[u[i], v[i]]
        tree = cKDTree(uv[top])
        d, k = tree.query(uv, distance_upper_bound=radius)
        near = np.isfinite(d)
        dz = xyz[i, 2][near] - xyz[i[top], 2][k[near]]
        front = xyz[i, 2] - xyz[i[top], 2][tree.query(uv)[1]]
        good = size == size.max()
        for c in np.flatnonzero(size >= min_points):
            x = dz[lab[near] == c]
            good[c] |= (not len(x) or np.median(x) < behind) and np.median(front[lab == c]) > -behind
        keep[i[good[lab]]] = True
    return keep
