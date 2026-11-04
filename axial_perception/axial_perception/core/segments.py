import numpy as np


def fit_line(pts):
    c = pts.mean(axis=0)
    return c, np.linalg.svd(pts - c, full_matrices=False)[2][0]


def line_distance(pts, c, d):
    r = pts - c
    return np.linalg.norm(r - np.outer(r @ d, d), axis=1)


def gap_split(pts, labels, gap, min_points):
    """Cuts each line where consecutive points along it are more than gap apart and drops pieces under min_points."""
    out = np.zeros(len(pts), int)
    n = 0
    for k in np.unique(labels[labels > 0]):
        sel = np.flatnonzero(labels == k)
        c, d = fit_line(pts[sel])
        t = (pts[sel] - c) @ d
        order = np.argsort(t)
        piece = np.zeros(len(sel), int)
        if gap > 0:
            piece[1:] = np.cumsum(np.diff(t[order]) > gap)
        for p in range(piece[-1] + 1):
            idx = sel[order[piece == p]]
            if len(idx) >= min_points:
                n += 1
                out[idx] = n
    return out


def no_split(pts, tol, min_points, gap, rng):
    return np.full(len(pts), int(len(pts) >= min_points))


def ransac(pts, tol, min_points, gap, rng, iters=100):
    labels = np.zeros(len(pts), int)
    free = np.arange(len(pts))
    while len(free) >= max(min_points, 2):
        p = pts[free]
        i, j = rng.integers(len(p), size=(2, iters))
        d = p[j] - p[i]
        norm = np.linalg.norm(d, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            break
        i, d = i[ok], d[ok] / norm[ok, None]
        r = p[None] - p[i][:, None]
        dist2 = (r * r).sum(-1) - np.einsum("hnk,hk->hn", r, d) ** 2
        best = np.argmax((dist2 < tol * tol).sum(1))
        inl = dist2[best] < tol * tol
        if inl.sum() < min_points:
            break
        inl = line_distance(p, *fit_line(p[inl])) < tol
        if inl.sum() < min_points:
            break
        labels[free[inl]] = labels.max() + 1
        free = free[~inl]
    return gap_split(pts, labels, gap, min_points)


METHODS = {"none": no_split, "ransac": ransac}


def split_clusters(pts, cls, seg, method, tols, min_points, gap, seed=0):
    """Splits every mask (seg) into straight lines; returns per-point line ids 1..M (0 = unassigned)."""
    rng = np.random.default_rng(seed)
    line = np.zeros(len(pts), int)
    for k in np.unique(seg):
        sel = np.flatnonzero(seg == k)
        lab = METHODS[method](pts[sel], tols[int(cls[sel[0]])], min_points, gap, rng)
        line[sel[lab > 0]] = lab[lab > 0] + line.max()
    return line
