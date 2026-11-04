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


def knn(pts, k):
    sq = (pts * pts).sum(1)
    d = sq[:, None] + sq[None] - 2 * pts @ pts.T
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k - 1, axis=1)[:, :k]


def random_lines(pts, n, rng):
    i, j = rng.integers(len(pts), size=(2, n))
    d = pts[j] - pts[i]
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 1e-6
    return pts[i[ok]], d[ok] / norm[ok, None]


def dist2(pts, c, d):
    """Squared distances (lines x points) from pts to the lines through c along unit d."""
    r = pts[None] - c[:, None]
    return (r * r).sum(-1) - np.einsum("hnk,hk->hn", r, d) ** 2


def no_split(pts, tol, min_points, gap, rng):
    return np.full(len(pts), int(len(pts) >= min_points))


def ransac(pts, tol, min_points, gap, rng, iters=100):
    labels = np.zeros(len(pts), int)
    free = np.arange(len(pts))
    while len(free) >= max(min_points, 2):
        p = pts[free]
        c, d = random_lines(p, iters, rng)
        if not len(c):
            break
        inl = dist2(p, c, d) < tol * tol
        inl = inl[np.argmax(inl.sum(1))]
        if inl.sum() < min_points:
            break
        inl = line_distance(p, *fit_line(p[inl])) < tol
        if inl.sum() < min_points:
            break
        labels[free[inl]] = labels.max() + 1
        free = free[~inl]
    return gap_split(pts, labels, gap, min_points)


def pearl(pts, tol, min_points, gap, rng, proposals=30, k=6, smooth=0.3, label_cost=5.0, deadzone=0.7, sweeps=10):
    """PEARL-style: line proposals, then ICM on data cost + Potts on a k-NN graph + a label cost per line."""
    def data(d2):
        return (np.maximum(np.sqrt(np.maximum(d2, 0)) - deadzone * tol, 0) / ((1 - deadzone) * tol)) ** 2

    n = len(pts)
    if n < max(min_points, 3):
        return np.zeros(n, int)
    seed = ransac(pts, tol, min_points, 0, rng)
    lines = [fit_line(pts[seed == s]) for s in range(1, seed.max() + 1)]
    c, d = random_lines(pts, proposals, rng)
    c = np.concatenate([np.reshape([l[0] for l in lines], (-1, 3)), c])
    d = np.concatenate([np.reshape([l[1] for l in lines], (-1, 3)), d])
    if not len(c):
        return np.zeros(n, int)
    k = min(k, n - 1)
    nbr = knn(pts, k)
    cost = np.column_stack([np.ones(n), data(dist2(pts, c, d)).T])
    labels = cost.argmin(axis=1)
    for _ in range(sweeps):
        keep = prune(cost, label_cost)
        cost = cost[:, keep]
        labels = np.where(keep[labels], np.cumsum(keep)[labels] - 1, -1)
        counts = (labels[nbr][..., None] == np.arange(cost.shape[1])).sum(1)
        new = (cost + smooth * (k - counts)).argmin(axis=1)
        for l in np.flatnonzero(np.bincount(new, minlength=cost.shape[1])[1:] >= 2) + 1:
            cost[:, l] = data(line_distance(pts, *fit_line(pts[new == l])) ** 2)
        if np.array_equal(new, labels):
            break
        labels = new
    return gap_split(pts, labels, gap, min_points)


def prune(cost, label_cost):
    """Columns to keep: drops unused lines, then the line that saves least over the next best while under label_cost."""
    keep = np.ones(cost.shape[1], bool)
    while keep.sum() > 1:
        idx = np.flatnonzero(keep)
        c = cost[:, idx]
        best = c.argmin(axis=1)
        used = np.bincount(best, minlength=len(idx)) > 0
        used[0] = True
        keep[idx[~used]] = False
        if keep.sum() == 1:
            break
        c = c[:, used]
        best = c.argmin(axis=1)
        order = np.partition(c, 1, axis=1)
        saving = np.bincount(best, order[:, 1] - order[:, 0], minlength=c.shape[1])[1:]
        if saving.min() >= label_cost:
            break
        keep[idx[used][saving.argmin() + 1]] = False
    return keep


METHODS = {"none": no_split, "ransac": ransac, "pearl": pearl}


def split_clusters(pts, cls, seg, method, tols, min_points, gap, seed=0):
    """Splits every mask (seg) into straight lines; returns per-point line ids 1..M (0 = unassigned)."""
    rng = np.random.default_rng(seed)
    line = np.zeros(len(pts), int)
    for k in np.unique(seg):
        sel = np.flatnonzero(seg == k)
        lab = METHODS[method](pts[sel], tols[int(cls[sel[0]])], min_points, gap, rng)
        line[sel[lab > 0]] = lab[lab > 0] + line.max()
    return line
