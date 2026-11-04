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


def merge_lines(pts, labels, tol, share=0.95):
    """Greedily merges two lines while one line fits at least share of their points within tol."""
    labels = labels.copy()
    while True:
        ids = np.unique(labels[labels > 0])
        best, pair = share, None
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sel = (labels == a) | (labels == b)
                fit = (line_distance(pts[sel], *fit_line(pts[sel])) < tol).mean()
                if fit >= best:
                    best, pair = fit, (a, b)
        if pair is None:
            return labels
        labels[labels == pair[1]] = pair[0]


def finish(pts, labels, tol, min_points, gap):
    return gap_split(pts, merge_lines(pts, labels, tol), gap, min_points)


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
    return finish(pts, labels, tol, min_points, gap)


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
    return finish(pts, labels, tol, min_points, gap)


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


def jlinkage(pts, tol, min_points, gap, rng, proposals=100, local=10):
    """J-Linkage: agglomerative merging of points by Jaccard distance of their preference sets."""
    n = len(pts)
    if n < max(min_points, 3):
        return np.zeros(n, int)
    nbr = knn(pts, min(local, n - 1))
    i = rng.integers(n, size=proposals)
    d = pts[nbr[i, rng.integers(nbr.shape[1], size=proposals)]] - pts[i]
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 1e-6
    sets = (dist2(pts, pts[i[ok]], d[ok] / norm[ok, None]) < tol * tol).T.astype(float)
    size = sets.sum(1)
    inter = sets @ sets.T
    jac = 1 - inter / np.maximum(size[:, None] + size[None] - inter, 1)
    np.fill_diagonal(jac, np.inf)
    owner = np.arange(n)
    alive = np.ones(n, bool)
    rmin, rarg = jac.min(1), jac.argmin(1)
    while True:
        a = np.argmin(rmin)
        b = rarg[a]
        if rmin[a] >= 1:
            break
        sets[a] *= sets[b]
        owner[owner == b] = a
        alive[b] = False
        jac[b], jac[:, b], rmin[b] = np.inf, np.inf, np.inf
        size[a] = sets[a].sum()
        inter = sets @ sets[a]
        row = np.where(alive, 1 - inter / np.maximum(size + size[a] - inter, 1), np.inf)
        row[a] = np.inf
        jac[a], jac[:, a] = row, row
        stale = alive & ((rarg == a) | (rarg == b))
        stale[a] = True
        rmin[stale], rarg[stale] = jac[stale].min(1), jac[stale].argmin(1)
        better = alive & (jac[:, a] < rmin)
        rmin[better], rarg[better] = jac[better, a], a
    _, labels, counts = np.unique(owner, return_inverse=True, return_counts=True)
    labels = np.where(counts[labels] >= min_points, labels + 1, 0)
    return finish(pts, labels, tol, min_points, gap)


def mst(pts):
    """Prim's minimum spanning tree: parent and edge length per point, and the order points joined the tree."""
    n = len(pts)
    parent, w, done = np.zeros(n, int), np.zeros(n), np.zeros(n, bool)
    dist = np.linalg.norm(pts - pts[0], axis=1)
    done[0], dist[0] = True, np.inf
    order = [0]
    for _ in range(n - 1):
        j = np.argmin(dist)
        w[j], done[j] = dist[j], True
        order.append(j)
        nd = np.linalg.norm(pts - pts[j], axis=1)
        upd = (nd < dist) & ~done
        dist[upd], parent[upd] = nd[upd], j
        dist[j] = np.inf
    return parent, w, order


def longest_path(nodes, parent, pts):
    adj = {v: [] for v in nodes}
    for v in nodes:
        if parent[v] in adj and parent[v] != v:
            adj[v].append(parent[v])
            adj[parent[v]].append(v)

    def farthest(src):
        dist, prev, stack = {src: 0.0}, {src: -1}, [src]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in dist:
                    dist[v], prev[v] = dist[u] + np.linalg.norm(pts[u] - pts[v]), u
                    stack.append(v)
        end = max(dist, key=dist.get)
        return end, prev

    a, _ = farthest(nodes[0])
    b, prev = farthest(a)
    path = [b]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])
    return np.array(path)


def segmented_lsq(p, penalty):
    """Optimal split of an ordered point sequence into lines: sum of squared residuals + penalty per line."""
    m = len(p)
    s1 = np.vstack([np.zeros(3), np.cumsum(p, axis=0)])
    s2 = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(p[:, :, None] * p[:, None], axis=0)])
    i, j = np.triu_indices(m + 1, 1)
    a, b = s1[j] - s1[i], s2[j] - s2[i]
    ev = np.linalg.eigvalsh(b - a[:, :, None] * a[:, None] / (j - i)[:, None, None])
    sse = np.full((m + 1, m + 1), np.inf)
    sse[i, j] = ev[:, 0] + ev[:, 1]
    best, arg = np.zeros(m + 1), np.zeros(m + 1, int)
    for e in range(1, m + 1):
        v = best[:e] + sse[:e, e] + penalty
        arg[e] = np.argmin(v)
        best[e] = v[arg[e]]
    labels, e, k = np.zeros(m, int), m, 0
    while e > 0:
        k += 1
        labels[arg[e]:e] = k
        e = arg[e]
    return labels


def mst_dp(pts, tol, min_points, gap, rng, penalty=5.0):
    """Orders points along the longest MST path and splits the order with segmented least squares."""
    n = len(pts)
    if n < max(min_points, 3):
        return np.zeros(n, int)
    parent, w, order = mst(pts)
    comp = np.arange(n)
    for v in order[1:]:
        if w[v] <= gap or gap <= 0:
            comp[v] = comp[parent[v]]
    labels = np.zeros(n, int)
    for c in np.unique(comp):
        nodes = np.flatnonzero(comp == c)
        if len(nodes) < min_points:
            continue
        path = longest_path(list(nodes), parent, pts)
        seg = segmented_lsq(pts[path], penalty * tol * tol)
        near = np.argmin(np.linalg.norm(pts[nodes][:, None] - pts[path][None], axis=-1), axis=1)
        labels[nodes] = seg[near] + labels.max()
    for k in np.unique(labels[labels > 0]):
        sel = np.flatnonzero(labels == k)
        if len(sel) >= 2:
            labels[sel[line_distance(pts[sel], *fit_line(pts[sel])) > tol]] = 0
    return finish(pts, labels, tol, min_points, gap)


METHODS = {"none": no_split, "ransac": ransac, "pearl": pearl, "jlinkage": jlinkage, "mst_dp": mst_dp}


def split_clusters(pts, cls, seg, method, tols, min_points, gap, seed=0):
    """Splits every mask (seg) into straight lines; returns per-point line ids 1..M (0 = unassigned)."""
    rng = np.random.default_rng(seed)
    line = np.zeros(len(pts), int)
    for k in np.unique(seg):
        sel = np.flatnonzero(seg == k)
        lab = METHODS[method](pts[sel], tols[int(cls[sel[0]])], min_points, gap, rng)
        line[sel[lab > 0]] = lab[lab > 0] + line.max()
    return line
