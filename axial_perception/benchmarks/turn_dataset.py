#!/usr/bin/env python3
"""Raw-frame dataset of pipe and duct turns, to score the occlusion filter and the line splitting offline.

--build picks the frames of a record_raw.py folder where a mask shows both arms of a joint (a few views per turn, at
least 1 m apart) and keeps their raw inputs: the cloud in the camera frame, the labels map, K and the camera pose.
Scoring runs the mask_projector steps on them with the params of the mask_projector params file.

    python3 turn_dataset.py --gt world_gt.yaml -c params.yaml -d turns.npz --build /tmp/raw20
    python3 turn_dataset.py --gt world_gt.yaml -c params.yaml -d turns.npz [--occlusion none zbuffer mask]
        [-m none ransac]
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import yaml

from axial_perception.core.camera import project
from axial_perception.core.clusters import label_points
from axial_perception.core.masks import decode_gz_instance
from axial_perception.core.occlusion import front_mask, surface_mask
from axial_perception.core.segments import (METHODS, drop_short, finish, fit_line, line_distance, longest_path, mst,
                                            split_clusters)
from line_segments_gt import load_sections, section_ids

ARM_POINTS = 20
OFFLINE = False
CORNER = 1.0


def params(config):
    p = yaml.safe_load(open(config))["mask_projector"]["ros__parameters"]
    classes = p.get("axial_classes", [4, 6, 8])
    return dict(classes=classes, depth_min=p.get("depth_min", 0.1), depth_max=p.get("depth_max", 30.0),
                window=p.get("occlusion_window", 9), occ_tol=p.get("occlusion_tol", 0.1),
                link=p.get("occlusion_link", 0.1), radius=p.get("occlusion_radius", 15),
                behind=p.get("occlusion_behind", 0.3),
                min_points=p.get("min_points", 5), tols=dict(zip(classes, p.get("split_tol", [0.07, 0.25, 0.3]))),
                gaps=dict(zip(classes, p.get("split_gap", [0.5, 1.5, 1.0]))),
                split_min_points=p.get("split_min_points", 5), min_length=p.get("split_min_length", 0.5))


def joints(sections):
    """(i, j, point) for every two pipe or duct sections of the same run that share an end."""
    a, b, r, c, run = sections
    out = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if run[i] != run[j]:
                continue
            ends = np.array([[np.linalg.norm(p - q) for q in (a[j], b[j])] for p in (a[i], b[i])])
            k = np.unravel_index(ends.argmin(), ends.shape)
            if ends[k] < 0.05:
                out.append((i, j, (a[i], b[i])[k[0]]))
    return out


def elbow_zone(p, g, sections, turns):
    """Points around a joint, closer to it than the arm radius + 5 cm: they belong to both arms."""
    r = sections[2]
    near = np.zeros(len(p), bool)
    for i, j, q in turns:
        near |= np.isin(g, (i, j)) & (np.linalg.norm(p - q, axis=1) < max(r[i], r[j]) + 0.05)
    return near


def gt_labels(world, cls, seg, sections, tol=0.1):
    """Section of each point: the nearest one of its mask's run (the run most of its points are on) if within tol,
    else -2 if within tol of another run, else -1."""
    run = sections[4]
    g = section_ids(world, cls, sections, tol)
    out = np.full(len(g), -1)
    for s in np.unique(seg):
        m = seg == s
        if not (g[m] >= 0).any():
            continue
        own = np.flatnonzero(run == np.bincount(run[g[m][g[m] >= 0]]).argmax())
        mine = section_ids(world[m], cls[m], tuple(x[own] for x in sections), tol)
        out[m] = np.where(mine >= 0, own[np.maximum(mine, 0)], np.where(g[m] >= 0, -2, -1))
    return out


def frame_points(f, p, occlusion):
    """mask_projector steps on one frame: projection, occlusion, labelling. Returns kept xyz, class and mask ids."""
    xyz = f["xyz"].astype(float)
    class_img, mask_img = decode_gz_instance(f["labels"].tobytes(), *f["labels"].shape[:2])
    h, w = class_img.shape
    u, v, idx = project(xyz, f["K"], w, h, p["depth_min"], p["depth_max"])
    if occlusion == "zbuffer":
        keep = front_mask(u, v, xyz[idx, 2], w, h, p["window"], p["occ_tol"])
        u, v, idx = u[keep], v[keep], idx[keep]
    elif occlusion == "mask":
        on = np.isin(class_img[v, u], p["classes"])
        u, v, idx = u[on], v[on], idx[on]
        keep = surface_mask(u, v, xyz[idx], mask_img, p["link"], p["radius"], p["behind"])
        u, v, idx = u[keep], v[keep], idx[keep]
    kept, cls, seg = label_points(u, v, idx, class_img, mask_img, p["classes"], p["min_points"])
    return xyz[kept], cls, seg


def thin(img):
    """Zhang-Suen thinning of a binary image."""
    img = np.pad(img.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            c = img[1:-1, 1:-1]
            n = [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2],
                 img[:-2, :-2]]
            b = sum(x.astype(int) for x in n)
            a = sum(((n[k] == 0) & (n[(k + 1) % 8] == 1)).astype(int) for k in range(8))
            if step == 0:
                side = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                side = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            drop = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & side
            if drop.any():
                c[drop] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def centerline_pieces(mask, eps_ratio=0.5, min_pixels=5):
    """Straight pieces (start, end, half width) in pixels of a binary mask: its centerline ordered along the longest
    path of each connected part, simplified with Douglas-Peucker to eps_ratio times the local half width."""
    radius = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    sk = np.argwhere(thin(mask))[:, ::-1].astype(float)
    if len(sk) < min_pixels:
        return []
    parent, w, order = mst(sk)
    comp = np.arange(len(sk))
    for k in order[1:]:
        if w[k] < 1.5:
            comp[k] = comp[parent[k]]
    pieces = []
    for c in np.unique(comp):
        nodes = np.flatnonzero(comp == c)
        if len(nodes) < min_pixels:
            continue
        path = sk[longest_path(list(nodes), parent, sk)]
        r = float(np.median(radius[path[:, 1].astype(int), path[:, 0].astype(int)]))
        poly = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), max(2.0, eps_ratio * r), False)
        poly = poly.reshape(-1, 2)
        pieces += [(a, b, r) for a, b in zip(poly[:-1], poly[1:])]
    return pieces


def split_2d(xyz, cls, seg, f, p):
    """Line ids from the masks: each mask's centerline pieces, every pixel to its nearest piece, every point to the
    piece of its pixel, then the 3D check of the other methods (off-line points out, collinear merge, gap split)."""
    class_img, mask_img = decode_gz_instance(f["labels"].tobytes(), *f["labels"].shape[:2])
    h, w = class_img.shape
    u, v, _ = project(xyz, f["K"], w, h, 0.0, np.inf)
    line = np.zeros(len(xyz), int)
    for s in np.unique(seg):
        sel = np.flatnonzero(seg == s)
        ys, xs = np.nonzero(mask_img == mask_img[v[sel[0]], u[sel[0]]])
        y0, x0 = ys.min(), xs.min()
        crop = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), bool)
        crop[ys - y0, xs - x0] = True
        pieces = centerline_pieces(crop)
        if not pieces:
            continue
        q = np.c_[u[sel] - x0, v[sel] - y0].astype(float)
        d = []
        for a, b, _ in pieces:
            t = np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
            d.append(np.linalg.norm(q - a - t[:, None] * (b - a), axis=1))
        lab = np.argmin(d, axis=0) + 1
        for k, (a, b, r) in enumerate(pieces):
            if np.linalg.norm(b - a) < CORNER * r:
                lab[lab == k + 1] = 0
            if k and np.array_equal(pieces[k - 1][1], a):
                lab[np.linalg.norm(q - a, axis=1) < CORNER * r] = 0
        c = int(cls[sel[0]])
        pts = xyz[sel]
        if OFFLINE:
            for k in np.unique(lab):
                m = np.flatnonzero(lab == k)
                if len(m) >= 4:
                    lab[m[line_distance(pts[m], *fit_line(pts[m])) > p["tols"][c]]] = 0
        lab = finish(pts, lab, p["tols"][c], p["split_min_points"], p["gaps"][c])
        line[sel[lab > 0]] = lab[lab > 0] + line.max()
    return drop_short(xyz, line, p["min_length"]) if p["min_length"] > 0 else line


def load_raw(folder):
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        yield dict(np.load(path))


def build(folder, out, sections, p, per_key=4, apart=1.0):
    turns = joints(sections)
    axis = sections[1] - sections[0]
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    kept_frames, views = [], {}
    n = 0
    for f in load_raw(folder):
        n += 1
        xyz, cls, seg = frame_points(f, p, "none")
        world = xyz @ f["R"].T + f["t"]
        g = gt_labels(world, cls, seg, sections)
        use = (g >= 0) & ~elbow_zone(world, g, sections, turns)
        keys = []
        for s in np.unique(seg):
            arms = set()
            for k in np.unique(g[use & (seg == s)]):
                t = world[use & (seg == s) & (g == k)] @ axis[k]
                if len(t) >= ARM_POINTS and t.max() - t.min() >= 0.15:
                    arms.add(int(k))
            keys += [j for j, (a, b, _) in enumerate(turns) if a in arms and b in arms]
        cam = f["t"]
        new = [k for k in keys if len(views.get(k, [])) < per_key and
               all(np.linalg.norm(cam - c) >= apart for c in views.get(k, []))]
        if new:
            for k in new:
                views.setdefault(k, []).append(cam)
            kept_frames.append(f)
    sizes = [len(f["xyz"]) for f in kept_frames]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, size=sizes, xyz=np.concatenate([f["xyz"] for f in kept_frames]),
                        **{k: np.stack([f[k] for f in kept_frames]) for k in ("labels", "K", "R", "t", "stamp")})
    print(f"{out}: {len(kept_frames)} of {n} frames, {len(views)} of {len(turns)} joints seen "
          f"({sum(map(len, views.values()))} views), {os.path.getsize(out) / 1e6:.1f} MB")


def load(path):
    d = np.load(path)
    xyz = np.split(d["xyz"], np.cumsum(d["size"])[:-1])
    return [dict(xyz=x, **{k: d[k][n] for k in ("labels", "K", "R", "t")}) for n, x in enumerate(xyz)]


def score_occlusion(frames, sections, p, occlusion):
    """Share of the kept mask points on their mask's own run (precision), and share of the points on the own run
    surface that survive the filter (recall, against no filter)."""
    own = other = off = base_own = 0
    times = []
    for f in frames:
        start = time.perf_counter()
        xyz, cls, seg = frame_points(f, p, occlusion)
        times.append(time.perf_counter() - start)
        g = gt_labels(xyz @ f["R"].T + f["t"], cls, seg, sections)
        own, other, off = own + (g >= 0).sum(), other + (g == -2).sum(), off + (g == -1).sum()
        if occlusion != "none":
            xyz0, cls0, seg0 = frame_points(f, p, "none")
            base_own += (gt_labels(xyz0 @ f["R"].T + f["t"], cls0, seg0, sections) >= 0).sum()
    total = max(own + other + off, 1)
    return dict(own=100 * own / total, other=100 * other / total, off=100 * off / total,
                recall=100 * own / max(base_own, 1) if base_own else 100.0, ms=1e3 * np.mean(times))


def score_split(frames, sections, p, occlusion, method):
    turns = joints(sections)
    at = {frozenset((i, j)): q for i, j, q in turns}
    axis = sections[1] - sections[0]
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    ok = merged = missed = total = wrong = scored = covered = arm_points = 0
    depth, angles, times = [], [], []
    for f in frames:
        xyz, cls, seg = frame_points(f, p, occlusion)
        world = xyz @ f["R"].T + f["t"]
        g = gt_labels(world, cls, seg, sections)
        start = time.perf_counter()
        if method == "mask2d":
            line = split_2d(xyz, cls, seg, f, p)
        else:
            line = split_clusters(xyz, cls, seg, method, p["tols"], p["split_min_points"], p["gaps"], p["min_length"])
        times.append(time.perf_counter() - start)
        use = (g >= 0) & ~elbow_zone(world, g, sections, turns)
        main = {}
        for k in np.unique(line[line > 0]):
            s = g[(line == k) & use]
            if not len(s):
                continue
            main[k] = np.bincount(s).argmax()
            bad = (line == k) & use & (g != main[k])
            wrong += bad.sum()
            scored += ((line == k) & use).sum()
            reach = 0.0
            for q, s2 in zip(world[bad], g[bad]):
                j = at.get(frozenset((main[k], s2)))
                reach = max(reach, abs((q - j) @ axis[s2]) if j is not None else np.inf)
            depth.append(reach)
            angles.append(np.degrees(np.arccos(min(abs(fit_line(world[line == k])[1] @ axis[main[k]]), 1.0))))
        arm_points += use.sum()
        covered += (use & (line > 0)).sum()
        for s in np.unique(seg):
            m = use & (seg == s)
            for i, j, _ in turns:
                if (g[m] == i).sum() < ARM_POINTS or (g[m] == j).sum() < ARM_POINTS:
                    continue
                total += 1
                li, lj = (np.bincount(line[m & (g == k)]).argmax() for k in (i, j))
                if not li or not lj:
                    missed += 1
                elif li == lj:
                    merged += 1
                elif all((g[(line == k) & use] == main[k]).mean() >= 0.98 for k in (li, lj)):
                    ok += 1
    depth = np.array(depth)
    split = max(total - missed, 1)
    return dict(joints=total, missed=100 * missed / max(total, 1), ok=100 * ok / split,
                dirty=100 * (split - ok - merged) / split, merged=100 * merged / split,
                wrong=100 * wrong / max(scored, 1), invading=100 * np.mean(depth > 0.1) if len(depth) else 0.0,
                depth=np.percentile(depth[np.isfinite(depth)], 95) if np.isfinite(depth).any() else 0.0,
                angle=np.percentile(angles, 95) if angles else 0.0, coverage=100 * covered / max(arm_points, 1),
                ms=1e3 * np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="record_raw.py folder to build the dataset from")
    ap.add_argument("-d", "--data", required=True, help="dataset .npz")
    ap.add_argument("--gt", required=True, help="ground-truth world .yaml")
    ap.add_argument("-c", "--config", required=True, help="mask_projector params file")
    ap.add_argument("--occlusion", nargs="+", default=["none", "zbuffer"], choices=["none", "zbuffer", "mask"])
    ap.add_argument("-m", "--methods", nargs="+", default=["none", "ransac"], choices=list(METHODS) + ["mask2d"])
    args = ap.parse_args()
    sections, p = load_sections(args.gt), params(args.config)
    if args.build:
        build(args.build, args.data, sections, p)
        return
    frames = load(args.data)
    print(f"{len(frames)} frames; occlusion window {p['window']} px, tol {p['occ_tol']} m; split tol {p['tols']}, "
          f"gap {p['gaps']}, min_points {p['split_min_points']}, min_length {p['min_length']}")
    print("occlusion   own run  other run  off surface  recall  ms/frame")
    for o in args.occlusion:
        s = score_occlusion(frames, sections, p, o)
        print(f"{o:10s} {s['own']:8.1f} {s['other']:10.1f} {s['off']:12.1f} {s['recall']:7.1f} {s['ms']:9.1f}")
    print("occlusion  method    joints  missed  clean  dirty  merged  wrong  invading  depth95  angle95  coverage  ms")
    for o in args.occlusion:
        for m in args.methods:
            s = score_split(frames, sections, p, o, m)
            print(f"{o:10s} {m:9s} {s['joints']:6d} {s['missed']:7.1f} {s['ok']:6.1f} {s['dirty']:6.1f} "
                  f"{s['merged']:7.1f} {s['wrong']:6.1f} {s['invading']:9.1f} {s['depth']:8.2f} {s['angle']:8.1f} "
                  f"{s['coverage']:9.1f} {s['ms']:5.1f}")
    print("own run / other run / off surface: kept mask points within 0.1 m of their mask's run, of another run, of "
          "neither;\nrecall: own-run points kept, against no filter. Turns: missed = an arm without a line; of the "
          "others,\nclean = each arm its own line, 98% or more on it; dirty = less pure; merged = both arms on one "
          "line;\nwrong: line points on another section; invading / depth95: lines reaching more than 0.1 m into "
          "another section,\n95th percentile of that reach in m; angle95: 95th percentile of line tilt in deg")


if __name__ == "__main__":
    main()
