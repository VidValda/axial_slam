#!/usr/bin/env python3
"""Benchmark the line-splitting methods of axial_perception against the ground-truth sections of a world file.
Records /axial/mask_clusters once and runs every method on the same frames, with the split parameters of the
mask_projector params file. --save/--load keep the frames, to compare settings on the same data.

    python3 line_segments_gt.py --gt world_gt.yaml -c params.yaml -n drone0 [-t 10] [--tol 0.1] [-m ransac pearl]
        [--by-class] [--save frames.npz | --load frames.npz]
"""
import argparse
import time

import numpy as np
import yaml

from axial_perception.core.segments import METHODS, fit_line, split_clusters
from mask_clusters_gt import AXIAL, class_names, record


def load_sections(path):
    """Start, end, radius, class and run (model) id of every pipe, duct and pillar section."""
    sec, runs = [], []
    for o in yaml.safe_load(open(path))["objects"]:
        if o["class"] in AXIAL[:2]:
            sec.append((o["start"], o["end"], o["radius"], o["class_id"]))
        elif o["class"] == "pillar":
            sec.append(([*o["centre"], 0.0], [*o["centre"], o["height"]], o["radius"], o["class_id"]))
        else:
            continue
        runs.append(o.get("model", o["name"]))
    return (*(np.array(x, float) for x in zip(*sec)), np.unique(runs, return_inverse=True)[1])


def section_ids(p, cls, sections, tol):
    """Index of the nearest section surface of the point's class, -1 if more than tol off."""
    a, b, r, c, _ = sections
    ab = b - a
    t = np.clip(((p[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    d = np.abs(np.linalg.norm(p[:, None] - (a + t[..., None] * ab), axis=-1) - r)
    d[cls[:, None] != c] = np.inf
    k = d.argmin(axis=1)
    return np.where(d[np.arange(len(p)), k] <= tol, k, -1)


def gt_labels(a, p, sections, tol):
    """section_ids, with points of another run than their mask's main run set to -2: those are projector errors."""
    g = section_ids(p, a["class_id"], sections, tol)
    run = sections[4]
    for s in np.unique(a["segment"]):
        on = (a["segment"] == s) & (g >= 0)
        if on.any():
            g[on & (run[g] != np.bincount(run[g[on]]).argmax())] = -2
    return g


def split_params(config):
    p = yaml.safe_load(open(config))["mask_projector"]["ros__parameters"]
    classes = p.get("axial_classes", [4, 6, 8])
    tols = dict(zip(classes, p.get("split_tol", [0.07, 0.25, 0.3])))
    gaps = dict(zip(classes, p.get("split_gap", [0.5, 1.5, 1.0])))
    return tols, gaps, p.get("split_min_points", 5), p.get("split_min_length", 0.5)


def save(path, frames):
    a = np.concatenate([a for a, _ in frames])
    np.savez_compressed(path, size=[len(f) for f, _ in frames], world=np.concatenate([p for _, p in frames]),
                        **{k: a[k] for k in ("x", "y", "z", "class_id", "segment")})


def load(path):
    d = np.load(path)
    dtype = np.dtype([(k, np.float32) for k in ("x", "y", "z", "class_id", "segment")])
    a = np.empty(len(d["world"]), dtype)
    for k in dtype.names:
        a[k] = d[k]
    cut = np.cumsum(d["size"])[:-1]
    return list(zip(np.split(a, cut), np.split(d["world"], cut)))


def score(frames, gt, sections, method, tols, min_points, gaps, min_length=0.0, cls=None, visible=0.5):
    lines = found = shown_total = over = mixed = bad = assigned = on = 0
    purity, angles, times = [], [], []
    axis = sections[1] - sections[0]
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    for a, p in frames:
        xyz = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
        start = time.perf_counter()
        line = split_clusters(xyz, a["class_id"], a["segment"].astype(int), method, tols, min_points, gaps,
                              min_length)
        times.append(time.perf_counter() - start)
        g = gt[id(a)]
        if cls is not None:
            line, g = np.where(a["class_id"] == cls, line, 0), np.where(a["class_id"] == cls, g, -1)
        major = {}
        for k in np.unique(line[line > 0]):
            s = g[(line == k) & (g >= 0)]
            lines += 1
            if len(s):
                m = np.bincount(s).max()
                purity.append(m / len(s))
                mixed += m / len(s) < 0.9
                main = np.bincount(s).argmax()
                major.setdefault(main, []).append(k)
                d = fit_line(p[line == k])[1]
                angles.append(np.degrees(np.arccos(min(abs(d @ axis[main]), 1.0))))
                bad += m / len(s) < 0.9 or angles[-1] > 5
        for s in np.unique(g[g >= 0]):
            t = p[g == s] @ axis[s]
            if t.max() - t.min() < visible:
                continue
            shown_total += 1
            found += s in major
            over += len(major.get(s, [])) > 1
        on += (g >= 0).sum()
        assigned += ((g >= 0) & (line > 0)).sum()
    t, angles = np.array(times) * 1e3, np.array(angles)
    return dict(lines=lines / len(frames), purity=100 * np.mean(purity) if purity else 0.0,
                mixed=100 * mixed / max(lines, 1), angle=np.median(angles) if len(angles) else 0.0,
                tilted=100 * np.mean(angles > 5) if len(angles) else 0.0, bad=100 * bad / max(lines, 1),
                found=100 * found / max(shown_total, 1), over=100 * over / max(shown_total, 1),
                assigned=100 * assigned / max(on, 1), ms=t.mean(), ms_max=t.max())


HEADER = "method    lines  purity  mixed  angle  tilted    bad  found   over  assigned  ms/fr    max"


def row(name, s):
    return (f"{name:9s} {s['lines']:6.1f} {s['purity']:7.1f} {s['mixed']:6.1f} {s['angle']:6.1f} {s['tilted']:7.1f} "
            f"{s['bad']:6.1f} {s['found']:6.1f} {s['over']:6.1f} {s['assigned']:7.1f} {s['ms']:6.1f} "
            f"{s['ms_max']:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", type=float, default=10.0, help="seconds to record")
    ap.add_argument("--tol", type=float, default=0.1, help="max distance in m from a point to its section surface")
    ap.add_argument("-m", "--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("-n", "--ns", help="drone namespace (base frame and ground_truth/pose topic), to record")
    ap.add_argument("--gt", required=True, help="ground-truth world .yaml")
    ap.add_argument("-c", "--config", required=True, help="params file with the mask_projector split params")
    ap.add_argument("--by-class", action="store_true", help="also score each class on its own")
    ap.add_argument("--save", help="write the recorded frames to this .npz")
    ap.add_argument("--load", help="score the frames of this .npz instead of recording")
    args = ap.parse_args()
    if not args.load and not args.ns:
        ap.error("-n is needed to record")
    frames = load(args.load) if args.load else record(args.t, args.ns)
    if not frames:
        return
    if args.save:
        save(args.save, frames)
    frames = [f for f in frames if len(f[0])]
    tols, gaps, min_points, min_length = split_params(args.config)
    sections, names = load_sections(args.gt), class_names(args.gt)
    gt = {id(a): gt_labels(a, p, sections, args.tol) for a, p in frames}
    g = np.concatenate(list(gt.values()))
    print(f"{len(frames)} frames with clusters, tol {tols}, gap {gaps}, min_points {min_points}, "
          f"min_length {min_length}; "
          f"{np.mean(g[g != -1] == -2) * 100:.1f}% of on-section points are on another run than their mask (not scored)")
    print(HEADER)
    for m in args.methods:
        print(row(m, score(frames, gt, sections, m, tols, min_points, gaps, min_length)))
        if args.by_class:
            for c in tols:
                print(row(f"  {names[c]}", score(frames, gt, sections, m, tols, min_points, gaps, min_length, c)))
    print("purity: share of a line's points on its main section; mixed: lines under 90% purity;\n"
          "angle: median angle in deg between a line's fit and its main section; tilted: lines more than 5 deg off;\n"
          "bad: mixed or tilted lines;\n"
          "found/over: sections seen over 0.5 m or more that have at least one / more than one line; assigned: on-section points on a line")


if __name__ == "__main__":
    main()
