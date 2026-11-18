#!/usr/bin/env python3
"""Line landmarks against the ground-truth pillar, pipe and duct axes.

    ros2 run axial_perception line_eval.py <map.npz> --gt <world_gt.yaml> [-l]

map.npz holds the /axial/landmark_map points (xyz, class_id, landmark), the trajectory flown and, optionally, the
DPS-SLAM lines (slam_ids, slam_points, slam_directions).
"""
import argparse
import math

import numpy as np
import yaml

CLASS_IDS = {"pipe": 4, "duct": 6, "pillar": 8}
NAMES = {v: k for k, v in CLASS_IDS.items()}
BUNDLE = 0.12


def gt_sections(path):
    """Straight axes of the GT pillars, pipes and ducts: class id, name, ends a and b, radius."""
    with open(path) as fh:
        objects = yaml.safe_load(fh)["objects"]
    out = []
    for o in objects:
        if o["class"] == "pillar":
            cx, cy = o["centre"]
            a, b = np.array([cx, cy, 0.0]), np.array([cx, cy, o["height"]])
        elif o["class"] in ("pipe", "duct"):
            a, b = np.array(o["start"], float), np.array(o["end"], float)
        else:
            continue
        if np.linalg.norm(b - a) > 1e-6:
            out.append({"cls": CLASS_IDS[o["class"]], "name": o["name"], "a": a, "b": b, "r": o["radius"]})
    return out


def unit(v):
    return v / np.linalg.norm(v)


def point_segment(p, a, b):
    """Closest points of segment ab to the points p (N, 3)."""
    d = b - a
    t = np.clip((p - a) @ d / (d @ d), 0.0, 1.0)
    return a + t[:, None] * d


def bundles(secs):
    """Group id per section; parallel sections of a class within BUNDLE of each other share one."""
    group = list(range(len(secs)))

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i, s in enumerate(secs):
        for j in range(i + 1, len(secs)):
            t = secs[j]
            if s["cls"] != t["cls"] or abs(unit(s["b"] - s["a"]) @ unit(t["b"] - t["a"])) < 0.99:
                continue
            mid = (t["a"] + t["b"]) / 2
            if np.linalg.norm(mid - point_segment(mid[None], s["a"], s["b"])[0]) < BUNDLE:
                group[root(j)] = root(i)
    return [root(i) for i in range(len(secs))]


def fit(points):
    """Centroid, unit direction and extent along it of a point set."""
    c = points.mean(axis=0)
    d = np.linalg.svd(points - c, full_matrices=False)[2][0]
    t = (points - c) @ d
    return c, d, t.min(), t.max()


def match(c, d, lo, hi, cls, secs, max_angle=10.0, n=11):
    """GT section of the class within max_angle that most samples along the line are nearest to, so a line over
    several collinear sections of a run is scored against each: (index, median offset vector, angle), or None."""
    samples = c + np.linspace(lo, hi, n)[:, None] * d
    cand = []
    for i, s in enumerate(secs):
        if s["cls"] != cls:
            continue
        u = unit(s["b"] - s["a"])
        angle = math.degrees(math.acos(min(1.0, abs(u @ d))))
        if angle <= max_angle:
            cand.append((i, samples - point_segment(samples, s["a"], s["b"]), angle))
    if not cand:
        return None
    dist = np.array([np.linalg.norm(v, axis=1) for _, v, _ in cand])
    near = dist.argmin(axis=0)
    k = np.bincount(near, minlength=len(cand)).argmax()
    i, _, angle = cand[k]
    v = np.array([cand[j][1][t] for t, j in enumerate(near)])
    return i, v[np.argsort(np.linalg.norm(v, axis=1))[len(v) // 2]], angle


def covered_length(lines, sec, tol=0.15, step=0.25, max_angle=10.0):
    """Length of a GT section within tol of the surface of a line of its class (lines may be axis or surface)."""
    u = unit(sec["b"] - sec["a"])
    length = np.linalg.norm(sec["b"] - sec["a"])
    t = np.arange(0.0, length + 1e-9, step)
    samples = sec["a"] + t[:, None] * u
    hit = np.zeros(len(t), bool)
    for ln in lines:
        cls, c, d, lo, hi = ln[:5]
        if cls != sec["cls"] or math.degrees(math.acos(min(1.0, abs(u @ d)))) > max_angle:
            continue
        s = np.clip((samples - c) @ d, lo, hi)
        dist = np.linalg.norm(samples - (c + s[:, None] * d), axis=1)
        hit |= dist < sec["r"] + tol
    return hit.mean() * length, length


def evaluate(lines, secs, trajectory=None, max_offset=0.5, reach=3.5, min_length=1.5):
    """Per class: matched lines, ghosts, duplicates, offsets and coverage of the GT near the trajectory.

    lines: [(class id, centroid, direction, lo, hi[, id, points])]; trajectory: (N, 3) GT positions.
    With points, surface is the median distance of a line's points to the GT surface (axis distance - radius).
    """
    group = bundles(secs)
    rows = {}
    for cls in sorted({s["cls"] for s in secs}):
        mine = [ln for ln in lines if ln[0] == cls]
        hits, offs, zs, hs, angles, surf = {}, [], [], [], [], []
        ghosts = 0
        for ln in mine:
            _, c, d, lo, hi = ln[:5]
            m = match(c, d, lo, hi, cls, secs)
            if m is None or np.linalg.norm(m[1]) > max_offset:
                ghosts += 1
                continue
            i, v, angle = m
            hits[group[i]] = hits.get(group[i], 0) + 1
            u = unit(secs[i]["b"] - secs[i]["a"])
            offs.append(np.linalg.norm(v))
            if abs(u[2]) < 0.5:
                zs.append(v[2])
            hs.append(np.linalg.norm(v[:2] - (v[:2] @ u[:2]) * u[:2] if abs(u[2]) < 0.5 else v[:2]))
            angles.append(angle)
            if len(ln) > 6:
                pts = ln[6]
                surf.append(np.median(np.linalg.norm(pts - point_segment(pts, secs[i]["a"], secs[i]["b"]), axis=1))
                            - secs[i]["r"])
        near = set()
        got, total = 0.0, 0.0
        if trajectory is not None and len(trajectory):
            for i, s in enumerate(secs):
                if s["cls"] != cls or np.linalg.norm(s["b"] - s["a"]) < min_length:
                    continue
                samples = s["a"] + np.linspace(0, 1, 20)[:, None] * (s["b"] - s["a"])
                dxy = np.linalg.norm(samples[:, None, :2] - trajectory[None, ::20, :2], axis=2).min()
                if dxy < reach:
                    near.add(group[i])
                    g, t = covered_length(mine, s)
                    got, total = got + g, total + t
        offs, zs, hs = np.array(offs), np.array(zs), np.array(hs)
        rows[cls] = {
            "lines": len(mine), "matched": len(offs), "ghosts": ghosts,
            "duplicates": sum(n - 1 for n in hits.values()),
            "near": len(near), "covered": len(near & set(hits)),
            "length_covered": got / total if total else math.nan,
            "offset_median": float(np.median(offs)) if len(offs) else math.nan,
            "offset_p90": float(np.percentile(offs, 90)) if len(offs) else math.nan,
            "z_median": float(np.median(zs)) if len(zs) else math.nan,
            "z_abs_median": float(np.median(np.abs(zs))) if len(zs) else math.nan,
            "lateral_median": float(np.median(hs)) if len(hs) else math.nan,
            "angle_median": float(np.median(angles)) if angles else math.nan,
            "surface_median": float(np.median(surf)) if surf else math.nan,
        }
    return rows


def table(rows, title):
    out = [f"{title}",
           "| class | lines | matched | ghosts | duplicates | covered / near GT | length covered | offset median [m] | p90 [m] "
           "| z median [m] | abs z median [m] | lateral median [m] | angle median [deg] | surface median [m] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cls, r in rows.items():
        out.append(f"| {NAMES[cls]} | {r['lines']} | {r['matched']} | {r['ghosts']} | {r['duplicates']} "
                   f"| {r['covered']} / {r['near']} | {r['length_covered']:.0%} | {r['offset_median']:.3f} | {r['offset_p90']:.3f} "
                   f"| {r['z_median']:.3f} | {r['z_abs_median']:.3f} | {r['lateral_median']:.3f} "
                   f"| {r['angle_median']:.2f} | {r['surface_median']:.3f} |")
    return "\n".join(out)


def listing(lines, secs, max_offset=0.5):
    """One row per line: class, id, centre, length, tilt and its nearest GT section with the offset."""
    out = ["| class | id | centre [m] | length [m] | tilt [deg] | GT | offset [m] | z [m] | lateral [m] |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ln in lines:
        cls, c, d, lo, hi = ln[:5]
        gid = ln[5] if len(ln) > 5 else -1
        mid = c + (lo + hi) / 2 * d
        tilt = math.degrees(math.asin(min(1.0, abs(d[2]))))
        m = match(c, d, lo, hi, cls, secs)
        name, off, z, lat = "-", math.nan, math.nan, math.nan
        if m is not None:
            i, v, _ = m
            u = unit(secs[i]["b"] - secs[i]["a"])
            off = np.linalg.norm(v)
            name = secs[i]["name"] + ("" if off <= max_offset else " (ghost)")
            z = v[2] if abs(u[2]) < 0.5 else math.nan
            lat = np.linalg.norm(v[:2] - (v[:2] @ u[:2]) * u[:2] if abs(u[2]) < 0.5 else v[:2])
        out.append(f"| {NAMES[cls]} | {gid} | {mid[0]:.2f}, {mid[1]:.2f}, {mid[2]:.2f} | {hi - lo:.2f} | {tilt:.0f} "
                   f"| {name} | {off:.3f} | {z:.3f} | {lat:.3f} |")
    return "\n".join(out)


def map_lines(xyz, cls, landmark, min_points=10):
    """Line per landmark of the axial_frontend map (positive landmark ids): class, fit, landmark id."""
    out = []
    for gid in np.unique(landmark[landmark > 0]):
        sel = landmark == gid
        if sel.sum() >= min_points:
            out.append((int(cls[sel][0]),) + fit(xyz[sel]) + (int(gid), xyz[sel]))
    return out


def slam_lines(ids, points, directions, extents):
    """DPS-SLAM lines (axis estimates) over the extent of the map landmark of the same id."""
    out = []
    for i, p, d in zip(ids, points, directions):
        name, _, gid = i.rpartition("_")
        if name in CLASS_IDS and gid.isdigit() and int(gid) in extents:
            c, dm, lo, hi = extents[int(gid)]
            d = unit(d) * (1.0 if d @ dm >= 0.0 else -1.0)
            foot = p + ((c - p) @ d) * d
            out.append((CLASS_IDS[name], foot, d, lo, hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="npz with the landmark map, the trajectory and optionally the DPS-SLAM lines")
    ap.add_argument("-g", "--gt", required=True, help="ground-truth world file")
    ap.add_argument("-l", "--list", action="store_true", help="also list every pipe and duct line")
    args = ap.parse_args()
    m = np.load(args.map)
    secs = gt_sections(args.gt)
    lines = map_lines(m["xyz"], m["class_id"], m["landmark"])
    print(table(evaluate(lines, secs, m["trajectory"]), "axial_frontend landmark map"))
    if args.list:
        print(listing([ln for ln in lines if ln[0] != CLASS_IDS["pillar"]], secs))
    if "slam_ids" in m and len(m["slam_ids"]):
        ext = {ln[5]: ln[1:5] for ln in lines}
        slam = slam_lines(m["slam_ids"], m["slam_points"], m["slam_directions"], ext)
        print(table(evaluate(slam, secs, m["trajectory"]), "\nDPS-SLAM lines"))


if __name__ == "__main__":
    main()
