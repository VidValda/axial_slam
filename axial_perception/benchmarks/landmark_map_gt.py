#!/usr/bin/env python3
"""Replays frames through the landmark map and scores it against the ground-truth pillars: ids per pillar (split),
pillars per id (merged), axis offset and tilt per landmark, and how far the published points are from the surface.

Frames are a record_raw.py folder (run through the mask_projector steps) or a record_clusters.py file (full rate).

    python3 landmark_map_gt.py --raw /tmp/raw20 --gt world_gt.yaml -c params.yaml
    python3 landmark_map_gt.py --clusters clusters.npz --gt world_gt.yaml -c params.yaml
"""
import argparse
from collections import Counter

import numpy as np
import yaml

from axial_perception.core.camera import quat_to_matrix
from axial_perception.core.landmarks import LineLandmarks
from axial_perception.core.segments import split_clusters
from turn_dataset import frame_points, load_raw, params

PILLAR = 8


def load_pillars(path):
    ps = [o for o in yaml.safe_load(open(path))["objects"] if o["class"] == "pillar"]
    return np.array([o["centre"] for o in ps], float), np.array([o["radius"] for o in ps]), \
        np.array([o["height"] for o in ps])


def raw_frames(folder, cfg, p):
    for f in load_raw(folder):
        xyz, cls, seg = frame_points(f, p, cfg.get("occlusion_method", "zbuffer"))
        line = split_clusters(xyz, cls, seg, cfg.get("split_method", "ransac"), p["tols"], p["split_min_points"],
                              p["gaps"], p["min_length"])
        yield xyz @ f["R"].T + f["t"], cls, line


def cluster_frames(path):
    d = np.load(path)
    for a, q, t in zip(np.split(d["pts"], np.cumsum(d["size"])[:-1]), d["q"], d["t"]):
        if len(a):
            yield a[:, :3] @ quat_to_matrix(q).T + t, a[:, 3].astype(int), a[:, 5].astype(int)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", help="record_raw.py folder")
    src.add_argument("--clusters", help="record_clusters.py file")
    ap.add_argument("--gt", required=True, help="ground-truth world .yaml")
    ap.add_argument("-c", "--config", required=True, help="mask_projector params file")
    ap.add_argument("--match", type=float, default=0.3, help="max gap in m from a point to its pillar surface")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["mask_projector"]["ros__parameters"]
    p = params(args.config)
    lm = LineLandmarks(cfg.get("map_spawn_threshold", 0.5), cfg.get("map_min_separation", 1.0),
                       cfg.get("map_max_angle", 0.35), cfg.get("map_voxel", 0.03), cfg.get("map_min_elongation", 3.0))
    centre, radius, height = load_pillars(args.gt)
    frames = lines = 0
    for xyz, cls, line in raw_frames(args.raw, cfg, p) if args.raw else cluster_frames(args.clusters):
        frames += 1
        lines += len(lm.update(xyz, cls, line))

    ids_of, offset, tilt, gaps = {}, [], [], []
    for gid, a in lm.landmarks.items():
        if a.class_id != PILLAR:
            continue
        pts = lm.dense_points(gid, cfg.get("map_min_neighbours", 8))
        if len(pts) == 0:
            continue
        rd = np.linalg.norm(pts[:, None, :2] - centre[None], axis=2) - radius[None]
        gaps.append(rd.min(axis=1))
        j = Counter(rd.argmin(axis=1)[np.abs(rd.min(axis=1)) < args.match]).most_common(1)
        j = j[0][0] if j else -1
        ids_of.setdefault(j, []).append(gid)
        if j >= 0:
            q = a.point + (height[j] / 2 - a.point[2]) / a.direction[2] * a.direction
            offset.append(np.linalg.norm(q[:2] - centre[j]))
            tilt.append(np.degrees(np.arccos(min(abs(a.direction[2]), 1.0))))
    split = {j: g for j, g in ids_of.items() if j >= 0 and len(g) > 1}
    offset, tilt = np.array(offset), np.array(tilt)
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)

    print(f"{frames} frames, {lines} accepted lines, {sum(map(len, ids_of.values()))} pillar landmarks, "
          f"{len([j for j in ids_of if j >= 0])} of {len(centre)} pillars mapped")
    print(f"off any pillar: {len(ids_of.get(-1, []))} landmarks")
    print(f"split pillars: {len(split)} {dict(sorted(split.items()))}")
    if len(offset):
        print(f"axis offset from GT centre (m): median {np.median(offset):.3f}, p95 {np.percentile(offset, 95):.3f}, "
              f"max {offset.max():.3f} (radius {radius.min():.2f}-{radius.max():.2f})")
        print(f"tilt (deg): median {np.median(tilt):.2f}, p95 {np.percentile(tilt, 95):.2f}, max {tilt.max():.2f}")
    if len(gaps):
        print(f"points: {len(gaps)}, within 3 cm of the surface {100 * np.mean(np.abs(gaps) < 0.03):.1f}%, "
              f"5 cm {100 * np.mean(np.abs(gaps) < 0.05):.1f}%, more than 10 cm out {100 * np.mean(gaps > 0.1):.2f}%")


if __name__ == "__main__":
    main()
