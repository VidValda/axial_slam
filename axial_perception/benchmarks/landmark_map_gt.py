#!/usr/bin/env python3
"""Replays a record_raw.py folder through the mask_projector steps and the landmark map, and scores the map against
the ground-truth pillars: ids per pillar (split), pillars per id (merged), and the axis offset and tilt per landmark.

    python3 landmark_map_gt.py /tmp/raw20 --gt world_gt.yaml -c params.yaml
"""
import argparse
from collections import Counter

import numpy as np
import yaml

from axial_perception.core.landmarks import LineLandmarks
from axial_perception.core.segments import fit_line, split_clusters
from turn_dataset import frame_points, load_raw, params


def load_pillars(path):
    ps = [o for o in yaml.safe_load(open(path))["objects"] if o["class"] == "pillar"]
    return np.array([o["centre"] for o in ps], float), np.array([o["radius"] for o in ps]), \
        np.array([o["height"] for o in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="record_raw.py folder")
    ap.add_argument("--gt", required=True, help="ground-truth world .yaml")
    ap.add_argument("-c", "--config", required=True, help="mask_projector params file")
    ap.add_argument("--match", type=float, default=0.3, help="max gap in m from a line to its pillar surface")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["mask_projector"]["ros__parameters"]
    p = params(args.config)
    lm = LineLandmarks(cfg.get("map_spawn_threshold", 0.5), cfg.get("map_min_separation", 1.0),
                       cfg.get("map_alpha", 0.2), cfg.get("map_max_angle", 0.35), cfg.get("map_voxel", 0.05))
    centre, radius, height = load_pillars(args.gt)
    seen = Counter()
    frames = lines = 0
    for f in load_raw(args.raw):
        frames += 1
        xyz, cls, seg = frame_points(f, p, cfg.get("occlusion_method", "zbuffer"))
        line = split_clusters(xyz, cls, seg, cfg.get("split_method", "ransac"), p["tols"], p["split_min_points"],
                              p["gaps"], p["min_length"])
        world = xyz @ f["R"].T + f["t"]
        for k, gid in lm.update(world, cls, line).items():
            if lm.landmarks[gid].class_id != 8:
                continue
            lines += 1
            c, _ = fit_line(world[line == k])
            d = np.linalg.norm(centre - c[:2], axis=1) - radius
            j = int(d.argmin())
            seen[gid, j if d[j] < args.match else -1] += 1

    gt_of = {}
    for (gid, j), n in seen.items():
        gt_of.setdefault(gid, Counter())[j] = n
    ids_of = {}
    for gid, c in gt_of.items():
        ids_of.setdefault(c.most_common(1)[0][0], []).append(gid)
    merged = [g for g, c in gt_of.items() if len([j for j in c if j >= 0]) > 1]
    split = {j: g for j, g in ids_of.items() if j >= 0 and len(g) > 1}
    offset, tilt = [], []
    for gid, c in gt_of.items():
        j = c.most_common(1)[0][0]
        if j < 0:
            continue
        a = lm.landmarks[gid]
        q = a.point + (height[j] / 2 - a.point[2]) / a.direction[2] * a.direction
        offset.append(np.linalg.norm(q[:2] - centre[j]))
        tilt.append(np.degrees(np.arccos(min(abs(a.direction[2]), 1.0))))
    offset, tilt = np.array(offset), np.array(tilt)

    print(f"{frames} frames, {lines} pillar lines, {len(gt_of)} landmarks, "
          f"{len([j for j in ids_of if j >= 0])} of {len(centre)} pillars mapped")
    print(f"off any pillar: {len(ids_of.get(-1, []))} landmarks, {sum(n for (g, j), n in seen.items() if j < 0)} lines")
    print(f"split pillars: {len(split)} {dict(sorted(split.items()))}")
    print(f"merged landmarks: {len(merged)} {merged}")
    if len(offset):
        print(f"axis offset from GT centre (m): median {np.median(offset):.3f}, p95 {np.percentile(offset, 95):.3f}, "
              f"max {offset.max():.3f} (radius {radius.min():.2f}-{radius.max():.2f})")
        print(f"tilt (deg): median {np.median(tilt):.2f}, p95 {np.percentile(tilt, 95):.2f}")


if __name__ == "__main__":
    main()
