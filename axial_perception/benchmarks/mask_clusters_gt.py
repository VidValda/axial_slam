#!/usr/bin/env python3
"""Score /axial/mask_clusters against a ground-truth world file: distance of every point to the surface of the
nearest ground-truth pipe, duct or pillar of its class. Clusters are placed with the ground-truth drone pose,
so odometry drift doesn't count.

    python3 mask_clusters_gt.py --gt world_gt.yaml -n drone0 [-t 10] [--tol 0.1]
"""
import argparse
import time

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener

from axial_perception.core.camera import quat_to_matrix, transform

AXIAL = ("pipe", "duct", "pillar")


def class_names(path):
    """Class id -> name of the axial classes of the ground-truth file."""
    return {i: n for n, i in yaml.safe_load(open(path))["classes"].items() if n in AXIAL}


def load_axes(path):
    axes = {}
    for o in yaml.safe_load(open(path))["objects"]:
        if o["class"] in AXIAL[:2]:
            a, b = o["start"], o["end"]
        elif o["class"] == "pillar":
            a, b = [*o["centre"], 0.0], [*o["centre"], o["height"]]
        else:
            continue
        axes.setdefault(o["class_id"], []).append((a, b, o["radius"]))
    return {c: tuple(np.array(x) for x in zip(*v)) for c, v in axes.items()}


def surface_distance(p, axes):
    a, b, r = axes
    ab = b - a
    t = np.clip(((p[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    return np.abs(np.linalg.norm(p[:, None] - (a + t[..., None] * ab), axis=-1) - r).min(axis=1)


def record(seconds, ns):
    """Records /axial/mask_clusters for seconds (or until Ctrl-C); returns (cluster array, points placed with the
    ground-truth pose) per frame."""
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("mask_clusters_gt", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    tf = Buffer(cache_time=Duration(seconds=seconds + 10))
    TransformListener(tf, node)
    msgs, poses = [], []
    node.create_subscription(PointCloud2, "/axial/mask_clusters", msgs.append, 10)
    node.create_subscription(PoseStamped, f"/{ns}/ground_truth/pose", poses.append, qos_profile_sensor_data)
    end = time.time() + seconds
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        print("recording stopped early")

    if not poses:
        print("no ground-truth poses received")
        return []
    gt_stamps = np.array([Time.from_msg(p.header.stamp).nanoseconds for p in poses])
    frames = []
    for m in msgs:
        try:
            t = tf.lookup_transform(ns, m.header.frame_id, Time.from_msg(m.header.stamp))
        except Exception:
            continue
        g = poses[np.argmin(np.abs(gt_stamps - Time.from_msg(m.header.stamp).nanoseconds))].pose
        a = np.frombuffer(m.data, np.dtype([(f.name, np.float32) for f in m.fields]))
        q, tr = t.transform.rotation, t.transform.translation
        xyz = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
        p = transform(xyz, quat_to_matrix((q.x, q.y, q.z, q.w)), (tr.x, tr.y, tr.z))
        o, gp = g.orientation, g.position
        frames.append((a, transform(p, quat_to_matrix((o.x, o.y, o.z, o.w)), (gp.x, gp.y, gp.z))))
    if not frames:
        print("no clusters received")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", type=float, default=10.0, help="seconds to record")
    ap.add_argument("--tol", type=float, default=0.1, help="off-surface threshold in m")
    ap.add_argument("-n", "--ns", required=True, help="drone namespace (base frame and ground_truth/pose topic)")
    ap.add_argument("--gt", required=True, help="ground-truth world .yaml")
    args = ap.parse_args()
    axes = load_axes(args.gt)
    frames = record(args.t, args.ns)
    if not frames:
        return
    rows = []
    for a, p in frames:
        for c in np.unique(a["class_id"]):
            sel = a["class_id"] == c
            rows.append(np.column_stack([a["class_id"][sel], a["z"][sel], surface_distance(p[sel], axes[int(c)])]))
    rows = np.concatenate(rows)
    n = len(frames)
    print(f"{n} frames, {len(rows) / n:.0f} points/frame, off-surface = more than {args.tol * 100:.0f} cm")
    for c, name in class_names(args.gt).items():
        d = rows[rows[:, 0] == c]
        if len(d):
            off = d[:, 2] > args.tol
            print(f"  {name:6s} {len(d) / n:6.0f} pts/frame, off-surface {off.mean() * 100:5.1f}%, "
                  f"median distance {np.median(d[:, 2]) * 100:.1f} cm")
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 30)):
        d = rows[(rows[:, 1] >= lo) & (rows[:, 1] < hi)]
        if len(d):
            print(f"  depth {lo:2d}-{hi:2d} m: off-surface {np.mean(d[:, 2] > args.tol) * 100:5.1f}%")


if __name__ == "__main__":
    main()
