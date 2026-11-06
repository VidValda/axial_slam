#!/usr/bin/env python3
"""Records the raw inputs of mask_projector for offline work: per image, the stacked cloud in the camera frame (no
occlusion, cropped to the pixels near pipe, duct and pillar masks), the labels map, K and the camera pose in earth.
One .npz per frame in the output folder. Uses the mask_projector params (topics, n_scans, depth range).

    python3 record_raw.py -o /tmp/raw -t 600 --ros-args --params-file params.yaml -p use_sim_time:=true
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args

from axial_perception.core.camera import project, transform
from axial_perception.core.masks import decode_gz_instance
from axial_perception.nodes.mask_projector import MaskProjector


def dilate(img, r):
    """True where a True pixel is within r pixels (square window)."""
    c = np.pad(img, r + 1).astype(np.int32).cumsum(0).cumsum(1)
    h, w = img.shape
    k = 2 * r + 1
    return (c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) > 0


class Recorder(MaskProjector):
    def __init__(self, out, every, margin):
        super().__init__()
        self.out, self.every, self.margin = out, every, margin
        self.last, self.n = None, 0

    def process(self, msg, frame, tf):
        stamp = Time.from_msg(msg.header.stamp)
        if self.last is not None and (stamp - self.last).nanoseconds < self.every * 1e9:
            return
        pose = self.lookup("earth", frame, msg.header.stamp)
        if pose is None:
            return
        self.last = stamp
        pts = transform(self.scans.points(), *tf)
        u, v, idx = project(pts, self.K, msg.width, msg.height, self.p["depth_min"], self.p["depth_max"])
        class_img, _ = decode_gz_instance(msg.data, msg.height, msg.width)
        near = dilate(np.isin(class_img, self.p["axial_classes"]), self.margin)[v, u]
        np.savez_compressed(os.path.join(self.out, f"{self.n:05d}.npz"), xyz=pts[idx[near]].astype(np.float32),
                            labels=np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3),
                            K=np.array(self.K), R=pose[0], t=np.array(pose[1]), stamp=stamp.nanoseconds,
                            n_points=len(pts))
        self.n += 1
        if self.n % 20 == 0:
            self.get_logger().info(f"{self.n} frames, last {near.sum()} of {len(idx)} points in view")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("-t", type=float, default=600.0, help="seconds to record")
    ap.add_argument("--every", type=float, default=0.5, help="min seconds between frames")
    ap.add_argument("--margin", type=int, default=15, help="keep points this many pixels around the masks")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])
    os.makedirs(args.out, exist_ok=True)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Recorder(args.out, args.every, args.margin)
    end = time.monotonic() + args.t
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    print(f"{node.n} frames in {args.out}")


if __name__ == "__main__":
    main()
