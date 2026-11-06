#!/usr/bin/env python3
"""Records every /axial/mask_clusters frame with its camera pose in a fixed frame, for full-rate offline replays of
the landmark map (landmark_map_gt.py --clusters).

    python3 record_clusters.py -o clusters.npz -t 300 [-f earth] --ros-args -p use_sim_time:=true
"""
import argparse
import sys
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from tf2_ros import Buffer, TransformListener


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="output .npz")
    ap.add_argument("-t", type=float, default=300.0, help="seconds to record")
    ap.add_argument("-f", "--frame", default="earth", help="fixed frame of the poses")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("record_clusters")
    tf = Buffer(cache_time=Duration(seconds=30))
    TransformListener(tf, node)
    pending, frames = deque(maxlen=50), []
    node.create_subscription(PointCloud2, "/axial/mask_clusters", pending.append, 50)
    end = time.monotonic() + args.t
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
            while pending:
                msg = pending[0]
                stamp = Time.from_msg(msg.header.stamp)
                if not tf.can_transform(args.frame, msg.header.frame_id, stamp):
                    break
                pending.popleft()
                tr = tf.lookup_transform(args.frame, msg.header.frame_id, stamp).transform
                frames.append((read_points_numpy(msg, field_names=("x", "y", "z", "class_id", "segment", "line")),
                               (tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
                               (tr.translation.x, tr.translation.y, tr.translation.z)))
    except KeyboardInterrupt:
        pass
    a, q, t = zip(*frames) if frames else ([np.empty((0, 6))], [(0, 0, 0, 1)], [(0, 0, 0)])
    np.savez_compressed(args.out, size=[len(x) for x in a], pts=np.concatenate(a), q=np.array(q), t=np.array(t))
    print(f"{len(frames)} frames, {sum(len(x) for x in a)} points in {args.out}")


if __name__ == "__main__":
    main()
