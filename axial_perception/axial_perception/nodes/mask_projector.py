import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points_numpy
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from axial_perception.core.camera import project, quat_to_matrix, transform
from axial_perception.core.clusters import label_points
from axial_perception.core.landmarks import LineLandmarks
from axial_perception.core.masks import decode_gz_instance
from axial_perception.core.occlusion import front_mask, surface_mask
from axial_perception.core.scan_buffer import ScanBuffer
from axial_perception.core.segments import split_clusters

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40],
], np.uint8)

CLUSTER_DTYPE = np.dtype([(n, np.float32) for n in ("x", "y", "z", "class_id", "segment", "line", "rgb")])
MAP_DTYPE = np.dtype([(n, np.float32) for n in ("x", "y", "z", "class_id", "landmark", "rgb")])


def float_fields(dtype):
    return [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(dtype.names)]


CLUSTER_FIELDS = float_fields(CLUSTER_DTYPE)
MAP_FIELDS = float_fields(MAP_DTYPE)


def pack_rgb(rgb):
    rgb = rgb.astype(np.uint32)
    return ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).view(np.float32)


def line_colours(line):
    c = PALETTE[(line - 1) % len(PALETTE)]
    c[line == 0] = 160
    return c


class MaskProjector(Node):
    def __init__(self):
        super().__init__("mask_projector")
        p = {
            "labels_topic": "labels_map",
            "cloud_topic": "points",
            "camera_info_topic": "camera_info",
            "image_topic": "image_raw",
            "camera_frame": "",
            "odom_frame": "odom",
            "n_scans": 5,
            "depth_min": 0.1,
            "depth_max": 30.0,
            "occlusion_method": "zbuffer",
            "occlusion_window": 9,
            "occlusion_tol": 0.1,
            "occlusion_link": 0.1,
            "occlusion_radius": 15,
            "occlusion_behind": 0.3,
            "axial_classes": [4, 6, 8],
            "class_names": ["pipe", "duct", "pillar"],
            "min_points": 5,
            "split_method": "ransac",
            "split_tol": [0.07, 0.25, 0.3],
            "split_gap": [0.5, 1.5, 1.0],
            "split_min_points": 5,
            "split_min_length": 0.5,
            "max_rate_hz": 10.0,
            "map_spawn_threshold": 0.5,
            "map_min_separation": 1.0,
            "map_max_angle": 0.35,
            "map_voxel": 0.03,
            "map_min_elongation": 3.0,
            "map_min_neighbours": 8,
            "map_rate_hz": 1.0,
        }
        for name, default in p.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in p}

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.K = None
        self.scans = ScanBuffer(self.p["n_scans"])
        self.pending_scans = deque(maxlen=20)
        self.pending_labels = deque(maxlen=10)
        self.image = None
        self.last_stamp = None

        sensor = qos_profile_sensor_data
        self.create_subscription(CameraInfo, self.p["camera_info_topic"], self.on_info, sensor)
        self.create_subscription(PointCloud2, self.p["cloud_topic"], self.on_cloud, sensor)
        self.create_subscription(Image, self.p["image_topic"], self.on_image, sensor)
        self.create_subscription(Image, self.p["labels_topic"], self.on_labels, sensor)
        self.pub_clusters = self.create_publisher(PointCloud2, "/axial/mask_clusters", 5)
        self.pub_overlay = self.create_publisher(Image, "/axial/mask_overlay", 5)
        self.pub_markers = self.create_publisher(MarkerArray, "/axial/mask_cluster_markers", 5)
        self.pub_map = self.create_publisher(PointCloud2, "/axial/landmark_map", 1)
        self.pub_map_markers = self.create_publisher(MarkerArray, "/axial/landmark_markers", 1)
        self.landmarks = LineLandmarks(self.p["map_spawn_threshold"], self.p["map_min_separation"],
                                       self.p["map_max_angle"], self.p["map_voxel"], self.p["map_min_elongation"])
        self.create_timer(1.0 / self.p["map_rate_hz"], self.publish_map)
        self.names = dict(zip(self.p["axial_classes"], self.p["class_names"]))
        self.tols = dict(zip(self.p["axial_classes"], self.p["split_tol"]))
        self.gaps = dict(zip(self.p["axial_classes"], self.p["split_gap"]))
        self.stats = []
        self.stats_since = time.monotonic()

    def on_info(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_cloud(self, msg):
        self.pending_scans.append(msg)
        self.flush()

    def on_image(self, msg):
        self.image = msg

    def on_labels(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        if self.last_stamp is None or (stamp - self.last_stamp).nanoseconds >= 1e9 / self.p["max_rate_hz"]:
            self.last_stamp = stamp
            if len(self.pending_labels) == self.pending_labels.maxlen:
                self.get_logger().warn(f"no TF for {self.pending_labels.maxlen} images in a row", throttle_duration_sec=5.0)
            self.pending_labels.append(msg)
        self.flush()

    def flush(self):
        self.flush_scans()
        self.flush_labels()

    def flush_scans(self):
        while self.pending_scans:
            msg = self.pending_scans[0]
            tf = self.lookup(self.p["odom_frame"], msg.header.frame_id, msg.header.stamp)
            if tf is None:
                return
            self.pending_scans.popleft()
            xyz = read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
            self.scans.add(transform(xyz, *tf))

    def flush_labels(self):
        if self.K is None or len(self.scans) == 0:
            return
        for i in reversed(range(len(self.pending_labels))):
            msg = self.pending_labels[i]
            frame = self.p["camera_frame"] or msg.header.frame_id
            tf = self.lookup(frame, self.p["odom_frame"], msg.header.stamp)
            if tf is not None:
                for _ in range(i + 1):
                    self.pending_labels.popleft()
                self.process(msg, frame, tf)
                return

    def process(self, msg, frame, tf):
        start = time.perf_counter()
        odom = self.scans.points()
        pts = transform(odom, *tf)
        class_img, mask_img = decode_gz_instance(msg.data, msg.height, msg.width)
        u, v, idx = project(pts, self.K, msg.width, msg.height, self.p["depth_min"], self.p["depth_max"])
        if self.p["occlusion_method"] == "zbuffer":
            front = front_mask(u, v, pts[idx, 2], msg.width, msg.height,
                               self.p["occlusion_window"], self.p["occlusion_tol"])
            u, v, idx = u[front], v[front], idx[front]
        elif self.p["occlusion_method"] == "mask":
            on = np.isin(class_img[v, u], self.p["axial_classes"])
            u, v, idx = u[on], v[on], idx[on]
            front = surface_mask(u, v, pts[idx], mask_img, self.p["occlusion_link"], self.p["occlusion_radius"],
                                 self.p["occlusion_behind"])
            u, v, idx = u[front], v[front], idx[front]
        kept, cls, seg = label_points(u, v, idx, class_img, mask_img, self.p["axial_classes"], self.p["min_points"])
        split_start = time.perf_counter()
        line = split_clusters(pts[kept], cls, seg, self.p["split_method"], self.tols,
                              self.p["split_min_points"], self.gaps, self.p["split_min_length"])
        split_dt = time.perf_counter() - split_start
        self.landmarks.update(odom[kept], cls, line)

        self.publish_clusters(msg.header.stamp, frame, pts[kept], cls, seg, line)
        if self.pub_overlay.get_subscription_count() > 0 and self.image is not None:
            self.publish_overlay(u, v, idx, kept, line)
        if self.pub_markers.get_subscription_count() > 0:
            self.publish_markers(msg.header.stamp, frame, pts[kept], cls, line)
        self.log_stats(len(pts), len(kept), int(seg.max(initial=0)), int(line.max(initial=0)),
                       split_dt, time.perf_counter() - start)

    def publish_map(self):
        stamp, frame = self.get_clock().now().to_msg(), self.p["odom_frame"]
        lms = self.landmarks.landmarks
        if not lms:
            return
        dense = {gid: self.landmarks.dense_points(gid, self.p["map_min_neighbours"]) for gid in lms}
        if self.pub_map.get_subscription_count() > 0:
            parts = []
            for gid, lm in lms.items():
                arr = np.empty(len(dense[gid]), MAP_DTYPE)
                arr["x"], arr["y"], arr["z"] = dense[gid].T
                arr["class_id"], arr["landmark"] = lm.class_id, gid
                arr["rgb"] = pack_rgb(line_colours(np.full(len(arr), gid)))
                parts.append(arr)
            arr = np.concatenate(parts)
            out = PointCloud2(height=1, width=len(arr), fields=MAP_FIELDS, is_bigendian=False,
                              point_step=MAP_DTYPE.itemsize, row_step=MAP_DTYPE.itemsize * len(arr),
                              is_dense=True, data=arr.tobytes())
            out.header.stamp, out.header.frame_id = stamp, frame
            self.pub_map.publish(out)
        if self.pub_map_markers.get_subscription_count() > 0:
            out = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
            for gid, lm in lms.items():
                if len(dense[gid]) == 0:
                    continue
                t = (dense[gid] - lm.point) @ lm.direction
                ends = [lm.point + t.min() * lm.direction, lm.point + t.max() * lm.direction]
                r, g, b = line_colours(np.array([gid]))[0] / 255.0
                axis = Marker(ns="landmark_axes", id=gid, type=Marker.LINE_LIST, action=Marker.ADD)
                axis.points = [Point(x=float(e[0]), y=float(e[1]), z=float(e[2])) for e in ends]
                axis.scale.x = 0.05
                label = Marker(ns="landmark_labels", id=gid, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                               text=f"{self.names.get(lm.class_id, lm.class_id)} {gid} ({lm.hits})")
                label.pose.position.x, label.pose.position.y, label.pose.position.z = (float(v) for v in ends[1])
                label.pose.position.z += 0.2
                label.scale.z = 0.25
                for m in (axis, label):
                    m.header.stamp, m.header.frame_id = stamp, frame
                    m.pose.orientation.w = 1.0
                    m.color.r, m.color.g, m.color.b, m.color.a = float(r), float(g), float(b), 1.0
                    out.markers.append(m)
            self.pub_map_markers.publish(out)

    def log_stats(self, *row):
        self.stats.append(row)
        if time.monotonic() - self.stats_since < 5.0:
            return
        s = np.array(self.stats)
        self.get_logger().info(f"{len(s)} frames: {s[:, 0].mean():.0f} pts in, {s[:, 1].mean():.0f} kept in "
                               f"{s[:, 2].mean():.1f} masks, {s[:, 3].mean():.1f} lines, "
                               f"split {s[:, 4].mean() * 1e3:.1f} ms, {s[:, 5].mean() * 1e3:.1f} ms mean, "
                               f"{s[:, 5].max() * 1e3:.1f} ms max")
        self.stats, self.stats_since = [], time.monotonic()

    def lookup(self, target, source, stamp):
        # odometry TF arrives after the sensor data it is stamped with, so wait without blocking
        stamp = Time.from_msg(stamp)
        if not self.tf.can_transform(target, source, stamp):
            return None
        tf = self.tf.lookup_transform(target, source, stamp)
        q, t = tf.transform.rotation, tf.transform.translation
        return quat_to_matrix((q.x, q.y, q.z, q.w)), (t.x, t.y, t.z)

    def publish_clusters(self, stamp, frame, xyz, cls, seg, line):
        arr = np.empty(len(xyz), CLUSTER_DTYPE)
        arr["x"], arr["y"], arr["z"] = xyz.T
        arr["class_id"], arr["segment"], arr["line"] = cls, seg, line
        arr["rgb"] = pack_rgb(line_colours(line))
        out = PointCloud2(height=1, width=len(arr), fields=CLUSTER_FIELDS, is_bigendian=False,
                          point_step=CLUSTER_DTYPE.itemsize, row_step=CLUSTER_DTYPE.itemsize * len(arr),
                          is_dense=True, data=arr.tobytes())
        out.header.stamp, out.header.frame_id = stamp, frame
        self.pub_clusters.publish(out)

    def publish_overlay(self, u, v, idx, kept, line):
        img = self.image
        if img.encoding != "rgb8":
            self.get_logger().warn(f"overlay needs rgb8, got {img.encoding}", throttle_duration_sec=5.0)
            return
        out = np.frombuffer(img.data, np.uint8).reshape(img.height, img.width, 3) // 2
        colour = np.full((len(idx), 3), 160, np.uint8)
        colour[np.searchsorted(idx, kept)] = line_colours(line)
        for du in (0, 1):
            for dv in (0, 1):
                out[np.minimum(v + dv, img.height - 1), np.minimum(u + du, img.width - 1)] = colour
        msg = Image(height=img.height, width=img.width, encoding="rgb8", step=img.width * 3, data=out.tobytes())
        msg.header = img.header
        self.pub_overlay.publish(msg)

    def publish_markers(self, stamp, frame, xyz, cls, line):
        out = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for k in range(1, int(line.max(initial=0)) + 1):
            sel = line == k
            p = xyz[sel]
            c = p[np.argmin(np.linalg.norm(p - p.mean(axis=0), axis=1))]
            r, g, b = line_colours(np.array([k]))[0] / 255.0
            m = Marker(ns="mask_clusters", id=k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                       text=f"{self.names.get(int(cls[sel][0]), int(cls[sel][0]))} {k} ({sel.sum()})")
            m.header.stamp, m.header.frame_id = stamp, frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, c)
            m.pose.orientation.w = 1.0
            m.scale.z = 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = float(r), float(g), float(b), 1.0
            out.markers.append(m)
        self.pub_markers.publish(out)


def main():
    rclpy.init()
    rclpy.spin(MaskProjector())
