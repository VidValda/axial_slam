import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points_numpy
from tf2_ros import Buffer, TransformException, TransformListener

from axial_perception.core.camera import project, quat_to_matrix, transform
from axial_perception.core.clusters import label_points
from axial_perception.core.masks import decode_gz_instance

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40],
], np.uint8)

CLUSTER_DTYPE = np.dtype([(n, np.float32) for n in ("x", "y", "z", "class_id", "segment")])
CLUSTER_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(CLUSTER_DTYPE.names)]


class MaskProjector(Node):
    def __init__(self):
        super().__init__("mask_projector")
        p = {
            "labels_topic": "labels_map",
            "cloud_topic": "points",
            "camera_info_topic": "camera_info",
            "image_topic": "image_raw",
            "camera_frame": "",
            "depth_min": 0.1,
            "depth_max": 30.0,
            "axial_classes": [4, 6, 8],
            "min_points": 5,
            "max_rate_hz": 10.0,
        }
        for name, default in p.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in p}

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.K = None
        self.cloud = None
        self.image = None
        self.last_stamp = None

        sensor = qos_profile_sensor_data
        self.create_subscription(CameraInfo, self.p["camera_info_topic"], self.on_info, sensor)
        self.create_subscription(PointCloud2, self.p["cloud_topic"], self.on_cloud, sensor)
        self.create_subscription(Image, self.p["image_topic"], self.on_image, sensor)
        self.create_subscription(Image, self.p["labels_topic"], self.on_labels, sensor)
        self.pub_clusters = self.create_publisher(PointCloud2, "/axial/mask_clusters", 5)
        self.pub_overlay = self.create_publisher(Image, "/axial/mask_overlay", 5)

    def on_info(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_cloud(self, msg):
        self.cloud = msg

    def on_image(self, msg):
        self.image = msg

    def on_labels(self, msg):
        if self.K is None or self.cloud is None:
            return
        stamp = Time.from_msg(msg.header.stamp)
        if self.last_stamp is not None and (stamp - self.last_stamp).nanoseconds < 1e9 / self.p["max_rate_hz"]:
            return
        self.last_stamp = stamp

        frame = self.p["camera_frame"] or msg.header.frame_id
        try:
            tf = self.tf.lookup_transform(frame, self.cloud.header.frame_id, Time())
        except TransformException as e:
            self.get_logger().warn(f"no TF {frame} <- {self.cloud.header.frame_id}: {e}", throttle_duration_sec=5.0)
            return
        q, t = tf.transform.rotation, tf.transform.translation
        xyz = read_points_numpy(self.cloud, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
        pts = transform(xyz, quat_to_matrix((q.x, q.y, q.z, q.w)), (t.x, t.y, t.z))

        class_img, mask_img = decode_gz_instance(msg.data, msg.height, msg.width)
        u, v, idx = project(pts, self.K, msg.width, msg.height, self.p["depth_min"], self.p["depth_max"])
        kept, cls, seg = label_points(u, v, idx, class_img, mask_img, self.p["axial_classes"], self.p["min_points"])

        self.publish_clusters(msg.header.stamp, frame, pts[kept], cls, seg)
        if self.pub_overlay.get_subscription_count() > 0 and self.image is not None:
            self.publish_overlay(u, v, idx, kept, seg)

    def publish_clusters(self, stamp, frame, xyz, cls, seg):
        arr = np.empty(len(xyz), CLUSTER_DTYPE)
        arr["x"], arr["y"], arr["z"] = xyz.T
        arr["class_id"], arr["segment"] = cls, seg
        out = PointCloud2(height=1, width=len(arr), fields=CLUSTER_FIELDS, is_bigendian=False,
                          point_step=CLUSTER_DTYPE.itemsize, row_step=CLUSTER_DTYPE.itemsize * len(arr),
                          is_dense=True, data=arr.tobytes())
        out.header.stamp, out.header.frame_id = stamp, frame
        self.pub_clusters.publish(out)

    def publish_overlay(self, u, v, idx, kept, seg):
        img = self.image
        if img.encoding != "rgb8":
            self.get_logger().warn(f"overlay needs rgb8, got {img.encoding}", throttle_duration_sec=5.0)
            return
        out = np.frombuffer(img.data, np.uint8).reshape(img.height, img.width, 3) // 2
        colour = np.full((len(idx), 3), 160, np.uint8)
        colour[np.searchsorted(idx, kept)] = PALETTE[(seg - 1) % len(PALETTE)]
        for du in (0, 1):
            for dv in (0, 1):
                out[np.minimum(v + dv, img.height - 1), np.minimum(u + du, img.width - 1)] = colour
        msg = Image(height=img.height, width=img.width, encoding="rgb8", step=img.width * 3, data=out.tobytes())
        msg.header = img.header
        self.pub_overlay.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(MaskProjector())
