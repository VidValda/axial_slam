# axial_slam

Camera-LiDAR fusion for axial landmarks (pipes, ducts, pillars) as 4-DoF infinite lines for DPS-SLAM.

## axial_perception

`mask_projector` turns instance masks and a LiDAR cloud into one point cluster per mask:

1. Each scan is moved into `odom_frame` with TF at its stamp, and the last `n_scans` are kept.
2. For each mask image, the buffer is moved into `camera_frame` with TF at the image stamp and projected with the
   camera intrinsics. Scans and images wait in short queues until odometry TF covers their stamp.
3. Points hidden behind a nearer point within `occlusion_window` pixels (by more than `occlusion_tol`) are dropped.
4. Points on masks of `axial_classes` are kept; masks with fewer than `min_points` hits are dropped.

The ROS-free maths lives in `axial_perception/core` and is covered by `colcon test`.

| Topic | Type | |
|---|---|---|
| `labels_topic` | `sensor_msgs/Image` | rgb8 instance map, class in B and instance id in `(G << 8) \| R` (Gazebo instance camera) |
| `cloud_topic` | `sensor_msgs/PointCloud2` | LiDAR scans |
| `camera_info_topic` | `sensor_msgs/CameraInfo` | intrinsics of the mask camera |
| `image_topic` | `sensor_msgs/Image` | rgb8 image, only for the overlay |
| `/axial/mask_clusters` | `sensor_msgs/PointCloud2` | `x y z class_id segment rgb` in `camera_frame`; `segment` numbers the masks of one frame from 1 |
| `/axial/mask_overlay` | `sensor_msgs/Image` | projected points on the image, coloured by segment (grey if dropped) |
| `/axial/mask_cluster_markers` | `visualization_msgs/MarkerArray` | one label per cluster |

Parameters, with defaults:

| Parameter | Default | |
|---|---|---|
| `camera_frame` | mask image frame | optical frame (x right, y down, z forward) to project in |
| `odom_frame` | `odom` | frame the scans are accumulated in |
| `n_scans` | 5 | scans kept |
| `depth_min`, `depth_max` | 0.1, 30.0 | depth range in m |
| `occlusion_window`, `occlusion_tol` | 9, 0.1 | z-buffer window in px (0 turns it off) and tolerance in m |
| `axial_classes`, `class_names` | `[4, 6, 8]`, `[pipe, duct, pillar]` | classes kept, and their names for the labels |
| `min_points` | 5 | smallest cluster |
| `max_rate_hz` | 10.0 | mask images processed per second of stamp time |

```bash
ros2 launch axial_perception mask_projection.launch.py params_file:=<yaml> [rviz_config:=<rviz>] [use_sim_time:=true]
```

The node logs the input and kept point counts and the processing time every 5 s.
