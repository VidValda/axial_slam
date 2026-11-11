# axial_slam

Camera-LiDAR fusion for axial landmarks (pipes, ducts, pillars) as 4-DoF infinite lines for DPS-SLAM.

## axial_perception

`mask_projector` turns instance masks and a LiDAR cloud into one point cluster per mask:

1. Each scan is moved into `odom_frame` with TF at its stamp, and the last `n_scans` are kept.
2. For each mask image, the buffer is moved into `camera_frame` with TF at the image stamp and projected with the
   camera intrinsics. Scans and images wait in short queues until odometry TF covers their stamp.
3. Hidden points are dropped with `occlusion_method`:
   - `zbuffer`: points more than `occlusion_tol` behind the nearest point within `occlusion_window` pixels.
   - `mask`: per mask, points are grouped by linking them in 3D within `occlusion_link`. The largest group is the
     object; other groups are dropped when they lie more than `occlusion_behind` behind it within `occlusion_radius`
     pixels (background seen through the mask). Keeps more of slanted and receding pipes than `zbuffer`.
   - `none`.
4. Points on masks of `axial_classes` are kept; masks with fewer than `min_points` hits are dropped.
5. Each mask is split into straight lines with `split_method` (`src/segments.cpp`). `ransac` fits lines one after
   the other within `split_tol` of the class. For every method, two lines are merged when one line fits 95% of their
   points within `split_tol`, then each line is cut where its points are more than `split_gap` apart. Lines with
   fewer than `split_min_points` points or shorter than `split_min_length` are dropped and their points get `line` 0.
   Short lines give poor directions: their points cover only one or two scan lines across the pipe.

   The other methods are there to benchmark `ransac` against:
   - `pearl`: PEARL-style; RANSAC and random proposals, then points are re-assigned by minimising distance cost +
     k-NN smoothness + a cost per line.
   - `jlinkage`: J-Linkage; points are merged by the Jaccard distance of the proposals they fit. Too slow online.
   - `mst_dp`: points ordered along the longest path of their minimum spanning tree, then split with segmented least
     squares.

The node is C++ (`src/mask_projector.cpp`). The ROS-free maths lives in the other `src/*.cpp` and is covered by
`colcon test`.

`split_method: none` turns the splitting off: every mask is one line.

`axial_perception/benchmarks` scores the node against a ground-truth world file (`--gt`) with the params of a
params file (`-c`), run with `ros2 run axial_perception <benchmark>`: `mask_clusters_gt` (points off their surface),
`line_segments_gt` (split methods, live or from a saved `.bin`), `record_clusters` and `landmark_map_gt` (the
landmark map against the pillars), `record_raw` and `turn_dataset` (raw frames of pipe and duct turns, scored
offline). Datasets are zlib-compressed `.bin` files, passed with `-d`, `--raw`, `--clusters` or `--load`, and are kept
outside the repo.

| Topic | Type | |
|---|---|---|
| `labels_topic` | `sensor_msgs/Image` | rgb8 instance map, class in B and instance id in `(G << 8) \| R` (Gazebo instance camera) |
| `cloud_topic` | `sensor_msgs/PointCloud2` | LiDAR scans |
| `camera_info_topic` | `sensor_msgs/CameraInfo` | intrinsics of the mask camera |
| `image_topic` | `sensor_msgs/Image` | rgb8 image, only for the overlay |
| `/axial/mask_clusters` | `sensor_msgs/PointCloud2` | `x y z class_id segment line rgb` in `camera_frame`; `segment` numbers the masks and `line` the straight lines of one frame from 1 (0 = on no line) |
| `/axial/mask_overlay` | `sensor_msgs/Image` | projected points on the image, coloured by line (grey if dropped) |
| `/axial/mask_cluster_markers` | `visualization_msgs/MarkerArray` | one label per line |

Parameters, with defaults:

| Parameter | Default | |
|---|---|---|
| `camera_frame` | mask image frame | optical frame (x right, y down, z forward) to project in |
| `odom_frame` | `odom` | frame the scans are accumulated in |
| `n_scans` | 5 | scans kept |
| `depth_min`, `depth_max` | 0.1, 30.0 | depth range in m |
| `occlusion_method` | `zbuffer` | `zbuffer`, `mask` or `none` |
| `occlusion_window`, `occlusion_tol` | 9, 0.1 | `zbuffer`: window in px and tolerance in m |
| `occlusion_link`, `occlusion_radius`, `occlusion_behind` | 0.1, 15, 0.3 | `mask`: 3D link in m, image radius in px, depth behind the object in m |
| `axial_classes`, `class_names` | `[4, 6, 8]`, `[pipe, duct, pillar]` | classes kept, and their names for the labels |
| `min_points` | 5 | smallest cluster |
| `split_method` | `ransac` | `none`, `ransac`, `pearl`, `jlinkage`, `mst_dp` (`split_methods()` in `src/segments.cpp`) |
| `split_tol` | `[0.07, 0.25, 0.3]` | point-to-line tolerance in m per class of `axial_classes` |
| `split_gap` | `[0.5, 1.5, 1.0]` | gap in m along a line that cuts it, per class (0 turns it off) |
| `split_min_points`, `split_min_length` | 5, 0.5 | smallest line in points and in m |
| `max_rate_hz` | 10.0 | mask images processed per second of stamp time |

```bash
ros2 launch axial_perception mask_projection.launch.py params_file:=<yaml> [rviz:=true fixed_frame:=<frame>] [use_sim_time:=true]
```

`rviz:=true` opens `rviz/mask_projection.rviz` (clusters, labels and overlay) with `fixed_frame` (default `odom`) as the fixed frame.

The node logs the input and kept point counts, masks, lines and the split and processing times every 5 s.
