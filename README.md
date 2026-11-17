# axial_slam

Camera-LiDAR fusion for axial landmarks (pipes, ducts, pillars) as 4-DoF infinite lines for DPS-SLAM.

## axial_segmentation

YOLO instance segmentation (ultralytics seg model exported to ONNX with `nms=False`) run with ONNX Runtime. It
publishes the `labels_map` that `axial_frontend` reads, so the same front end runs on Gazebo ground truth or on
the model, on simulated or real images.

The class table is the node's `class_ids`, `class_names` and `class_geometry` (`axis` or `point`): YOLO output `i`
is `class_names[i]` and is written to `labels_map` as `class_ids[i]`. The node refuses a model whose `names`
metadata differs from `class_names`. Overlapping masks go to the higher score; instance ids count from 1 per
class by score.

| Topic | Type | |
|---|---|---|
| `image_topic` | `sensor_msgs/Image` or `CompressedImage` (name ending in `/compressed`) | camera |
| `labels_topic` | `sensor_msgs/Image` | rgb8 `labels_map`, same header as the image |
| `detections_topic` | `vision_msgs/Detection2DArray` | box, class name and score per instance |
| `overlay_topic` | `sensor_msgs/Image` | masks over the image, only when subscribed (`""` disables) |

Other parameters:
- `model_path`: a file, or `package://<package>/<path>` inside an installed package's share directory. The
  default is the model shipped in `models/`.
- `device`: `cuda` (default) runs ONNX Runtime's CUDA provider and falls back to `cpu` with a warning when it
  cannot load; the startup log names the one in use.
- `conf` (0.4), `iou` (0.6), `max_det` (100).
- `threads` (4): CPU threads, which sleep when idle.
- `max_rate_hz` (10): images closer in stamp are skipped, as in `axial_frontend`; 0 takes every image.
- `image_reliable` (false): best-effort image subscription.

Only the newest image is kept while the model runs.

`models/sherec_seg_sim_p2b_n.onnx` is a yolo11n-seg at 640 px for pillar, pipe, duct, fire_extinguisher and door
(class ids 8, 4, 6, 9, 10 in the sherec sim). It was trained in pillar-perception-ws on images of randomized sherec
worlds (layout, colours per pillar, lighting, Gazebo Fuel doors and extinguishers) and scores, on worlds held out
from training, mask mAP50-95 0.65 for pillars with 0.1% of pillars taken for doors. It runs in about 10 ms per
image on an RTX 4070 laptop GPU.

`seg_infer <model.onnx> <out_dir> <image>...` runs the same code on image files (labels PNGs, detections, timing);
on the CPU its output matches ultralytics `predict(retina_masks=True)` on the same ONNX model. ONNX Runtime is found
under `ONNXRUNTIME_ROOT` (default `/opt/onnxruntime`; the project Dockerfile installs the GPU build, which needs CUDA 12
and cuDNN 9 on the library path).

## axial_perception

`axial_frontend` turns instance masks and a LiDAR cloud into one point cluster per mask:

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

The node is C++ (`src/axial_frontend.cpp`). The ROS-free maths lives in the other `src/*.cpp` and is covered by
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
ros2 launch axial_perception axial_frontend.launch.py params_file:=<yaml> [rviz:=true fixed_frame:=<frame>] [use_sim_time:=true]
```

`rviz:=true` opens `rviz/axial_frontend.rviz` (clusters, labels and overlay) with `fixed_frame` (default `odom`) as the fixed frame.

The node logs the input and kept point counts, masks, lines and the split and processing times every 5 s.
