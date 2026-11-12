#include "axial_perception/axial_frontend.hpp"

#include <algorithm>
#include <charconv>
#include <exception>
#include <stdexcept>
#include <utility>

#include <geometry_msgs/msg/point.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer_interface.h>

#include "axial_perception/cloud_msg.hpp"
#include "axial_perception/masks.hpp"
#include "axial_perception/segments.hpp"

namespace axial_perception
{

namespace
{

constexpr std::array<std::array<uint8_t, 3>, 12> kPalette{{
  {230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200}, {245, 130, 48}, {145, 30, 180},
  {70, 240, 240}, {240, 50, 230}, {210, 245, 60}, {250, 190, 212}, {0, 128, 128}, {170, 110, 40}}};

std::array<uint8_t, 3> colour(int line)
{
  if (line <= 0) {
    return {160, 160, 160};
  }
  return kPalette[static_cast<size_t>(line - 1) % kPalette.size()];
}

geometry_msgs::msg::Point point(const Eigen::Vector3d & p)
{
  geometry_msgs::msg::Point out;
  out.x = p[0];
  out.y = p[1];
  out.z = p[2];
  return out;
}

template<class T>
std::map<int, T> zip(const std::vector<int64_t> & keys, const std::vector<T> & values)
{
  std::map<int, T> out;
  for (size_t i = 0; i < std::min(keys.size(), values.size()); ++i) {
    out.emplace(static_cast<int>(keys[i]), values[i]);
  }
  return out;
}

double positive(double value, const std::string & name)
{
  if (!(value > 0)) {
    throw std::invalid_argument(name + " must be positive");
  }
  return value;
}

}  // namespace

AxialFrontend::AxialFrontend(const rclcpp::NodeOptions & options)
: rclcpp::Node("axial_frontend", options),
  scans_(static_cast<size_t>(positive(static_cast<double>(declare_parameter<int64_t>("n_scans", 5)), "n_scans"))),
  landmarks_(
    declare_parameter<double>("map_spawn_threshold", 0.5), declare_parameter<double>("map_min_separation", 1.0),
    declare_parameter<double>("map_max_angle", 0.35), declare_parameter<double>("map_voxel", 0.03),
    declare_parameter<double>("map_min_elongation", 3.0))
{
  const auto labels_topic = declare_parameter<std::string>("labels_topic", "labels_map");
  const auto cloud_topic = declare_parameter<std::string>("cloud_topic", "points");
  const auto camera_info_topic = declare_parameter<std::string>("camera_info_topic", "camera_info");
  const auto image_topic = declare_parameter<std::string>("image_topic", "image_raw");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
  projector_.depth_min = declare_parameter<double>("depth_min", 0.1);
  projector_.depth_max = declare_parameter<double>("depth_max", 30.0);
  projector_.occlusion_method = declare_parameter<std::string>("occlusion_method", "zbuffer");
  projector_.occlusion_window = static_cast<int>(declare_parameter<int64_t>("occlusion_window", 9));
  projector_.occlusion_tol = declare_parameter<double>("occlusion_tol", 0.1);
  projector_.occlusion_link = declare_parameter<double>("occlusion_link", 0.1);
  projector_.occlusion_radius = static_cast<double>(declare_parameter<int64_t>("occlusion_radius", 15));
  projector_.occlusion_behind = declare_parameter<double>("occlusion_behind", 0.3);
  const auto classes = declare_parameter<std::vector<int64_t>>("axial_classes", {4, 6, 8});
  const auto class_names = declare_parameter<std::vector<std::string>>("class_names", {"pipe", "duct", "pillar"});
  const auto class_radius = declare_parameter<std::vector<double>>("class_radius", {0.0});
  projector_.min_points = static_cast<int>(declare_parameter<int64_t>("min_points", 5));
  split_method_ = declare_parameter<std::string>("split_method", "ransac");
  const auto split_tol = declare_parameter<std::vector<double>>("split_tol", {0.07, 0.25, 0.3});
  const auto split_gap = declare_parameter<std::vector<double>>("split_gap", {0.5, 1.5, 1.0});
  split_min_points_ = static_cast<int>(declare_parameter<int64_t>("split_min_points", 5));
  split_min_length_ = declare_parameter<double>("split_min_length", 0.5);
  max_rate_hz_ = positive(declare_parameter<double>("max_rate_hz", 10.0), "max_rate_hz");
  map_min_neighbours_ = static_cast<int>(declare_parameter<int64_t>("map_min_neighbours", 8));
  const double map_rate_hz = positive(declare_parameter<double>("map_rate_hz", 1.0), "map_rate_hz");
  map_frame_ = declare_parameter<std::string>("map_frame", "");
  const auto slam_optimized_topic = declare_parameter<std::string>("slam_optimized_topic", "");
  detection_frame_ = declare_parameter<std::string>("detection_frame", "");
  detection_frame_id_ = declare_parameter<std::string>("detection_frame_id", "");
  const double dir_std = declare_parameter<double>("line_dir_std", 0.05);
  const double offset_std = declare_parameter<double>("line_offset_std", 0.1);

  check_occlusion_method(projector_.occlusion_method);
  split_method(split_method_);
  if (projector_.occlusion_window < 1) {
    throw std::invalid_argument("occlusion_window must be at least 1");
  }
  if (split_tol.size() < classes.size() || split_gap.size() < classes.size()) {
    throw std::invalid_argument("split_tol and split_gap need one value per axial class");
  }
  projector_.classes.assign(classes.begin(), classes.end());
  names_ = zip(classes, class_names);
  radius_ = zip(classes, class_radius);
  tols_ = zip(classes, split_tol);
  gaps_ = zip(classes, split_gap);
  if (map_frame_.empty()) {
    map_frame_ = odom_frame_;
  }
  const std::array<double, 4> var{dir_std * dir_std, dir_std * dir_std, offset_std * offset_std,
    offset_std * offset_std};
  for (size_t i = 0; i < 4; ++i) {
    line_cov_[5 * i] = var[i];
  }

  tf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_, this);

  const auto sensor = rclcpp::SensorDataQoS();
  sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, sensor, [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr & m) {on_info(m);});
  sub_cloud_ = create_subscription<PointCloud2>(
    cloud_topic, sensor, [this](const PointCloud2::ConstSharedPtr & m) {on_cloud(m);});
  sub_image_ = create_subscription<Image>(
    image_topic, sensor, [this](const Image::ConstSharedPtr & m) {on_image(m);});
  sub_labels_ = create_subscription<Image>(
    labels_topic, sensor, [this](const Image::ConstSharedPtr & m) {on_labels(m);});
  pub_clusters_ = create_publisher<PointCloud2>("/axial/mask_clusters", 5);
  pub_overlay_ = create_publisher<Image>("/axial/mask_overlay", 5);
  pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/axial/mask_cluster_markers", 5);
  pub_map_ = create_publisher<PointCloud2>("/axial/landmark_map", 1);
  pub_map_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/axial/landmark_markers", 1);
  pub_detections_ = create_publisher<Detections>("/axial/line_detections", 10);
  if (!slam_optimized_topic.empty()) {
    sub_optimized_ = create_subscription<Detections>(
      slam_optimized_topic, 10, [this](const Detections::ConstSharedPtr & m) {on_optimized(m);});
  }
  map_timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Duration::from_seconds(1.0 / map_rate_hz), [this]() {publish_map();});
  stats_since_ = std::chrono::steady_clock::now();
}

std::string AxialFrontend::name_of(int cls) const
{
  const auto it = names_.find(cls);
  return it == names_.end() ? std::to_string(cls) : it->second;
}

void AxialFrontend::on_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  K_ = Intrinsics{msg->k[0], msg->k[4], msg->k[2], msg->k[5]};
}

void AxialFrontend::on_cloud(const PointCloud2::ConstSharedPtr & msg)
{
  if (pending_scans_.size() == 20) {
    pending_scans_.pop_front();
  }
  pending_scans_.push_back(msg);
  flush();
}

void AxialFrontend::on_image(const Image::ConstSharedPtr & msg)
{
  image_ = msg;
}

void AxialFrontend::on_labels(const Image::ConstSharedPtr & msg)
{
  const rclcpp::Time stamp(msg->header.stamp, RCL_ROS_TIME);
  if (!last_stamp_ || static_cast<double>((stamp - *last_stamp_).nanoseconds()) >= 1e9 / max_rate_hz_) {
    last_stamp_ = stamp;
    if (pending_labels_.size() == 10) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no TF for 10 images in a row");
      pending_labels_.pop_front();
    }
    pending_labels_.push_back(msg);
  }
  flush();
}

void AxialFrontend::flush()
{
  flush_scans();
  flush_labels();
}

void AxialFrontend::flush_scans()
{
  while (!pending_scans_.empty()) {
    const PointCloud2::ConstSharedPtr msg = pending_scans_.front();
    const auto tf = lookup(odom_frame_, msg->header.frame_id, msg->header.stamp);
    if (!tf) {
      return;
    }
    pending_scans_.pop_front();
    try {
      const std::vector<double> xyz = read_fields(*msg, {"x", "y", "z"}, true);
      const Cloud pts = Eigen::Map<const Cloud>(xyz.data(), 3, static_cast<Eigen::Index>(xyz.size() / 3));
      scans_.add(tf->apply(pts));
    } catch (const std::invalid_argument & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "scan dropped: %s", e.what());
    }
  }
}

void AxialFrontend::flush_labels()
{
  if (!K_ || scans_.size() == 0) {
    return;
  }
  for (size_t i = pending_labels_.size(); i-- > 0; ) {
    const Image::ConstSharedPtr msg = pending_labels_[i];
    const std::string frame = camera_frame_.empty() ? msg->header.frame_id : camera_frame_;
    const auto tf = lookup(frame, odom_frame_, msg->header.stamp);
    const auto to_map = lookup(map_frame_, odom_frame_, msg->header.stamp);
    if (tf && to_map) {
      pending_labels_.erase(pending_labels_.begin(), pending_labels_.begin() + static_cast<std::ptrdiff_t>(i) + 1);
      try {
        process(*msg, frame, *tf, *to_map);
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "labels dropped: %s", e.what());
      }
      return;
    }
  }
}

void AxialFrontend::process(const Image & msg, const std::string & frame, const Rigid & tf, const Rigid & to_map)
{
  const auto start = std::chrono::steady_clock::now();
  const Cloud odom = scans_.points();
  const Cloud pts = tf.apply(odom);
  const InstanceImage img = decode_gz_instance(msg.data, static_cast<int>(msg.height), static_cast<int>(msg.width),
      msg.step);
  const MaskPoints mp = mask_points(pts, img, *K_, projector_);
  const Labelled & l = mp.labelled;
  const Cloud kept = select(pts, l.idx);
  const auto split_start = std::chrono::steady_clock::now();
  const Labels line = split_clusters(kept, l.cls, l.seg, split_method_, tols_, split_min_points_, gaps_,
      split_min_length_);
  const double split_dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - split_start).count();
  const auto ids = landmarks_.update(to_map.apply(select(odom, l.idx)), l.cls, line);
  if (!ids.empty()) {
    publish_detections(msg.header.stamp, frame, kept, l.cls, line, ids);
  }
  publish_clusters(msg.header.stamp, frame, kept, l.cls, l.seg, line);
  if (pub_overlay_->get_subscription_count() > 0 && image_) {
    publish_overlay(mp.proj, l.idx, line);
  }
  if (pub_markers_->get_subscription_count() > 0) {
    publish_markers(msg.header.stamp, frame, kept, l.cls, line);
  }
  Stats s;
  s.pts = static_cast<double>(pts.cols());
  s.kept = static_cast<double>(kept.cols());
  s.masks = l.seg.empty() ? 0 : *std::max_element(l.seg.begin(), l.seg.end());
  s.lines = line.empty() ? 0 : std::max(0, *std::max_element(line.begin(), line.end()));
  s.split = split_dt;
  s.total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  log_stats(s);
}

void AxialFrontend::publish_detections(
  const builtin_interfaces::msg::Time & stamp, const std::string & frame, const Cloud & xyz,
  const std::vector<int> & cls, const Labels & line, const std::map<int, int> & ids)
{
  const std::string target = detection_frame_.empty() ? frame : detection_frame_;
  const auto tf = target == frame ? std::optional<Rigid>(Rigid{}) : lookup(target, frame, stamp);
  if (!tf) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "no TF %s -> %s, line detections dropped", frame.c_str(), target.c_str());
    return;
  }
  const auto members = groups(line);
  Detections out;
  out.header.stamp = stamp;
  out.header.frame_id = detection_frame_id_.empty() ? target : detection_frame_id_;
  for (const auto & [k, gid] : ids) {
    const Indices & sel = members.at(k);
    const int c = cls[static_cast<size_t>(sel.front())];
    const Line fit = fit_line(select(xyz, sel));
    const auto r = radius_.find(c);
    const Eigen::Vector3d p = tf->apply(axis_from_surface(fit.c, fit.d, r == radius_.end() ? 0.0 : r->second));
    const Eigen::Vector3d d = tf->R * fit.d;
    dps_slam_msgs::msg::DetectionWithID det;
    det.label = name_of(c);
    det.id = det.label + "_" + std::to_string(gid);
    det.geometry.type = dps_slam_msgs::msg::Geometry::LINE3D;
    det.geometry.line3d.point = point(p);
    det.geometry.line3d.direction.x = d[0];
    det.geometry.line3d.direction.y = d[1];
    det.geometry.line3d.direction.z = d[2];
    std::copy(line_cov_.begin(), line_cov_.end(), det.geometry.line3d.covariance.begin());
    out.detections.push_back(std::move(det));
  }
  pub_detections_->publish(out);
}

void AxialFrontend::on_optimized(const Detections::ConstSharedPtr & msg)
{
  const auto tf = lookup(map_frame_, msg->header.frame_id, msg->header.stamp);
  if (!tf) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "no TF %s -> %s, landmarks not snapped", msg->header.frame_id.c_str(),
      map_frame_.c_str());
    return;
  }
  std::map<std::string, int> classes;
  for (const auto & [c, name] : names_) {
    classes[name] = c;
  }
  for (const auto & det : msg->detections) {
    const size_t cut = det.id.rfind('_');
    const std::string name = cut == std::string::npos ? "" : det.id.substr(0, cut);
    const std::string id = cut == std::string::npos ? det.id : det.id.substr(cut + 1);
    int gid = 0;
    const auto [end, err] = std::from_chars(id.data(), id.data() + id.size(), gid);
    if (id.empty() || err != std::errc() || end != id.data() + id.size() || id[0] == '-' || id[0] == '+') {
      continue;
    }
    const auto lm = landmarks_.landmarks().find(gid);
    const auto c = classes.find(name);
    if (det.geometry.type != dps_slam_msgs::msg::Geometry::LINE3D || lm == landmarks_.landmarks().end() ||
      c == classes.end() || c->second != lm->second.class_id)
    {
      continue;
    }
    const auto & l = det.geometry.line3d;
    landmarks_.snap(
      gid, tf->apply(Eigen::Vector3d(l.point.x, l.point.y, l.point.z)),
      tf->R * Eigen::Vector3d(l.direction.x, l.direction.y, l.direction.z));
  }
}

void AxialFrontend::publish_map()
{
  const auto & lms = landmarks_.landmarks();
  if (lms.empty()) {
    return;
  }
  std_msgs::msg::Header header;
  header.stamp = now();
  header.frame_id = map_frame_;
  std::map<int, Cloud> dense;
  for (const auto & [gid, lm] : lms) {
    dense.emplace(gid, landmarks_.dense_points(gid, map_min_neighbours_));
  }
  if (pub_map_->get_subscription_count() > 0) {
    std::vector<float> data;
    for (const auto & [gid, lm] : lms) {
      const auto rgb = colour(gid);
      const float packed = pack_rgb(rgb[0], rgb[1], rgb[2]);
      const Cloud & pts = dense.at(gid);
      for (Eigen::Index i = 0; i < pts.cols(); ++i) {
        data.insert(data.end(), {static_cast<float>(pts(0, i)), static_cast<float>(pts(1, i)),
          static_cast<float>(pts(2, i)), static_cast<float>(lm.class_id), static_cast<float>(gid), packed});
      }
    }
    pub_map_->publish(float_cloud({"x", "y", "z", "class_id", "landmark", "rgb"}, data, header));
  }
  if (pub_map_markers_->get_subscription_count() > 0) {
    visualization_msgs::msg::MarkerArray out;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    out.markers.push_back(clear);
    for (const auto & [gid, lm] : lms) {
      const Cloud & pts = dense.at(gid);
      if (pts.cols() == 0) {
        continue;
      }
      const Eigen::VectorXd t = lm.direction.transpose() * (pts.colwise() - lm.point);
      const Eigen::Vector3d lo = lm.point + t.minCoeff() * lm.direction;
      const Eigen::Vector3d hi = lm.point + t.maxCoeff() * lm.direction;
      const auto rgb = colour(gid);
      visualization_msgs::msg::Marker axis;
      axis.ns = "landmark_axes";
      axis.type = visualization_msgs::msg::Marker::LINE_LIST;
      axis.points = {point(lo), point(hi)};
      axis.scale.x = 0.05;
      visualization_msgs::msg::Marker label;
      label.ns = "landmark_labels";
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.text = name_of(lm.class_id) + " " + std::to_string(gid) + " (" + std::to_string(lm.hits) + ")";
      label.pose.position = point(hi);
      label.pose.position.z += 0.2;
      label.scale.z = 0.25;
      for (auto * m : {&axis, &label}) {
        m->id = gid;
        m->action = visualization_msgs::msg::Marker::ADD;
        m->header = header;
        m->pose.orientation.w = 1.0;
        m->color.r = static_cast<float>(rgb[0] / 255.0);
        m->color.g = static_cast<float>(rgb[1] / 255.0);
        m->color.b = static_cast<float>(rgb[2] / 255.0);
        m->color.a = 1.0f;
        out.markers.push_back(*m);
      }
    }
    pub_map_markers_->publish(out);
  }
}

void AxialFrontend::log_stats(const Stats & frame)
{
  stats_.pts += frame.pts;
  stats_.kept += frame.kept;
  stats_.masks += frame.masks;
  stats_.lines += frame.lines;
  stats_.split += frame.split;
  stats_.total += frame.total;
  stats_.max_total = std::max(stats_.max_total, frame.total);
  ++stats_.frames;
  if (std::chrono::steady_clock::now() - stats_since_ < std::chrono::seconds(5)) {
    return;
  }
  const double n = stats_.frames;
  RCLCPP_INFO(
    get_logger(), "%d frames: %.0f pts in, %.0f kept in %.1f masks, %.1f lines, split %.1f ms, %.1f ms mean, "
    "%.1f ms max", stats_.frames, stats_.pts / n, stats_.kept / n, stats_.masks / n, stats_.lines / n,
    stats_.split / n * 1e3, stats_.total / n * 1e3, stats_.max_total * 1e3);
  stats_ = Stats{};
  stats_since_ = std::chrono::steady_clock::now();
}

std::optional<Rigid> AxialFrontend::lookup(
  const std::string & target, const std::string & source, const builtin_interfaces::msg::Time & stamp) const
{
  const tf2::TimePoint t = tf2_ros::fromMsg(stamp);
  if (!tf_->canTransform(target, source, t)) {
    return std::nullopt;
  }
  try {
    const auto tr = tf_->lookupTransform(target, source, t).transform;
    return Rigid{quat_to_matrix(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
      {tr.translation.x, tr.translation.y, tr.translation.z}};
  } catch (const tf2::TransformException &) {
    return std::nullopt;
  }
}

void AxialFrontend::publish_clusters(
  const builtin_interfaces::msg::Time & stamp, const std::string & frame, const Cloud & xyz,
  const std::vector<int> & cls, const std::vector<int> & seg, const Labels & line)
{
  std::vector<float> data;
  data.reserve(static_cast<size_t>(xyz.cols()) * 7);
  for (Eigen::Index i = 0; i < xyz.cols(); ++i) {
    const size_t j = static_cast<size_t>(i);
    const auto rgb = colour(line[j]);
    data.insert(data.end(), {static_cast<float>(xyz(0, i)), static_cast<float>(xyz(1, i)),
      static_cast<float>(xyz(2, i)), static_cast<float>(cls[j]), static_cast<float>(seg[j]),
      static_cast<float>(line[j]), pack_rgb(rgb[0], rgb[1], rgb[2])});
  }
  std_msgs::msg::Header header;
  header.stamp = stamp;
  header.frame_id = frame;
  pub_clusters_->publish(float_cloud({"x", "y", "z", "class_id", "segment", "line", "rgb"}, data, header));
}

void AxialFrontend::publish_overlay(const Projection & proj, const Indices & kept, const Labels & line)
{
  const Image & img = *image_;
  if (img.encoding != "rgb8") {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "overlay needs rgb8, got %s", img.encoding.c_str());
    return;
  }
  const size_t w = img.width;
  const size_t h = img.height;
  if (w == 0 || h == 0 || img.step < 3 * w || img.data.size() < static_cast<size_t>(img.step) * h) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "overlay image smaller than its size and step");
    return;
  }
  Image out;
  out.header = img.header;
  out.height = img.height;
  out.width = img.width;
  out.encoding = "rgb8";
  out.step = static_cast<uint32_t>(3 * w);
  out.data.resize(3 * w * h);
  for (size_t r = 0; r < h; ++r) {
    for (size_t c = 0; c < 3 * w; ++c) {
      out.data[r * 3 * w + c] = img.data[r * img.step + c] / 2;
    }
  }
  std::vector<std::array<uint8_t, 3>> col(proj.size(), colour(0));
  for (size_t k = 0; k < kept.size(); ++k) {
    const auto it = std::lower_bound(proj.idx.begin(), proj.idx.end(), kept[k]);
    if (it != proj.idx.end() && *it == kept[k]) {
      col[static_cast<size_t>(it - proj.idx.begin())] = colour(line[k]);
    }
  }
  for (size_t i = 0; i < proj.size(); ++i) {
    for (size_t du = 0; du < 2; ++du) {
      for (size_t dv = 0; dv < 2; ++dv) {
        const size_t u = std::min(static_cast<size_t>(proj.u[i]) + du, w - 1);
        const size_t v = std::min(static_cast<size_t>(proj.v[i]) + dv, h - 1);
        std::copy(col[i].begin(), col[i].end(), out.data.begin() + static_cast<std::ptrdiff_t>(3 * (v * w + u)));
      }
    }
  }
  pub_overlay_->publish(out);
}

void AxialFrontend::publish_markers(
  const builtin_interfaces::msg::Time & stamp, const std::string & frame, const Cloud & xyz,
  const std::vector<int> & cls, const Labels & line)
{
  visualization_msgs::msg::MarkerArray out;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  out.markers.push_back(clear);
  for (const auto & [k, sel] : groups(line)) {
    const Cloud p = select(xyz, sel);
    Eigen::Index centre;
    (p.colwise() - p.rowwise().mean()).colwise().norm().minCoeff(&centre);
    const auto rgb = colour(k);
    visualization_msgs::msg::Marker m;
    m.ns = "mask_clusters";
    m.id = k;
    m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.text = name_of(cls[static_cast<size_t>(sel.front())]) + " " + std::to_string(k) + " (" +
      std::to_string(sel.size()) + ")";
    m.header.stamp = stamp;
    m.header.frame_id = frame;
    m.pose.position = point(p.col(centre));
    m.pose.orientation.w = 1.0;
    m.scale.z = 0.2;
    m.color.r = static_cast<float>(rgb[0] / 255.0);
    m.color.g = static_cast<float>(rgb[1] / 255.0);
    m.color.b = static_cast<float>(rgb[2] / 255.0);
    m.color.a = 1.0f;
    out.markers.push_back(m);
  }
  pub_markers_->publish(out);
}

}  // namespace axial_perception
