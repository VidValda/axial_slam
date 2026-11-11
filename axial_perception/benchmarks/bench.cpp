#include "bench.hpp"

#include <yaml-cpp/yaml.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_listener.h>

#include "axial_perception/cloud_msg.hpp"
#include "axial_perception/masks.hpp"

namespace axial_perception::bench
{

namespace
{

constexpr char kMagic[4] = {'A', 'X', 'D', 'S'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kMaxRecord = uint64_t{1} << 31;

class Bytes
{
public:
  template<class T>
  void put(const T & v)
  {
    static_assert(std::is_trivially_copyable<T>::value, "plain values only");
    const auto * p = reinterpret_cast<const uint8_t *>(&v);
    data.insert(data.end(), p, p + sizeof(T));
  }

  template<class T>
  void put_vector(const std::vector<T> & v)
  {
    put(static_cast<uint64_t>(v.size()));
    const auto * p = reinterpret_cast<const uint8_t *>(v.data());
    data.insert(data.end(), p, p + v.size() * sizeof(T));
  }

  void put_cloud(const Cloud & c)
  {
    std::vector<float> f(static_cast<size_t>(c.size()));
    for (Eigen::Index i = 0; i < c.size(); ++i) {
      f[static_cast<size_t>(i)] = static_cast<float>(c.data()[i]);
    }
    put_vector(f);
  }

  void put_rigid(const Rigid & r)
  {
    for (Eigen::Index i = 0; i < 9; ++i) {
      put(r.R.data()[i]);
    }
    for (Eigen::Index i = 0; i < 3; ++i) {
      put(r.t[i]);
    }
  }

  std::vector<uint8_t> data;
};

class Parser
{
public:
  explicit Parser(const std::vector<uint8_t> & d)
  : d_(d) {}

  template<class T>
  T get()
  {
    T v;
    take(&v, sizeof(T));
    return v;
  }

  template<class T>
  std::vector<T> get_vector()
  {
    const uint64_t n = get<uint64_t>();
    if (n > (d_.size() - at_) / sizeof(T)) {
      throw std::runtime_error("corrupt dataset record");
    }
    std::vector<T> v(static_cast<size_t>(n));
    take(v.data(), v.size() * sizeof(T));
    return v;
  }

  Cloud get_cloud()
  {
    const auto f = get_vector<float>();
    if (f.size() % 3) {
      throw std::runtime_error("corrupt dataset cloud");
    }
    Cloud c(3, static_cast<Eigen::Index>(f.size() / 3));
    for (size_t i = 0; i < f.size(); ++i) {
      c.data()[i] = f[i];
    }
    return c;
  }

  Rigid get_rigid()
  {
    Rigid r;
    for (Eigen::Index i = 0; i < 9; ++i) {
      r.R.data()[i] = get<double>();
    }
    for (Eigen::Index i = 0; i < 3; ++i) {
      r.t[i] = get<double>();
    }
    return r;
  }

  bool done() const {return at_ == d_.size();}

private:
  void take(void * out, size_t n)
  {
    if (n > d_.size() - at_) {
      throw std::runtime_error("corrupt dataset record");
    }
    if (n > 0) {
      std::memcpy(out, d_.data() + at_, n);
    }
    at_ += n;
  }

  const std::vector<uint8_t> & d_;
  size_t at_ = 0;
};

bool is_number(const std::string & s)
{
  char * end = nullptr;
  std::strtod(s.c_str(), &end);
  return !s.empty() && end == s.c_str() + s.size();
}

template<class T>
T param(const YAML::Node & p, const std::string & key, const T & fallback)
{
  return p[key] ? p[key].as<T>() : fallback;
}

}  // namespace

Args::Args(const std::vector<std::string> & argv, const std::vector<std::vector<std::string>> & keys)
{
  for (const auto & names : keys) {
    for (const auto & n : names) {
      canonical_[n] = names.back();
    }
  }
  std::string current;
  for (size_t i = 1; i < argv.size(); ++i) {
    const auto it = canonical_.find(argv[i]);
    if (it != canonical_.end()) {
      current = it->second;
      values_[current];
    } else if (argv[i].rfind('-', 0) == 0 && !is_number(argv[i])) {
      throw std::invalid_argument("unknown option " + argv[i]);
    } else if (current.empty()) {
      throw std::invalid_argument("unexpected argument " + argv[i]);
    } else {
      values_[current].push_back(argv[i]);
    }
  }
}

bool Args::has(const std::string & key) const
{
  return values_.count(canonical_.at(key)) > 0;
}

std::string Args::str(const std::string & key, const std::string & fallback) const
{
  const auto it = values_.find(canonical_.at(key));
  if (it == values_.end()) {
    return fallback;
  }
  if (it->second.size() != 1) {
    throw std::invalid_argument(key + " takes one value");
  }
  return it->second.front();
}

std::string Args::required(const std::string & key) const
{
  if (!has(key)) {
    throw std::invalid_argument(key + " is required");
  }
  return str(key);
}

double Args::num(const std::string & key, double fallback) const
{
  return has(key) ? std::stod(str(key)) : fallback;
}

std::vector<std::string> Args::list(const std::string & key, const std::vector<std::string> & fallback) const
{
  const auto it = values_.find(canonical_.at(key));
  return it == values_.end() ? fallback : it->second;
}

double Sections::distance(size_t k, const Eigen::Vector3d & p) const
{
  const Eigen::Vector3d ab = b[k] - a[k];
  const double t = std::clamp((p - a[k]).dot(ab) / ab.dot(ab), 0.0, 1.0);
  return std::abs((p - (a[k] + t * ab)).norm() - r[k]);
}

std::string World::name_of(int cls) const
{
  for (const auto & [c, n] : axial_names) {
    if (c == cls) {
      return n;
    }
  }
  return std::to_string(cls);
}

World load_world(const std::string & path)
{
  const YAML::Node root = YAML::LoadFile(path);
  World w;
  for (const auto & kv : root["classes"]) {
    const auto name = kv.first.as<std::string>();
    if (name == "pipe" || name == "duct" || name == "pillar") {
      w.axial_names.emplace_back(kv.second.as<int>(), name);
    }
  }
  std::vector<std::string> runs;
  for (const auto & o : root["objects"]) {
    const auto cls = o["class"].as<std::string>();
    if (cls == "pipe" || cls == "duct") {
      const auto s = o["start"].as<std::vector<double>>();
      const auto e = o["end"].as<std::vector<double>>();
      w.sections.a.emplace_back(s.at(0), s.at(1), s.at(2));
      w.sections.b.emplace_back(e.at(0), e.at(1), e.at(2));
    } else if (cls == "pillar") {
      const auto c = o["centre"].as<std::vector<double>>();
      const double h = o["height"].as<double>();
      w.sections.a.emplace_back(c.at(0), c.at(1), 0.0);
      w.sections.b.emplace_back(c.at(0), c.at(1), h);
      w.pillars.push_back({{c.at(0), c.at(1)}, o["radius"].as<double>(), h});
    } else {
      continue;
    }
    w.sections.r.push_back(o["radius"].as<double>());
    w.sections.cls.push_back(o["class_id"].as<int>());
    runs.push_back(o["model"] ? o["model"].as<std::string>() : o["name"].as<std::string>());
  }
  std::vector<std::string> unique = runs;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  for (const auto & r : runs) {
    w.sections.run.push_back(static_cast<int>(std::lower_bound(unique.begin(), unique.end(), r) - unique.begin()));
  }
  return w;
}

std::vector<int> section_ids(
  const Cloud & world, const std::vector<int> & cls, const Sections & s, double tol, const Indices & allowed)
{
  Indices all = allowed;
  if (all.empty()) {
    all.resize(s.size());
    std::iota(all.begin(), all.end(), 0);
  }
  std::vector<int> out(static_cast<size_t>(world.cols()), -1);
  for (Eigen::Index i = 0; i < world.cols(); ++i) {
    double best = std::numeric_limits<double>::infinity();
    int arg = -1;
    for (int k : all) {
      if (s.cls[static_cast<size_t>(k)] != cls[static_cast<size_t>(i)]) {
        continue;
      }
      const double d = s.distance(static_cast<size_t>(k), world.col(i));
      if (d < best) {
        best = d;
        arg = k;
      }
    }
    if (arg >= 0 && best <= tol) {
      out[static_cast<size_t>(i)] = arg;
    }
  }
  return out;
}

Params load_params(const std::string & path)
{
  const YAML::Node p = YAML::LoadFile(path)["mask_projector"]["ros__parameters"];
  Params out;
  auto & pr = out.projector;
  pr.depth_min = param(p, "depth_min", pr.depth_min);
  pr.depth_max = param(p, "depth_max", pr.depth_max);
  pr.occlusion_method = param<std::string>(p, "occlusion_method", pr.occlusion_method);
  pr.occlusion_window = param(p, "occlusion_window", pr.occlusion_window);
  pr.occlusion_tol = param(p, "occlusion_tol", pr.occlusion_tol);
  pr.occlusion_link = param(p, "occlusion_link", pr.occlusion_link);
  pr.occlusion_radius = param(p, "occlusion_radius", pr.occlusion_radius);
  pr.occlusion_behind = param(p, "occlusion_behind", pr.occlusion_behind);
  pr.classes = param(p, "axial_classes", pr.classes);
  pr.min_points = param(p, "min_points", pr.min_points);
  out.split_method = param<std::string>(p, "split_method", out.split_method);
  const auto tols = param(p, "split_tol", std::vector<double>{0.07, 0.25, 0.3});
  const auto gaps = param(p, "split_gap", std::vector<double>{0.5, 1.5, 1.0});
  for (size_t i = 0; i < pr.classes.size(); ++i) {
    if (i < tols.size()) {
      out.tols[pr.classes[i]] = tols[i];
    }
    if (i < gaps.size()) {
      out.gaps[pr.classes[i]] = gaps[i];
    }
  }
  out.split_min_points = param(p, "split_min_points", out.split_min_points);
  out.min_length = param(p, "split_min_length", out.min_length);
  out.map_spawn_threshold = param(p, "map_spawn_threshold", out.map_spawn_threshold);
  out.map_min_separation = param(p, "map_min_separation", out.map_min_separation);
  out.map_max_angle = param(p, "map_max_angle", out.map_max_angle);
  out.map_voxel = param(p, "map_voxel", out.map_voxel);
  out.map_min_elongation = param(p, "map_min_elongation", out.map_min_elongation);
  out.map_min_neighbours = param(p, "map_min_neighbours", out.map_min_neighbours);
  return out;
}

DatasetWriter::DatasetWriter(const std::string & path, Kind kind)
: out_(path, std::ios::binary | std::ios::trunc), kind_(kind)
{
  if (!out_) {
    throw std::runtime_error("cannot write " + path);
  }
  out_.write(kMagic, sizeof kMagic);
  const uint32_t header[2] = {kVersion, static_cast<uint32_t>(kind)};
  out_.write(reinterpret_cast<const char *>(header), sizeof header);
}

void DatasetWriter::write(const std::vector<uint8_t> & bytes)
{
  uLongf size = compressBound(static_cast<uLong>(bytes.size()));
  std::vector<uint8_t> packed(size);
  if (compress2(packed.data(), &size, bytes.data(), static_cast<uLong>(bytes.size()), 6) != Z_OK) {
    throw std::runtime_error("dataset compression failed");
  }
  const uint64_t lengths[2] = {bytes.size(), size};
  out_.write(reinterpret_cast<const char *>(lengths), sizeof lengths);
  out_.write(reinterpret_cast<const char *>(packed.data()), static_cast<std::streamsize>(size));
  out_.flush();
  if (!out_) {
    throw std::runtime_error("dataset write failed");
  }
  ++frames_;
}

void DatasetWriter::add(const RawFrame & f)
{
  if (kind_ != Kind::Raw) {
    throw std::logic_error("not a raw dataset");
  }
  Bytes b;
  b.put_cloud(f.xyz);
  b.put(static_cast<int32_t>(f.height));
  b.put(static_cast<int32_t>(f.width));
  b.put_vector(f.labels);
  b.put(f.K);
  b.put_rigid(f.pose);
  b.put(f.stamp);
  b.put(f.n_points);
  write(b.data);
}

void DatasetWriter::add(const ClusterFrame & f)
{
  if (kind_ != Kind::Clusters) {
    throw std::logic_error("not a clusters dataset");
  }
  Bytes b;
  b.put_cloud(f.xyz);
  b.put_vector(f.cls);
  b.put_vector(f.seg);
  b.put_vector(f.line);
  b.put_rigid(f.pose);
  write(b.data);
}

DatasetReader::DatasetReader(const std::string & path, Kind kind)
: in_(path, std::ios::binary), kind_(kind)
{
  char magic[4];
  uint32_t header[2];
  in_.read(magic, sizeof magic);
  in_.read(reinterpret_cast<char *>(header), sizeof header);
  if (!in_ || std::memcmp(magic, kMagic, sizeof kMagic) != 0 || header[0] != kVersion) {
    throw std::runtime_error(path + " is not an axial_perception dataset");
  }
  if (header[1] != static_cast<uint32_t>(kind)) {
    throw std::runtime_error(path + " holds another kind of frames");
  }
}

std::optional<std::vector<uint8_t>> DatasetReader::read()
{
  uint64_t lengths[2];
  in_.read(reinterpret_cast<char *>(lengths), sizeof lengths);
  if (in_.gcount() == 0 && in_.eof()) {
    return std::nullopt;
  }
  if (!in_ || lengths[0] > kMaxRecord || lengths[1] > kMaxRecord) {
    throw std::runtime_error("truncated or corrupt dataset");
  }
  std::vector<uint8_t> packed(static_cast<size_t>(lengths[1]));
  in_.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
  if (!in_) {
    throw std::runtime_error("truncated dataset");
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(lengths[0]));
  uLongf size = static_cast<uLongf>(bytes.size());
  if (uncompress(bytes.data(), &size, packed.data(), static_cast<uLong>(packed.size())) != Z_OK ||
    size != bytes.size())
  {
    throw std::runtime_error("corrupt dataset record");
  }
  return bytes;
}

bool DatasetReader::next(RawFrame & f)
{
  const auto bytes = read();
  if (!bytes) {
    return false;
  }
  Parser p(*bytes);
  f.xyz = p.get_cloud();
  f.height = p.get<int32_t>();
  f.width = p.get<int32_t>();
  f.labels = p.get_vector<uint8_t>();
  f.K = p.get<Intrinsics>();
  f.pose = p.get_rigid();
  f.stamp = p.get<int64_t>();
  f.n_points = p.get<int64_t>();
  if (!p.done() || f.height < 0 || f.width < 0 ||
    f.labels.size() != 3 * static_cast<size_t>(f.height) * static_cast<size_t>(f.width))
  {
    throw std::runtime_error("corrupt raw frame");
  }
  return true;
}

bool DatasetReader::next(ClusterFrame & f)
{
  const auto bytes = read();
  if (!bytes) {
    return false;
  }
  Parser p(*bytes);
  f.xyz = p.get_cloud();
  f.cls = p.get_vector<int>();
  f.seg = p.get_vector<int>();
  f.line = p.get_vector<int>();
  f.pose = p.get_rigid();
  const size_t n = static_cast<size_t>(f.xyz.cols());
  if (!p.done() || f.cls.size() != n || f.seg.size() != n || f.line.size() != n) {
    throw std::runtime_error("corrupt cluster frame");
  }
  return true;
}

std::vector<ClusterFrame> read_clusters(const std::string & path)
{
  DatasetReader in(path, Kind::Clusters);
  std::vector<ClusterFrame> out;
  ClusterFrame f;
  while (in.next(f)) {
    out.push_back(f);
  }
  return out;
}

FramePoints frame_points(const RawFrame & f, const Params & p, const std::string & occlusion)
{
  ProjectorParams pp = p.projector;
  pp.occlusion_method = occlusion;
  const InstanceImage img = decode_gz_instance(f.labels, f.height, f.width, 3 * static_cast<size_t>(f.width));
  const MaskPoints mp = mask_points(f.xyz, img, f.K, pp);
  return {select(f.xyz, mp.labelled.idx), mp.labelled.cls, mp.labelled.seg};
}

std::vector<ClusterFrame> record_clusters(double seconds, const std::string & ns)
{
  using Cloud2 = sensor_msgs::msg::PointCloud2;
  using Pose = geometry_msgs::msg::PoseStamped;
  auto node = std::make_shared<rclcpp::Node>(
    "mask_clusters_gt", rclcpp::NodeOptions().parameter_overrides({{"use_sim_time", true}}));
  auto tf = std::make_shared<tf2_ros::Buffer>(
    node->get_clock(), tf2::durationFromSec(seconds + 10));
  tf2_ros::TransformListener listener(*tf, node.get());
  std::vector<Cloud2::ConstSharedPtr> msgs;
  std::vector<Pose::ConstSharedPtr> poses;
  auto sub_clusters = node->create_subscription<Cloud2>(
    "/axial/mask_clusters", 10, [&](const Cloud2::ConstSharedPtr & m) {msgs.push_back(m);});
  auto sub_poses = node->create_subscription<Pose>(
    "/" + ns + "/ground_truth/pose", rclcpp::SensorDataQoS(), [&](const Pose::ConstSharedPtr & m) {
      poses.push_back(m);
    });
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < end) {
    exec.spin_some(std::chrono::milliseconds(50));
  }
  if (!rclcpp::ok()) {
    std::cout << "recording stopped early" << std::endl;
  }
  std::vector<ClusterFrame> frames;
  if (poses.empty()) {
    std::cout << "no ground-truth poses received" << std::endl;
    return frames;
  }
  for (const auto & m : msgs) {
    Rigid cam;
    try {
      const auto tr = tf->lookupTransform(ns, m->header.frame_id, tf2_ros::fromMsg(m->header.stamp)).transform;
      cam = {quat_to_matrix(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
        {tr.translation.x, tr.translation.y, tr.translation.z}};
    } catch (const tf2::TransformException &) {
      continue;
    }
    const int64_t stamp = rclcpp::Time(m->header.stamp).nanoseconds();
    const Pose * best = nullptr;
    int64_t best_dt = std::numeric_limits<int64_t>::max();
    for (const auto & p : poses) {
      const int64_t dt = std::abs(rclcpp::Time(p->header.stamp).nanoseconds() - stamp);
      if (dt < best_dt) {
        best_dt = dt;
        best = p.get();
      }
    }
    const auto & o = best->pose.orientation;
    const Rigid g{quat_to_matrix(o.x, o.y, o.z, o.w), {best->pose.position.x, best->pose.position.y,
        best->pose.position.z}};
    const auto v = read_fields(*m, {"x", "y", "z", "class_id", "segment", "line"}, false);
    ClusterFrame f;
    const size_t n = v.size() / 6;
    f.xyz.resize(3, static_cast<Eigen::Index>(n));
    for (size_t i = 0; i < n; ++i) {
      f.xyz.col(static_cast<Eigen::Index>(i)) << v[6 * i], v[6 * i + 1], v[6 * i + 2];
      f.cls.push_back(static_cast<int>(v[6 * i + 3]));
      f.seg.push_back(static_cast<int>(v[6 * i + 4]));
      f.line.push_back(static_cast<int>(v[6 * i + 5]));
    }
    f.pose = {g.R * cam.R, g.R * cam.t + g.t};
    frames.push_back(std::move(f));
  }
  if (frames.empty()) {
    std::cout << "no clusters received" << std::endl;
  }
  return frames;
}

double percentile(std::vector<double> x, double q)
{
  if (x.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(x.begin(), x.end());
  const double pos = q / 100.0 * static_cast<double>(x.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, x.size() - 1);
  return x[lo] + (pos - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

double mean(const std::vector<double> & x)
{
  return x.empty() ? std::numeric_limits<double>::quiet_NaN() :
         std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

int argmax_count(const std::vector<int> & values)
{
  std::map<int, int> count;
  for (int v : values) {
    ++count[v];
  }
  int best = 0;
  int arg = 0;
  for (const auto & [v, c] : count) {
    if (c > best) {
      best = c;
      arg = v;
    }
  }
  return arg;
}

}  // namespace axial_perception::bench
