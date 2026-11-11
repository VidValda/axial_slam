#include "axial_perception/landmarks.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "axial_perception/kdtree.hpp"

namespace axial_perception
{

namespace
{

struct VoxelHash
{
  size_t operator()(const std::array<int64_t, 3> & k) const
  {
    uint64_t h = 1469598103934665603ULL;
    for (int64_t x : k) {
      h = (h ^ static_cast<uint64_t>(x)) * 1099511628211ULL;
    }
    return static_cast<size_t>(h);
  }
};

}  // namespace

Eigen::Vector3d canonical(const Eigen::Vector3d & d)
{
  const Eigen::Vector3d u = d.normalized();
  return u[2] < 0 ? Eigen::Vector3d(-u) : u;
}

Eigen::Vector3d perpendicular(const Eigen::Vector3d & v, const Eigen::Vector3d & d)
{
  return v - v.dot(d) * d;
}

Eigen::Matrix3d rotation(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  const Eigen::Vector3d v = a.cross(b);
  const double c = a.dot(b);
  if (c < -1 + 1e-9) {
    return Eigen::Matrix3d::Identity();
  }
  Eigen::Matrix3d k;
  k << 0.0, -v[2], v[1],
    v[2], 0.0, -v[0],
    -v[1], v[0], 0.0;
  return Eigen::Matrix3d::Identity() + k + k * k / (1 + c);
}

LineLandmarks::LineLandmarks(
  double spawn_threshold, std::optional<double> min_anchor_separation, double max_angle, double voxel,
  double min_elongation)
: spawn_threshold_(spawn_threshold),
  min_anchor_separation_(min_anchor_separation.value_or(spawn_threshold)),
  cos_max_angle_(std::cos(max_angle)),
  voxel_(voxel),
  min_elongation_(min_elongation)
{
  if (min_anchor_separation_ < spawn_threshold_) {
    throw std::invalid_argument(
            "min_anchor_separation (" + std::to_string(min_anchor_separation_) +
            ") must be >= spawn_threshold (" + std::to_string(spawn_threshold_) + ")");
  }
}

bool LineLandmarks::accepts(const Cloud & pts, const Eigen::Vector3d & c) const
{
  if (pts.cols() < 2) {
    return false;
  }
  const Eigen::Vector3d s = spread(pts, c);
  return s[0] >= min_elongation_ * s[1];
}

std::map<int, int> LineLandmarks::update(const Cloud & xyz, const std::vector<int> & cls, const Labels & line)
{
  std::map<int, int> ids;
  for (const auto & [k, sel] : groups(line)) {
    const Cloud pts = select(xyz, sel);
    const Line l = fit_line(pts);
    if (accepts(pts, l.c)) {
      ids[k] = assign(cls[static_cast<size_t>(sel.front())], l.c, l.d, pts);
    }
  }
  return ids;
}

std::pair<int, double> LineLandmarks::nearest(
  int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & d) const
{
  int best = -1;
  double best_dist = std::numeric_limits<double>::infinity();
  for (const auto & [gid, lm] : landmarks_) {
    if (lm.class_id != class_id || std::abs(d.dot(lm.direction)) < cos_max_angle_) {
      continue;
    }
    const double dist = perpendicular(c - lm.point, lm.direction).norm();
    if (dist < best_dist) {
      best = gid;
      best_dist = dist;
    }
  }
  return {best, best_dist};
}

int LineLandmarks::assign(int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & dir, const Cloud & pts)
{
  const Eigen::Vector3d d = canonical(dir);
  auto [gid, dist] = nearest(class_id, c, d);
  if (gid >= 0 && dist < spawn_threshold_) {
    Landmark & lm = landmarks_.at(gid);
    if (pts.cols() > 0) {
      Cloud both(3, lm.points.cols() + pts.cols());
      both << lm.points, pts;
      lm.points = voxelize(both);
      const Line l = fit_line(lm.points);
      lm.point = l.c;
      lm.direction = canonical(l.d);
    }
  } else if (gid < 0 || dist >= min_anchor_separation_) {
    gid = next_id_++;
    Landmark lm;
    lm.class_id = class_id;
    lm.point = c;
    lm.direction = d;
    lm.points = voxelize(pts);
    landmarks_.emplace(gid, std::move(lm));
  }
  ++landmarks_.at(gid).hits;
  return gid;
}

void LineLandmarks::set_anchor(
  int gid, const Eigen::Vector3d & point, const Eigen::Vector3d & dir, int class_id)
{
  const Eigen::Vector3d direction = canonical(dir);
  const auto it = landmarks_.find(gid);
  if (it == landmarks_.end()) {
    Landmark lm;
    lm.class_id = class_id;
    lm.point = point;
    lm.direction = direction;
    landmarks_.emplace(gid, std::move(lm));
  } else {
    Landmark & lm = it->second;
    const Eigen::Vector3d c = point + (lm.point - point).dot(direction) * direction;
    lm.points = (rotation(lm.direction, direction) * (lm.points.colwise() - lm.point)).colwise() + c;
    lm.point = c;
    lm.direction = direction;
  }
  next_id_ = std::max(next_id_, gid + 1);
}

bool LineLandmarks::snap(int gid, const Eigen::Vector3d & point, const Eigen::Vector3d & dir)
{
  const auto it = landmarks_.find(gid);
  if (it == landmarks_.end()) {
    return false;
  }
  const Eigen::Vector3d direction = canonical(dir);
  const Landmark & lm = it->second;
  if (std::abs(direction.dot(lm.direction)) < cos_max_angle_ ||
    perpendicular(lm.point - point, direction).norm() > min_anchor_separation_)
  {
    return false;
  }
  set_anchor(gid, point, direction);
  return true;
}

Cloud LineLandmarks::voxelize(const Cloud & pts) const
{
  if (voxel_ <= 0 || pts.cols() == 0) {
    return pts;
  }
  std::unordered_set<std::array<int64_t, 3>, VoxelHash> seen;
  seen.reserve(static_cast<size_t>(pts.cols()));
  Indices first;
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    const std::array<int64_t, 3> key{
      static_cast<int64_t>(std::floor(pts(0, i) / voxel_)),
      static_cast<int64_t>(std::floor(pts(1, i) / voxel_)),
      static_cast<int64_t>(std::floor(pts(2, i) / voxel_))};
    if (seen.insert(key).second) {
      first.push_back(static_cast<int>(i));
    }
  }
  return select(pts, first);
}

Cloud LineLandmarks::dense_points(int gid, int min_neighbours) const
{
  const Cloud & pts = landmarks_.at(gid).points;
  if (min_neighbours <= 0 || pts.cols() == 0) {
    return pts;
  }
  KdTree<3> tree(pts);
  Indices keep;
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    if (static_cast<int>(tree.within(pts.col(i), 1.8 * voxel_).size()) - 1 >= min_neighbours) {
      keep.push_back(static_cast<int>(i));
    }
  }
  return select(pts, keep);
}

}  // namespace axial_perception
