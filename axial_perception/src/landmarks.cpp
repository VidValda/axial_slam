#include "axial_perception/landmarks.hpp"
#include "axial_perception/occlusion.hpp"

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

void LineLandmarks::set_class_thresholds(int class_id, double spawn_threshold, double min_anchor_separation)
{
  if (min_anchor_separation < spawn_threshold) {
    throw std::invalid_argument(
            "min_anchor_separation (" + std::to_string(min_anchor_separation) + ") must be >= spawn_threshold (" +
            std::to_string(spawn_threshold) + ") for class " + std::to_string(class_id));
  }
  class_thresholds_[class_id] = {spawn_threshold, min_anchor_separation};
}

std::pair<double, double> LineLandmarks::thresholds(int class_id) const
{
  const auto it = class_thresholds_.find(class_id);
  return it == class_thresholds_.end() ? std::make_pair(spawn_threshold_, min_anchor_separation_) : it->second;
}

bool LineLandmarks::accepts(const Cloud & pts, const Eigen::Vector3d & c) const
{
  if (pts.cols() < 2) {
    return false;
  }
  const Eigen::Vector3d s = spread(pts, c);
  return s[0] >= min_elongation_ * s[1];
}

void LineLandmarks::set_radius_fit(double band, double min_span, double max_range, int refit_hits)
{
  radius_band_ = band;
  radius_min_span_ = min_span;
  radius_max_range_ = max_range;
  radius_refit_hits_ = std::max(1, refit_hits);
}

double LineLandmarks::radius(int gid)
{
  const auto it = landmarks_.find(gid);
  if (it == landmarks_.end()) {
    return 0.0;
  }
  Landmark & lm = it->second;
  if (lm.radius > 0.0 || lm.class_radius > 0.0) {
    return lm.radius > 0.0 ? lm.radius : lm.class_radius;
  }
  std::vector<double> learned;
  for (const auto & [id, other] : landmarks_) {
    if (other.class_id == lm.class_id && other.radius > 0.0) {
      learned.push_back(other.radius);
    }
  }
  lm.class_radius = learned.empty() ? 0.0 : median(learned);
  return lm.class_radius;
}

void LineLandmarks::add_section(
  Landmark & lm, const Cloud & pts, const std::optional<Eigen::Vector3d> & viewer) const
{
  if (radius_band_ <= 0.0 || lm.radius > 0.0) {
    return;
  }
  int added = 0;
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    if (viewer && radius_max_range_ > 0.0 && (pts.col(i) - *viewer).norm() > radius_max_range_) {
      continue;
    }
    ++added;
    const std::array<int64_t, 3> key{
      static_cast<int64_t>(std::floor(pts(0, i) / radius_band_)),
      static_cast<int64_t>(std::floor(pts(1, i) / radius_band_)),
      static_cast<int64_t>(std::floor(pts(2, i) / radius_band_))};
    auto & cell = lm.cells.try_emplace(key, Eigen::Vector3d::Zero(), 0).first->second;
    cell.first += pts.col(i);
    ++cell.second;
  }
  if (viewer && added > 0) {
    const Eigen::Vector3d v = perpendicular(*viewer - pts.rowwise().mean(), lm.direction);
    if (v.norm() > 1e-9) {
      lm.seen_from += v.normalized();
    }
  }
  if (lm.hits < lm.fitted_at + radius_refit_hits_) {
    return;
  }
  lm.fitted_at = lm.hits;
  const Eigen::Matrix3d rot = rotation(lm.direction, Eigen::Vector3d::UnitZ());
  const size_t stride = std::max<size_t>(1, lm.cells.size() / 3000);
  if (lm.cells.empty()) {
    return;
  }
  Eigen::Matrix2Xd q(2, static_cast<Eigen::Index>((lm.cells.size() + stride - 1) / stride));
  Eigen::Index j = 0;
  size_t k = 0;
  for (const auto & [key, cell] : lm.cells) {
    if (k++ % stride == 0) {
      q.col(j++) = (rot * (cell.first / cell.second)).head<2>();
    }
  }
  const auto c = fit_circle(q.leftCols(j), radius_band_);
  const Eigen::Vector2d seen = (rot * lm.seen_from).head<2>();
  const Eigen::Vector2d mean = q.leftCols(j).rowwise().mean();
  if (c && c->span >= radius_min_span_ && (seen.norm() < 1e-9 || (mean - c->centre).dot(seen) > 0.0)) {
    lm.radius = c->radius;
  }
}

void LineLandmarks::set_min_width(double min_width)
{
  min_width_ = min_width;
}

std::map<int, int> LineLandmarks::update(
  const Cloud & xyz, const std::vector<int> & cls, const Labels & line, const std::map<int, double> & widths,
  const std::optional<Eigen::Vector3d> & viewer)
{
  std::map<int, int> ids;
  for (const auto & [k, sel] : groups(line)) {
    const Cloud pts = select(xyz, sel);
    const Line l = fit_line(pts);
    if (accepts(pts, l.c)) {
      const auto w = widths.find(k);
      const int gid = assign(
        cls[static_cast<size_t>(sel.front())], l.c, l.d, pts, w == widths.end() ? 0.0 : w->second, viewer);
      if (gid >= 0) {
        ids[k] = gid;
      }
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

int LineLandmarks::assign(
  int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & dir, const Cloud & pts, double width,
  const std::optional<Eigen::Vector3d> & viewer)
{
  const Eigen::Vector3d d = canonical(dir);
  const auto [spawn, separation] = thresholds(class_id);
  auto [gid, dist] = nearest(class_id, c, d);
  if (gid >= 0 && dist < separation && width > 0.0 && min_width_ > 0.0 && width < min_width_ * radius(gid)) {
    return -1;
  }
  const bool merge = gid >= 0 && dist < spawn;
  if (merge) {
    Landmark & lm = landmarks_.at(gid);
    if (pts.cols() > 0) {
      Cloud both(3, lm.points.cols() + pts.cols());
      both << lm.points, pts;
      lm.points = voxelize(both);
      const Line l = fit_line(lm.points);
      lm.point = l.c;
      lm.direction = canonical(l.d);
    }
  } else if (gid < 0 || dist >= separation) {
    gid = next_id_++;
    Landmark lm;
    lm.class_id = class_id;
    lm.point = c;
    lm.direction = d;
    lm.points = voxelize(pts);
    landmarks_.emplace(gid, std::move(lm));
  }
  Landmark & lm = landmarks_.at(gid);
  ++lm.hits;
  if (merge || lm.hits == 1) {
    add_section(lm, pts, viewer);
  }
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
    perpendicular(lm.point - point, direction).norm() > thresholds(lm.class_id).second)
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
    if (static_cast<int>(tree.count_within(pts.col(i), 1.8 * voxel_)) - 1 >= min_neighbours) {
      keep.push_back(static_cast<int>(i));
    }
  }
  return select(pts, keep);
}

}  // namespace axial_perception
