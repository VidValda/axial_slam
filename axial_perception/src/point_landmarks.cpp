#include "axial_perception/point_landmarks.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>

namespace axial_perception
{

Eigen::Vector3d median_point(const Cloud & pts)
{
  if (pts.cols() == 0) {
    throw std::invalid_argument("median of no points");
  }
  Eigen::Vector3d out;
  std::vector<double> v(static_cast<size_t>(pts.cols()));
  for (int k = 0; k < 3; ++k) {
    for (Eigen::Index i = 0; i < pts.cols(); ++i) {
      v[static_cast<size_t>(i)] = pts(k, i);
    }
    auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    out[k] = *mid;
  }
  return out;
}

PointLandmarks::PointLandmarks(double match_radius, double voxel, int max_points)
: match_radius_(match_radius), voxel_(voxel), max_points_(max_points)
{
  if (!(match_radius > 0) || !(voxel > 0) || max_points < 1) {
    throw std::invalid_argument("point landmarks: match radius, voxel and max points must be positive");
  }
}

std::map<int, int> PointLandmarks::update(const Cloud & xyz, const std::vector<int> & cls, const std::vector<int> & seg)
{
  std::map<int, Indices> members;
  for (size_t i = 0; i < seg.size(); ++i) {
    members[seg[i]].push_back(static_cast<int>(i));
  }
  auto key = [this](const Eigen::Vector3d & q) {
      return std::make_tuple(std::lround(q[0] / voxel_), std::lround(q[1] / voxel_), std::lround(q[2] / voxel_));
    };
  std::map<int, int> out;
  for (const auto & [s, sel] : members) {
    const int c = cls[static_cast<size_t>(sel.front())];
    const Cloud pts = select(xyz, sel);
    const Eigen::Vector3d p = median_point(pts);
    int best = 0;
    double best_d = match_radius_;
    for (const auto & [gid, lm] : landmarks_) {
      const double d = (lm.point - p).norm();
      if (lm.class_id == c && d < best_d) {
        best = gid;
        best_d = d;
      }
    }
    if (best == 0) {
      best = next_id_++;
      landmarks_[best].class_id = c;
    }
    PointLandmark & lm = landmarks_[best];
    lm.point = (lm.point * lm.hits + p) / (lm.hits + 1);
    ++lm.hits;
    // keep one point per voxel, up to max_points
    std::set<std::tuple<long, long, long>> occupied;
    for (Eigen::Index i = 0; i < lm.points.cols(); ++i) {
      occupied.insert(key(lm.points.col(i)));
    }
    std::vector<Eigen::Vector3d> added;
    for (Eigen::Index i = 0; i < pts.cols(); ++i) {
      if (lm.points.cols() + static_cast<Eigen::Index>(added.size()) >= max_points_) {
        break;
      }
      if (occupied.insert(key(pts.col(i))).second) {
        added.push_back(pts.col(i));
      }
    }
    const Eigen::Index n0 = lm.points.cols();
    lm.points.conservativeResize(3, n0 + static_cast<Eigen::Index>(added.size()));
    for (size_t i = 0; i < added.size(); ++i) {
      lm.points.col(n0 + static_cast<Eigen::Index>(i)) = added[i];
    }
    out[s] = best;
  }
  return out;
}

}  // namespace axial_perception
