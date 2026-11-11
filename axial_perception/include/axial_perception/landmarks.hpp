#pragma once

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "axial_perception/segments.hpp"
#include "axial_perception/types.hpp"

namespace axial_perception
{

struct Landmark
{
  int class_id = 0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
  Cloud points = Cloud(3, 0);
  int hits = 0;
};

Eigen::Vector3d canonical(const Eigen::Vector3d & d);
Eigen::Vector3d perpendicular(const Eigen::Vector3d & v, const Eigen::Vector3d & d);
Eigen::Matrix3d rotation(const Eigen::Vector3d & a, const Eigen::Vector3d & b);

class LineLandmarks
{
public:
  explicit LineLandmarks(
    double spawn_threshold, std::optional<double> min_anchor_separation = std::nullopt,
    double max_angle = 0.35, double voxel = 0.05, double min_elongation = 0.0);

  bool accepts(const Cloud & pts, const Eigen::Vector3d & c) const;
  std::map<int, int> update(const Cloud & xyz, const std::vector<int> & cls, const Labels & line);
  std::pair<int, double> nearest(int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & d) const;
  int assign(int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & d, const Cloud & pts = Cloud(3, 0));
  void set_anchor(int gid, const Eigen::Vector3d & point, const Eigen::Vector3d & direction, int class_id = 0);
  bool snap(int gid, const Eigen::Vector3d & point, const Eigen::Vector3d & direction);
  Cloud voxelize(const Cloud & pts) const;
  Cloud dense_points(int gid, int min_neighbours) const;

  const std::map<int, Landmark> & landmarks() const {return landmarks_;}

private:
  double spawn_threshold_;
  double min_anchor_separation_;
  double cos_max_angle_;
  double voxel_;
  double min_elongation_;
  std::map<int, Landmark> landmarks_;
  int next_id_ = 1;
};

}  // namespace axial_perception
