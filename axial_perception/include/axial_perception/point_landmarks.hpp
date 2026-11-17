#pragma once

#include <map>
#include <vector>

#include "axial_perception/types.hpp"

namespace axial_perception
{

struct PointLandmark
{
  int class_id = 0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  Cloud points = Cloud(3, 0);
  int hits = 0;
};

// Per-axis median, robust to background points seen through a mask.
Eigen::Vector3d median_point(const Cloud & pts);

// Map of point-like objects (doors, extinguishers): one centroid per mask, matched by class within a radius.
class PointLandmarks
{
public:
  PointLandmarks(double match_radius, double voxel, int max_points = 2000);

  // xyz in the map frame; seg numbers the masks. Returns seg -> landmark id.
  std::map<int, int> update(const Cloud & xyz, const std::vector<int> & cls, const std::vector<int> & seg);

  const std::map<int, PointLandmark> & landmarks() const {return landmarks_;}

private:
  double match_radius_;
  double voxel_;
  int max_points_;
  std::map<int, PointLandmark> landmarks_;
  int next_id_ = 1;
};

}  // namespace axial_perception
