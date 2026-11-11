#pragma once

#include <vector>

#include "axial_perception/types.hpp"

namespace axial_perception
{

struct Intrinsics
{
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

struct Projection
{
  std::vector<int> u;
  std::vector<int> v;
  Indices idx;

  size_t size() const {return idx.size();}
  Projection select(const std::vector<bool> & keep) const;
};

Eigen::Matrix3d quat_to_matrix(double x, double y, double z, double w);

Projection project(
  const Cloud & pts, const Intrinsics & K, int width, int height, double depth_min,
  double depth_max);

}  // namespace axial_perception
