#pragma once

#include <Eigen/Core>
#include <vector>

namespace axial_perception
{

using Cloud = Eigen::Matrix3Xd;
using Indices = std::vector<int>;

struct Rigid
{
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Cloud apply(const Cloud & pts) const {return (R * pts).colwise() + t;}
  Eigen::Vector3d apply(const Eigen::Vector3d & p) const {return R * p + t;}
};

inline Cloud select(const Cloud & pts, const Indices & idx)
{
  Cloud out(3, static_cast<Eigen::Index>(idx.size()));
  for (size_t i = 0; i < idx.size(); ++i) {
    out.col(static_cast<Eigen::Index>(i)) = pts.col(idx[i]);
  }
  return out;
}

}  // namespace axial_perception
