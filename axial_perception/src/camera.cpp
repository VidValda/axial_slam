#include "axial_perception/camera.hpp"

namespace axial_perception
{

Projection Projection::select(const std::vector<bool> & keep) const
{
  Projection out;
  for (size_t i = 0; i < idx.size(); ++i) {
    if (keep[i]) {
      out.u.push_back(u[i]);
      out.v.push_back(v[i]);
      out.idx.push_back(idx[i]);
    }
  }
  return out;
}

Eigen::Matrix3d quat_to_matrix(double x, double y, double z, double w)
{
  Eigen::Matrix3d R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
  return R;
}

Projection project(
  const Cloud & pts, const Intrinsics & K, int width, int height, double depth_min,
  double depth_max)
{
  Projection out;
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    const double z = pts(2, i);
    if (!(z >= depth_min && z <= depth_max)) {
      continue;
    }
    const double u = K.fx * pts(0, i) / z + K.cx + 0.5;
    const double v = K.fy * pts(1, i) / z + K.cy + 0.5;
    if (u >= 0 && u < width && v >= 0 && v < height) {
      out.u.push_back(static_cast<int>(u));
      out.v.push_back(static_cast<int>(v));
      out.idx.push_back(static_cast<int>(i));
    }
  }
  return out;
}

}  // namespace axial_perception
