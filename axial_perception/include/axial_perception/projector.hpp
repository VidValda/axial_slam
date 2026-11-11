#pragma once

#include <string>
#include <vector>

#include "axial_perception/camera.hpp"
#include "axial_perception/clusters.hpp"
#include "axial_perception/masks.hpp"

namespace axial_perception
{

struct ProjectorParams
{
  double depth_min = 0.1;
  double depth_max = 30.0;
  std::string occlusion_method = "zbuffer";
  int occlusion_window = 9;
  double occlusion_tol = 0.1;
  double occlusion_link = 0.1;
  double occlusion_radius = 15.0;
  double occlusion_behind = 0.3;
  std::vector<int> classes{4, 6, 8};
  int min_points = 5;
};

struct MaskPoints
{
  Projection proj;
  Labelled labelled;
};

void check_occlusion_method(const std::string & method);

MaskPoints mask_points(
  const Cloud & pts, const InstanceImage & img, const Intrinsics & K, const ProjectorParams & p);

}  // namespace axial_perception
