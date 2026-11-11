#pragma once

#include <vector>

#include "axial_perception/camera.hpp"
#include "axial_perception/masks.hpp"

namespace axial_perception
{

struct Labelled
{
  Indices idx;
  std::vector<int> cls;
  std::vector<int> seg;
};

Labelled label_points(
  const Projection & proj, const InstanceImage & img, const std::vector<int> & classes,
  int min_points);

}  // namespace axial_perception
