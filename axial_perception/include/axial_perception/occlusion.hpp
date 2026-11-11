#pragma once

#include <vector>

#include "axial_perception/masks.hpp"
#include "axial_perception/types.hpp"

namespace axial_perception
{

std::vector<bool> front_mask(
  const std::vector<int> & u, const std::vector<int> & v, const std::vector<double> & z,
  int width, int height, int window, double tol);

std::vector<bool> surface_mask(
  const std::vector<int> & u, const std::vector<int> & v, const Cloud & xyz,
  const InstanceImage & img, double link, double radius, double behind, int min_points = 10);

double median(std::vector<double> x);

}  // namespace axial_perception
