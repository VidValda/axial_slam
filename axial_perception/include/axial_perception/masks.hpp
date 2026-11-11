#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace axial_perception
{

struct InstanceImage
{
  int width = 0;
  int height = 0;
  std::vector<uint8_t> cls;
  std::vector<int32_t> mask;

  uint8_t class_at(int u, int v) const {return cls[static_cast<size_t>(v) * width + u];}
  int32_t mask_at(int u, int v) const {return mask[static_cast<size_t>(v) * width + u];}
};

InstanceImage decode_gz_instance(
  const std::vector<uint8_t> & data, int height, int width, size_t step);

}  // namespace axial_perception
