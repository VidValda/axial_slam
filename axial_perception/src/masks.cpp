#include "axial_perception/masks.hpp"

#include <stdexcept>

namespace axial_perception
{

InstanceImage decode_gz_instance(
  const std::vector<uint8_t> & data, int height, int width, size_t step)
{
  if (height < 0 || width < 0) {
    throw std::invalid_argument("negative image size");
  }
  const size_t w = static_cast<size_t>(width);
  const size_t h = static_cast<size_t>(height);
  if (step < 3 * w || data.size() < step * h) {
    throw std::invalid_argument("labels image smaller than its size and step");
  }
  InstanceImage out;
  out.width = width;
  out.height = height;
  out.cls.resize(w * h);
  out.mask.resize(w * h);
  for (size_t r = 0; r < h; ++r) {
    const uint8_t * row = data.data() + r * step;
    for (size_t c = 0; c < w; ++c) {
      const uint8_t * px = row + 3 * c;
      const int32_t cls = px[2];
      out.cls[r * w + c] = px[2];
      out.mask[r * w + c] = cls == 0 ? 0 : (cls << 16) | (px[1] << 8) | px[0];
    }
  }
  return out;
}

}  // namespace axial_perception
