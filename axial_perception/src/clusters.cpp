#include "axial_perception/clusters.hpp"

#include <algorithm>
#include <map>

namespace axial_perception
{

Labelled label_points(
  const Projection & proj, const InstanceImage & img, const std::vector<int> & classes,
  int min_points)
{
  std::vector<size_t> on;
  std::map<int32_t, int> counts;
  for (size_t i = 0; i < proj.size(); ++i) {
    const int cls = img.class_at(proj.u[i], proj.v[i]);
    const int32_t mask = img.mask_at(proj.u[i], proj.v[i]);
    if (mask > 0 && std::find(classes.begin(), classes.end(), cls) != classes.end()) {
      on.push_back(i);
      ++counts[mask];
    }
  }
  std::map<int32_t, int> segment;
  for (const auto & [mask, n] : counts) {
    if (n >= min_points) {
      segment.emplace(mask, static_cast<int>(segment.size()) + 1);
    }
  }
  Labelled out;
  for (size_t i : on) {
    const auto it = segment.find(img.mask_at(proj.u[i], proj.v[i]));
    if (it != segment.end()) {
      out.idx.push_back(proj.idx[i]);
      out.cls.push_back(img.class_at(proj.u[i], proj.v[i]));
      out.seg.push_back(it->second);
    }
  }
  return out;
}

}  // namespace axial_perception
