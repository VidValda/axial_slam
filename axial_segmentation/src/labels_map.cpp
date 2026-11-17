#include "axial_segmentation/labels_map.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace axial_segmentation
{

std::vector<SegClass> make_classes(
  const std::vector<int64_t> & ids, const std::vector<std::string> & names,
  const std::vector<std::string> & geometry)
{
  if (ids.size() != names.size() || ids.size() != geometry.size() || ids.empty()) {
    throw std::invalid_argument("class_ids, class_names and class_geometry must be non-empty and the same length");
  }
  std::set<int64_t> seen_ids;
  std::set<std::string> seen_names;
  std::vector<SegClass> out;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 1 || ids[i] > 255) {
      throw std::invalid_argument("class id " + std::to_string(ids[i]) + " outside 1..255");
    }
    if (geometry[i] != "axis" && geometry[i] != "point") {
      throw std::invalid_argument("class_geometry '" + geometry[i] + "' is not axis or point");
    }
    if (!seen_ids.insert(ids[i]).second || !seen_names.insert(names[i]).second) {
      throw std::invalid_argument("duplicate class id or name: " + names[i]);
    }
    out.push_back({static_cast<int>(i), static_cast<int>(ids[i]), names[i], geometry[i]});
  }
  return out;
}

cv::Mat encode_labels(const std::vector<Instance> & instances, const std::vector<SegClass> & classes,
                      int width, int height)
{
  cv::Mat out(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<size_t> order(instances.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
    [&](size_t a, size_t b) {return instances[a].score < instances[b].score;});
  // ids by descending score, painted lowest first so the best instance wins overlaps
  std::vector<int> ids(instances.size());
  std::map<int, int> next;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    ids[*it] = ++next[instances[*it].index];
  }
  for (const size_t i : order) {
    const auto & inst = instances[i];
    if (inst.index < 0 || inst.index >= static_cast<int>(classes.size())) {
      continue;
    }
    const int id = ids[i];
    if (id > 0xFFFF) {
      continue;
    }
    const cv::Vec3b px(static_cast<uint8_t>(id & 0xFF), static_cast<uint8_t>(id >> 8),
      static_cast<uint8_t>(classes[inst.index].id));   // rgb8 bytes: R, G, B
    out(inst.box_px).setTo(px, inst.mask);
  }
  return out;
}

cv::Mat draw_overlay(const cv::Mat & bgr, const std::vector<Instance> & instances,
                     const std::vector<SegClass> & classes)
{
  static const cv::Scalar palette[] = {
    {60, 180, 75}, {200, 130, 0}, {48, 130, 245}, {75, 25, 230}, {180, 30, 145},
    {240, 240, 70}, {230, 50, 240}, {0, 128, 128}, {40, 110, 170}, {128, 0, 0}};
  cv::Mat out = bgr.clone(), tint = bgr.clone();
  for (const auto & inst : instances) {
    const auto & colour = palette[inst.index % 10];
    tint(inst.box_px).setTo(colour, inst.mask);
  }
  cv::addWeighted(tint, 0.45, out, 0.55, 0.0, out);
  for (const auto & inst : instances) {
    const auto & colour = palette[inst.index % 10];
    cv::rectangle(out, inst.box, colour, 1);
    const std::string name = inst.index < static_cast<int>(classes.size()) ?
      classes[inst.index].name : std::to_string(inst.index);
    cv::putText(out, name + " " + std::to_string(static_cast<int>(inst.score * 100)),
      cv::Point(static_cast<int>(inst.box.x), std::max(10, static_cast<int>(inst.box.y) - 3)),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, colour, 1);
  }
  return out;
}

}  // namespace axial_segmentation
