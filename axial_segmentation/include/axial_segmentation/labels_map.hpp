#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "axial_segmentation/yolo_seg.hpp"

namespace axial_segmentation
{

struct SegClass
{
  int index = 0;          // YOLO output index
  int id = 0;             // project class id, written to labels_map
  std::string name;
  std::string geometry;   // "axis" or "point"
};

// Class table from the parallel class_ids / class_names / class_geometry lists.
std::vector<SegClass> make_classes(
  const std::vector<int64_t> & ids, const std::vector<std::string> & names,
  const std::vector<std::string> & geometry);

// rgb8 instance map: class id in B, per-class instance id (G << 8) | R from 1, higher scores on top.
cv::Mat encode_labels(const std::vector<Instance> & instances, const std::vector<SegClass> & classes,
                      int width, int height);

// BGR overlay of the instances on the image, for display.
cv::Mat draw_overlay(const cv::Mat & bgr, const std::vector<Instance> & instances,
                     const std::vector<SegClass> & classes);

}  // namespace axial_segmentation
