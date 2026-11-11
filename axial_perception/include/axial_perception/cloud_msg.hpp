#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

namespace axial_perception
{

sensor_msgs::msg::PointCloud2 float_cloud(
  const std::vector<std::string> & fields, const std::vector<float> & data, const std_msgs::msg::Header & header);

std::vector<double> read_fields(
  const sensor_msgs::msg::PointCloud2 & msg, const std::vector<std::string> & fields, bool skip_nonfinite);

float pack_rgb(uint8_t r, uint8_t g, uint8_t b);

}  // namespace axial_perception
