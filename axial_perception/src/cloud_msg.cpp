#include "axial_perception/cloud_msg.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace axial_perception
{

sensor_msgs::msg::PointCloud2 float_cloud(
  const std::vector<std::string> & fields, const std::vector<float> & data, const std_msgs::msg::Header & header)
{
  if (fields.empty() || data.size() % fields.size() != 0) {
    throw std::invalid_argument("cloud data does not match its fields");
  }
  sensor_msgs::msg::PointCloud2 out;
  out.header = header;
  out.height = 1;
  out.width = static_cast<uint32_t>(data.size() / fields.size());
  for (size_t i = 0; i < fields.size(); ++i) {
    sensor_msgs::msg::PointField f;
    f.name = fields[i];
    f.offset = static_cast<uint32_t>(4 * i);
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    out.fields.push_back(f);
  }
  out.is_bigendian = false;
  out.point_step = static_cast<uint32_t>(4 * fields.size());
  out.row_step = out.point_step * out.width;
  out.is_dense = true;
  out.data.resize(data.size() * sizeof(float));
  if (!data.empty()) {
    std::memcpy(out.data.data(), data.data(), out.data.size());
  }
  return out;
}

std::vector<double> read_fields(
  const sensor_msgs::msg::PointCloud2 & msg, const std::vector<std::string> & fields, bool skip_nonfinite)
{
  if (msg.is_bigendian) {
    throw std::invalid_argument("big-endian clouds are not supported");
  }
  std::vector<uint32_t> offset;
  std::vector<uint8_t> type;
  for (const auto & name : fields) {
    const sensor_msgs::msg::PointField * found = nullptr;
    for (const auto & f : msg.fields) {
      if (f.name == name) {
        found = &f;
      }
    }
    if (!found) {
      throw std::invalid_argument("cloud has no field " + name);
    }
    const size_t size = found->datatype == sensor_msgs::msg::PointField::FLOAT32 ? 4 :
      found->datatype == sensor_msgs::msg::PointField::FLOAT64 ? 8 : 0;
    if (size == 0) {
      throw std::invalid_argument("cloud field " + name + " is not float32 or float64");
    }
    if (static_cast<size_t>(found->offset) + size > msg.point_step) {
      throw std::invalid_argument("cloud field " + name + " overruns the point step");
    }
    offset.push_back(found->offset);
    type.push_back(found->datatype);
  }
  const size_t width = msg.width;
  const size_t height = msg.height;
  if (static_cast<size_t>(msg.point_step) * width > msg.row_step ||
    static_cast<size_t>(msg.row_step) * height > msg.data.size())
  {
    throw std::invalid_argument("cloud data smaller than its size and steps");
  }
  std::vector<double> out;
  out.reserve(width * height * fields.size());
  std::vector<double> point(fields.size());
  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      const uint8_t * p = msg.data.data() + r * msg.row_step + c * msg.point_step;
      bool finite = true;
      for (size_t k = 0; k < fields.size(); ++k) {
        if (type[k] == sensor_msgs::msg::PointField::FLOAT32) {
          float v;
          std::memcpy(&v, p + offset[k], sizeof v);
          point[k] = v;
        } else {
          std::memcpy(&point[k], p + offset[k], sizeof(double));
        }
        finite = finite && std::isfinite(point[k]);
      }
      if (finite || !skip_nonfinite) {
        out.insert(out.end(), point.begin(), point.end());
      }
    }
  }
  return out;
}

float pack_rgb(uint8_t r, uint8_t g, uint8_t b)
{
  const uint32_t rgb = (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | b;
  float out;
  std::memcpy(&out, &rgb, sizeof out);
  return out;
}

}  // namespace axial_perception
