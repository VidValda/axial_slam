#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "bench.hpp"
#include "axial_perception/masks.hpp"
#include "axial_perception/axial_frontend.hpp"

using namespace axial_perception;
using namespace axial_perception::bench;

namespace
{

std::vector<bool> dilate(const InstanceImage & img, const std::vector<int> & classes, int r)
{
  const int w = img.width;
  const int h = img.height;
  std::vector<int> sum(static_cast<size_t>((w + 1) * (h + 1)), 0);
  auto at = [&](int y, int x) -> int & {return sum[static_cast<size_t>(y * (w + 1) + x)];};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const bool on = std::find(classes.begin(), classes.end(), img.class_at(x, y)) != classes.end();
      at(y + 1, x + 1) = on + at(y, x + 1) + at(y + 1, x) - at(y, x);
    }
  }
  std::vector<bool> out(static_cast<size_t>(w * h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int y0 = std::max(0, y - r), y1 = std::min(h, y + r + 1);
      const int x0 = std::max(0, x - r), x1 = std::min(w, x + r + 1);
      out[static_cast<size_t>(y * w + x)] = at(y1, x1) - at(y0, x1) - at(y1, x0) + at(y0, x0) > 0;
    }
  }
  return out;
}

class Recorder : public AxialFrontend
{
public:
  Recorder(const std::string & out, double every, int margin)
  : out_(out, Kind::Raw), every_(every), margin_(margin) {}

  size_t frames() const {return out_.frames();}

protected:
  void process(const Image & msg, const std::string & frame, const Rigid & tf, const Rigid &) override
  {
    const rclcpp::Time stamp(msg.header.stamp, RCL_ROS_TIME);
    if (last_ && (stamp - *last_).seconds() < every_) {
      return;
    }
    const auto pose = lookup("earth", frame, msg.header.stamp);
    if (!pose) {
      return;
    }
    last_ = stamp;
    const Cloud pts = tf.apply(scans_.points());
    const InstanceImage img = decode_gz_instance(
      msg.data, static_cast<int>(msg.height), static_cast<int>(msg.width), msg.step);
    const Projection proj = project(pts, *K_, img.width, img.height, projector_.depth_min, projector_.depth_max);
    const std::vector<bool> near = dilate(img, projector_.classes, margin_);
    Indices keep;
    for (size_t i = 0; i < proj.size(); ++i) {
      if (near[static_cast<size_t>(proj.v[i] * img.width + proj.u[i])]) {
        keep.push_back(proj.idx[i]);
      }
    }
    RawFrame f;
    f.xyz = select(pts, keep);
    f.height = img.height;
    f.width = img.width;
    for (int y = 0; y < img.height; ++y) {
      const auto row = msg.data.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(y) * msg.step);
      f.labels.insert(f.labels.end(), row, row + 3 * img.width);
    }
    f.K = *K_;
    f.pose = *pose;
    f.stamp = stamp.nanoseconds();
    f.n_points = pts.cols();
    out_.add(f);
    if (out_.frames() % 20 == 0) {
      RCLCPP_INFO(
        get_logger(), "%zu frames, last %zu of %zu points in view", out_.frames(), keep.size(), proj.size());
    }
  }

private:
  DatasetWriter out_;
  double every_;
  int margin_;
  std::optional<rclcpp::Time> last_;
};

void run(const Args & args)
{
  auto node = std::make_shared<Recorder>(
    args.required("--out"), args.num("--every", 0.5), static_cast<int>(args.num("--margin", 15)));
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(args.num("-t", 600.0));
  while (rclcpp::ok() && std::chrono::steady_clock::now() < end) {
    exec.spin_some(std::chrono::milliseconds(100));
  }
  std::cout << node->frames() << " frames in " << args.str("--out") << std::endl;
}

}  // namespace

int main(int argc, char ** argv)
{
  const auto argv_clean = rclcpp::init_and_remove_ros_arguments(argc, argv);
  int code = 0;
  try {
    run(Args(argv_clean, {{"-o", "--out"}, {"-t"}, {"--every"}, {"--margin"}}));
  } catch (const std::exception & e) {
    std::cerr << "record_raw: " << e.what() << "\n"
              << "usage: record_raw -o raw.bin [-t 600] [--every 0.5] [--margin 15] --ros-args --params-file "
      "params.yaml -p use_sim_time:=true" << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
