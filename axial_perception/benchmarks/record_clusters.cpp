#include <chrono>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_listener.h>

#include "bench.hpp"
#include "axial_perception/cloud_msg.hpp"

using namespace axial_perception;
using namespace axial_perception::bench;

namespace
{

void run(const Args & args)
{
  using Cloud2 = sensor_msgs::msg::PointCloud2;
  const std::string frame = args.str("--frame", "earth");
  DatasetWriter out(args.required("--out"), Kind::Clusters);
  auto node = std::make_shared<rclcpp::Node>("record_clusters");
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock(), tf2::durationFromSec(30));
  tf2_ros::TransformListener listener(*tf, node.get());
  std::deque<Cloud2::ConstSharedPtr> pending;
  auto sub = node->create_subscription<Cloud2>(
    "/axial/mask_clusters", 50, [&](const Cloud2::ConstSharedPtr & m) {
      if (pending.size() == 50) {
        pending.pop_front();
      }
      pending.push_back(m);
    });
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  size_t points = 0;
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(args.num("-t", 300.0));
  while (rclcpp::ok() && std::chrono::steady_clock::now() < end) {
    exec.spin_some(std::chrono::milliseconds(10));
    while (!pending.empty()) {
      const auto msg = pending.front();
      const tf2::TimePoint stamp = tf2_ros::fromMsg(msg->header.stamp);
      if (!tf->canTransform(frame, msg->header.frame_id, stamp)) {
        break;
      }
      pending.pop_front();
      geometry_msgs::msg::Transform tr;
      try {
        tr = tf->lookupTransform(frame, msg->header.frame_id, stamp).transform;
      } catch (const tf2::TransformException &) {
        continue;
      }
      const auto v = read_fields(*msg, {"x", "y", "z", "class_id", "segment", "line"}, false);
      ClusterFrame f;
      const size_t n = v.size() / 6;
      f.xyz.resize(3, static_cast<Eigen::Index>(n));
      for (size_t i = 0; i < n; ++i) {
        f.xyz.col(static_cast<Eigen::Index>(i)) << v[6 * i], v[6 * i + 1], v[6 * i + 2];
        f.cls.push_back(static_cast<int>(v[6 * i + 3]));
        f.seg.push_back(static_cast<int>(v[6 * i + 4]));
        f.line.push_back(static_cast<int>(v[6 * i + 5]));
      }
      f.pose = {quat_to_matrix(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
        {tr.translation.x, tr.translation.y, tr.translation.z}};
      out.add(f);
      points += n;
    }
  }
  std::cout << out.frames() << " frames, " << points << " points in " << args.str("--out") << std::endl;
}

}  // namespace

int main(int argc, char ** argv)
{
  const auto argv_clean = rclcpp::init_and_remove_ros_arguments(argc, argv);
  int code = 0;
  try {
    run(Args(argv_clean, {{"-o", "--out"}, {"-t"}, {"-f", "--frame"}}));
  } catch (const std::exception & e) {
    std::cerr << "record_clusters: " << e.what() << "\n"
              << "usage: record_clusters -o clusters.bin [-t 300] [-f earth] --ros-args -p use_sim_time:=true"
              << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
