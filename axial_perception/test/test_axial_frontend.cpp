#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include "axial_perception/cloud_msg.hpp"
#include "axial_perception/axial_frontend.hpp"

using axial_perception::AxialFrontend;
using Detections = dps_slam_msgs::msg::DetectionWithIDArray;
using sensor_msgs::msg::PointCloud2;

namespace
{

constexpr int kSize = 100;

rclcpp::NodeOptions options(const std::vector<rclcpp::Parameter> & extra = {})
{
  std::vector<rclcpp::Parameter> params{
    {"labels_topic", "test_labels"}, {"cloud_topic", "test_points"}, {"camera_info_topic", "test_info"},
    {"image_topic", "test_image"}, {"camera_frame", "test_cam"}, {"odom_frame", "test_odom"},
    {"n_scans", 2}, {"axial_classes", std::vector<int64_t>{8}}, {"class_names", std::vector<std::string>{"pillar"}},
    {"class_radius", std::vector<double>{0.1}}, {"split_tol", std::vector<double>{0.3}},
    {"split_gap", std::vector<double>{1.0}}, {"split_method", "none"}, {"occlusion_method", "mask"},
    {"split_min_points", 5}, {"split_min_length", 0.3}};
  params.insert(params.end(), extra.begin(), extra.end());
  return rclcpp::NodeOptions().parameter_overrides(params).arguments(
    {"--ros-args", "-r", "/axial/mask_clusters:=/axial_test/mask_clusters", "-r",
      "/axial/line_detections:=/axial_test/line_detections", "-r", "/axial/mask_overlay:=/axial_test/mask_overlay",
      "-r", "/axial/landmark_map:=/axial_test/landmark_map"});
}

PointCloud2 pillar_cloud(const builtin_interfaces::msg::Time & stamp)
{
  std::vector<float> xyz;
  for (int i = 0; i < 200; ++i) {
    const double a = -0.8 + 1.6 * (i % 8) / 7.0;
    xyz.insert(xyz.end(), {static_cast<float>(0.2 + 0.1 * std::sin(a)), static_cast<float>(-1.0 + 0.01 * i),
      static_cast<float>(3.0 - 0.1 * std::cos(a))});
  }
  const float nan = std::nanf("");
  xyz.insert(xyz.end(), {nan, 0.0f, 3.0f, 0.0f, 0.0f, 8.0f});
  std_msgs::msg::Header h;
  h.stamp = stamp;
  h.frame_id = "test_cam";
  return axial_perception::float_cloud({"x", "y", "z"}, xyz, h);
}

sensor_msgs::msg::Image labels(const builtin_interfaces::msg::Time & stamp)
{
  sensor_msgs::msg::Image img;
  img.header.stamp = stamp;
  img.header.frame_id = "test_cam";
  img.height = img.width = kSize;
  img.encoding = "rgb8";
  img.step = 3 * kSize;
  img.data.assign(3 * kSize * kSize, 0);
  for (int v = 10; v < 90; ++v) {
    for (int u = 50; u < 62; ++u) {
      img.data[static_cast<size_t>(3 * (v * kSize + u))] = 1;
      img.data[static_cast<size_t>(3 * (v * kSize + u) + 2)] = 8;
    }
  }
  return img;
}

class AxialFrontendTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

}  // namespace

TEST_F(AxialFrontendTest, PublishesClustersAndLineDetections)
{
  auto frontend = std::make_shared<AxialFrontend>(options());
  auto io = std::make_shared<rclcpp::Node>("axial_frontend_test_io");
  tf2_ros::StaticTransformBroadcaster tf(io);
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = "test_odom";
  t.child_frame_id = "test_cam";
  t.transform.rotation.w = 1.0;
  tf.sendTransform(t);

  auto pub_info = io->create_publisher<sensor_msgs::msg::CameraInfo>("test_info", 10);
  auto pub_cloud = io->create_publisher<PointCloud2>("test_points", 10);
  auto pub_labels = io->create_publisher<sensor_msgs::msg::Image>("test_labels", 10);
  PointCloud2::ConstSharedPtr clusters;
  Detections::ConstSharedPtr detections;
  auto sub_clusters = io->create_subscription<PointCloud2>(
    "/axial_test/mask_clusters", 10, [&](const PointCloud2::ConstSharedPtr & m) {clusters = m;});
  auto sub_detections = io->create_subscription<Detections>(
    "/axial_test/line_detections", 10, [&](const Detections::ConstSharedPtr & m) {detections = m;});

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(frontend);
  exec.add_node(io);
  sensor_msgs::msg::CameraInfo info;
  info.k = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  int64_t ns = 1'000'000'000;
  while ((!clusters || !detections) && std::chrono::steady_clock::now() < deadline) {
    const builtin_interfaces::msg::Time stamp = rclcpp::Time(ns += 200'000'000);
    pub_info->publish(info);
    pub_cloud->publish(pillar_cloud(stamp));
    pub_labels->publish(labels(stamp));
    exec.spin_some(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(clusters);
  ASSERT_TRUE(detections);

  const auto v = axial_perception::read_fields(*clusters, {"x", "y", "z", "class_id", "segment", "line"}, false);
  ASSERT_GE(v.size() / 6, 100u);
  for (size_t i = 0; i < v.size(); i += 6) {
    EXPECT_EQ(v[i + 3], 8.0);
    EXPECT_EQ(v[i + 4], 1.0);
    EXPECT_EQ(v[i + 5], 1.0);
  }
  EXPECT_EQ(clusters->header.frame_id, "test_cam");

  ASSERT_EQ(detections->detections.size(), 1u);
  const auto & det = detections->detections.front();
  EXPECT_EQ(det.id, "pillar_1");
  EXPECT_EQ(det.label, "pillar");
  EXPECT_EQ(det.geometry.type, dps_slam_msgs::msg::Geometry::LINE3D);
  const auto & l = det.geometry.line3d;
  EXPECT_NEAR(std::abs(l.direction.y), 1.0, 1e-3);
  EXPECT_NEAR(l.point.z, 3.0, 0.05);
  EXPECT_NEAR(l.point.x, 0.2, 0.05);
  EXPECT_NEAR(l.covariance[0], 0.05 * 0.05, 1e-12);
  EXPECT_NEAR(l.covariance[15], 0.1 * 0.1, 1e-12);
  EXPECT_EQ(l.covariance[1], 0.0);
}

TEST_F(AxialFrontendTest, RejectsBadParameters)
{
  EXPECT_THROW(AxialFrontend(options({{"split_method", "nope"}})), std::invalid_argument);
  EXPECT_THROW(AxialFrontend(options({{"occlusion_method", "nope"}})), std::invalid_argument);
  EXPECT_THROW(AxialFrontend(options({{"n_scans", 0}})), std::invalid_argument);
  EXPECT_THROW(AxialFrontend(options({{"split_tol", std::vector<double>{}}})), std::invalid_argument);
  EXPECT_THROW(
    AxialFrontend(options({{"map_spawn_threshold", 2.0}, {"map_min_separation", 1.0}})), std::invalid_argument);
}
