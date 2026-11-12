#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <dps_slam_msgs/msg/detection_with_id_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "axial_perception/camera.hpp"
#include "axial_perception/landmarks.hpp"
#include "axial_perception/projector.hpp"
#include "axial_perception/scan_buffer.hpp"

namespace axial_perception
{

class AxialFrontend : public rclcpp::Node
{
public:
  explicit AxialFrontend(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

protected:
  using Image = sensor_msgs::msg::Image;
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using Detections = dps_slam_msgs::msg::DetectionWithIDArray;

  virtual void process(const Image & msg, const std::string & frame, const Rigid & tf, const Rigid & to_map);
  std::optional<Rigid> lookup(
    const std::string & target, const std::string & source, const builtin_interfaces::msg::Time & stamp) const;

  std::string odom_frame_;
  std::string camera_frame_;
  std::string map_frame_;
  ProjectorParams projector_;
  std::optional<Intrinsics> K_;
  ScanBuffer scans_;

private:
  struct Stats
  {
    double pts = 0, kept = 0, masks = 0, lines = 0, split = 0, total = 0, max_total = 0;
    int frames = 0;
  };

  void on_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void on_cloud(const PointCloud2::ConstSharedPtr & msg);
  void on_image(const Image::ConstSharedPtr & msg);
  void on_labels(const Image::ConstSharedPtr & msg);
  void on_optimized(const Detections::ConstSharedPtr & msg);
  void flush();
  void flush_scans();
  void flush_labels();
  void publish_detections(
    const builtin_interfaces::msg::Time & stamp, const std::string & frame, const Cloud & xyz,
    const std::vector<int> & cls, const Labels & line, const std::map<int, int> & ids);
  void publish_clusters(
    const builtin_interfaces::msg::Time & stamp, const std::string & frame, const Cloud & xyz,
    const std::vector<int> & cls, const std::vector<int> & seg, const Labels & line);
  void publish_overlay(const Projection & proj, const Indices & kept, const Labels & line);
  void publish_markers(
    const builtin_interfaces::msg::Time & stamp, const std::string & frame, const Cloud & xyz,
    const std::vector<int> & cls, const Labels & line);
  void publish_map();
  void log_stats(const Stats & frame);
  std::string name_of(int cls) const;

  std::string split_method_;
  std::map<int, double> tols_;
  std::map<int, double> gaps_;
  std::map<int, double> radius_;
  std::map<int, std::string> names_;
  int split_min_points_;
  double split_min_length_;
  double max_rate_hz_;
  int map_min_neighbours_;
  std::string detection_frame_;
  std::string detection_frame_id_;
  std::array<double, 16> line_cov_{};

  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  LineLandmarks landmarks_;
  std::deque<PointCloud2::ConstSharedPtr> pending_scans_;
  std::deque<Image::ConstSharedPtr> pending_labels_;
  Image::ConstSharedPtr image_;
  std::optional<rclcpp::Time> last_stamp_;
  Stats stats_;
  std::chrono::steady_clock::time_point stats_since_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<Image>::SharedPtr sub_image_;
  rclcpp::Subscription<Image>::SharedPtr sub_labels_;
  rclcpp::Subscription<Detections>::SharedPtr sub_optimized_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_clusters_;
  rclcpp::Publisher<Image>::SharedPtr pub_overlay_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_map_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
  rclcpp::Publisher<Detections>::SharedPtr pub_detections_;
  rclcpp::TimerBase::SharedPtr map_timer_;
};

}  // namespace axial_perception
