#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "axial_perception/segments.hpp"
#include "axial_perception/types.hpp"

namespace axial_perception
{

struct Landmark
{
  int class_id = 0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
  Cloud points = Cloud(3, 0);
  std::map<std::array<int64_t, 3>, std::pair<Eigen::Vector3d, int>> cells;
  double radius = 0.0;
  double class_radius = 0.0;
  Eigen::Vector3d seen_from = Eigen::Vector3d::Zero();
  int fitted_at = 0;
  int hits = 0;
};

Eigen::Vector3d canonical(const Eigen::Vector3d & d);
Eigen::Vector3d perpendicular(const Eigen::Vector3d & v, const Eigen::Vector3d & d);
Eigen::Matrix3d rotation(const Eigen::Vector3d & a, const Eigen::Vector3d & b);

class LineLandmarks
{
public:
  explicit LineLandmarks(
    double spawn_threshold, std::optional<double> min_anchor_separation = std::nullopt,
    double max_angle = 0.35, double voxel = 0.05, double min_elongation = 0.0);

  void set_class_thresholds(int class_id, double spawn_threshold, double min_anchor_separation);
  void set_radius_fit(double band, double min_span, double max_range, int refit_hits = 10);
  double radius(int gid);
  void set_min_width(double min_width);
  bool accepts(const Cloud & pts, const Eigen::Vector3d & c) const;
  std::map<int, int> update(
    const Cloud & xyz, const std::vector<int> & cls, const Labels & line, const std::map<int, double> & widths = {},
    const std::optional<Eigen::Vector3d> & viewer = std::nullopt);
  std::pair<int, double> nearest(int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & d) const;
  int assign(
    int class_id, const Eigen::Vector3d & c, const Eigen::Vector3d & d, const Cloud & pts = Cloud(3, 0),
    double width = 0.0, const std::optional<Eigen::Vector3d> & viewer = std::nullopt);
  void set_anchor(int gid, const Eigen::Vector3d & point, const Eigen::Vector3d & direction, int class_id = 0);
  bool snap(int gid, const Eigen::Vector3d & point, const Eigen::Vector3d & direction);
  Cloud voxelize(const Cloud & pts) const;
  Cloud dense_points(int gid, int min_neighbours) const;

  const std::map<int, Landmark> & landmarks() const {return landmarks_;}

private:
  std::pair<double, double> thresholds(int class_id) const;
  void add_section(Landmark & lm, const Cloud & pts, const std::optional<Eigen::Vector3d> & viewer) const;

  double spawn_threshold_;
  double min_anchor_separation_;
  double cos_max_angle_;
  double voxel_;
  double min_elongation_;
  double radius_band_ = 0.0;
  double radius_min_span_ = 0.0;
  double radius_max_range_ = 0.0;
  int radius_refit_hits_ = 10;
  double min_width_ = 0.0;
  std::map<int, std::pair<double, double>> class_thresholds_;
  std::map<int, Landmark> landmarks_;
  int next_id_ = 1;
};

}  // namespace axial_perception
