#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "axial_perception/types.hpp"

namespace axial_perception
{

using Labels = std::vector<int>;
using Rng = std::mt19937_64;

struct Line
{
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  Eigen::Vector3d d = Eigen::Vector3d::UnitX();
};

struct Section
{
  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
  Eigen::Vector3d to_cam = Eigen::Vector3d::Zero();
  double front = 0.0;
  double band = 0.0;
  double width = 0.0;
};

struct Circle
{
  Eigen::Vector2d centre = Eigen::Vector2d::Zero();
  double radius = 0.0;
  double span = 0.0;
  int inliers = 0;
};

struct Mst
{
  std::vector<int> parent;
  std::vector<double> w;
  Indices order;
};

using SplitFn = Labels (*)(const Cloud & pts, double tol, int min_points, double gap, Rng & rng);

Line fit_line(const Cloud & pts);
Eigen::Vector3d spread(const Cloud & pts, const Eigen::Vector3d & c);
Section section(const Cloud & pts, const Line & line, double tol, double trim);
double seen_width(const Cloud & pts);
Eigen::Vector3d axis_from_surface(const Eigen::Vector3d & c, const Eigen::Vector3d & d, double radius);
Eigen::Vector3d axis_from_arc(const Cloud & pts, const Line & line, double radius, double trim = 0.0);
std::optional<Circle> fit_circle(const Eigen::Matrix2Xd & q, double band, int iterations = 300, uint64_t seed = 0);
Eigen::ArrayXd line_distance(const Cloud & pts, const Line & line);
std::map<int, Indices> groups(const Labels & labels);

Labels gap_split(const Cloud & pts, const Labels & labels, double gap, int min_points);
Labels merge_lines(const Cloud & pts, Labels labels, double tol, double share = 0.95);
Labels finish(const Cloud & pts, const Labels & labels, double tol, int min_points, double gap);
Labels drop_short(const Cloud & pts, const Labels & labels, double min_length);
Labels surface_filter(
  const Cloud & pts, const std::vector<int> & cls, const std::vector<int> & seg, const Labels & labels,
  const std::vector<int> & classes, double tol, double trim, int min_points,
  const std::map<int, double> & radius = {}, double min_width = 0.0);

Mst mst(const Cloud & pts);
Indices longest_path(const Indices & nodes, const std::vector<int> & parent, const Cloud & pts);
Labels segmented_lsq(const Cloud & p, double penalty);

Labels no_split(const Cloud & pts, double tol, int min_points, double gap, Rng & rng);
Labels ransac(const Cloud & pts, double tol, int min_points, double gap, Rng & rng);
Labels pearl(const Cloud & pts, double tol, int min_points, double gap, Rng & rng);
Labels jlinkage(const Cloud & pts, double tol, int min_points, double gap, Rng & rng);
Labels mst_dp(const Cloud & pts, double tol, int min_points, double gap, Rng & rng);

const std::vector<std::string> & split_methods();
SplitFn split_method(const std::string & name);

Labels split_clusters(
  const Cloud & pts, const std::vector<int> & cls, const std::vector<int> & seg,
  const std::string & method, const std::map<int, double> & tols, int min_points,
  const std::map<int, double> & gaps, double min_length = 0.0, uint64_t seed = 0);

}  // namespace axial_perception
