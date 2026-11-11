#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "axial_perception/camera.hpp"
#include "axial_perception/projector.hpp"
#include "axial_perception/types.hpp"

namespace axial_perception::bench
{

class Args
{
public:
  Args(const std::vector<std::string> & argv, const std::vector<std::vector<std::string>> & keys);

  bool has(const std::string & key) const;
  std::string str(const std::string & key, const std::string & fallback = "") const;
  std::string required(const std::string & key) const;
  double num(const std::string & key, double fallback) const;
  std::vector<std::string> list(const std::string & key, const std::vector<std::string> & fallback) const;

private:
  std::map<std::string, std::vector<std::string>> values_;
  std::map<std::string, std::string> canonical_;
};

struct Sections
{
  std::vector<Eigen::Vector3d> a;
  std::vector<Eigen::Vector3d> b;
  std::vector<double> r;
  std::vector<int> cls;
  std::vector<int> run;

  size_t size() const {return a.size();}
  Eigen::Vector3d axis(size_t k) const {return (b[k] - a[k]).normalized();}
  double distance(size_t k, const Eigen::Vector3d & p) const;
};

struct Pillar
{
  Eigen::Vector2d centre;
  double radius;
  double height;
};

struct World
{
  std::vector<std::pair<int, std::string>> axial_names;
  Sections sections;
  std::vector<Pillar> pillars;

  std::string name_of(int cls) const;
};

World load_world(const std::string & path);

std::vector<int> section_ids(
  const Cloud & world, const std::vector<int> & cls, const Sections & s, double tol,
  const Indices & allowed = {});

struct Params
{
  ProjectorParams projector;
  std::string split_method = "ransac";
  std::map<int, double> tols;
  std::map<int, double> gaps;
  int split_min_points = 5;
  double min_length = 0.5;
  double map_spawn_threshold = 0.5;
  double map_min_separation = 1.0;
  double map_max_angle = 0.35;
  double map_voxel = 0.03;
  double map_min_elongation = 3.0;
  int map_min_neighbours = 8;
};

Params load_params(const std::string & path);

struct RawFrame
{
  Cloud xyz = Cloud(3, 0);
  int height = 0;
  int width = 0;
  std::vector<uint8_t> labels;
  Intrinsics K;
  Rigid pose;
  int64_t stamp = 0;
  int64_t n_points = 0;
};

struct ClusterFrame
{
  Cloud xyz = Cloud(3, 0);
  std::vector<int> cls;
  std::vector<int> seg;
  std::vector<int> line;
  Rigid pose;

  Cloud world() const {return pose.apply(xyz);}
};

enum class Kind : uint32_t { Raw = 1, Clusters = 2 };

class DatasetWriter
{
public:
  DatasetWriter(const std::string & path, Kind kind);
  void add(const RawFrame & f);
  void add(const ClusterFrame & f);
  size_t frames() const {return frames_;}

private:
  void write(const std::vector<uint8_t> & bytes);
  std::ofstream out_;
  Kind kind_;
  size_t frames_ = 0;
};

class DatasetReader
{
public:
  DatasetReader(const std::string & path, Kind kind);
  bool next(RawFrame & f);
  bool next(ClusterFrame & f);

private:
  std::optional<std::vector<uint8_t>> read();
  std::ifstream in_;
  Kind kind_;
};

std::vector<ClusterFrame> read_clusters(const std::string & path);

struct FramePoints
{
  Cloud xyz;
  std::vector<int> cls;
  std::vector<int> seg;
};

FramePoints frame_points(const RawFrame & f, const Params & p, const std::string & occlusion);

std::vector<ClusterFrame> record_clusters(double seconds, const std::string & ns);

double percentile(std::vector<double> x, double q);
double mean(const std::vector<double> & x);
int argmax_count(const std::vector<int> & values);

}  // namespace axial_perception::bench
