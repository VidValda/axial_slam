#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "bench.hpp"
#include "axial_perception/landmarks.hpp"
#include "axial_perception/segments.hpp"

using namespace axial_perception;
using namespace axial_perception::bench;

namespace
{

constexpr int kPillar = 8;

using FrameFn = std::function<void(const Cloud &, const std::vector<int> &, const Labels &)>;

void raw_frames(const std::string & path, const Params & p, const FrameFn & fn)
{
  DatasetReader in(path, Kind::Raw);
  RawFrame f;
  while (in.next(f)) {
    const FramePoints fp = frame_points(f, p, p.projector.occlusion_method);
    const Labels line = split_clusters(
      fp.xyz, fp.cls, fp.seg, p.split_method, p.tols, p.split_min_points, p.gaps, p.min_length);
    fn(f.pose.apply(fp.xyz), fp.cls, line);
  }
}

void cluster_frames(const std::string & path, const FrameFn & fn)
{
  DatasetReader in(path, Kind::Clusters);
  ClusterFrame f;
  while (in.next(f)) {
    if (f.xyz.cols() > 0) {
      fn(f.world(), f.cls, f.line);
    }
  }
}

std::string ids_repr(const std::map<int, std::vector<int>> & split)
{
  std::string out = "{";
  for (const auto & [j, g] : split) {
    out += (out.size() > 1 ? ", " : "") + std::to_string(j) + ": [";
    for (size_t k = 0; k < g.size(); ++k) {
      out += (k ? ", " : "") + std::to_string(g[k]);
    }
    out += "]";
  }
  return out + "}";
}

double share(const std::vector<double> & x, const std::function<bool(double)> & pred)
{
  return 100.0 * static_cast<double>(std::count_if(x.begin(), x.end(), pred)) / static_cast<double>(x.size());
}

void run(const Args & args)
{
  if (args.has("--raw") == args.has("--clusters")) {
    throw std::invalid_argument("one of --raw and --clusters is required");
  }
  const double match = args.num("--match", 0.3);
  Params p = load_params(args.required("--config"));
  const World world = load_world(args.required("--gt"));
  LineLandmarks lm(p.map_spawn_threshold, p.map_min_separation, p.map_max_angle, p.map_voxel, p.map_min_elongation);
  size_t frames = 0;
  size_t lines = 0;
  const FrameFn update = [&](const Cloud & xyz, const std::vector<int> & cls, const Labels & line) {
      ++frames;
      lines += lm.update(xyz, cls, line).size();
    };
  if (args.has("--raw")) {
    raw_frames(args.str("--raw"), p, update);
  } else {
    cluster_frames(args.str("--clusters"), update);
  }

  const auto & pillars = world.pillars;
  std::map<int, std::vector<int>> ids_of;
  std::vector<double> offset, tilt, gaps;
  for (const auto & [gid, a] : lm.landmarks()) {
    if (a.class_id != kPillar) {
      continue;
    }
    const Cloud pts = lm.dense_points(gid, p.map_min_neighbours);
    if (pts.cols() == 0 || pillars.empty()) {
      continue;
    }
    std::vector<int> order;
    std::map<int, int> votes;
    for (Eigen::Index i = 0; i < pts.cols(); ++i) {
      double best = std::numeric_limits<double>::infinity();
      int arg = 0;
      for (size_t j = 0; j < pillars.size(); ++j) {
        const double rd = (pts.col(i).head<2>() - pillars[j].centre).norm() - pillars[j].radius;
        if (rd < best) {
          best = rd;
          arg = static_cast<int>(j);
        }
      }
      gaps.push_back(best);
      if (std::abs(best) < match) {
        if (!votes.count(arg)) {
          order.push_back(arg);
        }
        ++votes[arg];
      }
    }
    int j = -1;
    for (int k : order) {
      if (j < 0 || votes[k] > votes[j]) {
        j = k;
      }
    }
    ids_of[j].push_back(gid);
    if (j >= 0) {
      const Pillar & pl = pillars[static_cast<size_t>(j)];
      const Eigen::Vector3d q = a.point + (pl.height / 2 - a.point[2]) / a.direction[2] * a.direction;
      offset.push_back((q.head<2>() - pl.centre).norm());
      tilt.push_back(std::acos(std::min(std::abs(a.direction[2]), 1.0)) * 180 / M_PI);
    }
  }
  std::map<int, std::vector<int>> split;
  size_t landmarks = 0;
  size_t mapped = 0;
  for (const auto & [j, g] : ids_of) {
    landmarks += g.size();
    mapped += j >= 0;
    if (j >= 0 && g.size() > 1) {
      split[j] = g;
    }
  }
  std::printf(
    "%zu frames, %zu accepted lines, %zu pillar landmarks, %zu of %zu pillars mapped\n", frames, lines, landmarks,
    mapped, pillars.size());
  std::printf("off any pillar: %zu landmarks\n", ids_of.count(-1) ? ids_of.at(-1).size() : 0);
  std::printf("split pillars: %zu %s\n", split.size(), ids_repr(split).c_str());
  if (!offset.empty()) {
    double rmin = pillars.front().radius, rmax = rmin;
    for (const auto & pl : pillars) {
      rmin = std::min(rmin, pl.radius);
      rmax = std::max(rmax, pl.radius);
    }
    std::printf(
      "axis offset from GT centre (m): median %.3f, p95 %.3f, max %.3f (radius %.2f-%.2f)\n", percentile(offset, 50),
      percentile(offset, 95), *std::max_element(offset.begin(), offset.end()), rmin, rmax);
    std::printf(
      "tilt (deg): median %.2f, p95 %.2f, max %.2f\n", percentile(tilt, 50), percentile(tilt, 95),
      *std::max_element(tilt.begin(), tilt.end()));
  }
  if (!gaps.empty()) {
    std::printf(
      "points: %zu, within 3 cm of the surface %.1f%%, 5 cm %.1f%%, more than 10 cm out %.2f%%\n", gaps.size(),
      share(gaps, [](double g) {return std::abs(g) < 0.03;}), share(gaps, [](double g) {return std::abs(g) < 0.05;}),
      share(gaps, [](double g) {return g > 0.1;}));
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  std::vector<std::string> argv_all(argv, argv + argc);
  try {
    run(Args(argv_all, {{"--raw"}, {"--clusters"}, {"--gt"}, {"-c", "--config"}, {"--match"}}));
  } catch (const std::exception & e) {
    std::cerr << "landmark_map_gt: " << e.what() << "\n"
              << "usage: landmark_map_gt (--raw raw.bin | --clusters clusters.bin) --gt world_gt.yaml -c params.yaml "
      "[--match 0.3]" << std::endl;
    return 1;
  }
  return 0;
}
