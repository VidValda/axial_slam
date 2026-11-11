#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "bench.hpp"
#include "axial_perception/occlusion.hpp"
#include "axial_perception/segments.hpp"

using namespace axial_perception;
using namespace axial_perception::bench;

namespace
{

std::vector<int> gt_labels(const ClusterFrame & f, const Sections & s, double tol)
{
  std::vector<int> g = section_ids(f.world(), f.cls, s, tol);
  std::map<int, std::vector<int>> runs;
  for (size_t i = 0; i < g.size(); ++i) {
    if (g[i] >= 0) {
      runs[f.seg[i]].push_back(s.run[static_cast<size_t>(g[i])]);
    }
  }
  for (const auto & [seg, r] : runs) {
    const int main = argmax_count(r);
    for (size_t i = 0; i < g.size(); ++i) {
      if (f.seg[i] == seg && g[i] >= 0 && s.run[static_cast<size_t>(g[i])] != main) {
        g[i] = -2;
      }
    }
  }
  return g;
}

struct Score
{
  double lines, purity, mixed, angle, tilted, bad, found, over, assigned, ms, ms_max;
};

Score score(
  const std::vector<ClusterFrame> & frames, const std::vector<std::vector<int>> & gt, const Sections & sections,
  const std::string & method, const Params & p, std::optional<int> only = std::nullopt, double visible = 0.5)
{
  double lines = 0, found = 0, shown_total = 0, over = 0, mixed = 0, bad = 0, assigned = 0, on = 0;
  std::vector<double> purity, angles, times;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    const auto & f = frames[fi];
    const Cloud world = f.world();
    const auto start = std::chrono::steady_clock::now();
    Labels line = split_clusters(f.xyz, f.cls, f.seg, method, p.tols, p.split_min_points, p.gaps, p.min_length);
    times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    std::vector<int> g = gt[fi];
    if (only) {
      for (size_t i = 0; i < g.size(); ++i) {
        if (f.cls[i] != *only) {
          line[i] = 0;
          g[i] = -1;
        }
      }
    }
    std::map<int, std::vector<int>> major;
    for (const auto & [k, sel] : groups(line)) {
      std::vector<int> s;
      for (int i : sel) {
        if (g[static_cast<size_t>(i)] >= 0) {
          s.push_back(g[static_cast<size_t>(i)]);
        }
      }
      ++lines;
      if (s.empty()) {
        continue;
      }
      const int main = argmax_count(s);
      const double share = static_cast<double>(std::count(s.begin(), s.end(), main)) / static_cast<double>(s.size());
      purity.push_back(share);
      mixed += share < 0.9;
      major[main].push_back(k);
      const Eigen::Vector3d d = fit_line(select(world, sel)).d;
      angles.push_back(std::acos(std::min(std::abs(d.dot(sections.axis(static_cast<size_t>(main)))), 1.0)) * 180 /
        M_PI);
      bad += share < 0.9 || angles.back() > 5;
    }
    std::map<int, std::vector<double>> along;
    for (size_t i = 0; i < g.size(); ++i) {
      if (g[i] >= 0) {
        along[g[i]].push_back(world.col(static_cast<Eigen::Index>(i)).dot(sections.axis(static_cast<size_t>(g[i]))));
        ++on;
        assigned += line[i] > 0;
      }
    }
    for (const auto & [s, t] : along) {
      if (*std::max_element(t.begin(), t.end()) - *std::min_element(t.begin(), t.end()) < visible) {
        continue;
      }
      ++shown_total;
      found += major.count(s) > 0;
      over += major.count(s) && major.at(s).size() > 1;
    }
  }
  size_t tilted = 0;
  for (double a : angles) {
    tilted += a > 5;
  }
  const double nl = std::max(lines, 1.0);
  const double ns = std::max(shown_total, 1.0);
  return {
    lines / static_cast<double>(frames.size()), purity.empty() ? 0.0 : 100 * mean(purity), 100 * mixed / nl,
    angles.empty() ? 0.0 : median(angles),
    angles.empty() ? 0.0 : 100.0 * static_cast<double>(tilted) / static_cast<double>(angles.size()),
    100 * bad / nl, 100 * found / ns, 100 * over / ns, 100 * assigned / std::max(on, 1.0), 1e3 * mean(times),
    1e3 * *std::max_element(times.begin(), times.end())};
}

void row(const std::string & name, const Score & s)
{
  std::printf(
    "%-9s %6.1f %7.1f %6.1f %6.1f %7.1f %6.1f %6.1f %6.1f %7.1f %6.1f %6.1f\n", name.c_str(), s.lines, s.purity,
    s.mixed, s.angle, s.tilted, s.bad, s.found, s.over, s.assigned, s.ms, s.ms_max);
}

std::string join(const std::map<int, double> & m)
{
  std::string out = "{";
  for (const auto & [k, v] : m) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s%d: %g", out.size() > 1 ? ", " : "", k, v);
    out += buf;
  }
  return out + "}";
}

void run(const Args & args)
{
  const double tol = args.num("--tol", 0.1);
  const auto methods = args.list("--methods", split_methods());
  for (const auto & m : methods) {
    split_method(m);
  }
  if (!args.has("--load") && !args.has("--ns")) {
    throw std::invalid_argument("-n is needed to record");
  }
  std::vector<ClusterFrame> frames = args.has("--load") ? read_clusters(args.str("--load")) :
    record_clusters(args.num("-t", 10.0), args.str("--ns"));
  if (frames.empty()) {
    return;
  }
  if (args.has("--save")) {
    DatasetWriter out(args.str("--save"), Kind::Clusters);
    for (const auto & f : frames) {
      out.add(f);
    }
  }
  frames.erase(
    std::remove_if(frames.begin(), frames.end(), [](const ClusterFrame & f) {return f.xyz.cols() == 0;}),
    frames.end());
  if (frames.empty()) {
    return;
  }
  const Params p = load_params(args.required("--config"));
  const World world = load_world(args.required("--gt"));
  std::vector<std::vector<int>> gt;
  double other = 0, on_section = 0;
  for (const auto & f : frames) {
    gt.push_back(gt_labels(f, world.sections, tol));
    for (int g : gt.back()) {
      on_section += g != -1;
      other += g == -2;
    }
  }
  std::printf(
    "%zu frames with clusters, tol %s, gap %s, min_points %d, min_length %g; %.1f%% of on-section points are on "
    "another run than their mask (not scored)\n", frames.size(), join(p.tols).c_str(), join(p.gaps).c_str(),
    p.split_min_points, p.min_length, 100 * other / on_section);
  std::printf("method    lines  purity  mixed  angle  tilted    bad  found   over  assigned  ms/fr    max\n");
  for (const auto & m : methods) {
    row(m, score(frames, gt, world.sections, m, p));
    if (args.has("--by-class")) {
      for (const auto & [c, t] : p.tols) {
        row("  " + world.name_of(c), score(frames, gt, world.sections, m, p, c));
      }
    }
  }
  std::printf(
    "purity: share of a line's points on its main section; mixed: lines under 90%% purity;\n"
    "angle: median angle in deg between a line's fit and its main section; tilted: lines more than 5 deg off;\n"
    "bad: mixed or tilted lines;\n"
    "found/over: sections seen over 0.5 m or more that have at least one / more than one line; assigned: "
    "on-section points on a line\n");
}

}  // namespace

int main(int argc, char ** argv)
{
  const auto argv_clean = rclcpp::init_and_remove_ros_arguments(argc, argv);
  int code = 0;
  try {
    run(Args(argv_clean, {{"-t"}, {"--tol"}, {"-m", "--methods"}, {"-n", "--ns"}, {"--gt"}, {"-c", "--config"},
      {"--by-class"}, {"--save"}, {"--load"}}));
  } catch (const std::exception & e) {
    std::cerr << "line_segments_gt: " << e.what() << "\n"
              << "usage: line_segments_gt --gt world_gt.yaml -c params.yaml (-n drone0 [-t 10] [--save frames.bin] | "
      "--load frames.bin) [--tol 0.1] [-m ransac pearl] [--by-class]" << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
