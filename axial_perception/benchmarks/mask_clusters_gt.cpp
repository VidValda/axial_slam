#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "bench.hpp"
#include "axial_perception/occlusion.hpp"

using namespace axial_perception;
using namespace axial_perception::bench;

namespace
{

struct Row
{
  int cls;
  double z;
  double d;
};

void run(const Args & args)
{
  const double tol = args.num("--tol", 0.1);
  const World world = load_world(args.required("--gt"));
  const auto frames = record_clusters(args.num("-t", 10.0), args.required("--ns"));
  if (frames.empty()) {
    return;
  }
  std::vector<Row> rows;
  for (const auto & f : frames) {
    const Cloud p = f.world();
    for (Eigen::Index i = 0; i < p.cols(); ++i) {
      double d = std::numeric_limits<double>::infinity();
      for (size_t k = 0; k < world.sections.size(); ++k) {
        if (world.sections.cls[k] == f.cls[static_cast<size_t>(i)]) {
          d = std::min(d, world.sections.distance(k, p.col(i)));
        }
      }
      rows.push_back({f.cls[static_cast<size_t>(i)], f.xyz(2, i), d});
    }
  }
  const double n = static_cast<double>(frames.size());
  std::printf(
    "%zu frames, %.0f points/frame, off-surface = more than %.0f cm\n", frames.size(),
    static_cast<double>(rows.size()) / n, tol * 100);
  for (const auto & [c, name] : world.axial_names) {
    std::vector<double> d;
    for (const auto & r : rows) {
      if (r.cls == c) {
        d.push_back(r.d);
      }
    }
    if (d.empty()) {
      continue;
    }
    size_t off = 0;
    for (double x : d) {
      off += x > tol;
    }
    std::printf(
      "  %-6s %6.0f pts/frame, off-surface %5.1f%%, median distance %.1f cm\n", name.c_str(),
      static_cast<double>(d.size()) / n, 100.0 * static_cast<double>(off) / static_cast<double>(d.size()),
      median(d) * 100);
  }
  for (const auto & [lo, hi] : std::vector<std::pair<int, int>>{{0, 4}, {4, 8}, {8, 12}, {12, 30}}) {
    size_t count = 0;
    size_t off = 0;
    for (const auto & r : rows) {
      if (r.z >= lo && r.z < hi) {
        ++count;
        off += r.d > tol;
      }
    }
    if (count) {
      std::printf(
        "  depth %2d-%2d m: off-surface %5.1f%%\n", lo, hi,
        100.0 * static_cast<double>(off) / static_cast<double>(count));
    }
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  const auto argv_clean = rclcpp::init_and_remove_ros_arguments(argc, argv);
  int code = 0;
  try {
    run(Args(argv_clean, {{"-t"}, {"--tol"}, {"-n", "--ns"}, {"--gt"}}));
  } catch (const std::exception & e) {
    std::cerr << "mask_clusters_gt: " << e.what() << "\n"
              << "usage: mask_clusters_gt --gt world_gt.yaml -n drone0 [-t 10] [--tol 0.1]" << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
