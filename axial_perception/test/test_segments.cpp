#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "axial_perception/segments.hpp"

using axial_perception::Cloud;
using axial_perception::Labels;
using axial_perception::Rng;

namespace
{

struct Polyline
{
  Cloud pts;
  std::vector<int> truth;
};

Polyline polyline(const std::vector<Eigen::Vector3d> & verts, double step = 0.05, double noise = 0.01)
{
  Rng rng(0);
  std::normal_distribution<double> n(0.0, noise);
  std::vector<Eigen::Vector3d> pts;
  Polyline out;
  for (size_t k = 0; k + 1 < verts.size(); ++k) {
    const Eigen::Vector3d a = verts[k];
    const Eigen::Vector3d b = verts[k + 1];
    const double dt = step / (b - a).norm();
    for (double t = 0; t < 1 - 1e-12; t += dt) {
      pts.push_back(a + t * (b - a) + Eigen::Vector3d(n(rng), n(rng), n(rng)));
      out.truth.push_back(static_cast<int>(k));
    }
  }
  out.pts.resize(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    out.pts.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  return out;
}

Cloud points(const std::vector<Eigen::Vector3d> & pts)
{
  Cloud out(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    out.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  return out;
}

double purity(const Labels & labels, const std::vector<int> & truth)
{
  std::map<int, std::map<int, int>> votes;
  int hit = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0) {
      ++votes[labels[i]][truth[i]];
      ++hit;
    }
  }
  int best = 0;
  for (const auto & [k, v] : votes) {
    int m = 0;
    for (const auto & [t, c] : v) {
      m = std::max(m, c);
    }
    best += m;
  }
  return static_cast<double>(best) / std::max(hit, 1);
}

std::set<int> values(const Labels & l)
{
  return {l.begin(), l.end()};
}

int max_of(const Labels & l)
{
  return l.empty() ? 0 : *std::max_element(l.begin(), l.end());
}

Labels split(const std::string & method, const Cloud & pts, double tol = 0.05, int min_points = 5, double gap = 0.5)
{
  Rng rng(0);
  return axial_perception::split_method(method)(pts, tol, min_points, gap, rng);
}

class AllMethods : public ::testing::TestWithParam<std::string> {};
class Splitters : public ::testing::TestWithParam<std::string> {};

}  // namespace

TEST_P(AllMethods, StraightIsOneLine)
{
  const auto p = polyline({{0, 0, 0}, {3, 0, 0}});
  EXPECT_EQ(values(split(GetParam(), p.pts)), (std::set<int>{1}));
}

TEST_P(AllMethods, ThickHalfCylinderIsOneLine)
{
  std::vector<Eigen::Vector3d> pts;
  for (int i = 0; i < 8; ++i) {
    const double a = M_PI * i / 7;
    for (double z = 0; z < 3 - 1e-9; z += 0.05) {
      pts.emplace_back(0.2 * std::cos(a), 0.2 * std::sin(a), z);
    }
  }
  EXPECT_EQ(values(split(GetParam(), points(pts), 0.3)), (std::set<int>{1}));
}

TEST_P(AllMethods, SmallClusterIsUnassigned)
{
  const auto p = polyline({{0, 0, 0}, {0.15, 0, 0}});
  EXPECT_EQ(values(split(GetParam(), p.pts)), (std::set<int>{0}));
}

TEST_P(AllMethods, Empty)
{
  EXPECT_TRUE(split(GetParam(), Cloud(3, 0)).empty());
}

TEST_P(Splitters, LSplitsAtCorner)
{
  const auto p = polyline({{0, 0, 0}, {2, 0, 0}, {2, 0, 1.5}});
  const Labels labels = split(GetParam(), p.pts);
  EXPECT_EQ(max_of(labels), 2);
  EXPECT_GT(purity(labels, p.truth), 0.9);
}

TEST_P(Splitters, UDipSplitsCollinearPieces)
{
  const auto p = polyline({{0, 0, 0}, {2, 0, 0}, {2, 0, -0.35}, {4.5, 0, -0.35}, {4.5, 0, 0}, {6.5, 0, 0}});
  const Labels labels = split(GetParam(), p.pts);
  EXPECT_GE(max_of(labels), 3);
  EXPECT_GT(purity(labels, p.truth), 0.9);
}

INSTANTIATE_TEST_SUITE_P(
  Segments, AllMethods, ::testing::ValuesIn(axial_perception::split_methods()),
  [](const auto & info) {return info.param;});
INSTANTIATE_TEST_SUITE_P(
  Segments, Splitters, ::testing::Values("ransac", "pearl", "jlinkage", "mst_dp"),
  [](const auto & info) {return std::string(info.param);});

TEST(Segments, GapSplitCutsCollinearPieces)
{
  std::vector<Eigen::Vector3d> v;
  for (double x : {0.0, 0.1, 0.2, 2.0, 2.1, 2.2}) {
    v.emplace_back(x, 0, 0);
  }
  const Cloud pts = points(v);
  EXPECT_EQ(axial_perception::gap_split(pts, Labels(6, 1), 0.5, 3), (Labels{1, 1, 1, 2, 2, 2}));
  EXPECT_EQ(axial_perception::gap_split(pts, Labels(6, 1), 0, 3), Labels(6, 1));
}

TEST(Segments, MergeLinesJoinsParallelButNotPerpendicular)
{
  std::vector<Eigen::Vector3d> near, cross;
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 20; ++i) {
      const double x = 0.1 * i;
      near.emplace_back(x, 0.1 * k, 0);
      cross.push_back(k == 0 ? Eigen::Vector3d(x, 0, 0) : Eigen::Vector3d(0, x, 0));
    }
  }
  Labels labels(40, 1);
  std::fill(labels.begin() + 20, labels.end(), 2);
  EXPECT_EQ(values(axial_perception::merge_lines(points(near), labels, 0.3)), (std::set<int>{1}));
  EXPECT_EQ(values(axial_perception::merge_lines(points(cross), labels, 0.3)), (std::set<int>{1, 2}));
}

TEST(Segments, DropShortUnassignsAndRenumbers)
{
  const Cloud pts = points({{0, 0, 0}, {0.1, 0, 0}, {0.2, 0, 0}, {5, 0, 0}, {5, 0.5, 0}, {5, 1, 0}});
  EXPECT_EQ(axial_perception::drop_short(pts, {1, 1, 1, 2, 2, 2}, 0.5), (Labels{0, 0, 0, 1, 1, 1}));
}

TEST(Segments, SplitClustersNumbersLinesPerFrame)
{
  const auto a = polyline({{0, 0, 0}, {2, 0, 0}, {2, 0, 1.5}});
  const auto b = polyline({{0, 1, 0}, {0, 1, 2}});
  Cloud pts(3, a.pts.cols() + b.pts.cols());
  pts << a.pts, b.pts;
  std::vector<int> seg(static_cast<size_t>(pts.cols()), 1);
  std::vector<int> cls(seg.size(), 4);
  std::fill(seg.begin() + a.pts.cols(), seg.end(), 2);
  std::fill(cls.begin() + a.pts.cols(), cls.end(), 8);
  const Labels line = axial_perception::split_clusters(
    pts, cls, seg, "ransac", {{4, 0.05}, {8, 0.1}}, 5, {{4, 0.5}, {8, 0.5}});
  std::set<int> first, second;
  for (size_t i = 0; i < line.size(); ++i) {
    (seg[i] == 1 ? first : second).insert(line[i]);
  }
  EXPECT_EQ(first, (std::set<int>{1, 2}));
  EXPECT_EQ(second, (std::set<int>{3}));
}

TEST(Segments, SplitClustersUnknownMethodThrows)
{
  EXPECT_THROW(
    axial_perception::split_clusters(Cloud(3, 0), {}, {}, "nope", {}, 5, {}), std::invalid_argument);
}

TEST(Segments, AxisFromSurfaceMovesFaceLineOntoAxis)
{
  const Eigen::Vector3d axis(3.0, 1.0, 0.5);
  const double r = 0.14;
  const Eigen::Vector3d face = axis - r * Eigen::Vector3d(3.0, 1.0, 0.0) / std::hypot(3.0, 1.0);
  const Eigen::Vector3d d = Eigen::Vector3d::UnitZ();
  EXPECT_LT((axial_perception::axis_from_surface(face, d, r).head<2>() - axis.head<2>()).norm(), 1e-12);
  EXPECT_LT((axial_perception::axis_from_surface(face, -d, r).head<2>() - axis.head<2>()).norm(), 1e-12);
  EXPECT_EQ(axial_perception::axis_from_surface(face, d, 0.0), face);
}

TEST(Segments, SegmentedLsqSplitsAnL)
{
  std::vector<Eigen::Vector3d> v;
  for (int i = 0; i < 10; ++i) {
    v.emplace_back(0.1 * i, 0, 0);
  }
  for (int i = 1; i <= 10; ++i) {
    v.emplace_back(0.9, 0.1 * i, 0);
  }
  const Labels l = axial_perception::segmented_lsq(points(v), 0.01);
  EXPECT_EQ(values(l), (std::set<int>{1, 2}));
  EXPECT_EQ(l.front(), 2);
  EXPECT_EQ(l.back(), 1);
}
