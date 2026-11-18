#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

TEST(AlignedFilter, DropsLinesNeitherHorizontalNorVertical)
{
  Eigen::Matrix3d to_world;
  to_world << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  const auto tilt = [](double deg) {
      return Eigen::Vector3d(std::cos(deg * M_PI / 180.0), 0.0, std::sin(deg * M_PI / 180.0));
    };
  const std::vector<std::pair<Eigen::Vector3d, int>> lines = {
    {tilt(0.0), 6}, {tilt(5.0), 6}, {tilt(15.0), 6}, {tilt(87.0), 6}, {tilt(15.0), 4}};
  Cloud pts(3, 50 * static_cast<Eigen::Index>(lines.size()));
  std::vector<int> cls;
  Labels labels;
  for (size_t k = 0; k < lines.size(); ++k) {
    const Eigen::Vector3d start(4.0, 1.0 * static_cast<double>(k), 1.0);
    for (int i = 0; i < 50; ++i) {
      pts.col(static_cast<Eigen::Index>(50 * k) + i) = to_world.transpose() * (start + 0.04 * i * lines[k].first);
      cls.push_back(lines[k].second);
      labels.push_back(static_cast<int>(k) + 1);
    }
  }
  const Labels out = axial_perception::aligned_filter(pts, cls, labels, to_world, {6}, 8.0 * M_PI / 180.0);
  for (size_t k = 0; k < lines.size(); ++k) {
    EXPECT_EQ(out[50 * k], k == 2 ? 0 : static_cast<int>(k) + 1) << k;
  }
}

TEST(SnapAligned, SnapsNearVerticalAndNearHorizontalDirections)
{
  const double tol = 5.0 * M_PI / 180.0;
  const Eigen::Vector3d up = axial_perception::snap_aligned(Eigen::Vector3d(0.03, -0.02, -1.0), tol);
  EXPECT_TRUE(up.isApprox(Eigen::Vector3d(0.0, 0.0, -1.0)));
  const Eigen::Vector3d flat = axial_perception::snap_aligned(Eigen::Vector3d(3.0, 4.0, 0.2), tol);
  EXPECT_TRUE(flat.isApprox(Eigen::Vector3d(0.6, 0.8, 0.0)));
  const Eigen::Vector3d tilted(1.0, 0.0, 1.0);
  EXPECT_TRUE(axial_perception::snap_aligned(tilted, tol).isApprox(tilted.normalized()));
}

TEST(TiltCovariance, TightOnlyAlongTheTiltOfAHorizontalLine)
{
  const Eigen::Vector3d up(std::sin(0.1), 0.0, std::cos(0.1));
  const Eigen::Vector3d d(0.0, 1.0, 0.0);
  const Eigen::Matrix2d cov = axial_perception::tilt_covariance(d, up, 0.03, 0.001);
  const Eigen::Vector3d b1 = d.cross(Eigen::Vector3d::UnitZ()).normalized();
  const Eigen::Vector3d b2 = d.cross(b1);
  const Eigen::Vector2d tilt = Eigen::Vector2d(b1.dot(up), b2.dot(up)).normalized();
  const Eigen::Vector2d heading(-tilt.y(), tilt.x());
  EXPECT_NEAR(tilt.dot(cov * tilt), 0.001 * 0.001, 1e-12);
  EXPECT_NEAR(heading.dot(cov * heading), 0.03 * 0.03, 1e-12);
  EXPECT_NEAR(tilt.dot(cov * heading), 0.0, 1e-12);
  const Eigen::Matrix2d vertical = axial_perception::tilt_covariance(up, up, 0.03, 0.001);
  EXPECT_TRUE(vertical.isApprox(0.001 * 0.001 * Eigen::Matrix2d::Identity()));
}

TEST(SurfaceFilter, DropsTheBackgroundSeenPastTheEdgeOfARoundObject)
{
  const double r = 0.2;
  const Eigen::Vector3d axis(0.0, -1.5, 3.0);
  const Eigen::Vector3d to_cam = -Eigen::Vector3d(0.0, axis.y(), axis.z()).normalized();
  const Eigen::Vector3d side = Eigen::Vector3d::UnitX().cross(to_cam);
  std::vector<Eigen::Vector3d> pts;
  for (double x = -1.0; x <= 1.0; x += 0.05) {
    for (double a = -1.5; a <= 1.5; a += 0.15) {
      pts.push_back(axis + Eigen::Vector3d(x, 0, 0) + r * (std::cos(a) * to_cam + std::sin(a) * side));
    }
    for (double e : {-1.0, 1.0}) {
      for (double t = 0.1; t <= 0.4; t += 0.1) {
        pts.push_back(axis + Eigen::Vector3d(x, 0, 0) + e * r * side - (r + t) * to_cam);
      }
    }
  }
  Cloud cloud(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    cloud.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  const std::vector<int> cls(pts.size(), 6);
  const std::vector<int> seg(pts.size(), 1);
  const Labels line(pts.size(), 1);
  const Labels filtered = axial_perception::surface_filter(cloud, cls, seg, line, {6}, 0.05, 0.02, 10);
  size_t surface = 0, surface_kept = 0, background_kept = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const Eigen::Vector3d v = pts[i] - axis;
    if (std::abs((v - v.x() * Eigen::Vector3d::UnitX()).norm() - r) < 1e-9) {
      ++surface;
      surface_kept += filtered[i] > 0;
    } else {
      background_kept += filtered[i] > 0;
    }
  }
  EXPECT_EQ(surface_kept, surface);
  EXPECT_EQ(background_kept, 0u);
  EXPECT_EQ(axial_perception::surface_filter(cloud, cls, seg, line, {4}, 0.05, 0.02, 10), line);
}

TEST(SurfaceFilter, DropsAParallelLineOfTheSameMaskBehindTheObject)
{
  const double r = 0.2;
  const Eigen::Vector3d axis(0.0, -1.5, 3.0);
  const Eigen::Vector3d to_cam = -Eigen::Vector3d(0.0, axis.y(), axis.z()).normalized();
  const Eigen::Vector3d side = Eigen::Vector3d::UnitX().cross(to_cam);
  std::vector<Eigen::Vector3d> pts;
  Labels line;
  for (double x = -1.0; x <= 1.0; x += 0.05) {
    for (double a = -1.2; a <= 1.2; a += 0.2) {
      pts.push_back(axis + Eigen::Vector3d(x, 0, 0) + r * (std::cos(a) * to_cam + std::sin(a) * side));
      line.push_back(1);
    }
    for (double t = -0.3; t <= 0.3; t += 0.1) {
      pts.push_back(axis + Eigen::Vector3d(x, 0, 0) - 0.5 * to_cam + t * side);
      line.push_back(2);
    }
  }
  Cloud cloud(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    cloud.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  const std::vector<int> cls(pts.size(), 6);
  const auto out = axial_perception::surface_filter(
    cloud, cls, std::vector<int>(pts.size(), 1), line, {6}, 0.05, 0.02, 10);
  for (size_t i = 0; i < pts.size(); ++i) {
    EXPECT_EQ(out[i], line[i] == 1 ? 1 : 0);
  }
  std::vector<int> seg(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    seg[i] = line[i];
  }
  EXPECT_EQ(axial_perception::surface_filter(cloud, cls, seg, line, {6}, 0.05, 0.02, 10), line);
}

TEST(SurfaceFilter, WithARadiusDropsAStripNarrowerThanTheObject)
{
  const double r = 0.2;
  const Eigen::Vector3d axis(0.0, -1.5, 3.0);
  const Eigen::Vector3d to_cam = -Eigen::Vector3d(0.0, axis.y(), axis.z()).normalized();
  const Eigen::Vector3d side = Eigen::Vector3d::UnitX().cross(to_cam);
  std::vector<Eigen::Vector3d> pts;
  Labels line;
  for (double x = -1.0; x <= 1.0; x += 0.05) {
    for (double a = -1.2; a <= 1.2; a += 0.2) {
      pts.push_back(axis + Eigen::Vector3d(x, 0, 0) + r * (std::cos(a) * to_cam + std::sin(a) * side));
      line.push_back(1);
    }
    for (double t = -0.03; t <= 0.03; t += 0.01) {
      pts.push_back(axis + Eigen::Vector3d(x, 2.0, 0) - 0.3 * to_cam + t * side);
      line.push_back(2);
    }
  }
  Cloud cloud(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    cloud.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  const std::vector<int> cls(pts.size(), 6);
  std::vector<int> seg(line.begin(), line.end());
  const auto out = axial_perception::surface_filter(cloud, cls, seg, line, {6}, 0.05, 0.05, 10, {{6, r}}, 0.5);
  for (size_t i = 0; i < pts.size(); ++i) {
    EXPECT_EQ(out[i], line[i] == 1 ? 1 : 0);
  }
  EXPECT_EQ(axial_perception::surface_filter(cloud, cls, seg, line, {6}, 0.05, 0.05, 10, {{6, r}}), line);
}

TEST(SurfaceFilter, KeepsCollinearPiecesOfOneObject)
{
  std::vector<Eigen::Vector3d> pts;
  Labels line;
  for (double z = 2.0; z <= 4.0; z += 0.05) {
    pts.emplace_back(-0.5 + 0.06 * (z - 2.0), -1.5, z);
    line.push_back(1);
  }
  for (double z = 6.0; z <= 9.0; z += 0.05) {
    pts.emplace_back(-0.5, -1.5, z);
    line.push_back(2);
  }
  Cloud cloud(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    cloud.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  const std::vector<int> cls(pts.size(), 6);
  EXPECT_EQ(axial_perception::surface_filter(
      cloud, cls, std::vector<int>(pts.size(), 1), line, {6}, 0.05, 0.02, 10), line);
}

namespace
{

using Profile = std::vector<Eigen::Vector2d>;

Profile circle(double r)
{
  Profile out;
  for (double a = 0.0; a < 2 * M_PI; a += 0.002) {
    out.emplace_back(r * std::cos(a), r * std::sin(a));
  }
  return out;
}

Profile rectangle(double w, double h, double yaw)
{
  const Eigen::Rotation2Dd rot(yaw);
  const std::array<Eigen::Vector2d, 4> c{
    Eigen::Vector2d(w / 2, h / 2), Eigen::Vector2d(-w / 2, h / 2), Eigen::Vector2d(-w / 2, -h / 2),
    Eigen::Vector2d(w / 2, -h / 2)};
  Profile out;
  for (size_t i = 0; i < 4; ++i) {
    const Eigen::Vector2d a = c[i], b = c[(i + 1) % 4];
    const int n = static_cast<int>((b - a).norm() / 0.002);
    for (int k = 0; k < n; ++k) {
      out.push_back(rot * (a + (b - a) * k / n));
    }
  }
  return out;
}

Profile scan(const Profile & profile, const Eigen::Vector2d & centre)
{
  double lo = M_PI, hi = -M_PI;
  for (const auto & q : profile) {
    const double a = std::atan2(centre.x() + q.x(), centre.y() + q.y());
    lo = std::min(lo, a);
    hi = std::max(hi, a);
  }
  Profile out;
  for (double a = lo; a <= hi; a += 0.4 * M_PI / 180.0) {
    const Eigen::Vector2d ray(std::sin(a), std::cos(a));
    double best = std::numeric_limits<double>::infinity();
    for (const auto & q : profile) {
      const Eigen::Vector2d p = centre + q;
      if (std::abs(ray.x() * p.y() - ray.y() * p.x()) < 0.003 && p.norm() < best) {
        best = p.norm();
      }
    }
    if (std::isfinite(best)) {
      out.push_back(best * ray);
    }
  }
  return out;
}

Profile embedded(double r, const Eigen::Vector2d & centre, double wall_yaw)
{
  Profile shape = circle(r);
  const Eigen::Vector2d along(std::cos(wall_yaw), std::sin(wall_yaw));
  const Eigen::Vector2d normal(-along.y(), along.x());
  const double facing = normal.dot(-centre) > 0 ? 1.0 : -1.0;
  shape.erase(
    std::remove_if(shape.begin(), shape.end(), [&](const Eigen::Vector2d & q) {return facing * normal.dot(q) < 0;}),
    shape.end());
  for (double t = r; t <= 1.0; t += 0.002) {
    shape.push_back(t * along);
    shape.push_back(-t * along);
  }
  Profile out;
  for (const auto & p : scan(shape, centre)) {
    if ((p - centre).norm() < r + 0.03) {
      out.push_back(p - centre);
    }
  }
  return out;
}

Cloud extrude(const Profile & profile, const Eigen::Vector2d & centre, bool cast = true)
{
  axial_perception::Rng rng(3);
  std::normal_distribution<double> noise(0.0, 0.005);
  std::vector<Eigen::Vector3d> pts;
  const Profile seen = cast ? scan(profile, centre) : profile;
  for (double y = -1.0; y <= 1.0; y += 0.1) {
    for (const auto & q : seen) {
      const Eigen::Vector2d p = cast ? q : Eigen::Vector2d(centre + q);
      pts.emplace_back(p.x() + noise(rng), y, p.y() + noise(rng));
    }
  }
  Cloud cloud(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    cloud.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  return cloud;
}

double error(const Eigen::Vector3d & axis, const Eigen::Vector2d & centre)
{
  return (Eigen::Vector2d(axis.x(), axis.z()) - centre).norm();
}

}  // namespace

Eigen::Matrix2Xd arc(double r, const Eigen::Vector2d & centre, double from, double to, Rng & rng)
{
  std::normal_distribution<double> noise(0.0, 0.005);
  std::vector<Eigen::Vector2d> pts;
  for (double a = from; a <= to; a += 0.01) {
    pts.push_back(centre + (r + noise(rng)) * Eigen::Vector2d(std::cos(a), std::sin(a)));
  }
  Eigen::Matrix2Xd q(2, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    q.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  return q;
}

TEST(FitCircle, FindsAHalfCircleOfAnyRadiusBesideAWall)
{
  Rng rng(1);
  for (const double r : {0.05, 0.14, 0.21, 0.35}) {
    const Eigen::Vector2d centre(1.0, -2.0);
    const Eigen::Matrix2Xd half = arc(r, centre, 0.0, M_PI, rng);
    Eigen::Matrix2Xd q(2, half.cols() + 60);
    q.leftCols(half.cols()) = half;
    for (int i = 0; i < 30; ++i) {
      q.col(half.cols() + 2 * i) = centre + Eigen::Vector2d(r + 0.01 * i, 0.0);
      q.col(half.cols() + 2 * i + 1) = centre - Eigen::Vector2d(r + 0.01 * i, 0.0);
    }
    const auto c = axial_perception::fit_circle(q, 0.02);
    ASSERT_TRUE(c) << r;
    EXPECT_LT(std::abs(c->radius - r), 0.01) << r;
    EXPECT_LT((c->centre - centre).norm(), 0.01) << r;
    EXPECT_NEAR(c->span, M_PI, 0.15) << r;
  }
}

TEST(FitCircle, SpanIsTheArcCoveredByTheInliers)
{
  Rng rng(2);
  const auto c = axial_perception::fit_circle(arc(0.2, Eigen::Vector2d::Zero(), 2.0, 3.0, rng), 0.02);
  ASSERT_TRUE(c);
  EXPECT_NEAR(c->span, 1.0, 0.05);
  EXPECT_FALSE(axial_perception::fit_circle(Eigen::Matrix2Xd(2, 5), 0.02));
}

TEST(AxisFromArc, WithTheLearnedRadiusFindsTheAxisOfAPillarHalfInAWall)
{
  for (const double r : {0.14, 0.21}) {
    for (const double yaw : {0.0, 0.6, 1.2}) {
      const Eigen::Vector2d centre(0.8, 3.0);
      const Cloud cloud = extrude(embedded(r, centre, yaw), centre, false);
      const auto line = axial_perception::fit_line(cloud);
      EXPECT_LT(error(axial_perception::axis_from_arc(cloud, line, r, 0.05), centre), 0.03) << r << " " << yaw;
      EXPECT_GT(error(line.c, centre), 0.05) << r << " " << yaw;
    }
  }
}

TEST(AxisFromArc, ASquareWithTheRadiusFittedToAllItsSidesStaysWithinAFewCentimetres)
{
  const Eigen::Vector2d centre(0.5, 3.0);
  std::vector<Eigen::Vector2d> all;
  for (double yaw = 0.0; yaw < 2 * M_PI; yaw += 0.2) {
    for (const auto & p : scan(rectangle(0.4, 0.4, yaw), centre)) {
      all.push_back(Eigen::Rotation2Dd(-yaw) * (p - centre));
    }
  }
  Eigen::Matrix2Xd q(2, static_cast<Eigen::Index>(all.size()));
  for (size_t i = 0; i < all.size(); ++i) {
    q.col(static_cast<Eigen::Index>(i)) = all[i];
  }
  const auto c = axial_perception::fit_circle(q, 0.05);
  ASSERT_TRUE(c);
  EXPECT_GT(c->radius, 0.2);
  EXPECT_LT(c->radius, 0.2 * std::sqrt(2.0));
  for (double yaw = 0.0; yaw < M_PI / 2; yaw += 0.1) {
    const Cloud cloud = extrude(rectangle(0.4, 0.4, yaw), centre);
    const auto line = axial_perception::fit_line(cloud);
    EXPECT_LT(error(axial_perception::axis_from_arc(cloud, line, c->radius, 0.05), centre), 0.06) << yaw;
  }
}
