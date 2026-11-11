#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "axial_perception/landmarks.hpp"

using axial_perception::Cloud;
using axial_perception::Labels;
using axial_perception::LineLandmarks;

namespace
{

const Eigen::Vector3d UP = Eigen::Vector3d::UnitZ();

Cloud pillar(double x, double y, double z0 = 0.0, double z1 = 3.0, double step = 0.02, double radius = 0.0)
{
  std::vector<Eigen::Vector3d> pts;
  const double da = step / std::max(radius, step);
  for (double h = z0; h < z1 - 1e-12; h += step) {
    if (radius > 0) {
      for (double t = -1.2; t < 1.2 - 1e-12; t += da) {
        pts.emplace_back(x + radius * std::cos(t), y + radius * std::sin(t), h);
      }
    } else {
      pts.emplace_back(x, y, h);
    }
  }
  Cloud out(3, static_cast<Eigen::Index>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    out.col(static_cast<Eigen::Index>(i)) = pts[i];
  }
  return out;
}

struct Frame
{
  Cloud xyz;
  std::vector<int> cls;
  Labels line;
};

Frame frame(const std::vector<Cloud> & clouds)
{
  Frame f;
  Eigen::Index n = 0;
  for (const auto & c : clouds) {
    n += c.cols();
  }
  f.xyz.resize(3, n);
  Eigen::Index at = 0;
  for (size_t k = 0; k < clouds.size(); ++k) {
    f.xyz.middleCols(at, clouds[k].cols()) = clouds[k];
    at += clouds[k].cols();
    f.line.insert(f.line.end(), static_cast<size_t>(clouds[k].cols()), static_cast<int>(k) + 1);
  }
  f.cls.assign(static_cast<size_t>(n), 8);
  return f;
}

Eigen::Vector3d mean(const Cloud & pts)
{
  return pts.rowwise().mean();
}

}  // namespace

TEST(Landmarks, ShiftAlongAxisKeepsId)
{
  LineLandmarks lm(0.3);
  const int a = lm.assign(8, {1.0, 2.0, 0.5}, UP);
  const int b = lm.assign(8, {1.05, 2.0, 2.5}, -UP);
  EXPECT_EQ(a, b);
  EXPECT_EQ(lm.landmarks().size(), 1u);
}

TEST(Landmarks, FarLineSpawns)
{
  LineLandmarks lm(0.3);
  EXPECT_NE(lm.assign(8, {0, 0, 1}, UP), lm.assign(8, {1, 0, 1}, UP));
}

TEST(Landmarks, OtherClassSpawns)
{
  LineLandmarks lm(0.3);
  EXPECT_NE(lm.assign(8, {0, 0, 1}, UP), lm.assign(4, {0, 0, 1}, UP));
}

TEST(Landmarks, AngleGateSpawns)
{
  LineLandmarks lm(0.3, std::nullopt, 0.2);
  EXPECT_NE(lm.assign(8, {0, 0, 1}, UP), lm.assign(8, {0, 0, 1}, Eigen::Vector3d::UnitX()));
}

TEST(Landmarks, ForceMatchBandDoesNotTouchLandmark)
{
  LineLandmarks lm(0.3, 1.0);
  const int a = lm.assign(8, {0, 0, 1.5}, UP, pillar(0, 0));
  const auto n = lm.landmarks().at(a).points.cols();
  EXPECT_EQ(lm.assign(8, {0.6, 0, 1.5}, UP, pillar(0.6, 0)), a);
  EXPECT_LT(lm.landmarks().at(a).point.head<2>().norm(), 1e-9);
  EXPECT_EQ(lm.landmarks().at(a).points.cols(), n);
}

TEST(Landmarks, MatchRefitsLineToAllPoints)
{
  LineLandmarks lm(0.3, std::nullopt, 0.35, 0.0);
  const int a = lm.assign(8, {0, 0, 1.5}, {0.1, 0, 1}, pillar(0, 0, 0, 1));
  lm.assign(8, {0, 0, 2}, UP, pillar(0, 0, 1, 3));
  EXPECT_LT((lm.landmarks().at(a).direction - UP).norm(), 1e-9);
}

TEST(Landmarks, SeparationBelowThresholdThrows)
{
  EXPECT_THROW(LineLandmarks(0.5, 0.2), std::invalid_argument);
}

TEST(Landmarks, VoxelKeepsCloudBounded)
{
  LineLandmarks lm(0.3, std::nullopt, 0.35, 0.05);
  const Cloud pts = pillar(0, 0);
  int gid = 0;
  for (int i = 0; i < 10; ++i) {
    gid = lm.assign(8, mean(pts), UP, pts);
  }
  EXPECT_EQ(lm.landmarks().at(gid).points.cols(), lm.voxelize(pts).cols());
}

TEST(Landmarks, SetAnchorReservesId)
{
  LineLandmarks lm(0.3);
  lm.set_anchor(7, {5, 0, 0}, UP, 8);
  EXPECT_EQ(lm.assign(8, {5.1, 0, 1}, UP), 7);
  EXPECT_EQ(lm.assign(8, {9, 0, 1}, UP), 8);
}

TEST(Landmarks, SetAnchorMovesPointsWithTheLine)
{
  LineLandmarks lm(0.3, std::nullopt, 0.35, 0.0);
  const int gid = lm.assign(8, {0, 0, 1.5}, UP, pillar(0, 0));
  const Eigen::Vector3d tilted = Eigen::Vector3d(0.1, 0, 1).normalized();
  lm.set_anchor(gid, {2, 1, 0}, tilted);
  const auto l = axial_perception::fit_line(lm.landmarks().at(gid).points);
  EXPECT_NEAR(std::abs(l.d.dot(tilted)), 1.0, 1e-9);
  EXPECT_LT(axial_perception::perpendicular(l.c - Eigen::Vector3d(2, 1, 0), tilted).norm(), 1e-9);
  EXPECT_EQ(lm.assign(8, {2.15, 1, 0}, tilted, pillar(2, 1, 0, 3, 0.02, 0.15)), gid);
}

TEST(Landmarks, SnapSkipsEstimatesBeyondTheSeparation)
{
  LineLandmarks lm(0.3, 1.0, 0.2);
  const int gid = lm.assign(8, {0, 0, 1.5}, UP, pillar(0, 0));
  EXPECT_FALSE(lm.snap(gid, {1.5, 0, 0}, UP));
  EXPECT_FALSE(lm.snap(gid, {0, 0, 0}, Eigen::Vector3d::UnitX()));
  EXPECT_LT(lm.landmarks().at(gid).point.head<2>().norm(), 1e-9);
  EXPECT_TRUE(lm.snap(gid, {0.8, 0, 0}, UP));
  EXPECT_LT((lm.landmarks().at(gid).point.head<2>() - Eigen::Vector2d(0.8, 0)).norm(), 1e-9);
  EXPECT_FALSE(lm.snap(99, {0, 0, 0}, UP));
}

TEST(Landmarks, UpdateMapsLinesOfAFrame)
{
  LineLandmarks lm(0.3);
  const Frame f = frame({pillar(0, 0), pillar(2, 0)});
  const auto first = lm.update(f.xyz, f.cls, f.line);
  EXPECT_EQ(lm.update(f.xyz, f.cls, f.line), first);
  EXPECT_EQ(first, (std::map<int, int>{{1, 1}, {2, 2}}));
}

TEST(Landmarks, UpdateDropsLinesWithoutADirection)
{
  LineLandmarks lm(0.3, std::nullopt, 0.35, 0.05, 3.0);
  const Frame f = frame({pillar(0, 0, 0, 0.3, 0.02, 0.2), pillar(2, 0, 0, 3, 0.02, 0.15)});
  EXPECT_EQ(lm.update(f.xyz, f.cls, f.line), (std::map<int, int>{{2, 1}}));
}

TEST(Landmarks, DensePointsDropStrays)
{
  LineLandmarks lm(0.3, std::nullopt, 0.35, 0.03);
  const Cloud pts = pillar(0, 0, 0, 3, 0.01, 0.15);
  Cloud all(3, pts.cols() + 2);
  all << pts, Eigen::Vector3d(0.4, 0, 1), Eigen::Vector3d(0, 0.5, 2);
  const int gid = lm.assign(8, mean(pts), UP, all);
  const Cloud kept = lm.dense_points(gid, 4);
  EXPECT_GE(kept.cols(), lm.landmarks().at(gid).points.cols() - 2 - 20);
  for (const Eigen::Vector3d & s : {Eigen::Vector3d(0.4, 0, 1), Eigen::Vector3d(0, 0.5, 2)}) {
    EXPECT_GT((kept.colwise() - s).colwise().norm().minCoeff(), 0.1);
  }
}

TEST(Landmarks, RotationTakesAOntoB)
{
  const Eigen::Vector3d a = Eigen::Vector3d(1, 2, 3).normalized();
  const Eigen::Vector3d b = Eigen::Vector3d(-1, 0.5, 0.2).normalized();
  EXPECT_LT((axial_perception::rotation(a, b) * a - b).norm(), 1e-12);
  EXPECT_TRUE(axial_perception::rotation(a, -a).isIdentity());
}
