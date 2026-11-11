#include <gtest/gtest.h>

#include <Eigen/LU>

#include <cmath>
#include <limits>
#include <random>

#include "axial_perception/camera.hpp"

using axial_perception::Cloud;
using axial_perception::Intrinsics;

namespace
{

const Intrinsics K{400.0, 400.0, 320.0, 240.0};
constexpr int W = 640;
constexpr int H = 480;

Cloud cloud(std::initializer_list<Eigen::Vector3d> pts)
{
  Cloud out(3, static_cast<Eigen::Index>(pts.size()));
  Eigen::Index i = 0;
  for (const auto & p : pts) {
    out.col(i++) = p;
  }
  return out;
}

}  // namespace

TEST(Camera, QuatIdentity)
{
  EXPECT_TRUE(axial_perception::quat_to_matrix(0, 0, 0, 1).isApprox(Eigen::Matrix3d::Identity()));
}

TEST(Camera, QuatYaw90MapsXToY)
{
  const double s = std::sqrt(0.5);
  const Eigen::Vector3d y = axial_perception::quat_to_matrix(0, 0, s, s) * Eigen::Vector3d::UnitX();
  EXPECT_LT((y - Eigen::Vector3d::UnitY()).norm(), 1e-12);
}

TEST(Camera, QuatRandomIsRotation)
{
  std::mt19937 rng(0);
  std::normal_distribution<double> n;
  Eigen::Vector4d q(n(rng), n(rng), n(rng), n(rng));
  q.normalize();
  const Eigen::Matrix3d R = axial_perception::quat_to_matrix(q[0], q[1], q[2], q[3]);
  EXPECT_LT((R * R.transpose() - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
}

TEST(Camera, Transform)
{
  const double s = std::sqrt(0.5);
  axial_perception::Rigid tf{axial_perception::quat_to_matrix(0, 0, s, s), {1, 2, 3}};
  const Cloud out = tf.apply(cloud({{1, 0, 0}, {0, 0, 1}}));
  EXPECT_LT((out - cloud({{1, 3, 3}, {1, 2, 4}})).norm(), 1e-12);
}

TEST(Camera, ProjectKnownPixels)
{
  const auto p = axial_perception::project(cloud({{0, 0, 2}, {1, 0.5, 2}}), K, W, H, 0.1, 30.0);
  EXPECT_EQ(p.u, (std::vector<int>{320, 520}));
  EXPECT_EQ(p.v, (std::vector<int>{240, 340}));
  EXPECT_EQ(p.idx, (std::vector<int>{0, 1}));
}

TEST(Camera, ProjectDropsBehindOutOfFovAndOutOfRange)
{
  const auto p = axial_perception::project(
    cloud({{0, 0, -1}, {5, 0, 1}, {0, 0, 0.05}, {0, 0, 40}, {0, 0, 5}}), K, W, H, 0.1, 30.0);
  EXPECT_EQ(p.idx, (std::vector<int>{4}));
}

TEST(Camera, ProjectPixelEdges)
{
  auto at = [](double col) {return Eigen::Vector3d((col - K.cx) / K.fx, 0.0, 1.0);};
  const auto p = axial_perception::project(
    cloud({at(-0.4), at(-0.6), at(W - 0.6), at(W - 0.4)}), K, W, H, 0.1, 30.0);
  EXPECT_EQ(p.idx, (std::vector<int>{0, 2}));
  EXPECT_EQ(p.u, (std::vector<int>{0, W - 1}));
}

TEST(Camera, ProjectSkipsNan)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto p = axial_perception::project(cloud({{nan, 0, 2}, {0, 0, nan}, {0, 0, 2}}), K, W, H, 0.1, 30.0);
  EXPECT_EQ(p.idx, (std::vector<int>{2}));
}
