#include <gtest/gtest.h>

#include <random>

#include "axial_perception/kdtree.hpp"

using axial_perception::Indices;
using axial_perception::KdTree;

TEST(KdTree, WithinIsInclusive)
{
  Eigen::Matrix3Xd pts(3, 3);
  pts << 0, 1, 2,
    0, 0, 0,
    0, 0, 0;
  KdTree<3> tree(pts);
  EXPECT_EQ(tree.within(Eigen::Vector3d::Zero(), 1.0), (Indices{0, 1}));
  EXPECT_EQ(tree.within(Eigen::Vector3d::Zero(), 0.5), (Indices{0}));
}

TEST(KdTree, NearestBoundIsStrict)
{
  Eigen::Matrix2Xd pts(2, 2);
  pts << 0, 3,
    0, 4;
  KdTree<2> tree(pts);
  EXPECT_EQ(tree.nearest(Eigen::Vector2d(6, 8), 5.0).first, -1);
  const auto [k, d] = tree.nearest(Eigen::Vector2d(6, 8), 5.0001);
  EXPECT_EQ(k, 1);
  EXPECT_DOUBLE_EQ(d, 5.0);
}

TEST(KdTree, NearestTiesGoToLowestIndex)
{
  std::mt19937 rng(1);
  std::uniform_int_distribution<int> px(0, 6);
  Eigen::Matrix2Xd pts(2, 300);
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    pts.col(i) << px(rng), px(rng);
  }
  KdTree<2> tree(pts);
  for (int qx = -1; qx < 8; ++qx) {
    for (int qy = -1; qy < 8; ++qy) {
      const Eigen::Vector2d q(qx + 0.5 * (qx % 2), qy);
      const Eigen::VectorXd d = (pts.colwise() - q).colwise().squaredNorm();
      Eigen::Index best;
      d.minCoeff(&best);
      EXPECT_EQ(tree.nearest(q).first, best);
    }
  }
}

TEST(KdTree, WithinMatchesBruteForce)
{
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> x(0.0, 1.0);
  Eigen::Matrix3Xd pts(3, 500);
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    pts.col(i) << x(rng), x(rng), x(rng);
  }
  KdTree<3> tree(pts);
  for (Eigen::Index q = 0; q < 20; ++q) {
    Indices brute;
    for (Eigen::Index i = 0; i < pts.cols(); ++i) {
      if ((pts.col(i) - pts.col(q)).norm() <= 0.2) {
        brute.push_back(static_cast<int>(i));
      }
    }
    EXPECT_EQ(tree.within(pts.col(q), 0.2), brute);
  }
}

TEST(KdTree, Empty)
{
  KdTree<3> tree(Eigen::Matrix3Xd(3, 0));
  EXPECT_TRUE(tree.within(Eigen::Vector3d::Zero(), 1.0).empty());
  EXPECT_EQ(tree.nearest(Eigen::Vector3d::Zero()).first, -1);
}
