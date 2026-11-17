#include <gtest/gtest.h>

#include "axial_perception/point_landmarks.hpp"

using axial_perception::Cloud;
using axial_perception::median_point;
using axial_perception::PointLandmarks;

namespace
{

Cloud blob(const Eigen::Vector3d & c, int n, double outlier = 0.0)
{
  Cloud pts(3, n);
  for (int i = 0; i < n; ++i) {
    pts.col(i) = c + Eigen::Vector3d(0.01 * (i % 5), 0.01 * (i % 3), 0.01 * (i % 7));
  }
  if (outlier != 0.0) {
    pts.col(0) = c + Eigen::Vector3d(outlier, 0, 0);   // background seen through the mask
  }
  return pts;
}

}  // namespace

TEST(PointLandmarks, MedianIgnoresOutliers)
{
  const Eigen::Vector3d p = median_point(blob({1, 2, 3}, 21, 5.0));
  EXPECT_NEAR(p.x(), 1.02, 0.021);
  EXPECT_THROW(median_point(Cloud(3, 0)), std::invalid_argument);
}

TEST(PointLandmarks, MatchesByClassAndRadius)
{
  PointLandmarks map(1.0, 0.05);
  Cloud xyz(3, 30);
  xyz << blob({0, 0, 1}, 10), blob({0.2, 0, 1}, 10), blob({5, 0, 1}, 10);
  std::vector<int> cls(30, 9), seg(30, 1);
  for (int i = 10; i < 20; ++i) {
    cls[i] = 10;           // a door next to the extinguisher: separate landmark
    seg[i] = 2;
  }
  for (int i = 20; i < 30; ++i) {
    seg[i] = 3;
  }
  const auto ids = map.update(xyz, cls, seg);
  EXPECT_EQ(map.landmarks().size(), 3u);
  EXPECT_NE(ids.at(1), ids.at(2));

  Cloud again = blob({0.3, 0, 1}, 10);
  const auto ids2 = map.update(again, std::vector<int>(10, 9), std::vector<int>(10, 1));
  EXPECT_EQ(ids2.at(1), ids.at(1));
  const auto & lm = map.landmarks().at(ids.at(1));
  EXPECT_EQ(lm.hits, 2);
  EXPECT_NEAR(lm.point.x(), 0.17, 0.02);
  EXPECT_LE(lm.points.cols(), 20);
}
