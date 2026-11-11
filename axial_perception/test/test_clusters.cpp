#include <gtest/gtest.h>

#include <set>

#include "axial_perception/clusters.hpp"

using axial_perception::InstanceImage;
using axial_perception::Labelled;
using axial_perception::Projection;

namespace
{

InstanceImage image()
{
  InstanceImage img;
  img.width = 4;
  img.height = 2;
  img.cls = {0, 4, 4, 8, 0, 4, 4, 1};
  img.mask = {0, 10, 10, 20, 0, 11, 11, 30};
  return img;
}

Labelled run(const std::vector<int> & u, const std::vector<int> & v, int min_points = 1)
{
  Projection p;
  p.u = u;
  p.v = v;
  for (size_t i = 0; i < u.size(); ++i) {
    p.idx.push_back(static_cast<int>(i) + 100);
  }
  return axial_perception::label_points(p, image(), {4, 8}, min_points);
}

}  // namespace

TEST(Clusters, KeepsOnlyAxialClassesAndMasks)
{
  const auto out = run({0, 1, 3, 3}, {0, 0, 0, 1});
  EXPECT_EQ(out.idx, (std::vector<int>{101, 102}));
  EXPECT_EQ(out.cls, (std::vector<int>{4, 8}));
}

TEST(Clusters, SegmentsFollowMasks)
{
  const auto s = run({1, 2, 1, 2, 3}, {0, 0, 1, 1, 0}).seg;
  EXPECT_EQ(s[0], s[1]);
  EXPECT_NE(s[1], s[2]);
  EXPECT_EQ(s[2], s[3]);
  EXPECT_NE(s[3], s[4]);
  EXPECT_EQ(std::set<int>(s.begin(), s.end()), (std::set<int>{1, 2, 3}));
}

TEST(Clusters, MinPointsDropsSmallMasks)
{
  const auto out = run({1, 2, 1, 3}, {0, 0, 1, 0}, 2);
  EXPECT_EQ(out.idx, (std::vector<int>{100, 101}));
  EXPECT_EQ(out.seg, (std::vector<int>{1, 1}));
}

TEST(Clusters, Empty)
{
  const auto out = run({}, {});
  EXPECT_TRUE(out.idx.empty() && out.cls.empty() && out.seg.empty());
}
