#include <gtest/gtest.h>

#include "axial_perception/scan_buffer.hpp"

using axial_perception::Cloud;
using axial_perception::ScanBuffer;

TEST(ScanBuffer, Empty)
{
  EXPECT_EQ(ScanBuffer(3).points().cols(), 0);
}

TEST(ScanBuffer, KeepsLastNInOrder)
{
  ScanBuffer buf(2);
  for (int k = 0; k < 4; ++k) {
    buf.add(Cloud::Constant(3, k + 1, k));
  }
  EXPECT_EQ(buf.size(), 2u);
  const Eigen::RowVectorXd x = buf.points().row(0);
  EXPECT_EQ(x, (Eigen::RowVectorXd(7) << 2, 2, 2, 3, 3, 3, 3).finished());
}

TEST(ScanBuffer, ZeroKeepsNothing)
{
  ScanBuffer buf(0);
  buf.add(Cloud::Zero(3, 4));
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.points().cols(), 0);
}
