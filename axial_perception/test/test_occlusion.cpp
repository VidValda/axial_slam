#include <gtest/gtest.h>

#include <algorithm>

#include "axial_perception/occlusion.hpp"

using axial_perception::Cloud;
using axial_perception::InstanceImage;
using axial_perception::front_mask;
using axial_perception::surface_mask;

namespace
{

constexpr int W = 40;
constexpr int H = 20;

bool all(const std::vector<bool> & k, size_t from, size_t to)
{
  return std::all_of(k.begin() + from, k.begin() + to, [](bool b) {return b;});
}

bool none(const std::vector<bool> & k, size_t from, size_t to)
{
  return std::none_of(k.begin() + from, k.begin() + to, [](bool b) {return b;});
}

Eigen::VectorXd linspace(double a, double b, int n)
{
  return Eigen::VectorXd::LinSpaced(n, a, b);
}

Cloud stack(const Cloud & a, const Cloud & b)
{
  Cloud out(3, a.cols() + b.cols());
  out << a, b;
  return out;
}

Cloud columns(const Eigen::VectorXd & x, const Eigen::VectorXd & y, const Eigen::VectorXd & z)
{
  Cloud out(3, x.size());
  out.row(0) = x.transpose();
  out.row(1) = y.transpose();
  out.row(2) = z.transpose();
  return out;
}

struct MaskFrame
{
  std::vector<int> u;
  std::vector<int> v;
  InstanceImage img;
};

MaskFrame mask_frame(const Cloud & pts, int32_t id)
{
  MaskFrame f;
  f.img.width = 100;
  f.img.height = 100;
  f.img.cls.assign(100 * 100, 0);
  f.img.mask.assign(100 * 100, 0);
  for (Eigen::Index i = 0; i < pts.cols(); ++i) {
    f.u.push_back(static_cast<int>(100 * pts(0, i) / pts(2, i) + 50));
    f.v.push_back(static_cast<int>(100 * pts(1, i) / pts(2, i) + 50));
    f.img.mask[static_cast<size_t>(f.v.back()) * 100 + static_cast<size_t>(f.u.back())] = id;
  }
  return f;
}

std::vector<bool> surface(const Cloud & pts, double radius)
{
  const auto f = mask_frame(pts, 1);
  return surface_mask(f.u, f.v, pts, f.img, 0.1, radius, 0.3);
}

}  // namespace

TEST(Occlusion, WallBehindPipeIsRejectedNextToItIsKept)
{
  std::vector<int> u, v;
  std::vector<double> z;
  for (int x = 18; x < 22; ++x) {
    u.push_back(x);
    z.push_back(2.0);
  }
  for (int x = 0; x < W; ++x) {
    u.push_back(x);
    z.push_back(2.3);
  }
  v.assign(u.size(), 10);
  const auto keep = front_mask(u, v, z, W, H, 5, 0.1);
  EXPECT_TRUE(all(keep, 0, 4));
  EXPECT_TRUE(none(keep, 4 + 16, 4 + 24));
  EXPECT_TRUE(all(keep, 4, 4 + 16));
  EXPECT_TRUE(all(keep, 4 + 24, keep.size()));
}

TEST(Occlusion, WithinToleranceIsKept)
{
  const auto keep = front_mask({5, 6}, {5, 5}, {2.0, 2.08}, W, H, 3, 0.1);
  EXPECT_TRUE(all(keep, 0, 2));
}

TEST(Occlusion, SamePixelKeepsOnlyFront)
{
  EXPECT_EQ(front_mask({5, 5}, {5, 5}, {3.0, 1.0}, W, H, 1, 0.1), (std::vector<bool>{false, true}));
}

TEST(Occlusion, ImageBorder)
{
  const auto keep = front_mask({0, W - 1}, {0, H - 1}, {1.0, 1.0}, W, H, 5, 0.1);
  EXPECT_TRUE(all(keep, 0, 2));
}

TEST(Occlusion, EvenWindowLooksOneMorePixelBack)
{
  EXPECT_EQ(front_mask({4, 5}, {5, 5}, {1.0, 2.0}, W, H, 2, 0.1), (std::vector<bool>{true, false}));
  EXPECT_EQ(front_mask({6, 5}, {5, 5}, {1.0, 2.0}, W, H, 2, 0.1), (std::vector<bool>{true, true}));
}

TEST(Occlusion, SurfaceMaskDropsWallBehindPipe)
{
  const Eigen::VectorXd x = linspace(-0.5, 0.5, 60);
  Eigen::VectorXd wx(20);
  for (int i = 0; i < 20; ++i) {
    wx[i] = x[3 * i] * 1.2;
  }
  const Cloud pts = stack(
    columns(x, Eigen::VectorXd::Zero(60), Eigen::VectorXd::Constant(60, 2.0)),
    columns(wx, Eigen::VectorXd::Constant(20, 0.01), Eigen::VectorXd::Constant(20, 2.6)));
  const auto keep = surface(pts, 15);
  EXPECT_TRUE(all(keep, 0, 60));
  EXPECT_TRUE(none(keep, 60, 80));
}

TEST(Occlusion, SurfaceMaskKeepsBothArmsOfARecedingElbow)
{
  const Eigen::VectorXd t = linspace(0, 1, 50);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(50);
  const Cloud pts = stack(
    columns(t.array() - 1, zero, Eigen::VectorXd::Constant(50, 2.0)),
    columns(zero, zero, 2.0 + 2 * t.array()));
  EXPECT_TRUE(all(surface(pts, 15), 0, 100));
}

TEST(Occlusion, SurfaceMaskKeepsASeparatePieceAtTheSameDepth)
{
  const Eigen::VectorXd x = linspace(-0.4, -0.1, 30);
  const Cloud pts = stack(
    columns(x, Eigen::VectorXd::Zero(30), Eigen::VectorXd::Constant(30, 2.0)),
    columns(-x.head(20), Eigen::VectorXd::Zero(20), Eigen::VectorXd::Constant(20, 2.05)));
  EXPECT_TRUE(all(surface(pts, 15), 0, 50));
}

TEST(Occlusion, SurfaceMaskEmpty)
{
  InstanceImage img;
  img.width = img.height = 4;
  img.cls.assign(16, 0);
  img.mask.assign(16, 0);
  EXPECT_TRUE(surface_mask({}, {}, Cloud(3, 0), img, 0.1, 15, 0.3).empty());
}

TEST(Occlusion, SurfaceMaskDropsAPieceInFrontAwayFromTheSurface)
{
  const Cloud pts = stack(
    columns(Eigen::VectorXd::Zero(60), linspace(-0.4, 0.4, 60), Eigen::VectorXd::Constant(60, 4.0)),
    columns(linspace(-0.05, 0.05, 12), Eigen::VectorXd::Constant(12, -0.75), Eigen::VectorXd::Constant(12, 2.0)));
  const auto keep = surface(pts, 5);
  EXPECT_TRUE(all(keep, 0, 60));
  EXPECT_TRUE(none(keep, 60, 72));
}

TEST(Occlusion, Median)
{
  EXPECT_DOUBLE_EQ(axial_perception::median({3.0, 1.0, 2.0}), 2.0);
  EXPECT_DOUBLE_EQ(axial_perception::median({4.0, 1.0, 3.0, 2.0}), 2.5);
}
