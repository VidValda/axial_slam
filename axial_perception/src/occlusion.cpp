#include "axial_perception/occlusion.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

#include "axial_perception/kdtree.hpp"

namespace axial_perception
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

class UnionFind
{
public:
  explicit UnionFind(size_t n)
  : parent_(n) {std::iota(parent_.begin(), parent_.end(), 0);}

  size_t find(size_t a)
  {
    while (parent_[a] != a) {
      parent_[a] = parent_[parent_[a]];
      a = parent_[a];
    }
    return a;
  }

  void join(size_t a, size_t b) {parent_[find(a)] = find(b);}

private:
  std::vector<size_t> parent_;
};

std::vector<int> components(const Cloud & pts, double link)
{
  const size_t n = static_cast<size_t>(pts.cols());
  UnionFind uf(n);
  KdTree<3> tree(pts);
  for (size_t j = 0; j < n; ++j) {
    tree.visit_within(pts.col(static_cast<Eigen::Index>(j)), link, [&](int k) {
        if (static_cast<size_t>(k) > j) {
          uf.join(j, static_cast<size_t>(k));
        }
      });
  }
  std::vector<int> label(n);
  std::map<size_t, int> root_label;
  for (size_t j = 0; j < n; ++j) {
    label[j] = root_label.emplace(uf.find(j), static_cast<int>(root_label.size())).first->second;
  }
  return label;
}

}  // namespace

double median(std::vector<double> x)
{
  if (x.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const size_t mid = x.size() / 2;
  std::nth_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(mid), x.end());
  const double hi = x[mid];
  if (x.size() % 2) {
    return hi;
  }
  const double lo = *std::max_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(mid));
  return (lo + hi) / 2;
}

std::vector<bool> front_mask(
  const std::vector<int> & u, const std::vector<int> & v, const std::vector<double> & z,
  int width, int height, int window, double tol)
{
  if (window < 1) {
    throw std::invalid_argument("occlusion window must be at least 1 px");
  }
  const size_t w = static_cast<size_t>(width);
  std::vector<double> zbuf(w * static_cast<size_t>(height), kInf);
  for (size_t i = 0; i < z.size(); ++i) {
    double & cell = zbuf[static_cast<size_t>(v[i]) * w + static_cast<size_t>(u[i])];
    cell = std::min(cell, z[i]);
  }
  const int r = window / 2;
  std::vector<double> rows(zbuf.size(), kInf);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double m = kInf;
      for (int k = std::max(0, r - x); k < std::min(window, width - x + r); ++k) {
        m = std::min(m, zbuf[static_cast<size_t>(y) * w + static_cast<size_t>(x + k - r)]);
      }
      rows[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = m;
    }
  }
  std::vector<bool> keep(z.size());
  for (size_t i = 0; i < z.size(); ++i) {
    double m = kInf;
    for (int k = std::max(0, r - v[i]); k < std::min(window, height - v[i] + r); ++k) {
      m = std::min(m, rows[static_cast<size_t>(v[i] + k - r) * w + static_cast<size_t>(u[i])]);
    }
    keep[i] = z[i] <= m + tol;
  }
  return keep;
}

std::vector<bool> surface_mask(
  const std::vector<int> & u, const std::vector<int> & v, const Cloud & xyz,
  const InstanceImage & img, double link, double radius, double behind, int min_points)
{
  std::map<int32_t, Indices> by_mask;
  for (size_t j = 0; j < u.size(); ++j) {
    by_mask[img.mask_at(u[j], v[j])].push_back(static_cast<int>(j));
  }
  std::vector<bool> keep(u.size(), false);
  for (const auto & [mask, i] : by_mask) {
    const size_t n = i.size();
    const Cloud pts = select(xyz, i);
    const std::vector<int> lab = components(pts, link);
    std::vector<int> size(static_cast<size_t>(*std::max_element(lab.begin(), lab.end())) + 1, 0);
    for (int l : lab) {
      ++size[static_cast<size_t>(l)];
    }
    const int largest = *std::max_element(size.begin(), size.end());
    const int top = static_cast<int>(std::find(size.begin(), size.end(), largest) - size.begin());

    Indices top_idx;
    for (size_t j = 0; j < n; ++j) {
      if (lab[j] == top) {
        top_idx.push_back(static_cast<int>(j));
      }
    }
    Eigen::Matrix2Xd top_uv(2, static_cast<Eigen::Index>(top_idx.size()));
    for (size_t t = 0; t < top_idx.size(); ++t) {
      const int j = i[static_cast<size_t>(top_idx[t])];
      top_uv.col(static_cast<Eigen::Index>(t)) << u[static_cast<size_t>(j)], v[static_cast<size_t>(j)];
    }
    KdTree<2> tree(std::move(top_uv));

    std::vector<std::vector<double>> dz(size.size());
    std::vector<std::vector<double>> front(size.size());
    for (size_t j = 0; j < n; ++j) {
      const size_t p = static_cast<size_t>(i[j]);
      const auto [k, d] = tree.nearest(Eigen::Vector2d(u[p], v[p]));
      const double diff = pts(2, static_cast<Eigen::Index>(j)) -
        pts(2, top_idx[static_cast<size_t>(k)]);
      front[static_cast<size_t>(lab[j])].push_back(diff);
      if (d < radius) {
        dz[static_cast<size_t>(lab[j])].push_back(diff);
      }
    }

    std::vector<bool> good(size.size());
    for (size_t c = 0; c < size.size(); ++c) {
      good[c] = size[c] == largest || (size[c] >= min_points &&
        (dz[c].empty() || median(dz[c]) < behind) && median(front[c]) > -behind);
    }
    for (size_t j = 0; j < n; ++j) {
      if (good[static_cast<size_t>(lab[j])]) {
        keep[static_cast<size_t>(i[j])] = true;
      }
    }
  }
  return keep;
}

}  // namespace axial_perception
