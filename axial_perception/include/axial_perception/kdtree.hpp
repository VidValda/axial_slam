#pragma once

#include <nanoflann/nanoflann.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "axial_perception/types.hpp"

namespace axial_perception
{

template<int Dim>
class KdTree
{
public:
  using Points = Eigen::Matrix<double, Dim, Eigen::Dynamic>;
  using Point = Eigen::Matrix<double, Dim, 1>;

  explicit KdTree(Points pts)
  : pts_(std::move(pts)), adaptor_{&pts_}
  {
    if (pts_.cols() > 0) {
      index_ = std::make_unique<Index>(Dim, adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }
  }

  KdTree(const KdTree &) = delete;
  KdTree & operator=(const KdTree &) = delete;

  const Points & points() const {return pts_;}

  template<class F>
  void visit_within(const Point & q, double radius, F && f) const
  {
    if (!index_ || radius < 0) {
      return;
    }
    Visit<F> visit{std::nextafter(radius * radius, std::numeric_limits<double>::infinity()), f};
    index_->findNeighbors(visit, q.data(), nanoflann::SearchParameters(0, false));
  }

  Indices within(const Point & q, double radius) const
  {
    Indices out;
    visit_within(q, radius, [&out](int i) {out.push_back(i);});
    std::sort(out.begin(), out.end());
    return out;
  }

  size_t count_within(const Point & q, double radius) const
  {
    size_t n = 0;
    visit_within(q, radius, [&n](int) {++n;});
    return n;
  }

  std::pair<int, double> nearest(
    const Point & q, double bound = std::numeric_limits<double>::infinity()) const
  {
    Nearest result{bound * bound, -1};
    if (index_) {
      index_->findNeighbors(result, q.data());
    }
    return {static_cast<int>(result.index), result.index < 0 ? bound : std::sqrt(result.best)};
  }

private:
  struct Adaptor
  {
    const Points * pts;
    size_t kdtree_get_point_count() const {return static_cast<size_t>(pts->cols());}
    double kdtree_get_pt(size_t i, size_t d) const
    {
      return (*pts)(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(i));
    }
    template<class Box>
    bool kdtree_get_bbox(Box &) const {return false;}
  };

  template<class F>
  struct Visit
  {
    using DistanceType = double;
    using IndexType = uint32_t;

    double r2;
    F & f;

    bool addPoint(double d, uint32_t i)
    {
      if (d < r2) {
        f(static_cast<int>(i));
      }
      return true;
    }
    double worstDist() const {return r2;}
    bool full() const {return true;}
  };

  struct Nearest
  {
    using DistanceType = double;
    using IndexType = uint32_t;

    double best;
    int64_t index;

    bool addPoint(double d, uint32_t i)
    {
      if (d < best || (d == best && index >= 0 && i < index)) {
        best = d;
        index = i;
      }
      return true;
    }
    double worstDist() const
    {
      return index < 0 ? best : std::nextafter(best, std::numeric_limits<double>::infinity());
    }
    bool full() const {return true;}
  };

  using Index = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Adaptor>, Adaptor, Dim, uint32_t>;

  Points pts_;
  Adaptor adaptor_;
  std::unique_ptr<Index> index_;
};

}  // namespace axial_perception
