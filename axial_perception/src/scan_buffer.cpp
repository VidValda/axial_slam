#include "axial_perception/scan_buffer.hpp"

#include <utility>

namespace axial_perception
{

ScanBuffer::ScanBuffer(size_t n)
: n_(n) {}

void ScanBuffer::add(Cloud pts)
{
  if (n_ == 0) {
    return;
  }
  if (scans_.size() == n_) {
    scans_.pop_front();
  }
  scans_.push_back(std::move(pts));
}

Cloud ScanBuffer::points() const
{
  Eigen::Index n = 0;
  for (const auto & s : scans_) {
    n += s.cols();
  }
  Cloud out(3, n);
  Eigen::Index at = 0;
  for (const auto & s : scans_) {
    out.middleCols(at, s.cols()) = s;
    at += s.cols();
  }
  return out;
}

}  // namespace axial_perception
