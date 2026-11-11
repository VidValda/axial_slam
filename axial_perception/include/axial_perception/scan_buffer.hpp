#pragma once

#include <deque>

#include "axial_perception/types.hpp"

namespace axial_perception
{

class ScanBuffer
{
public:
  explicit ScanBuffer(size_t n);

  void add(Cloud pts);
  Cloud points() const;
  size_t size() const {return scans_.size();}

private:
  size_t n_;
  std::deque<Cloud> scans_;
};

}  // namespace axial_perception
