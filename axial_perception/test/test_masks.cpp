#include <gtest/gtest.h>

#include <set>
#include <stdexcept>

#include "axial_perception/masks.hpp"

using axial_perception::decode_gz_instance;

namespace
{

std::vector<uint8_t> encode(const std::vector<int> & cls, const std::vector<int> & instance)
{
  std::vector<uint8_t> out;
  for (size_t i = 0; i < cls.size(); ++i) {
    out.push_back(static_cast<uint8_t>(instance[i] & 0xFF));
    out.push_back(static_cast<uint8_t>(instance[i] >> 8));
    out.push_back(static_cast<uint8_t>(cls[i]));
  }
  return out;
}

}  // namespace

TEST(Masks, RoundTrip)
{
  const std::vector<int> cls{0, 4, 4, 8, 4, 6};
  const auto img = decode_gz_instance(encode(cls, {0, 1, 300, 1, 1, 2}), 2, 3, 9);
  for (size_t i = 0; i < cls.size(); ++i) {
    EXPECT_EQ(img.cls[i], cls[i]);
  }
  EXPECT_EQ(img.mask_at(0, 0), 0);
  EXPECT_EQ(img.mask_at(1, 0), img.mask_at(1, 1));
  std::set<int32_t> ids{img.mask_at(1, 0), img.mask_at(2, 0), img.mask_at(0, 1), img.mask_at(2, 1)};
  EXPECT_EQ(ids.size(), 4u);
}

TEST(Masks, SameInstanceIdInDifferentClassesAreDifferentMasks)
{
  const auto img = decode_gz_instance(encode({4, 8}, {1, 1}), 1, 2, 6);
  EXPECT_NE(img.mask_at(0, 0), img.mask_at(1, 0));
}

TEST(Masks, BackgroundIsZeroEvenWithInstanceBits)
{
  EXPECT_EQ(decode_gz_instance(encode({0}, {5}), 1, 1, 3).mask_at(0, 0), 0);
}

TEST(Masks, RowPadding)
{
  auto data = encode({4, 8}, {1, 2});
  data.insert(data.begin() + 3, {9, 9});
  data.insert(data.end(), {9, 9});
  const auto img = decode_gz_instance(data, 2, 1, 5);
  EXPECT_EQ(img.class_at(0, 0), 4);
  EXPECT_EQ(img.class_at(0, 1), 8);
}

TEST(Masks, ShortDataThrows)
{
  EXPECT_THROW(decode_gz_instance(encode({4}, {1}), 2, 1, 3), std::invalid_argument);
  EXPECT_THROW(decode_gz_instance(encode({4, 4}, {1, 1}), 1, 2, 3), std::invalid_argument);
}
