#include <gtest/gtest.h>

#include <stdexcept>

#include "axial_segmentation/labels_map.hpp"

using namespace axial_segmentation;

namespace
{

Instance box_instance(int index, float score, cv::Rect r)
{
  Instance i;
  i.index = index;
  i.score = score;
  i.box = cv::Rect2f(r);
  i.box_px = r;
  i.mask = cv::Mat::ones(r.height, r.width, CV_8U);
  return i;
}

}  // namespace

TEST(LabelsMap, ClassTableIsValidated)
{
  const auto c = make_classes({8, 4}, {"pillar", "pipe"}, {"axis", "axis"});
  EXPECT_EQ(c[1].index, 1);
  EXPECT_EQ(c[1].id, 4);
  EXPECT_THROW(make_classes({8}, {"pillar", "pipe"}, {"axis"}), std::invalid_argument);
  EXPECT_THROW(make_classes({8, 8}, {"a", "b"}, {"axis", "axis"}), std::invalid_argument);
  EXPECT_THROW(make_classes({0}, {"a"}, {"axis"}), std::invalid_argument);
  EXPECT_THROW(make_classes({9}, {"a"}, {"blob"}), std::invalid_argument);
}

TEST(LabelsMap, EncodesClassAndPerClassInstance)
{
  const auto classes = make_classes({8, 9}, {"pillar", "door"}, {"axis", "point"});
  const std::vector<Instance> inst{
    box_instance(0, 0.5f, {0, 0, 4, 4}),
    box_instance(0, 0.9f, {2, 0, 4, 4}),     // overlaps the first, higher score
    box_instance(1, 0.7f, {0, 6, 2, 2})};
  const cv::Mat m = encode_labels(inst, classes, 8, 8);
  ASSERT_EQ(m.type(), CV_8UC3);
  auto px = [&](int u, int v) {return m.at<cv::Vec3b>(v, u);};
  EXPECT_EQ(px(7, 7), cv::Vec3b(0, 0, 0));
  EXPECT_EQ(px(3, 0), cv::Vec3b(1, 0, 8));   // overlap goes to the 0.9 pillar, instance 1
  EXPECT_EQ(px(0, 0), cv::Vec3b(2, 0, 8));   // 0.5 pillar is instance 2
  EXPECT_EQ(px(1, 7), cv::Vec3b(1, 0, 9));   // door ids start again at 1
}
