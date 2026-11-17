#include <gtest/gtest.h>

#include <vector>

#include <opencv2/core.hpp>

#include "axial_segmentation/yolo_seg.hpp"

using namespace axial_segmentation;

TEST(Decode, LetterboxOf640x480)
{
  const auto lb = make_letterbox(640, 480, 640);
  EXPECT_DOUBLE_EQ(lb.gain, 1.0);
  EXPECT_EQ(lb.pad_x, 0);
  EXPECT_EQ(lb.pad_y, 80);
  const auto half = make_letterbox(1280, 720, 640);
  EXPECT_DOUBLE_EQ(half.gain, 0.5);
  EXPECT_EQ(half.pad_y, 140);
}

TEST(Decode, PreprocessPadsAndSwapsChannels)
{
  cv::Mat bgr(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
  const auto lb = make_letterbox(640, 480, 640);
  std::vector<float> in;
  preprocess(bgr, lb, in);
  const size_t plane = 640 * 640;
  ASSERT_EQ(in.size(), 3 * plane);
  EXPECT_FLOAT_EQ(in[0], 114 / 255.0f);                       // top padding
  const size_t px = 100 * 640 + 5;                            // inside the image
  EXPECT_FLOAT_EQ(in[px], 30 / 255.0f);                       // R
  EXPECT_FLOAT_EQ(in[2 * plane + px], 10 / 255.0f);           // B
}

TEST(Decode, NmsIsPerClass)
{
  const std::vector<cv::Rect2f> boxes{{0, 0, 10, 10}, {1, 0, 10, 10}, {1, 0, 10, 10}, {50, 50, 5, 5}};
  const std::vector<float> scores{0.9f, 0.8f, 0.7f, 0.6f};
  const std::vector<int> cls{0, 0, 1, 0};
  const auto keep = nms(boxes, scores, cls, 0.5f, 300);
  EXPECT_EQ(keep, (std::vector<int>{0, 2, 3}));
  EXPECT_EQ(nms(boxes, scores, cls, 0.5f, 1).size(), 1u);
}

TEST(Decode, MaskIsCroppedToBox)
{
  // one class, one prototype that is positive everywhere, 4 anchors of which one passes conf
  const int nc = 1, nm = 1, anchors = 4, rows = 4 + nc + nm, mh = 160, mw = 160;
  std::vector<float> out(rows * anchors, 0.0f);
  auto at = [&](int r, int a) -> float & {return out[r * anchors + a];};
  at(0, 2) = 320; at(1, 2) = 320; at(2, 2) = 100; at(3, 2) = 60;   // box 270..370 x 290..350 (letterbox)
  at(4, 2) = 0.9f;
  at(5, 2) = 1.0f;
  at(4, 0) = 0.1f;
  std::vector<float> protos(mh * mw, 2.0f);
  const auto lb = make_letterbox(640, 480, 640);
  DecodeParams p;
  p.conf = 0.4f;
  const auto inst = decode(out.data(), rows, anchors, protos.data(), nm, mh, mw, nc, lb, p);
  ASSERT_EQ(inst.size(), 1u);
  EXPECT_FLOAT_EQ(inst[0].score, 0.9f);
  EXPECT_NEAR(inst[0].box.x, 270, 1e-3);
  EXPECT_NEAR(inst[0].box.y, 210, 1e-3);                     // 290 - pad 80
  EXPECT_EQ(inst[0].box_px, cv::Rect(270, 210, 100, 60));
  EXPECT_EQ(cv::countNonZero(inst[0].mask), 100 * 60);
}

TEST(Decode, RejectsWrongRowCount)
{
  std::vector<float> out(10 * 2), protos(4);
  const auto lb = make_letterbox(640, 480, 640);
  EXPECT_THROW(decode(out.data(), 10, 2, protos.data(), 1, 2, 2, 3, lb, DecodeParams{}), std::invalid_argument);
}

TEST(Decode, ParsesUltralyticsNames)
{
  EXPECT_EQ(parse_names("{0: 'pillar', 1: 'pipe', 2: 'fire_extinguisher'}"),
    (std::vector<std::string>{"pillar", "pipe", "fire_extinguisher"}));
  EXPECT_THROW(parse_names("{0: 'a', 2: 'b'}"), std::invalid_argument);
}
