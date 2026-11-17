#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace axial_segmentation
{

// Letterbox of an image into a square network input.
struct Letterbox
{
  int size = 640;
  double gain = 1.0;
  int pad_x = 0;
  int pad_y = 0;
  int width = 0;   // original image
  int height = 0;
};

Letterbox make_letterbox(int width, int height, int size);

// BGR image -> NCHW float RGB in [0, 1], padded with 114 like ultralytics.
void preprocess(const cv::Mat & bgr, const Letterbox & lb, std::vector<float> & out);

struct Instance
{
  int index = 0;          // YOLO class index
  float score = 0.0f;
  cv::Rect2f box;         // original image pixels
  cv::Mat mask;           // CV_8U 0/1 over box_px
  cv::Rect box_px;        // integer pixel rect of mask, inside the image
};

struct DecodeParams
{
  float conf = 0.25f;
  float iou = 0.7f;
  int max_det = 300;
};

// output0 [4 + nc + nm, n] and protos [nm, mh, mw] -> instances, as ultralytics predict.
std::vector<Instance> decode(
  const float * output0, int rows, int anchors, const float * protos, int nm, int mh, int mw,
  int num_classes, const Letterbox & lb, const DecodeParams & p);

// Greedy per-class NMS on xyxy boxes sorted by score; returns kept indices.
std::vector<int> nms(const std::vector<cv::Rect2f> & boxes, const std::vector<float> & scores,
                     const std::vector<int> & cls, float iou, int max_det);

// Class names from the ultralytics "names" metadata, e.g. "{0: 'pillar', 1: 'pipe'}".
std::vector<std::string> parse_names(const std::string & meta);

class YoloSeg
{
public:
  // device "cuda" falls back to "cpu" when the CUDA provider cannot load; see provider()
  YoloSeg(const std::string & model_path, int threads, const std::string & device = "cuda");
  ~YoloSeg();

  std::vector<Instance> infer(const cv::Mat & bgr, const DecodeParams & p);
  const std::vector<std::string> & names() const {return names_;}
  int input_size() const {return size_;}
  const std::string & provider() const {return provider_;}
  const std::string & fallback_reason() const {return fallback_reason_;}

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::vector<std::string> names_;
  int size_ = 640;
  std::string provider_;
  std::string fallback_reason_;
  std::vector<float> input_;
};

}  // namespace axial_segmentation
