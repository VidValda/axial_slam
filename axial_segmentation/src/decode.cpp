#include <algorithm>
#include <cmath>
#include <numeric>
#include <regex>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "axial_segmentation/yolo_seg.hpp"

namespace axial_segmentation
{

Letterbox make_letterbox(int width, int height, int size)
{
  if (width <= 0 || height <= 0 || size <= 0) {
    throw std::invalid_argument("letterbox: non-positive size");
  }
  Letterbox lb;
  lb.size = size;
  lb.width = width;
  lb.height = height;
  lb.gain = std::min(static_cast<double>(size) / height, static_cast<double>(size) / width);
  const int new_w = static_cast<int>(std::round(width * lb.gain));
  const int new_h = static_cast<int>(std::round(height * lb.gain));
  lb.pad_x = static_cast<int>(std::round((size - new_w) / 2.0 - 0.1));
  lb.pad_y = static_cast<int>(std::round((size - new_h) / 2.0 - 0.1));
  return lb;
}

void preprocess(const cv::Mat & bgr, const Letterbox & lb, std::vector<float> & out)
{
  cv::Mat resized = bgr;
  const int new_w = static_cast<int>(std::round(lb.width * lb.gain));
  const int new_h = static_cast<int>(std::round(lb.height * lb.gain));
  if (bgr.cols != new_w || bgr.rows != new_h) {
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  }
  cv::Mat boxed(lb.size, lb.size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(boxed(cv::Rect(lb.pad_x, lb.pad_y, new_w, new_h)));
  const size_t plane = static_cast<size_t>(lb.size) * lb.size;
  out.resize(3 * plane);
  for (int r = 0; r < lb.size; ++r) {
    const uint8_t * px = boxed.ptr<uint8_t>(r);
    for (int c = 0; c < lb.size; ++c, px += 3) {
      const size_t i = static_cast<size_t>(r) * lb.size + c;
      out[i] = px[2] / 255.0f;
      out[plane + i] = px[1] / 255.0f;
      out[2 * plane + i] = px[0] / 255.0f;
    }
  }
}

namespace
{

int py_round(double v) {return static_cast<int>(std::nearbyint(v));}

// Prototype-resolution logits -> original image size, as ultralytics process_mask_native.
cv::Mat to_image(const cv::Mat & logits, int w, int h)
{
  const int mh = logits.rows, mw = logits.cols;
  const double g4 = std::min(4.0 * mh / h, 4.0 * mw / w);
  const int pad = (4 * mh - py_round(h * g4)) | (4 * mw - py_round(w * g4));
  const int f = 4 / std::gcd(4, pad | (pad / 2));
  cv::Mat src = logits;
  if (f > 1) {
    cv::resize(logits, src, cv::Size(mw * f, mh * f), 0, 0, cv::INTER_LINEAR);
  }
  if (src.rows == h && src.cols == w) {
    return src;
  }
  const double gain = std::min(static_cast<double>(src.rows) / h, static_cast<double>(src.cols) / w);
  const int top = py_round((src.rows - py_round(h * gain)) / 2.0 - 0.1);
  const int left = py_round((src.cols - py_round(w * gain)) / 2.0 - 0.1);
  const cv::Rect content(left, top, py_round(w * gain), py_round(h * gain));
  cv::Mat out;
  cv::resize(src(content), out, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  return out;
}

}  // namespace

std::vector<int> nms(const std::vector<cv::Rect2f> & boxes, const std::vector<float> & scores,
                     const std::vector<int> & cls, float iou, int max_det)
{
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return scores[a] > scores[b];});
  std::vector<int> keep;
  std::vector<bool> gone(boxes.size(), false);
  for (size_t i = 0; i < order.size() && static_cast<int>(keep.size()) < max_det; ++i) {
    const int a = order[i];
    if (gone[a]) {
      continue;
    }
    keep.push_back(a);
    for (size_t j = i + 1; j < order.size(); ++j) {
      const int b = order[j];
      if (gone[b] || cls[b] != cls[a]) {
        continue;
      }
      const float inter = (boxes[a] & boxes[b]).area();
      const float uni = boxes[a].area() + boxes[b].area() - inter;
      if (uni > 0 && inter / uni > iou) {
        gone[b] = true;
      }
    }
  }
  return keep;
}

std::vector<Instance> decode(
  const float * output0, int rows, int anchors, const float * protos, int nm, int mh, int mw,
  int num_classes, const Letterbox & lb, const DecodeParams & p)
{
  if (rows != 4 + num_classes + nm) {
    throw std::invalid_argument("decode: output0 has " + std::to_string(rows) + " rows, expected " +
            std::to_string(4 + num_classes + nm));
  }
  std::vector<cv::Rect2f> boxes;
  std::vector<float> scores;
  std::vector<int> cls, anchor;
  for (int a = 0; a < anchors; ++a) {
    int best = 0;
    float score = output0[4 * anchors + a];
    for (int c = 1; c < num_classes; ++c) {
      const float s = output0[(4 + c) * anchors + a];
      if (s > score) {
        score = s;
        best = c;
      }
    }
    if (score <= p.conf) {
      continue;
    }
    const float cx = output0[a], cy = output0[anchors + a];
    const float w = output0[2 * anchors + a], h = output0[3 * anchors + a];
    boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
    scores.push_back(score);
    cls.push_back(best);
    anchor.push_back(a);
  }
  const auto keep = nms(boxes, scores, cls, p.iou, p.max_det);

  const cv::Mat proto(nm, mh * mw, CV_32F, const_cast<float *>(protos));
  const cv::Rect image(0, 0, lb.width, lb.height);
  std::vector<Instance> out;
  out.reserve(keep.size());
  cv::Mat coeff(1, nm, CV_32F), flat, logits, native;
  for (const int k : keep) {
    Instance inst;
    inst.index = cls[k];
    inst.score = scores[k];
    const cv::Rect2f & b = boxes[k];
    // box in letterbox pixels -> original pixels, clipped
    const float x1 = std::clamp(static_cast<float>((b.x - lb.pad_x) / lb.gain), 0.0f, static_cast<float>(lb.width));
    const float y1 = std::clamp(static_cast<float>((b.y - lb.pad_y) / lb.gain), 0.0f, static_cast<float>(lb.height));
    const float x2 = std::clamp(static_cast<float>((b.x + b.width - lb.pad_x) / lb.gain), 0.0f,
        static_cast<float>(lb.width));
    const float y2 = std::clamp(static_cast<float>((b.y + b.height - lb.pad_y) / lb.gain), 0.0f,
        static_cast<float>(lb.height));
    inst.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

    for (int j = 0; j < nm; ++j) {
      coeff.at<float>(j) = output0[(4 + num_classes + j) * anchors + anchor[k]];
    }
    cv::gemm(coeff, proto, 1.0, cv::noArray(), 0.0, flat);
    logits = flat.reshape(1, mh);
    native = to_image(logits, lb.width, lb.height);
    // pixel i is inside when x1 <= i < x2, like ultralytics crop_mask
    const cv::Rect px(cv::Point(static_cast<int>(std::ceil(x1)), static_cast<int>(std::ceil(y1))),
      cv::Point(static_cast<int>(std::ceil(x2)), static_cast<int>(std::ceil(y2))));
    inst.box_px = px & image;
    if (inst.box_px.empty()) {
      continue;
    }
    cv::Mat m = native(inst.box_px) > 0.0f;
    m.convertTo(inst.mask, CV_8U, 1.0 / 255.0);
    out.push_back(std::move(inst));
  }
  return out;
}

std::vector<std::string> parse_names(const std::string & meta)
{
  std::vector<std::pair<int, std::string>> found;
  const std::regex item(R"((\d+)\s*:\s*['"]([^'"]*)['"])");
  for (auto it = std::sregex_iterator(meta.begin(), meta.end(), item); it != std::sregex_iterator(); ++it) {
    found.emplace_back(std::stoi((*it)[1]), (*it)[2]);
  }
  std::sort(found.begin(), found.end());
  std::vector<std::string> names;
  for (size_t i = 0; i < found.size(); ++i) {
    if (found[i].first != static_cast<int>(i)) {
      throw std::invalid_argument("model names are not 0..n-1: " + meta);
    }
    names.push_back(found[i].second);
  }
  return names;
}

}  // namespace axial_segmentation
