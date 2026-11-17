// Run the model on images: writes <out>/<stem>_labels.png (raw labels_map bytes) and <stem>.txt
// (index score x1 y1 x2 y2 area per instance), and prints the mean inference time.
//
//   seg_infer <model.onnx> <out_dir> [--conf 0.4] [--iou 0.6] [--threads 4] [--device cuda|cpu] <image>...

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "axial_segmentation/labels_map.hpp"
#include "axial_segmentation/yolo_seg.hpp"

namespace fs = std::filesystem;
using namespace axial_segmentation;

int main(int argc, char ** argv)
{
  if (argc < 4) {
    std::cerr << "usage: seg_infer <model.onnx> <out_dir> [--conf c] [--iou i] [--threads n] [--device d] <image>...\n";
    return 2;
  }
  DecodeParams p;
  p.conf = 0.4f;
  p.iou = 0.6f;
  int threads = 4;
  std::string device = "cuda";
  std::vector<std::string> images;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--conf" && i + 1 < argc) {
      p.conf = std::stof(argv[++i]);
    } else if (a == "--iou" && i + 1 < argc) {
      p.iou = std::stof(argv[++i]);
    } else if (a == "--threads" && i + 1 < argc) {
      threads = std::stoi(argv[++i]);
    } else if (a == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else {
      images.push_back(a);
    }
  }
  YoloSeg model(argv[1], threads, device);
  std::printf("running on %s%s%s\n", model.provider().c_str(), model.fallback_reason().empty() ? "" : ": ",
    model.fallback_reason().c_str());
  std::vector<SegClass> classes;
  for (size_t i = 0; i < model.names().size(); ++i) {
    classes.push_back({static_cast<int>(i), static_cast<int>(i) + 1, model.names()[i], "axis"});
  }
  const fs::path out(argv[2]);
  fs::create_directories(out);
  double total_ms = 0;
  int n = 0;
  for (const auto & path : images) {
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "cannot read " << path << "\n";
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto inst = model.infer(bgr, p);
    const auto labels = encode_labels(inst, classes, bgr.cols, bgr.rows);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (n++ > 0) {   // first run warms up
      total_ms += ms;
    }
    const auto stem = fs::path(path).stem().string();
    cv::imwrite((out / (stem + "_labels.png")).string(), labels);
    std::ofstream txt(out / (stem + ".txt"));
    for (const auto & d : inst) {
      char line[160];
      std::snprintf(line, sizeof(line), "%d %.5f %.3f %.3f %.3f %.3f %d\n", d.index, d.score, d.box.x, d.box.y,
        d.box.x + d.box.width, d.box.y + d.box.height, cv::countNonZero(d.mask));
      txt << line;
    }
  }
  if (n > 1) {
    std::printf("%d images, %.1f ms per image (model + decode + labels_map)\n", n, total_ms / (n - 1));
  }
  return 0;
}
