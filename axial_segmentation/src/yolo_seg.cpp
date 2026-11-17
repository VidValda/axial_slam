#include "axial_segmentation/yolo_seg.hpp"

#include <array>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace axial_segmentation
{

struct YoloSeg::Impl
{
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "axial_segmentation"};
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::string input_name;
  std::array<std::string, 2> output_names;
};

namespace
{

Ort::SessionOptions session_options(int threads)
{
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (threads > 0) {
    opts.SetIntraOpNumThreads(threads);
  }
  // idle worker threads sleep instead of busy-waiting, so the node does not starve the rest of the robot
  opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
  opts.AddConfigEntry("session.inter_op.allow_spinning", "0");
  return opts;
}

}  // namespace

YoloSeg::YoloSeg(const std::string & model_path, int threads, const std::string & device)
: impl_(std::make_unique<Impl>())
{
  if (device != "cuda" && device != "cpu") {
    throw std::invalid_argument("device must be cuda or cpu, not '" + device + "'");
  }
  provider_ = "cpu";
  if (device == "cuda") {
    try {
      auto opts = session_options(threads);
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = 0;
      opts.AppendExecutionProvider_CUDA(cuda);
      impl_->session = Ort::Session(impl_->env, model_path.c_str(), opts);
      provider_ = "cuda";
    } catch (const Ort::Exception & e) {
      fallback_reason_ = e.what();
    }
  }
  if (provider_ == "cpu") {
    impl_->session = Ort::Session(impl_->env, model_path.c_str(), session_options(threads));
  }
  auto & s = impl_->session;
  if (s.GetInputCount() != 1 || s.GetOutputCount() != 2) {
    throw std::runtime_error(model_path + ": expected 1 input and 2 outputs (a YOLO seg export with nms=False)");
  }
  Ort::AllocatorWithDefaultOptions alloc;
  impl_->input_name = s.GetInputNameAllocated(0, alloc).get();
  for (size_t i = 0; i < 2; ++i) {
    impl_->output_names[i] = s.GetOutputNameAllocated(i, alloc).get();
  }
  const auto shape = s.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 4 || shape[2] != shape[3] || shape[2] <= 0) {
    throw std::runtime_error(model_path + ": input must be a static square NCHW image");
  }
  size_ = static_cast<int>(shape[2]);
  const auto meta = s.GetModelMetadata();
  const auto names = meta.LookupCustomMetadataMapAllocated("names", alloc);
  if (!names) {
    throw std::runtime_error(model_path + ": no 'names' metadata");
  }
  names_ = parse_names(names.get());
}

YoloSeg::~YoloSeg() = default;

std::vector<Instance> YoloSeg::infer(const cv::Mat & bgr, const DecodeParams & p)
{
  const auto lb = make_letterbox(bgr.cols, bgr.rows, size_);
  preprocess(bgr, lb, input_);
  const std::array<int64_t, 4> in_shape{1, 3, size_, size_};
  auto input = Ort::Value::CreateTensor<float>(
    impl_->memory, input_.data(), input_.size(), in_shape.data(), in_shape.size());
  const char * in_names[] = {impl_->input_name.c_str()};
  const char * out_names[] = {impl_->output_names[0].c_str(), impl_->output_names[1].c_str()};
  auto out = impl_->session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 2);
  const auto s0 = out[0].GetTensorTypeAndShapeInfo().GetShape();   // [1, 4 + nc + nm, anchors]
  const auto s1 = out[1].GetTensorTypeAndShapeInfo().GetShape();   // [1, nm, mh, mw]
  return decode(out[0].GetTensorData<float>(), static_cast<int>(s0[1]), static_cast<int>(s0[2]),
           out[1].GetTensorData<float>(), static_cast<int>(s1[1]), static_cast<int>(s1[2]),
           static_cast<int>(s1[3]), static_cast<int>(names_.size()), lb, p);
}

}  // namespace axial_segmentation
