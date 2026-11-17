#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "axial_segmentation/labels_map.hpp"
#include "axial_segmentation/yolo_seg.hpp"

namespace axial_segmentation
{

using sensor_msgs::msg::CompressedImage;
using sensor_msgs::msg::Image;
using vision_msgs::msg::Detection2DArray;

namespace enc = sensor_msgs::image_encodings;

namespace
{

bool ends_with(const std::string & s, const std::string & suffix)
{
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

cv::Mat to_bgr(const Image & msg)
{
  int type = 0, code = -1;
  if (msg.encoding == enc::BGR8) {
    type = CV_8UC3;
  } else if (msg.encoding == enc::RGB8) {
    type = CV_8UC3;
    code = cv::COLOR_RGB2BGR;
  } else if (msg.encoding == enc::BGRA8) {
    type = CV_8UC4;
    code = cv::COLOR_BGRA2BGR;
  } else if (msg.encoding == enc::RGBA8) {
    type = CV_8UC4;
    code = cv::COLOR_RGBA2BGR;
  } else if (msg.encoding == enc::MONO8) {
    type = CV_8UC1;
    code = cv::COLOR_GRAY2BGR;
  } else {
    throw std::runtime_error("unsupported image encoding " + msg.encoding);
  }
  const cv::Mat view(static_cast<int>(msg.height), static_cast<int>(msg.width), type,
    const_cast<uint8_t *>(msg.data.data()), msg.step);
  cv::Mat out;
  if (code < 0) {
    out = view.clone();
  } else {
    cv::cvtColor(view, out, code);
  }
  return out;
}

// package://<package>/<path> -> <package share>/<path>; other paths are returned as they are
std::string resolve_path(const std::string & path)
{
  const std::string scheme = "package://";
  if (path.rfind(scheme, 0) != 0) {
    return path;
  }
  const auto slash = path.find('/', scheme.size());
  if (slash == std::string::npos) {
    throw std::runtime_error("model_path " + path + " has no file after the package name");
  }
  const auto package = path.substr(scheme.size(), slash - scheme.size());
  return ament_index_cpp::get_package_share_directory(package) + path.substr(slash);
}

Image to_msg(const cv::Mat & m, const std::string & encoding, const std_msgs::msg::Header & header)
{
  Image msg;
  msg.header = header;
  msg.height = static_cast<uint32_t>(m.rows);
  msg.width = static_cast<uint32_t>(m.cols);
  msg.encoding = encoding;
  msg.step = static_cast<uint32_t>(m.cols * m.elemSize());
  msg.data.resize(msg.step * msg.height);
  for (int r = 0; r < m.rows; ++r) {
    std::memcpy(msg.data.data() + r * msg.step, m.ptr(r), msg.step);
  }
  return msg;
}

}  // namespace

class SegmentationNode : public rclcpp::Node
{
public:
  SegmentationNode()
  : Node("axial_segmentation")
  {
    const auto model_path = resolve_path(declare_parameter<std::string>(
        "model_path", "package://axial_segmentation/models/sherec_seg_sim_p2b_n.onnx"));
    const auto image_topic = declare_parameter<std::string>("image_topic", "image_raw");
    const auto labels_topic = declare_parameter<std::string>("labels_topic", "segmentation/labels_map");
    const auto detections_topic = declare_parameter<std::string>("detections_topic", "segmentation/detections");
    const auto overlay_topic = declare_parameter<std::string>("overlay_topic", "segmentation/overlay");
    classes_ = make_classes(
      declare_parameter<std::vector<int64_t>>("class_ids", std::vector<int64_t>{}),
      declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{}),
      declare_parameter<std::vector<std::string>>("class_geometry", std::vector<std::string>{}));
    params_.conf = static_cast<float>(declare_parameter<double>("conf", 0.4));
    params_.iou = static_cast<float>(declare_parameter<double>("iou", 0.6));
    params_.max_det = static_cast<int>(declare_parameter<int64_t>("max_det", 100));
    const int threads = static_cast<int>(declare_parameter<int64_t>("threads", 4));
    const auto device = declare_parameter<std::string>("device", "cuda");
    const bool reliable = declare_parameter<bool>("image_reliable", false);
    const double max_rate_hz = declare_parameter<double>("max_rate_hz", 10.0);
    min_period_ns_ = max_rate_hz > 0 ? static_cast<int64_t>(1e9 / max_rate_hz) : 0;

    model_ = std::make_unique<YoloSeg>(model_path, threads, device);
    if (model_->provider() != device) {
      RCLCPP_WARN(get_logger(), "device %s unavailable, running on %s: %s", device.c_str(),
        model_->provider().c_str(), model_->fallback_reason().c_str());
    }
    if (model_->names().size() != classes_.size()) {
      throw std::runtime_error("model has " + std::to_string(model_->names().size()) + " classes, config has " +
              std::to_string(classes_.size()));
    }
    for (const auto & c : classes_) {
      if (model_->names()[c.index] != c.name) {
        throw std::runtime_error("model class " + std::to_string(c.index) + " is '" + model_->names()[c.index] +
                "', config says '" + c.name + "'");
      }
    }

    // images go out best effort like the camera, so a slow subscriber never throttles the publisher
    pub_labels_ = create_publisher<Image>(labels_topic, rclcpp::SensorDataQoS());
    pub_detections_ = create_publisher<Detection2DArray>(detections_topic, 5);
    if (!overlay_topic.empty()) {
      pub_overlay_ = create_publisher<Image>(overlay_topic, rclcpp::SensorDataQoS().keep_last(1));
    }
    auto qos = rclcpp::SensorDataQoS();
    if (reliable) {
      qos.reliable();
    }
    if (ends_with(image_topic, "/compressed")) {
      sub_compressed_ = create_subscription<CompressedImage>(image_topic, qos,
          [this](CompressedImage::ConstSharedPtr m) {push(Frame{m->header, m, nullptr});});
    } else {
      sub_image_ = create_subscription<Image>(image_topic, qos,
          [this](Image::ConstSharedPtr m) {push(Frame{m->header, nullptr, m});});
    }
    worker_ = std::thread([this] {run();});
    RCLCPP_INFO(get_logger(), "%s: %zu classes, input %d, on %s, image %s", model_path.c_str(), classes_.size(),
      model_->input_size(), model_->provider().c_str(), image_topic.c_str());
  }

  ~SegmentationNode() override
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }

private:
  struct Frame
  {
    std_msgs::msg::Header header;
    CompressedImage::ConstSharedPtr compressed;
    Image::ConstSharedPtr raw;
  };

  // keeps only the newest frame; decoding happens on the worker
  void push(Frame f)
  {
    // at most max_rate_hz images by stamp, as axial_frontend
    const int64_t stamp = rclcpp::Time(f.header.stamp).nanoseconds();
    if (last_stamp_ns_ && stamp - *last_stamp_ns_ < min_period_ns_ && stamp >= *last_stamp_ns_) {
      return;
    }
    last_stamp_ns_ = stamp;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (pending_) {
        ++dropped_;
      }
      pending_ = std::move(f);
    }
    cv_.notify_one();
  }

  void run()
  {
    while (true) {
      Frame f;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {return stop_ || pending_.has_value();});
        if (stop_) {
          return;
        }
        f = std::move(*pending_);
        pending_.reset();
      }
      try {
        process(f);
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "%s", e.what());
      }
    }
  }

  void process(const Frame & f)
  {
    const auto t0 = std::chrono::steady_clock::now();
    cv::Mat bgr;
    if (f.compressed) {
      bgr = cv::imdecode(f.compressed->data, cv::IMREAD_COLOR);
      if (bgr.empty()) {
        throw std::runtime_error("cannot decode compressed image (" + f.compressed->format + ")");
      }
    } else {
      bgr = to_bgr(*f.raw);
    }
    const auto instances = model_->infer(bgr, params_);
    const cv::Mat labels = encode_labels(instances, classes_, bgr.cols, bgr.rows);
    pub_labels_->publish(to_msg(labels, enc::RGB8, f.header));

    Detection2DArray dets;
    dets.header = f.header;
    for (const auto & inst : instances) {
      vision_msgs::msg::Detection2D d;
      d.header = f.header;
      d.bbox.center.position.x = inst.box.x + inst.box.width / 2.0;
      d.bbox.center.position.y = inst.box.y + inst.box.height / 2.0;
      d.bbox.size_x = inst.box.width;
      d.bbox.size_y = inst.box.height;
      vision_msgs::msg::ObjectHypothesisWithPose h;
      h.hypothesis.class_id = classes_[inst.index].name;
      h.hypothesis.score = inst.score;
      d.results.push_back(h);
      dets.detections.push_back(std::move(d));
    }
    pub_detections_->publish(dets);
    if (pub_overlay_ && pub_overlay_->get_subscription_count() > 0) {
      pub_overlay_->publish(to_msg(draw_overlay(bgr, instances, classes_), enc::BGR8, f.header));
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    total_ms_ += ms;
    if (++frames_ % 100 == 0) {
      RCLCPP_INFO(get_logger(), "%d frames, %.1f ms mean, %d dropped while busy", frames_, total_ms_ / frames_,
        dropped_);
    }
  }

  std::vector<SegClass> classes_;
  DecodeParams params_;
  std::unique_ptr<YoloSeg> model_;
  rclcpp::Publisher<Image>::SharedPtr pub_labels_;
  rclcpp::Publisher<Image>::SharedPtr pub_overlay_;
  rclcpp::Publisher<Detection2DArray>::SharedPtr pub_detections_;
  rclcpp::Subscription<Image>::SharedPtr sub_image_;
  rclcpp::Subscription<CompressedImage>::SharedPtr sub_compressed_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::optional<Frame> pending_;
  bool stop_ = false;
  std::thread worker_;
  int frames_ = 0;
  int dropped_ = 0;
  int64_t min_period_ns_ = 0;
  std::optional<int64_t> last_stamp_ns_;
  double total_ms_ = 0.0;
};

}  // namespace axial_segmentation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<axial_segmentation::SegmentationNode>());
  } catch (const std::exception & e) {
    std::cerr << "axial_segmentation: " << e.what() << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
