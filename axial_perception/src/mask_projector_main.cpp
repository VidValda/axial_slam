#include <exception>
#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "axial_perception/mask_projector.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<axial_perception::MaskProjector>());
  } catch (const std::exception & e) {
    std::cerr << "mask_projector: " << e.what() << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
