#include <exception>
#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "axial_perception/axial_frontend.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<axial_perception::AxialFrontend>());
  } catch (const std::exception & e) {
    std::cerr << "axial_frontend: " << e.what() << std::endl;
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
