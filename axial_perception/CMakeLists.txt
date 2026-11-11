cmake_minimum_required(VERSION 3.8)
project(axial_perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(dps_slam_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(axial_core STATIC
  src/camera.cpp
  src/clusters.cpp
  src/landmarks.cpp
  src/masks.cpp
  src/occlusion.cpp
  src/projector.cpp
  src/scan_buffer.cpp
  src/segments.cpp
)
set_target_properties(axial_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(axial_core PUBLIC include)
target_include_directories(axial_core SYSTEM PUBLIC third_party)
target_link_libraries(axial_core PUBLIC Eigen3::Eigen)

add_library(axial_ros STATIC
  src/cloud_msg.cpp
  src/mask_projector.cpp
)
target_link_libraries(axial_ros PUBLIC
  axial_core
  rclcpp::rclcpp
  tf2::tf2
  tf2_ros::tf2_ros
  ${builtin_interfaces_TARGETS}
  ${dps_slam_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
)

add_executable(mask_projector src/mask_projector_main.cpp)
target_link_libraries(mask_projector axial_ros)
install(TARGETS mask_projector DESTINATION lib/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/mask_projector_py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  foreach(name camera clusters kdtree landmarks masks occlusion scan_buffer segments)
    ament_add_gtest(test_${name}_cpp test/test_${name}.cpp)
    target_link_libraries(test_${name}_cpp axial_core)
  endforeach()
  ament_add_gtest(test_mask_projector_cpp test/test_mask_projector.cpp TIMEOUT 60)
  target_link_libraries(test_mask_projector_cpp axial_ros)
  ament_add_pytest_test(test_python test)
endif()

ament_package()
