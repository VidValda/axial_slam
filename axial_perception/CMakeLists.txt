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

add_library(axial_core STATIC
  src/camera.cpp
  src/clusters.cpp
  src/landmarks.cpp
  src/masks.cpp
  src/occlusion.cpp
  src/scan_buffer.cpp
  src/segments.cpp
)
set_target_properties(axial_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(axial_core PUBLIC include)
target_include_directories(axial_core SYSTEM PUBLIC third_party)
target_link_libraries(axial_core PUBLIC Eigen3::Eigen)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/mask_projector DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  foreach(name camera clusters kdtree landmarks masks occlusion scan_buffer segments)
    ament_add_gtest(test_${name}_cpp test/test_${name}.cpp)
    target_link_libraries(test_${name}_cpp axial_core)
  endforeach()
  ament_add_pytest_test(test_python test)
endif()

ament_package()
