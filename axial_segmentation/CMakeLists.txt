cmake_minimum_required(VERSION 3.8)
project(axial_segmentation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ONNX Runtime C++ release, unpacked under ONNXRUNTIME_ROOT
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime install prefix")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
endif()

add_library(seg_core STATIC
  src/decode.cpp
  src/labels_map.cpp
)
set_target_properties(seg_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(seg_core PUBLIC include)
target_link_libraries(seg_core PUBLIC ${OpenCV_LIBS})

add_library(seg_ort STATIC src/yolo_seg.cpp)
set_target_properties(seg_ort PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(seg_ort SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(seg_ort PUBLIC seg_core ${ONNXRUNTIME_LIBRARY})

add_executable(axial_segmentation src/segmentation_node.cpp)
target_link_libraries(axial_segmentation seg_ort rclcpp::rclcpp ament_index_cpp::ament_index_cpp ${sensor_msgs_TARGETS}
  ${vision_msgs_TARGETS})

add_executable(seg_infer tools/seg_infer.cpp)
target_link_libraries(seg_infer seg_ort)

set_target_properties(axial_segmentation seg_infer PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib" BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
install(TARGETS axial_segmentation seg_infer DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch models DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  foreach(name decode labels_map)
    ament_add_gtest(test_${name} test/test_${name}.cpp)
    target_link_libraries(test_${name} seg_core)
  endforeach()
endif()

ament_package()
