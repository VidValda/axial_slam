"""YOLO segmentation node; classes and topics come from the axial_segmentation section of params_file.

    ros2 launch axial_segmentation axial_segmentation.launch.py params_file:=<yaml> [model_path:=<onnx>] [use_sim_time:=true]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    model = LaunchConfiguration("model_path")
    params = [LaunchConfiguration("params_file"), {"use_sim_time": LaunchConfiguration("use_sim_time")}]
    return LaunchDescription([
        DeclareLaunchArgument("params_file"),
        DeclareLaunchArgument("model_path", default_value="", description="overrides model_path in params_file"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(package="axial_segmentation", executable="axial_segmentation", output="screen", parameters=params,
             condition=IfCondition(PythonExpression(["'", model, "' == ''"]))),
        Node(package="axial_segmentation", executable="axial_segmentation", output="screen",
             parameters=params + [{"model_path": model}],
             condition=IfCondition(PythonExpression(["'", model, "' != ''"]))),
    ])
