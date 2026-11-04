"""Mask projector, plus RViz when rviz_config is given.

    ros2 launch axial_perception mask_projection.launch.py params_file:=<yaml> [rviz_config:=<rviz>] [use_sim_time:=true]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    rviz_config = LaunchConfiguration("rviz_config")
    return LaunchDescription([
        DeclareLaunchArgument("params_file"),
        DeclareLaunchArgument("rviz_config", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(package="axial_perception", executable="mask_projector", output="screen",
             parameters=[LaunchConfiguration("params_file"), sim_time]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_config], parameters=[sim_time],
             condition=IfCondition(PythonExpression(["'", rviz_config, "' != ''"]))),
    ])
