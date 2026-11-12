"""Mask projector, plus RViz with rviz:=true.

    ros2 launch axial_perception axial_frontend.launch.py params_file:=<yaml> [rviz:=true fixed_frame:=<frame>] [use_sim_time:=true]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    rviz_config = PathJoinSubstitution([FindPackageShare("axial_perception"), "rviz", "axial_frontend.rviz"])
    return LaunchDescription([
        DeclareLaunchArgument("params_file"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("fixed_frame", default_value="odom"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(package="axial_perception", executable="axial_frontend", output="screen",
             parameters=[LaunchConfiguration("params_file"), sim_time]),
        Node(package="rviz2", executable="rviz2", parameters=[sim_time],
             arguments=["-d", rviz_config, "-f", LaunchConfiguration("fixed_frame")],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
