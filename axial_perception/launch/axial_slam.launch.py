"""axial_slam front end from one params file: axial_frontend, plus axial_segmentation when the front end reads its masks.

    ros2 launch axial_perception axial_slam.launch.py params_file:=<yaml> [labels_topic:=<topic>] [detections_topic:=<topic>]
        [use_sim_time:=true]
"""
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def section(params, node):
    return (params.get(node) or {}).get("ros__parameters") or {}


def launch_nodes(context):
    params_file = LaunchConfiguration("params_file").perform(context)
    labels_override = LaunchConfiguration("labels_topic").perform(context)
    detections = LaunchConfiguration("detections_topic").perform(context)
    sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time").perform(context) == "true"}
    with open(params_file) as f:
        params = yaml.safe_load(f) or {}
    labels = labels_override or section(params, "axial_frontend").get("labels_topic")
    frontend = [params_file, sim_time] + ([{"labels_topic": labels_override}] if labels_override else [])
    remap = [("/axial/line_detections", detections)] if detections else []
    nodes = [Node(package="axial_perception", executable="axial_frontend", output="screen", parameters=frontend,
                  remappings=remap)]
    if labels and labels == section(params, "axial_segmentation").get("labels_topic"):
        nodes.append(Node(package="axial_segmentation", executable="axial_segmentation", output="screen",
                          parameters=[params_file, sim_time]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_file"),
        DeclareLaunchArgument("labels_topic", default_value="", description="overrides axial_frontend's labels_topic"),
        DeclareLaunchArgument("detections_topic", default_value="", description="remaps /axial/line_detections"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=launch_nodes),
    ])
