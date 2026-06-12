from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    cmd_source = LaunchConfiguration("cmd_source")
    default_params = PathJoinSubstitution(
        [FindPackageShare("semantic_nav_ros"), "config", "semantic_nav_ros.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("cmd_source", default_value="waypoint"),
            Node(
                package="semantic_nav_ros",
                executable="semantic_nav_node",
                name="semantic_nav_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="semantic_nav_ros",
                executable="cmd_mux_node",
                name="semantic_nav_cmd_mux",
                output="screen",
                parameters=[params_file, {"default_source": cmd_source}],
            ),
            Node(
                package="semantic_nav_ros",
                executable="safety_gate_node",
                name="semantic_nav_safety_gate",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
