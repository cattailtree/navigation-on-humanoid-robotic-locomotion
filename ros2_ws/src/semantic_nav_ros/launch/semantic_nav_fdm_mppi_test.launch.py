from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    cmd_source = LaunchConfiguration("cmd_source")
    fdm_backend = LaunchConfiguration("fdm_backend")
    default_params = PathJoinSubstitution(
        [FindPackageShare("semantic_nav_ros"), "config", "semantic_nav_ros.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("cmd_source", default_value="fdm"),
            DeclareLaunchArgument("fdm_backend", default_value="fdm_mppi"),
            Node(
                package="semantic_nav_ros",
                executable="semantic_nav_node",
                name="semantic_nav_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="semantic_nav_ros",
                executable="fdm_mppi_ros2_node",
                name="fdm_mppi_ros2_node",
                output="screen",
                parameters=[params_file, {"backend": fdm_backend}],
            ),
            Node(
                package="semantic_nav_ros",
                executable="laser_height_scan_bridge",
                name="laser_height_scan_bridge",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="semantic_nav_ros",
                executable="fdm_observation_bridge",
                name="fdm_observation_bridge",
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
