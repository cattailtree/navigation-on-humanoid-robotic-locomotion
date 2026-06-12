from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    use_gdino_bridge = LaunchConfiguration("use_gdino_bridge")
    use_fdm_mppi = LaunchConfiguration("use_fdm_mppi")
    use_laser_height_scan = LaunchConfiguration("use_laser_height_scan")
    use_fdm_observation_bridge = LaunchConfiguration("use_fdm_observation_bridge")
    fdm_backend = LaunchConfiguration("fdm_backend")

    default_params = PathJoinSubstitution(
        [FindPackageShare("semantic_nav_ros"), "config", "semantic_nav_ros.yaml"]
    )

    nav_node = Node(
        package="semantic_nav_ros",
        executable="semantic_nav_node",
        name="semantic_nav_node",
        output="screen",
        parameters=[params_file],
    )
    cmd_mux = Node(
        package="semantic_nav_ros",
        executable="cmd_mux_node",
        name="semantic_nav_cmd_mux",
        output="screen",
        parameters=[params_file],
    )
    safety_gate = Node(
        package="semantic_nav_ros",
        executable="safety_gate_node",
        name="semantic_nav_safety_gate",
        output="screen",
        parameters=[params_file],
    )
    gdino_bridge = Node(
        package="semantic_nav_ros",
        executable="gdino_bridge_node",
        name="semantic_nav_gdino_bridge",
        output="screen",
        parameters=[params_file],
        condition=IfCondition(use_gdino_bridge),
    )
    fdm_mppi = Node(
        package="semantic_nav_ros",
        executable="fdm_mppi_ros2_node",
        name="fdm_mppi_ros2_node",
        output="screen",
        parameters=[params_file, {"backend": fdm_backend}],
        condition=IfCondition(use_fdm_mppi),
    )
    laser_height_scan = Node(
        package="semantic_nav_ros",
        executable="laser_height_scan_bridge",
        name="laser_height_scan_bridge",
        output="screen",
        parameters=[params_file],
        condition=IfCondition(use_laser_height_scan),
    )
    fdm_observation_bridge = Node(
        package="semantic_nav_ros",
        executable="fdm_observation_bridge",
        name="fdm_observation_bridge",
        output="screen",
        parameters=[params_file],
        condition=IfCondition(use_fdm_observation_bridge),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("use_gdino_bridge", default_value="true"),
            DeclareLaunchArgument("use_fdm_mppi", default_value="false"),
            DeclareLaunchArgument("use_laser_height_scan", default_value="false"),
            DeclareLaunchArgument("use_fdm_observation_bridge", default_value="false"),
            DeclareLaunchArgument("fdm_backend", default_value="fdm_mppi"),
            nav_node,
            cmd_mux,
            safety_gate,
            gdino_bridge,
            fdm_mppi,
            laser_height_scan,
            fdm_observation_bridge,
        ]
    )
