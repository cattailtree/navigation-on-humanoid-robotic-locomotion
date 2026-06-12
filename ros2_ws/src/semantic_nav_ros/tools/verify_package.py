from __future__ import annotations

import ast
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []
    setup_py = PACKAGE_ROOT / "setup.py"
    config_yaml = PACKAGE_ROOT / "config" / "semantic_nav_ros.yaml"
    launch_dir = PACKAGE_ROOT / "launch"
    docs_dir = PACKAGE_ROOT / "docs"
    launch_py = launch_dir / "semantic_nav_real.launch.py"
    package_xml = PACKAGE_ROOT / "package.xml"

    for path in (setup_py, config_yaml, launch_py, package_xml):
        if not path.exists():
            errors.append(f"missing {path.relative_to(PACKAGE_ROOT)}")

    if errors:
        return _finish(errors)

    setup_text = setup_py.read_text(encoding="utf-8")
    if 'glob("launch/*.launch.py")' not in setup_text:
        errors.append("setup.py does not install all launch files")
    if 'glob("config/*.yaml")' not in setup_text:
        errors.append("setup.py does not install all config files")
    if 'glob("docs/*.md")' not in setup_text:
        errors.append("setup.py does not install docs")
    if 'glob("tools/*.py")' not in setup_text:
        errors.append("setup.py does not install tools")
    expected_entrypoints = (
        "semantic_nav_node",
        "fdm_mppi_ros2_node",
        "fdm_mppi_smoke_inputs",
        "inspect_fdm_checkpoint",
        "laser_height_scan_bridge",
        "fdm_observation_bridge",
        "cmd_mux_node",
        "safety_gate_node",
        "gdino_bridge_node",
    )
    for entrypoint in expected_entrypoints:
        if f"{entrypoint} =" not in setup_text:
            errors.append(f"missing console script {entrypoint}")

    config_text = config_yaml.read_text(encoding="utf-8")
    expected_config_nodes = (
        "semantic_nav_node:",
        "semantic_nav_cmd_mux:",
        "semantic_nav_safety_gate:",
        "fdm_mppi_ros2_node:",
        "fdm_observation_bridge:",
        "laser_height_scan_bridge:",
        "semantic_nav_gdino_bridge:",
    )
    for node in expected_config_nodes:
        if node not in config_text:
            errors.append(f"missing config node {node}")
    for required_config in (
        "model_run_dir: logs/fdm/fdm_se2_prediction_depth/Jun11_14-20-48_fdm_train",
        "fdm_state_dim: 8",
        "fdm_proprioception_dim: 157",
        "proprio_layout_csv: velocity_commands,projected_gravity,base_lin_vel,base_ang_vel,joint_torque,joint_pos,joint_vel,last_actions,second_last_action",
        "state_position_frame: local",
        "require_observation_dims: true",
        "backend: fdm_mppi",
    ):
        if required_config not in config_text:
            errors.append(f"config missing {required_config}")

    expected_launch_files = (
        "semantic_nav_real.launch.py",
        "semantic_nav_waypoint.launch.py",
        "semantic_nav_fdm_mppi_test.launch.py",
    )
    for filename in expected_launch_files:
        if not (launch_dir / filename).exists():
            errors.append(f"missing launch/{filename}")

    launch_text = launch_py.read_text(encoding="utf-8")
    for launch_arg in ("use_gdino_bridge", "use_fdm_mppi", "use_laser_height_scan", "use_fdm_observation_bridge", "fdm_backend"):
        if launch_arg not in launch_text:
            errors.append(f"missing launch arg {launch_arg}")
    for executable in (
        "semantic_nav_node",
        "cmd_mux_node",
        "safety_gate_node",
        "fdm_mppi_ros2_node",
        "fdm_observation_bridge",
        "laser_height_scan_bridge",
        "gdino_bridge_node",
    ):
        if f'executable="{executable}"' not in launch_text:
            errors.append(f"launch does not start executable {executable}")

    package_xml_text = package_xml.read_text(encoding="utf-8")
    for dep in ("rclpy", "geometry_msgs", "nav_msgs", "sensor_msgs", "std_msgs", "python3-numpy"):
        if f"<exec_depend>{dep}</exec_depend>" not in package_xml_text:
            errors.append(f"missing exec_depend {dep}")

    interfaces_md = docs_dir / "interfaces.md"
    if not interfaces_md.exists():
        errors.append("missing docs/interfaces.md")
    else:
        interfaces_text = interfaces_md.read_text(encoding="utf-8")
        for required_text in (
            "semantic_nav_waypoint.launch.py",
            "semantic_nav_fdm_mppi_test.launch.py",
            "/semantic_nav/cmd_vel_raw",
            "/semantic_nav/fdm_cmd_vel_raw",
            "/semantic_nav/selected_cmd_vel_raw",
            "/semantic_nav/fdm_goal",
            "/semantic_nav/fdm_height_scan",
            "/semantic_nav/fdm_state",
            "/semantic_nav/fdm_proprioception",
            "/cmd_vel",
        ):
            if required_text not in interfaces_text:
                errors.append(f"interfaces.md missing {required_text}")

    for path in (*sorted(launch_dir.glob("*.py")), *sorted((PACKAGE_ROOT / "semantic_nav_ros").glob("*.py"))):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"syntax error in {path.relative_to(PACKAGE_ROOT)}: {exc}")

    return _finish(errors)


def _finish(errors: list[str]) -> int:
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("semantic_nav_ros package structure OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
