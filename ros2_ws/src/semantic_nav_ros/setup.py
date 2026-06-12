from glob import glob

from setuptools import find_packages, setup


package_name = "semantic_nav_ros"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/docs", glob("docs/*.md")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/tools", glob("tools/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fdm team",
    maintainer_email="admin@example.com",
    description="ROS2 bridge for semantic navigation and velocity-control deployment.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "semantic_nav_node = semantic_nav_ros.semantic_nav_node:main",
            "fdm_mppi_ros2_node = semantic_nav_ros.fdm_mppi_ros2_node:main",
            "fdm_mppi_smoke_inputs = semantic_nav_ros.fdm_mppi_smoke_inputs:main",
            "inspect_fdm_checkpoint = semantic_nav_ros.inspect_fdm_checkpoint:main",
            "laser_height_scan_bridge = semantic_nav_ros.laser_height_scan_bridge:main",
            "fdm_observation_bridge = semantic_nav_ros.fdm_observation_bridge:main",
            "cmd_mux_node = semantic_nav_ros.cmd_mux_node:main",
            "safety_gate_node = semantic_nav_ros.safety_gate_node:main",
            "gdino_bridge_node = semantic_nav_ros.gdino_bridge_node:main",
        ],
    },
)
