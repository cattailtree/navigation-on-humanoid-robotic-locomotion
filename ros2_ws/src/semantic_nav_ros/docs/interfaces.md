# semantic_nav_ros Interfaces

This file is the topic and launch contract for the ROS2 migration layer.

## Launch Files

| Launch | Purpose | Default command source |
| --- | --- | --- |
| `semantic_nav_waypoint.launch.py` | Semantic graph / open-vocabulary waypoint navigation through the safety gate | `waypoint` |
| `semantic_nav_fdm_mppi_test.launch.py` | Semantic waypoint goals feeding FDM-MPPI local planning, with scan/state/proprio bridges and safety gate | `fdm` |
| `semantic_nav_real.launch.py` | Composable bring-up with launch arguments for optional G-DINO, FDM-MPPI, height scan, and observation bridges | `waypoint` from YAML |

## Nodes

| Node | Executable | Responsibility |
| --- | --- | --- |
| `semantic_nav_node` | `semantic_nav_node` | Natural-language / graph target parsing, semantic path generation, waypoint raw command, active FDM goal publication |
| `semantic_nav_cmd_mux` | `cmd_mux_node` | Select one raw command source before safety gating |
| `semantic_nav_safety_gate` | `safety_gate_node` | Clamp velocity, stop on stale command, stale scan, front obstacle, or estop |
| `fdm_mppi_ros2_node` | `fdm_mppi_ros2_node` | ROS2 wrapper for MPPI-only and FDM-MPPI local planner |
| `laser_height_scan_bridge` | `laser_height_scan_bridge` | Project `/scan` to flattened FDM height scan |
| `fdm_observation_bridge` | `fdm_observation_bridge` | Build FDM state and proprioception vectors from standard ROS2 topics |
| `semantic_nav_gdino_bridge` | `gdino_bridge_node` | Bridge compressed camera images to GroundingDINO HTTP server |

## External Robot Inputs

| Topic | Type | Required by |
| --- | --- | --- |
| `/odom` | `nav_msgs/Odometry` | semantic nav, FDM-MPPI, FDM observation bridge |
| `/scan` | `sensor_msgs/LaserScan` | safety gate, laser height scan bridge |
| `/joint_states` | `sensor_msgs/JointState` | FDM observation bridge |
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | GroundingDINO bridge |
| `/cmd_vel` | `geometry_msgs/Twist` | robot low-level gait controller consumes this final command |

## Task And Control Topics

| Topic | Type | Direction | Meaning |
| --- | --- | --- | --- |
| `/semantic_nav/goal` | `std_msgs/String` | input | Natural language task |
| `/semantic_nav/target_node` | `std_msgs/String` | input | Direct semantic graph node target |
| `/semantic_nav/pause` | `std_msgs/Bool` | input | Pause semantic navigation |
| `/semantic_nav/estop` | `std_msgs/Bool` | input | Emergency stop safety gate |
| `/semantic_nav/cmd_source` | `std_msgs/String` | input | Select `waypoint` or `fdm` in command mux |
| `/semantic_nav/cmd_vel_raw` | `geometry_msgs/Twist` | internal output | Waypoint raw command |
| `/semantic_nav/fdm_cmd_vel_raw` | `geometry_msgs/Twist` | internal output | FDM-MPPI raw command |
| `/semantic_nav/selected_cmd_vel_raw` | `geometry_msgs/Twist` | internal output | Mux-selected raw command sent to safety gate |
| `/cmd_vel` | `geometry_msgs/Twist` | robot output | Final safety-gated command |

## FDM-MPPI Topics

| Topic | Type | Direction | Meaning |
| --- | --- | --- | --- |
| `/semantic_nav/fdm_goal` | `geometry_msgs/PoseStamped` | input to FDM | Active semantic waypoint as local planner goal |
| `/semantic_nav/fdm_height_scan` | `std_msgs/Float32MultiArray` | input to FDM | Flattened local height scan, default shape `[60, 46]` |
| `/semantic_nav/fdm_state` | `std_msgs/Float32MultiArray` | input to FDM | State vector matching checkpoint config; Jun11 G1 default is 8 values |
| `/semantic_nav/fdm_proprioception` | `std_msgs/Float32MultiArray` | input to FDM | Proprioception vector matching checkpoint config; Jun11 G1 default is 157 values and includes `base_lin_vel` |
| `/semantic_nav/fdm_path` | `nav_msgs/Path` | output | Selected local predicted path |
| `/semantic_nav/fdm_debug` | `std_msgs/String` | output | JSON debug terms |

## Status Topics

| Topic | Type |
| --- | --- |
| `/semantic_nav/status` | `std_msgs/String` |
| `/semantic_nav/cmd_mux_status` | `std_msgs/String` |
| `/semantic_nav/safety_status` | `std_msgs/String` |
| `/semantic_nav/fdm_status` | `std_msgs/String` |
| `/semantic_nav/height_scan_status` | `std_msgs/String` |
| `/semantic_nav/fdm_observation_status` | `std_msgs/String` |
| `/semantic_nav/detections_json` | `std_msgs/String` |

## Command Chain

```text
/semantic_nav/cmd_vel_raw      \
                                -> /semantic_nav/selected_cmd_vel_raw -> safety_gate -> /cmd_vel
/semantic_nav/fdm_cmd_vel_raw  /
```

Only `/cmd_vel` should be consumed by the robot low-level gait controller during real runs.
