# semantic_nav_ros

ROS2 deployment bridge for this repository's semantic navigation stack.

For the full topic contract, see `docs/interfaces.md`.

This package keeps the simulation code and the robot-facing ROS2 code separated:

- `semantic_nav_node`: parses a natural-language or graph target request, builds a semantic execution plan, subscribes to odometry, and publishes raw velocity commands.
- `fdm_mppi_ros2_node`: runs the ROS2 FDM/MPPI local-planner test backend.
- `fdm_observation_bridge`: builds FDM state/proprio vectors from standard ROS2 odometry, joint states, and selected commands.
- `laser_height_scan_bridge`: projects `/scan` into the FDM height-scan topic for local-planner smoke tests.
- `semantic_nav_cmd_mux`: selects waypoint or FDM-MPPI raw commands before the safety gate.
- `semantic_nav_safety_gate`: clamps/stops raw velocity commands using command timeout, scan timeout, front clearance, and estop.
- `semantic_nav_gdino_bridge`: forwards compressed camera images to the existing GroundingDINO HTTP service and publishes detections as JSON.

The current real-robot semantic backend is the deploy-safe waypoint velocity backend. FDM-MPPI is also exposed as a separate ROS2 test node, so waypoint navigation and FDM-MPPI local planning can be validated independently before being chained together.

## Expected Robot Interfaces

Required topics:

- `/odom` (`nav_msgs/Odometry`): robot pose in the `map` frame or a locally consistent world frame.
- `/scan` (`sensor_msgs/LaserScan`): front safety scan. Used by the safety gate.
- `/cmd_vel` (`geometry_msgs/Twist`): final velocity command consumed by the robot low-level gait controller.

Optional topics:

- `/camera/color/image_raw/compressed` (`sensor_msgs/CompressedImage`): JPEG stream for GroundingDINO.
- `/semantic_nav/estop` (`std_msgs/Bool`): `true` immediately forces zero velocity.

Published topics:

- `/semantic_nav/cmd_vel_raw` (`geometry_msgs/Twist`): semantic navigation command before the safety gate.
- `/semantic_nav/status` (`std_msgs/String`): task state.
- `/semantic_nav/safety_status` (`std_msgs/String`): safety gate state.
- `/semantic_nav/path` (`nav_msgs/Path`): active graph/exploration path.
- `/semantic_nav/detections_json` (`std_msgs/String`): GroundingDINO detections.
- `/semantic_nav/fdm_cmd_vel_raw` (`geometry_msgs/Twist`): FDM-MPPI raw velocity command.
- `/semantic_nav/fdm_status` (`std_msgs/String`): FDM-MPPI node state.
- `/semantic_nav/height_scan_status` (`std_msgs/String`): scan-to-height bridge state.
- `/semantic_nav/fdm_observation_status` (`std_msgs/String`): FDM state/proprio bridge state.
- `/semantic_nav/selected_cmd_vel_raw` (`geometry_msgs/Twist`): command selected by the mux and consumed by the safety gate.
- `/semantic_nav/cmd_mux_status` (`std_msgs/String`): selected command source and timeout state.

Task input topics:

- `/semantic_nav/goal` (`std_msgs/String`): natural language request, such as `find the elevator` or `find the fridge`.
- `/semantic_nav/target_node` (`std_msgs/String`): graph node id, such as `room_f1`.
- `/semantic_nav/pause` (`std_msgs/Bool`): pause/resume navigation.

## Build

From the ROS2 machine:

```bash
cd /path/to/fdm/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
export FDM_REPO_ROOT=/path/to/fdm
```

If your ROS2 distro is not Humble, source your distro setup file instead.

The package imports the repository Python modules from `scripts/semantic_nav`, so keep this ROS2 workspace inside the repo layout unless you update `semantic_nav_ros/ros_utils.py`.

Without a ROS2 runtime, you can still run the package-structure check:

```bash
python ros2_ws/src/semantic_nav_ros/tools/verify_package.py
```

## Start GroundingDINO

On the Windows/dev machine, keep using the existing service:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\Admin\fdm\scripts\semantic_nav\apexnav_bridge\start_gdino_server.ps1
```

The ROS2 bridge expects the service at:

```text
http://127.0.0.1:12181/gdino
```

If GroundingDINO runs on another host, edit `config/semantic_nav_ros.yaml` and set `semantic_nav_gdino_bridge.endpoint`.

## Launch

```bash
cd /path/to/fdm/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export FDM_REPO_ROOT=/path/to/fdm
ros2 launch semantic_nav_ros semantic_nav_real.launch.py
```

Waypoint-only bring-up:

```bash
ros2 launch semantic_nav_ros semantic_nav_waypoint.launch.py
```

This launch selects `cmd_source=waypoint` automatically.

Disable the G-DINO bridge when you only want graph navigation:

```bash
ros2 launch semantic_nav_ros semantic_nav_real.launch.py use_gdino_bridge:=false
```

Launch the FDM-MPPI ROS2 node as well:

```bash
ros2 launch semantic_nav_ros semantic_nav_real.launch.py use_fdm_mppi:=true
```

FDM-MPPI local-planner bring-up with LaserScan height bridge and observation bridge:

```bash
ros2 launch semantic_nav_ros semantic_nav_fdm_mppi_test.launch.py fdm_backend:=mppi_only
```

This launch selects `cmd_source=fdm` automatically.

For true FDM-MPPI with the latest configured model, omit the override:

```bash
ros2 launch semantic_nav_ros semantic_nav_fdm_mppi_test.launch.py
```

## Send Tasks

Find the elevator:

```bash
ros2 topic pub --once /semantic_nav/goal std_msgs/msg/String "{data: 'find the elevator'}"
```

Go to a graph target:

```bash
ros2 topic pub --once /semantic_nav/target_node std_msgs/msg/String "{data: 'room_f1'}"
```

Open-vocabulary search:

```bash
ros2 topic pub --once /semantic_nav/goal std_msgs/msg/String "{data: 'find the fridge'}"
```

If the object is not in the semantic graph, the node enters open-set search mode, publishes detector prompts, follows configured exploration nodes, and stops once GroundingDINO confirms the object for `open_set_confirmations` frames.

Pause/resume:

```bash
ros2 topic pub --once /semantic_nav/pause std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /semantic_nav/pause std_msgs/msg/Bool "{data: false}"
```

Emergency stop:

```bash
ros2 topic pub --once /semantic_nav/estop std_msgs/msg/Bool "{data: true}"
```

Clear estop:

```bash
ros2 topic pub --once /semantic_nav/estop std_msgs/msg/Bool "{data: false}"
```

Select waypoint or FDM-MPPI command source:

```bash
ros2 topic pub --once /semantic_nav/cmd_source std_msgs/msg/String "{data: 'waypoint'}"
ros2 topic pub --once /semantic_nav/cmd_source std_msgs/msg/String "{data: 'fdm'}"
```

## Tune For The Robot

Edit `config/semantic_nav_ros.yaml` before real runs:

- `max_vx`, `max_vy`, `max_wz`: start conservative. Current defaults are `0.35`, `0.05`, `0.55`.
- `stop_distance_m`: hard front stop distance.
- `slow_distance_m`: speed starts scaling down inside this distance.
- `forward_fov_deg`: front scan sector used for safety.
- `xy_tolerance`: graph waypoint completion tolerance.
- `exploration_node_ids`: graph nodes used during open-set object search.
- `building_config`: semantic graph matching the real test area.
- `odom_topic`, `scan_topic`, `image_topic`, `cmd_vel_topic`: match the robot's topic names.

## Real-Robot Checklist

Before enabling walking:

1. Verify odometry frame and graph frame agree:

   ```bash
   ros2 topic echo /odom
   ```

2. Verify the safety gate stops without scan:

   ```bash
   ros2 topic echo /semantic_nav/safety_status
   ```

3. Verify raw commands do not bypass the safety gate. The robot should consume `/cmd_vel`, not `/semantic_nav/cmd_vel_raw` or `/semantic_nav/fdm_cmd_vel_raw`.

4. Verify estop:

   ```bash
   ros2 topic pub --once /semantic_nav/estop std_msgs/msg/Bool "{data: true}"
   ```

5. Verify GroundingDINO output:

   ```bash
   ros2 topic echo /semantic_nav/detections_json
   ```

6. Test with robot lifted or motors disabled, then with `max_vx <= 0.15`, then raise limits gradually.

## FDM-MPPI Testing

This repository already contains an FDM-MPPI robot-facing node under `ros/fdm_navigation_ros`. That node is ROS1/catkin: it uses `rospy`, `dynamic_reconfigure`, `grid_map_msgs/GridMap`, `anymal_msgs/AnymalState`, and `series_elastic_actuator_msgs/SeActuatorReadings`.

The ROS2 package adds `fdm_mppi_ros2_node`, which wraps the Isaac-free FDM/MPPI adapter from `scripts/mujoco_sim2sim/fdm_adapter.py`.

The default FDM contract is the latest G1 29DOF run:

- run dir: `logs/fdm/fdm_se2_prediction_depth/Jun11_14-20-48_fdm_train`
- checkpoint: highest `model_collection_round_*.pth`, currently expected to be `model_collection_round_28.pth`
- geometry head: required, detected from `geometric_collision_*` checkpoint keys
- FDM state: 8 values, `base_position_local xyz + base_orientation xyzw + energy`
- FDM proprioception: 157 values, `velocity_commands, projected_gravity, base_lin_vel, base_ang_vel, joint_torque, joint_pos, joint_vel, last_actions, second_last_action`
- height scan: `[60, 46]`

Before running true FDM-MPPI on a new machine, inspect the checkpoint:

```bash
ros2 run semantic_nav_ros inspect_fdm_checkpoint
```

or from the source tree before `colcon build`:

```bash
python ~/fdm/ros2_ws/src/semantic_nav_ros/tools/inspect_fdm_checkpoint.py
```

FDM-MPPI input topics:

- `/odom` (`nav_msgs/Odometry`): current robot SE(2) pose.
- `/semantic_nav/fdm_goal` (`geometry_msgs/PoseStamped`): local goal. `semantic_nav_node` publishes its active semantic waypoint here when `publish_fdm_goal: true`.
- `/semantic_nav/fdm_height_scan` (`std_msgs/Float32MultiArray`): flattened height scan, default shape `[60, 46]`.
- `/semantic_nav/fdm_state` (`std_msgs/Float32MultiArray`): FDM state vector. `fdm_observation_bridge` can publish this from `/odom`.
- `/semantic_nav/fdm_proprioception` (`std_msgs/Float32MultiArray`): FDM proprio vector. `fdm_observation_bridge` can publish this from `/odom`, `/joint_states`, and selected commands.

FDM-MPPI outputs:

- `/semantic_nav/fdm_cmd_vel_raw` (`geometry_msgs/Twist`)
- `/semantic_nav/fdm_status` (`std_msgs/String`)
- `/semantic_nav/fdm_debug` (`std_msgs/String`, JSON debug terms)

### MPPI-only smoke test

Terminal 1:

```bash
ros2 launch semantic_nav_ros semantic_nav_fdm_mppi_test.launch.py
```

Terminal 2:

```bash
ros2 run semantic_nav_ros fdm_mppi_smoke_inputs
```

Terminal 3:

```bash
ros2 topic echo /semantic_nav/fdm_cmd_vel_raw
ros2 topic echo /semantic_nav/fdm_status
```

`fdm_backend:=mppi_only` verifies the ROS2 planner loop, goal input, height scan input, and velocity output without loading an FDM checkpoint. The default launch backend is `fdm_mppi`.

### LaserScan height-bridge test

If the robot only has `/scan` available during the first real-robot bring-up, run:

```bash
ros2 launch semantic_nav_ros semantic_nav_fdm_mppi_test.launch.py fdm_backend:=mppi_only
```

Then check:

```bash
ros2 topic echo /semantic_nav/height_scan_status
ros2 topic echo /semantic_nav/fdm_observation_status
ros2 topic echo /semantic_nav/fdm_status
ros2 topic echo /semantic_nav/fdm_cmd_vel_raw
```

This bridge is a deployment smoke bridge, not the final trained-model observation bridge: it projects 2D scan returns into a local FDM grid and marks occupied cells with a fixed obstacle height. It is appropriate for `mppi_only` and scan-obstacle-cost tests. For final `fdm_mppi`, replace or extend it with the real depth/elevation-map adapter matching the training setup.

`fdm_observation_bridge` is the standard-message FDM state/proprio bridge. Its defaults match the Jun11 G1/FDM checkpoint: `fdm_state_dim=8`, `fdm_proprioception_dim=157`, `action_dim=29`, and a proprio layout containing `base_lin_vel`. Configure `joint_names_csv` to the G1 29DOF order from `exts/fdm/fdm/env_cfg/robot_cfg_g1.py`. For a robot-specific state estimator, replace this bridge's inputs while keeping the same output topics.

### Semantic planner feeding FDM-MPPI

`semantic_nav_node` publishes its active waypoint to `/semantic_nav/fdm_goal`. This lets you run semantic planning and FDM-MPPI local planning in parallel:

```bash
ros2 launch semantic_nav_ros semantic_nav_fdm_mppi_test.launch.py

ros2 topic pub --once /semantic_nav/goal std_msgs/msg/String "{data: 'find the elevator'}"
```

For this test, route only one command source to the robot. Use `/semantic_nav/cmd_vel_raw` for waypoint testing, or `/semantic_nav/fdm_cmd_vel_raw` for FDM-MPPI local-planner testing.

With the default launch, command routing is:

```text
/semantic_nav/cmd_vel_raw      \
                                -> /semantic_nav/selected_cmd_vel_raw -> safety gate -> /cmd_vel
/semantic_nav/fdm_cmd_vel_raw  /
```

Switch the selected source with `/semantic_nav/cmd_source`.

### True FDM-MPPI test

The default config is already wired to true FDM-MPPI:

```yaml
fdm_mppi_ros2_node:
  ros__parameters:
    backend: fdm_mppi
    model_run_dir: C:/Users/Admin/fdm/logs/fdm/fdm_se2_prediction_depth/Jun11_14-20-48_fdm_train
    checkpoint: ""
```

If `checkpoint` is empty, the node picks the highest `model_collection_round_*.pth` from `model_run_dir`.

Run:

```bash
ros2 launch semantic_nav_ros semantic_nav_fdm_mppi_test.launch.py
```

For true FDM-MPPI, publish real height scan, FDM state, and proprioception vectors with the same dimensions/timing used by the trained model. If `require_observation_dims` is disabled, the adapter can fall back to padded zero history for smoke tests, but that is not a valid final robot deployment signal.

In true `backend: fdm_mppi` mode, `require_observation_dims: true` makes state/proprio dimension mismatches fail loudly. This is intentional: it prevents accidentally running the current G1 model with an older 100-dimensional proprio bridge, a non-geometry-head checkpoint, or a quadruped/ROS1 observation adapter.

### Connect FDM-MPPI through the safety gate

The default config routes both waypoint and FDM-MPPI commands through `semantic_nav_cmd_mux`, then into the safety gate. To test FDM-MPPI as the active source:

```bash
ros2 topic pub --once /semantic_nav/cmd_source std_msgs/msg/String "{data: 'fdm'}"
ros2 topic echo /semantic_nav/cmd_mux_status
ros2 topic echo /semantic_nav/safety_status
```

Switch back to semantic waypoint tracking:

```bash
ros2 topic pub --once /semantic_nav/cmd_source std_msgs/msg/String "{data: 'waypoint'}"
```

## Migration Status

Already reusable:

- Natural-language task parsing.
- Semantic graph planning.
- Open-vocabulary search prompts.
- Waypoint tracking command generation.
- GroundingDINO HTTP detector path.
- Safety gating and command timeout.
- ROS1 FDM-MPPI planner contract.
- ROS2 FDM-MPPI local planner smoke node.

Still required before final real FDM-MPPI deployment:

- Calibrate or replace the standard-message bridges so height scan, FDM state, and proprioception exactly match the training setup.
- Set the real robot joint order in `fdm_observation_bridge.joint_names_csv`.
- Set `fdm_observation_bridge.proprio_layout_csv` to the checkpoint's proprioception term order.
- Confirm history timing and dimensions against the checkpoint config.
- Latency checks between sensing, FDM planning, safety gate, and low-level gait.
- A hard safety supervisor independent of FDM predictions.

Until those are tuned on the robot, test waypoint and FDM-MPPI separately, and route both through the safety gate.
