# ApexNav-Inspired Long-Horizon FDM Navigation Plan

## Goal

We want to extend the current FDM navigation project from local obstacle avoidance to goal-directed long-horizon humanoid navigation. The target task is not low-level elevator operation. Instead, the system should understand tasks such as "go downstairs" as semantic/topological navigation problems:

```text
current floor -> elevator lobby -> floor transition -> target floor -> target area
```

The core idea is to use an ApexNav-style semantic exploration and subgoal planning layer as the high-level planner, while preserving our FDM and MPPI stack as the humanoid local feasibility and execution layer.

## What We Should Learn From ApexNav

ApexNav is useful to us mainly as a long-horizon semantic navigation framework, not as a dynamics or control solution. Its robot platform and local motion assumptions are much simpler than humanoid locomotion, so we should not directly copy its local planner as our main navigation method.

The useful ideas are:

1. Target-centric semantic mapping.
2. Goal-directed exploration instead of pure geometric exploration.
3. A fallback chain from known target, to likely target region, to frontier exploration.
4. A modular ROS-style separation between mapping, planning, path search, and execution.
5. Explicit evaluation of long-horizon object-goal navigation tasks.

Reference repository:

- https://github.com/Robotics-STAR-Lab/ApexNav

## ApexNav Modules To Study

Based on the repository structure, the most relevant parts are:

```text
src/planner/exploration_manager
src/planner/plan_env
src/planner/path_searching
src/planner/bspline
```

### exploration_manager

This is the most important module for us. It appears to organize the high-level decision process:

```text
target object available
  -> navigate to target

possible target / over-depth object available
  -> navigate to likely semantic region

otherwise
  -> select frontier for exploration
```

We should study its decision structure, not copy it line by line. Our version should replace wheel-robot reachability assumptions with FDM-based humanoid feasibility.

### plan_env

This module is useful for understanding how ApexNav stores and queries navigation-relevant environment information:

```text
occupancy / ESDF
object map
value map
frontier map
```

For our project, this should inspire a simpler semantic/topological map first:

```text
floor graph
room graph
corridor graph
elevator node
frontier node
semantic object node
```

### path_searching

ApexNav uses conventional path search as part of its navigation stack. We can reuse the concept but should not rely on it as the final local feasibility criterion, because 2D path validity does not imply humanoid dynamic feasibility.

For us, A* should produce candidate routes or subgoals. FDM should then score whether the humanoid can actually execute the local segment.

### bspline / trajectory generation

This is less central for our current goal. It can be used as a reference for smoothing geometric paths, but we should not make it the core of the humanoid navigation system.

## What Can Be Directly Taken

We do not treat the repository license difference as the main blocker here. The more important question is engineering fit: whether a module is tightly coupled to ApexNav's ROS/C++ stack, wheel-robot assumptions, and third-party perception pipeline.

Things we can directly take at the design level:

1. The high-level module split:

```text
mapping
semantic memory
frontier selection
path search
local execution
```

2. The target-first fallback logic:

```text
known target -> likely target -> semantic frontier -> geometric frontier
```

3. The idea of maintaining object/semantic confidence over time.

4. The evaluation style for long-horizon object-goal navigation.

5. The ROS-style interface separation between planner and local executor.

Things we should not directly take in the first version:

1. Large C++ source files.
2. ROS package internals that introduce heavy dependency coupling.
3. Wheel-robot local planner assumptions.
4. Platform-specific launch/config files.
5. Perception-model launch scripts that are specific to ApexNav's sensor and runtime setup.

Things that can be copied or ported more aggressively if needed:

1. Data structures for frontier/object/value maps, after simplifying the dependencies.
2. Frontier scoring logic.
3. Target-first decision flow in `exploration_manager`.
4. Path-search utility logic, if it is easier than rewriting.
5. Config organization and experiment entry structure.

## Proposed Local Project Structure

We should follow the same spirit as `scripts/mujoco_sim2sim`: build a separate research folder that calls into FDM through adapters.

Recommended first location:

```text
scripts/semantic_nav/
```

Proposed structure:

```text
scripts/semantic_nav/
  configs/
    building_2f.yaml
    elevator_task.yaml

  maps/
    semantic_graph.py
    topo_graph.py
    floor_graph.py

  planners/
    goal_parser.py
    subgoal_planner.py
    frontier_planner.py
    elevator_planner.py

  fdm_bridge/
    fdm_model_adapter.py
    fdm_feasibility_scorer.py
    mppi_executor_adapter.py

  envs/
    abstract_building_env.py
    elevator_transition_env.py

  baselines/
    astar_baseline.py
    geometry_only_baseline.py
    apexnav_style_baseline.py

  experiments/
    run_oracle_graph.py
    run_semantic_exploration.py
    run_elevator_task.py

  utils/
    metrics.py
    visualization.py
```

This folder should not modify the existing FDM training pipeline.

## System Architecture

The final system should look like this:

```text
Language / task goal
        |
        v
Goal parser
        |
        v
Semantic-topological map
        |
        v
Subgoal planner
        |
        v
FDM feasibility scorer
        |
        v
Local FDM-MPPI executor
        |
        v
Humanoid locomotion policy
```

The key difference from ApexNav is this part:

```text
candidate subgoal/path
        |
        v
FDM predicts:
  - future pose
  - collision risk
  - stop/failure risk
  - tracking deviation
        |
        v
humanoid feasibility cost
```

So our planner does not only ask:

```text
Is this path geometrically short?
```

It asks:

```text
Can this humanoid actually walk this local segment?
```

## Cost Function

ApexNav-style high-level planning can be adapted into the following cost:

```text
total_cost =
    distance_cost
  + semantic_goal_cost
  + exploration_value_cost
  + humanoid_feasibility_cost
```

The humanoid feasibility term should come from FDM:

```text
humanoid_feasibility_cost =
    w_collision * predicted_collision
  + w_stop * predicted_stop
  + w_tracking * predicted_tracking_error
  + w_terrain * terrain_difficulty
```

This is the main technical bridge between ApexNav and our project.

## Elevator Task Abstraction

We should not implement low-level elevator manipulation. Instead, the elevator should be modeled as a semantic floor-transition node:

```text
elevator_lobby_floor_1
        |
        v
elevator_transition_edge
        |
        v
elevator_lobby_floor_0
```

The local navigation problem is only:

```text
reach elevator lobby
```

Once the robot reaches the elevator transition region, the environment can trigger:

```text
floor_id = target_floor
robot_pose = elevator_exit_pose_on_target_floor
```

This allows us to study target-directed long-horizon navigation without solving elevator button pressing, door interaction, or cabin control.

## Development Stages

### Stage 1: Oracle Semantic Graph

Build a hand-authored semantic graph:

```text
floor_1:
  start -> corridor -> elevator_lobby

floor_b1:
  elevator_lobby -> target_room

transition:
  elevator_lobby_floor_1 -> elevator_lobby_floor_b1
```

This stage verifies:

1. The goal parser understands floor-transition tasks.
2. The subgoal planner can select the elevator node.
3. The executor can run local navigation segment by segment.
4. FDM can improve local path selection compared with pure geometry.

This stage is also a clean ablation setting, because the semantic graph is given.

### Stage 2: FDM Feasibility Scoring

Implement `fdm_feasibility_scorer.py`.

Inputs:

```text
current robot state
local height scan / map crop
candidate subgoal
candidate local path or action sequence
```

Outputs:

```text
collision risk
stop risk
tracking deviation
overall feasibility score
```

This scorer should call the existing FDM model through an adapter. It should not duplicate model code.

### Stage 3: Local Execution Bridge

Implement `mppi_executor_adapter.py`.

This module should translate a semantic subgoal into the current local planner interface:

```text
subgoal pose
  -> local FDM-MPPI objective
  -> action command
  -> locomotion policy
```

The first version can be simulation-only. ROS integration can come later.

### Stage 4: ApexNav-Style Semantic Exploration

Add a frontier planner inspired by ApexNav:

```text
if target object is known:
    navigate to target
elif target semantic region is likely:
    navigate to likely region
else:
    explore frontier
```

Our frontier score should include FDM:

```text
frontier_score =
    semantic_confidence
  + information_gain
  - distance_cost
  - humanoid_feasibility_cost
```

This is where we move from oracle graph to online semantic navigation.

### Stage 5: Long-Horizon Evaluation

Evaluate the system on:

1. Same-floor object navigation.
2. Downstairs object navigation with oracle elevator.
3. Unknown target floor navigation.
4. Multiple elevator candidates.
5. Geometry-valid but humanoid-risky local route.
6. Uneven terrain near doors, corridors, and elevator lobbies.

## Baselines And Ablations

We should compare:

```text
A* only
Geometry-only MPPI
ApexNav-style high-level planner without FDM
Ours without semantic exploration
Ours without FDM feasibility
Full method
```

Important ablation:

```text
Oracle semantic graph + no FDM
vs
Oracle semantic graph + FDM feasibility
```

This proves that even with perfect high-level semantics, humanoid navigation still needs dynamics-aware feasibility prediction.

## Metrics

Recommended metrics:

1. Task success rate.
2. SPL or path efficiency.
3. Floor-transition success rate.
4. Collision rate.
5. Fall rate.
6. Timeout rate.
7. Average number of subgoals.
8. Replanning count.
9. FDM rejected-route count.
10. Success rate on geometry-valid but humanoid-risky routes.

## Integration Rules

To keep the current project stable:

1. Do not modify FDM training code for the first prototype.
2. Do not modify replay buffer or trajectory dataset for semantic navigation.
3. Keep ApexNav-inspired logic in `scripts/semantic_nav`.
4. Use adapters to call existing FDM/MPPI code.
5. Keep ROS integration as a later stage.
6. Directly port ApexNav code only when the dependency boundary is clear and the module's assumptions match our humanoid setting.

## Immediate TODO

1. Create `scripts/semantic_nav`.
2. Implement a minimal semantic graph data structure.
3. Implement a simple goal parser for:

```text
"go to target on current floor"
"go downstairs"
"go to target downstairs"
```

4. Implement oracle elevator transition.
5. Implement geometry-only baseline.
6. Add FDM feasibility adapter.
7. Compare geometry-only route selection against FDM-scored route selection.
8. Add ApexNav-style frontier selection after the oracle graph works.

## Expected Research Claim

The final claim should be:

```text
We extend target-directed semantic navigation to humanoid robots by combining
ApexNav-style long-horizon semantic planning with FDM-based local feasibility
prediction. Unlike wheel-robot navigation systems that treat geometric
traversability as sufficient, our method evaluates whether a candidate local
route can actually be executed by a humanoid locomotion policy.
```
