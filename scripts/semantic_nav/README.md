# Semantic Navigation Prototype

This folder contains the first ApexNav-inspired long-horizon navigation prototype for the FDM project.

The current milestone is intentionally small:

```text
task: "go downstairs"
  -> parse as a floor-transition goal
  -> search the semantic graph for a reachable elevator lobby
  -> output semantic subgoals for the local FDM/MPPI layer
```

It does not modify FDM training, replay buffers, trajectory datasets, or existing MPPI code.

## Run

```bash
python scripts/semantic_nav/experiments/run_find_elevator.py --goal "go downstairs"
```

This now uses an ApexNav-style semantic selector followed by A*:

```text
ApexNav-style target selector -> elevator candidates -> SemanticAStarPlanner -> approach path
```

Expected behavior:

```text
selected elevator: elevator_f1
transition target: elevator_b1
subgoals:
  corridor_f1
  elevator_f1
```

To run the full abstract elevator task:

```bash
python scripts/semantic_nav/experiments/run_elevator_task.py --goal "go downstairs to target room"
```

The same planner also supports same-floor targets:

```bash
python scripts/semantic_nav/experiments/run_elevator_task.py --goal "go to room" --target room_f1
```

Expected behavior:

```text
selected elevator: elevator_f1
transition event: elevator_f1 -> elevator_b1
full semantic route:
  start_f1
  corridor_f1
  elevator_f1
  elevator_b1
  corridor_b1
  target_room_b1
execution steps:
  WALK_TO corridor_f1
  WALK_TO elevator_f1
  FLOOR_TRANSITION elevator_f1 -> elevator_b1
  WALK_TO corridor_b1
  WALK_TO target_room_b1
```

To smoke-test the high-level execution plan with an abstract waypoint follower:

```bash
python scripts/semantic_nav/experiments/run_abstract_rollout.py
```

This still does not use the real G1 gait policy. It only verifies that the semantic route can be consumed as executable `WALK_TO` and `FLOOR_TRANSITION` steps.

To run the same single-elevator task with the MuJoCo G1 robot and low-level gait policy:

```bash
python scripts/semantic_nav/experiments/run_g1_elevator_task.py --steps 3000
```

This still does not use FDM. The semantic waypoint executor emits `[vx, vy, wz]`, which is converted to the sim2sim `LowLevelCommand` and tracked by the G1 low-level gait controller.

The execution backend is now separated behind `RobotNavAdapter`:

```text
semantic execution loop
  -> RobotNavAdapter
      -> Sim2SimRobotAdapter
      -> LabRobotAdapter
      -> future RosRobotAdapter
```

This keeps the single-elevator task independent of MuJoCo. MuJoCo is only the first backend.

To run the same task in Isaac Lab:

```bash
python scripts/semantic_nav/experiments/run_lab_elevator_task.py --headless
```

The default detector is `apexnav`, which follows the target-first fallback idea from ApexNav:

```text
known target semantic node -> likely target region -> frontier fallback
```

You can still force the graph detector for debugging:

```bash
python scripts/semantic_nav/experiments/run_lab_elevator_task.py --headless --detector graph
```

To exercise the same code path that will later consume an external YOLO/GroundingDINO service, use the dependency-free client detector:

```bash
python scripts/semantic_nav/experiments/run_find_elevator.py --detector dummy_client
python scripts/semantic_nav/experiments/run_lab_elevator_task.py --headless --detector dummy_client
```

The `dummy_client` detector still reads the current semantic graph, but it does so through the external-perception interface:

```text
PerceptionClient -> PerceptionDetection -> ClientBackedSemanticDetector -> A*
```

When the real vision service is ready, keep the Isaac Lab environment clean and pass only the service URL into the main process. Prefer ApexNav's own VLM server routes when possible:

```bash
python scripts/semantic_nav/experiments/run_lab_elevator_task.py \
  --headless \
  --detector apexnav_gdino \
  --perception-endpoint http://127.0.0.1:12181/gdino
```

ApexNav's GroundingDINO/YOLOv7 services return `ObjectDetections`-style JSON:

```json
{
  "boxes": [[120, 80, 260, 240]],
  "logits": [0.86],
  "phrases": ["elevator door"]
}
```

The generic `http_client` path is still available for our own future service format, but the ApexNav-compatible clients are the preferred reuse path:

```text
apexnav_gdino  -> /gdino   -> boxes/logits/phrases
apexnav_yolov7 -> /yolov7  -> boxes/logits/phrases
```

Important boundary: these VLM clients need an image in `PerceptionRequest.image_jpeg_b64`. Until camera capture is wired into the Lab/robot adapter, use `dummy_client` for graph-only smoke tests. After camera capture is available, detections should be projected into semantic graph nodes using camera pose/depth; label-only matching is only a temporary fallback.

## LLM Task Parser

The task parser can also be moved out-of-process, matching the ApexNav pattern used for VLM services. By default, all scripts still use the original rule parser:

```bash
python scripts/semantic_nav/experiments/run_elevator_task.py --task-parser rule
```

To call a small task-parsing service, run:

```bash
python scripts/semantic_nav/experiments/run_elevator_task.py \
  --task-parser llm_http \
  --llm-endpoint http://127.0.0.1:12182/parse_task \
  --goal "take the elevator to the basement target room" \
  --target auto
```

The `llm_http` endpoint receives the goal text, current floor, start node, and semantic graph. It should return:

```json
{
  "intent": "floor_transition",
  "target_floor": "B1",
  "target_label": "target room",
  "target_node_id": "target_room_b1",
  "rationale": "The request asks for a basement target."
}
```

For local servers that implement the OpenAI chat-completions shape, use:

```bash
python scripts/semantic_nav/experiments/run_elevator_task.py \
  --task-parser openai_compatible \
  --llm-endpoint http://127.0.0.1:8000/v1 \
  --llm-model "$SEMANTIC_NAV_LLM_MODEL" \
  --goal "find the elevator" \
  --target auto
```

`--target auto` lets the LLM choose a graph node. If `--target` is a concrete node id, that explicit node wins and the LLM only supplies the task intent/floor. This keeps old smoke tests stable while allowing ApexNav-style language grounding when desired.

## ApexNav Reuse Boundary

Reuse directly where the abstraction matches:

```text
VLM server process
  - GroundingDINO route `/gdino`
  - YOLOv7 route `/yolov7`
  - ObjectDetections JSON: boxes/logits/phrases

Semantic navigation idea
  - target/object driven exploration
  - frontier fallback
  - graph search over semantic/topological nodes
```

Do not directly reuse the wheel-robot control layer:

```text
ApexNav robot controller / trajectory follower
  -> replaced by Isaac Lab G1 + NavSE2Action + future FDM local scoring
```

This uses the same semantic graph and execution loop, but the backend is `LabRobotAdapter`. It defaults to a flat Lab terrain so the first milestone tests the semantic task and NavSE2Action wiring rather than terrain difficulty. Use `--lab-terrain generator` only when we intentionally want the more complex terrain.

The default semantic building is defined in:

```text
scripts/semantic_nav/configs/single_elevator_building.json
```

All experiment entries accept:

```bash
--building-config scripts/semantic_nav/configs/single_elevator_building.json
```

## Next Steps

1. Add the external YOLO/GroundingDINO service outside the `fdm-mppi` environment.
2. Add an external LLM task parser service outside the `fdm-mppi` environment.
3. Add camera/depth observation capture from Isaac Lab or the real robot adapter.
4. Convert detections into semantic graph node updates instead of relying on fixed graph node IDs.
5. Add FDM feasibility scoring after the semantic pipeline is complete.
