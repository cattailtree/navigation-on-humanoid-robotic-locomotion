# ApexNav Bridge

This folder describes how we reuse ApexNav without moving its heavy vision stack into `fdm-mppi`.

## Boundary

Run ApexNav VLMs in a separate environment:

```text
ApexNav environment
  GroundingDINO server -> http://127.0.0.1:12181/gdino
  YOLOv7 server        -> http://127.0.0.1:12184/yolov7
```

Keep humanoid navigation in the existing environment:

```text
fdm-mppi / Isaac Lab
  semantic graph -> A* -> NavSE2Action -> G1 gait
```

## Local Weights

The local weights are currently:

```text
D:/groundingdino_swint_ogc.pth
D:/yolov7-e6e.pt
```

ApexNav's server code expects:

```text
<ApexNav>/data/groundingdino_swint_ogc.pth
<ApexNav>/data/yolov7-e6e.pt
```

So the ApexNav workspace should either copy or link the two files into its `data/` directory.

## Start Servers

The full ApexNav environment is Linux/Habitat/ROS oriented. On Windows, start with the VLM-only environment instead:

```powershell
scripts/semantic_nav/apexnav_bridge/setup_vlm_env_windows.ps1
conda activate apexnav-vlm
```

Then start GroundingDINO:

```powershell
scripts/semantic_nav/apexnav_bridge/start_gdino_server.ps1
```

In another terminal:

```powershell
scripts/semantic_nav/apexnav_bridge/start_yolov7_server.ps1
```

GroundingDINO is the first detector to wire into the elevator task because it accepts text prompts such as:

```text
elevator door . elevator sign . lift . doorway . corridor .
```

YOLOv7-e6e is useful for COCO-style objects unless it is fine-tuned for elevator classes.

## Smoke Test

From the `fdm-mppi` environment, after the VLM server is running:

```bash
python scripts/semantic_nav/experiments/run_apexnav_vlm_check.py \
  --backend gdino \
  --image path/to/test_image.jpg
```

Expected output is a list of `label/score/bbox/source` detections. The Isaac Lab task should only switch from `dummy_client` to `apexnav_gdino` after this check works and camera capture is wired into `PerceptionRequest.image_jpeg_b64`.
