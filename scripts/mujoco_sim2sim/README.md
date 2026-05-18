# MuJoCo Sim2Sim Scaffold

This folder is the MuJoCo-side shell for replaying the FDM/MPPI navigation stack
outside IsaacLab.

The scaffold intentionally separates the parts we already have in this repo from
the assets/policies that are still missing.

## What Is Included

- `mujoco_g1_env.py`: a thin MuJoCo environment wrapper for loading a G1 MJCF and stepping physics.
- `low_level_controller.py`: G1 29DOF TorchScript policy wrapper matching the Lab 480-dim low-level obs.
- `height_scan.py`: flat and MuJoCo raycast height-scan providers.
- `scene_builder.py`: CSV/CLI obstacles plus MuJoCo primitive terrain generated from FDM terrain cfg presets.
- `fdm_adapter.py`: planner adapters for zero command, fixed command, goal tracking, and an Isaac-free FDM/MPPI bridge.
- `run_g1_mppi.py`: command-line entry point for validation and smoke tests.
- `check_sim2sim_csv.py`: quick health check for pose/control/height-scan CSV logs.
- `probe_fdm_model.py`: checkpoint/config probe for the saved FDM model.

## Missing Inputs

Already staged under `D:\fdm_data\mujoco_sim2sim`:

- Official Unitree G1 29DOF MuJoCo model and meshes.
- Low-level G1 policy copied as `policies\g1_policy.pt`.
- FDM checkpoint copied as `fdm_checkpoints\model_collection_round_14.pth`.

Still missing for full IsaacLab parity:

- Final terrain/risk feature parity with IsaacLab height extero observations.
- Optional low-level yaw-drift compensation if the high-level planner cannot absorb it.
- Exact reuse of IsaacLab scene-dependent cost-map/terrain-analysis costs.

## First Commands

Validate paths and report missing inputs:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --check-only
```

Run a MuJoCo zero-command smoke test after adding a G1 MJCF:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --zero-planner --steps 1000 --policy-device cpu
```

Run a fixed-command locomotion test:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --test-command 0.2 0 0 --steps 1000 --policy-device cpu
```

Run the end-to-end MuJoCo stack with a lightweight SE2 goal follower:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --planner goal --goal 5 0 0 --steps 1500 --policy-device cpu --height-scan flat
```

Try raycast exteroception plumbing:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --planner goal --goal 5 0 0 --steps 1500 --policy-device cpu --height-scan raycast
```

Run with the `plan_test` FDM planner-eval terrain family:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --planner fdm --goal 5 0 0 --steps 1500 --policy-device cpu --height-scan raycast --fdm-terrain-cfg planner_eval --fdm-population-size 128 --fdm-mppi-iterations 8
```

`--fdm-terrain-cfg planner_eval` maps `PLANNER_EVAL_CFG` object families to
MuJoCo primitives: outdoor pillar boxes/cylinders, single box/cylinder/wall,
and cross-pattern boxes. This is Isaac-free and runnable in MuJoCo; exact
Isaac terrain mesh export is separate from this scaffold.

Check the latest CSV:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/check_sim2sim_csv.py
```

Probe the saved FDM checkpoint and input tensor shapes:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/probe_fdm_model.py --device cpu
```

Run the Isaac-free FDM/MPPI bridge:

```powershell
C:\Users\Admin\IsaacLab\isaaclab.bat -p scripts/mujoco_sim2sim/run_g1_mppi.py --planner fdm --goal 5 0 0 --steps 1500 --policy-device cpu --height-scan raycast --fdm-population-size 128 --fdm-mppi-iterations 8
```

This bridge keeps the saved May12 FDM model in the loop, feeds MuJoCo
state/proprioception/history into it, and uses a local batched MPPI loop to
warm-start, sample, score, and update command sequences. It mirrors the
IsaacLab planner's FDM rollout/objective split without importing Isaac scene
objects. The bridge also includes a goal-progress guard so it keeps moving while
the robot is outside the goal tolerance instead of stopping early near the
target.

FDM runs add planner diagnostics to the CSV, including selected risk, predicted
terminal pose, MPPI value statistics, per-term cost, and whether the progress
guard overrode an over-conservative sample. `check_sim2sim_csv.py` summarizes
these columns.

## Low-Level Policy Port

The default policy path is:

```text
D:\fdm_data\mujoco_sim2sim\policies\g1_policy.pt
```

`TorchPolicyPDController` loads this TorchScript policy, builds an Isaac-style
low-level observation, converts policy output into joint-position targets with
`scale=0.25`, and applies PD torques to MuJoCo motor actuators.

The current controller follows the local Lab G1 setup:

- 29DOF joint order from `robot_cfg_g1.py`.
- 480-dim low-level obs as term-wise history: `96 x 5`.
- `sim.dt=0.005`, low-level inference every 4 physics steps.
- action scale `0.25` with default joint offset.
- MuJoCo torque motors wrapped by Lab-matched PD gains and effort limits.

## Height Scan Alignment

The default MuJoCo height scan matches the saved G1/FDM training config:

- `shape=(60, 46)`
- `resolution=0.1`
- Lab `env_sensor` pattern size `(4.5, 5.9)`
- local x range starts at `-0.5m`, so the scan is biased forward like the FDM model expects.
