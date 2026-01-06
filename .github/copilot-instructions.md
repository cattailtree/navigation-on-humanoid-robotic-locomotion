<!-- Copied/merged guidance tailored for AI coding agents working on this repository -->
# Copilot instructions for contributors and AI agents

This project is a research-oriented Forward Dynamics Model (FDM) codebase packaged as an IsaacLab/IsaacSim extension. The notes below are a concentrated, actionable set of tips to help an AI coding agent be immediately productive.

- **Big picture**: The `exts/fdm` extension implements a learned forward dynamics predictor and local planner. Key runtime integration points:
  - The extension package: [exts/fdm](exts/fdm) and its python module `fdm`.
  - ROS bridge and planner node: [ros](ros) contains ROS launch files and integration notes.
  - High-level runner and experiments: top-level `scripts/` contains `train.py`, `eval.py` and other orchestration scripts.

- **Primary directories to inspect first**:
  - [exts/fdm/fdm](exts/fdm/fdm): core implementation (agents, model, planner, runner, env_cfg, sensors, utils).
  - [exts/fdm/docs/README.md](exts/fdm/docs/README.md): extension-specific overview and short module map.
  - [scripts](scripts): training/eval/utility entrypoints used in developer workflows.
  - [ros](ros): ROS-specific launch files and guidance for robot integration.

- **Architecture notes (how pieces connect)**:
  - `agents/` generate action sequences or baseline controllers consumed by the `planner/` and `runner/`.
  - `data_buffers/` and `data/*` hold trajectory data used by `runner` and training loops in `model/`.
  - `runner/` orchestrates environment steps, data collection, and calls into `model/` for learning/evaluation.
  - The extension is intended to run inside IsaacLab/IsaacSim or with ROS middleware depending on the experiment.

- **Developer workflows / useful commands**:
  - Install the extension in editable mode (local dev):

    pip install -e exts/fdm

  - Typical experiment entrypoints (inspect flags in the scripts):

    python3 scripts/train.py
    python3 scripts/eval.py

  - ROS planner (when using ROS integration): check `ros/planner.launch` and `ros/record.launch`.

- **Environment & dependencies**:
  - Python >= 3.10 (see `exts/fdm/setup.py` and `pyproject.toml`).
  - The extension expects IsaacLab/IsaacSim and related packages (see `exts/fdm/config/extension.toml` and `pyproject.toml` isort groups referencing `omni.isaac.*`).
  - Key Python deps listed in `exts/fdm/setup.py` (e.g. `pypose`, `torchmetrics`, `kornia`).

- **Project-specific conventions**:
  - Import ordering / linting: `pyproject.toml` customises `isort` with extra sections `ISAACLAB_THIRDPARTY` and `NAVSUITE_THIRDPARTY`. Follow that grouping when adding imports.
  - Type checking: `pyright` config in `pyproject.toml` uses `reportMissingImports = "none"` — third-party/CI environments may not have all native bindings; handle optional imports defensively.
  - Packaging: extension metadata lives in `exts/fdm/config/extension.toml` (version, repository, required isaaclab modules).

- **Patterns & examples to mimic**:
  - Agent base class pattern: look for `agents/base_agent.py` (abstract base class, extend for new agents).
  - Data buffers: `data_buffers/dataset` provides `TrajectoryDataset` and replay buffer utilities — use these for data-loading consistency.
  - Planner usage: `planner/planner.py` (trajectory optimization hooks) communicates with `env_cfg` for environment parameters — prefer using config-driven parameters rather than hard-coded constants.

- **Integration pitfalls to watch for**:
  - IsaacSim-specific imports (omni.*, pxr, warp) will fail outside that environment. Add guarded imports and informative error messages when writing code that may be run in a CI or headless environment.
  - Large binary data under `exts/fdm/data` should not be committed; prefer referencing or downloading in setup scripts.

- **What to update in this file when you change the repo**:
  - Add any new top-level scripts under `scripts/` to the “Developer workflows” list.
  - If the extension adds runtime requirements (IsaacSim version, new native libs), note them under “Environment & dependencies”.

- If anything here is unclear or you want more detail (example CLI flags, preferred test commands, or more file-level examples), tell me which area to expand.
