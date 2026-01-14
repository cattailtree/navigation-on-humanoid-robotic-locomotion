```instructions
<!-- Short, practical guidance for AI coding agents working on this repo. -->
# Copilot instructions for contributors and AI agents

This repository implements a Forward Dynamics Model (FDM) extension for IsaacLab/IsaacSim and related navigation tooling. The guidance below is concise and actionable to make AI coding agents productive quickly.

- **Big picture**: The runtime extension lives under `exts/fdm`. Major subsystems:
  - `fdm/model/` — learned dynamics and prediction models.
  - `fdm/planner/` — trajectory optimization and planner hooks.
  - `fdm/agents/` — agent implementations that generate candidate actions.
  - `fdm/data_buffers/` and `exts/fdm/data/` — datasets, replay buffers, and transforms.
  - `fdm/runner/` and top-level `scripts/` — experiment orchestration, training, evaluation.
  - `ros/` — ROS bridge, planner node, and launch files for robot integration.

- **Start here (priority files)**n+  
  - `exts/fdm/fdm/agents/base_agent.py` — agent lifecycle (`plan`, `plan_reset`, `debug_viz`).
  - `exts/fdm/fdm/data_buffers/dataset.py` — `TrajectoryDataset` and replay utilities.
  - `exts/fdm/fdm/planner/planner.py` — planner hooks and `env_cfg` usage.
  - `exts/fdm/fdm/runner/runner.py` — orchestration and data-collection flow.
  - `scripts/train.py`, `scripts/eval.py` — example entrypoints and common flags.
  - `ros/fdm_navigation_ros/scripts/planner_node.py` — ROS integration example.

- **Common commands & workflows**
  - Install for local development (repo root): `pip install -e exts/fdm`.
  - Run a training/eval script: `python3 scripts/train.py` or `python3 scripts/eval.py` (see `argparse` flags at top of each script).
  - IsaacSim runs: source the IsaacSim environment and use its `python.sh` wrapper. Example pattern (replace <ISAAC_SIM_PATH> with your path):

    source <ISAAC_SIM_PATH>/setup_python_env.sh
    <ISAAC_SIM_PATH>/python.sh scripts/train.py --cfg <your_cfg>

  - VSCode helper task exists: `setup_python_env` (check `.vscode/tasks.json` in workspace).

- **Project conventions & patterns**
  - Config classes end with `Cfg` and map to implementation modules (e.g. `PaperFigureAgentCfg` ↔ `paper_figure_agent.py`). Prefer changing configs in `env_cfg` rather than hard-coding values.
  - `env_cfg` is the common mechanism for parameterizing planners and environments — search for usages in `planner/` and `runner/`.
  - Data pipelines use `TrajectoryDataset` and transforms in `data_buffers/` — follow these utilities for dataset compatibility.
  - isort groups in `pyproject.toml` include `ISAACLAB_THIRDPARTY` and `NAVSUITE_THIRDPARTY`. Keep import grouping consistent.

- **Integration & external dependencies**
  - Native IsaacSim packages (`omni.*`, `pxr`, `warp`) are required for simulator runs and are unavailable in plain CI. Guard imports that reference these modules to keep testable code paths.
  - ROS nodes and launch files live in `ros/`. ROS integration expects model parameters and planner interfaces — see `ros/fdm_navigation_ros/config`.
  - Large datasets live in `exts/fdm/data/`. Do not commit large binaries; prefer referencing or downloaded artifacts.

- **Pitfalls to avoid**
  - Do not import IsaacSim native modules at top-level in files executed by CI/static analysis. Wrap with try/except and provide clear error messages.
  - Avoid changing import group names without updating `pyproject.toml` isort sections.

- **When to update this file**
  - Add new top-level scripts to "Common commands".
  - Note new runtime/native requirements (IsaacSim version, new native libs).

- **Quick tasks I can do next**
  1. Produce a one-page Quickstart for running an IsaacSim experiment.
  2. Annotate the top 6 priority files with inline comments and TODOs.
  3. Run a static check (isort/flake8) against `exts/fdm` and report issues.

If you'd like any of these, reply with the number and I'll proceed.
```<!-- Short, practical guidance for AI coding agents working on this repo. -->
# Copilot instructions for contributors and AI agents

This repository implements a Forward Dynamics Model (FDM) extension for IsaacLab/IsaacSim and related navigation tooling. The notes below are concise, focused, and actionable so an AI agent can quickly be productive.

- **Big picture**: `exts/fdm` contains the extension: model learning (`model/`), planning (`planner/`), agent implementations (`agents/`), data handling (`data_buffers/`, `data/`) and orchestrators (`runner/`, `scripts/`). ROS integration is in `ros/`.

- **Start here (priority files)**
  - `exts/fdm/fdm/agents/base_agent.py` — agent lifecycle (`plan`, `plan_reset`, `debug_viz`).
  - `exts/fdm/fdm/data_buffers/dataset.py` — `TrajectoryDataset` and replay utilities.
  - `exts/fdm/fdm/planner/planner.py` — planner hooks and env-config usage.
  - `scripts/train.py`, `scripts/eval.py`, `scripts/test.py` — example invocations and flag patterns.
  - `ros/fdm_navigation_ros/scripts/planner_node.py` — example ROS usage and `model_params` handling.

- **Common commands & workflows**
  - Local dev install: `pip install -e exts/fdm` (run at repo root).
  - Run experiments: `python3 scripts/train.py` or `python3 scripts/eval.py` (check `argparse` for flags).
  - IsaacLab/IsaacSim runs require the simulator environment (source `setup_python_env.sh` and use Isaac's `python.sh` where documented).

- **Codebase conventions**
  - Config classes end with `Cfg` and pair with implementation modules (e.g. `PaperFigureAgentCfg` ↔ `paper_figure_agent.py`).
  - Most behavior is config-driven: prefer adding/updating config entries over invasive code changes.
  - isort groups in `pyproject.toml` include `ISAACLAB_THIRDPARTY` and `NAVSUITE_THIRDPARTY` — keep import groups consistent.

- **Integration & dependencies**
  - Runtime: IsaacSim/IsaacLab native packages (`omni.*`, `pxr`, `warp`) — unavailable outside the simulator.
  - ROS bridge: `ros/` provides launch files and nodes expecting model parameters and planner interfaces.
  - Data: `exts/fdm/data/` contains large datasets; do not check large binaries into Git.

- **Patterns to follow**
  - Agent composition: see `exts/fdm/fdm/agents/mixed_agent.py` for splitting planning across terms.
  - Data loaders and transforms: use utilities in `data_buffers/` to ensure compatibility with training loops.
  - Planner uses `env_cfg` for parameterization — change configs rather than hard-coded constants.

- **Pitfalls to avoid**
  - Avoid importing IsaacSim native modules in code paths executed by CI/static-checks; wrap such imports with clear fallbacks.
  - When adding dependencies that belong to IsaacLab/NAVSUITE groups, update `pyproject.toml` isort sections.

- **Files to update when repo changes**
  - Add new user-facing scripts to this file's "Common commands" section.
  - Note new runtime/native requirements in the "Integration & dependencies" section.

If you'd like, I can now (choose one):
1. produce a one-page Quickstart with copy-paste commands for IsaacSim runs; or
2. annotate the top 5 files above with inline comments; or
3. run a quick static check (flake/isort) and report issues. Which do you want next?
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
<!-- Copilot instructions for contributors and AI agents -->
# Copilot instructions (concise)

This repository implements a Forward Dynamics Model (FDM) extension and associated navigation tooling. The notes below are focused, actionable and specific to this codebase so an AI coding agent can be productive quickly.

- **Big picture**: top-level extension in `exts/fdm`.
  - `exts/fdm/fdm/model/`: learned dynamics & prediction models.
  - `exts/fdm/fdm/planner/`: trajectory optimization & planner hooks.
  - `exts/fdm/fdm/agents/`: agents that propose or evaluate action sequences.
  - `exts/fdm/fdm/data_buffers/` + `exts/fdm/data/`: datasets, replay buffers, transforms.
  - `exts/fdm/fdm/runner/` + top-level `scripts/`: orchestration, training, evaluation.
  - `ros/`: ROS bridge, planner node, launch files for robot integration.

- **Start here (high-value files)**n  
  - [exts/fdm/fdm/agents/base_agent.py](exts/fdm/fdm/agents/base_agent.py): agent lifecycle (`plan`, `plan_reset`, `debug_viz`).
  - [exts/fdm/fdm/data_buffers/dataset.py](exts/fdm/fdm/data_buffers/dataset.py): `TrajectoryDataset` and replay utilities.
  - [exts/fdm/fdm/planner/planner.py](exts/fdm/fdm/planner/planner.py): planner hooks and `env_cfg` usage.
  - [exts/fdm/fdm/runner/runner.py](exts/fdm/fdm/runner/runner.py): orchestration and data-collection flow.
  - [scripts/train.py](scripts/train.py), [scripts/eval.py](scripts/eval.py): canonical entrypoints and common `argparse` flags.
  - [ros/fdm_navigation_ros/scripts/planner_node.py](ros/fdm_navigation_ros/scripts/planner_node.py): ROS integration example and parameter handling.

- **Common commands & workflows**
  - Install for local dev (repo root): `pip install -e exts/fdm`.
  - Run experiments locally (inspect flags in scripts): `python3 scripts/train.py` or `python3 scripts/eval.py`.
  - IsaacSim runs: source IsaacSim env, then use Isaac `python.sh` wrapper. Typical pattern:

    source <ISAAC_SIM_PATH>/setup_python_env.sh
    <ISAAC_SIM_PATH>/python.sh scripts/train.py --cfg <your_cfg>

  - VSCode helper: task `setup_python_env` in `.vscode/tasks.json` (used to configure IsaacSim Python).

- **Project-specific conventions (be literal)**
  - Config classes end with `Cfg` and map to modules (e.g. `PaperFigureAgentCfg` → `paper_figure_agent.py`). Prefer updating `env_cfg` rather than hard-coding.
  - `env_cfg` is the common injection point for environment and planner parameters—search usages in `planner/` and `runner/`.
  - Data pipelines are centralized: use `TrajectoryDataset` and transforms in `data_buffers/` for compatibility.
  - `pyproject.toml` contains `isort` groups `ISAACLAB_THIRDPARTY` and `NAVSUITE_THIRDPARTY`; keep import grouping consistent.

- **Integration & dependencies**
  - Runtime requires IsaacLab/IsaacSim native packages (`omni.*`, `pxr`, `warp`) for simulation runs. These are NOT available in plain CI—guard such imports with try/except and fallbacks.
  - ROS integration lives under `ros/` (look at `ros/fdm_navigation_ros/config` and launch files). ROS nodes expect model params and planner interfaces.
  - Large datasets live in `exts/fdm/data/` — do not commit large binaries; reference or download artifacts instead.

- **Patterns and examples**
  - Agents: extend `agents/base_agent.py`; see `agents/mixed_agent.py` for composition examples.
  - Planner: use `planner/planner.py` hooks and read parameters via `env_cfg`.
  - Runner: `runner/runner.py` coordinates stepping, data collection and calls into `model/` for training/eval.

- **Pitfalls observed in codebase**
  - Avoid top-level imports of IsaacSim native modules in files used during static analysis or CI; wrap with clear error messages.
  - When adding new third-party groups, update `pyproject.toml` isort sections to preserve import grouping.

- **When to update this file**
  - Add new top-level scripts under `scripts/` to the "Common commands" section.
  - Note any added runtime native requirements (IsaacSim version, new native libs) in "Integration & dependencies".

- **If you want me to do more (choose one)**
  1. Produce a one-page Quickstart for running experiments in IsaacSim (copy-paste commands).
  2. Annotate the top priority files with inline comments and TODOs.
  3. Run a static check (`isort`/`flake8`) on `exts/fdm` and report issues.

Please tell me which option (1/2/3) you want next, or request edits to this file.
