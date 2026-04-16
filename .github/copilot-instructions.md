```instructions
# Copilot instructions for contributors and AI agents

This repository implements a Forward Dynamics Model (FDM) extension for IsaacLab/IsaacSim together with a suite of navigation tools. The notes below are concise and focused to help an AI agent get productive fast.

## Big picture architecture 🧠

- Core extension lives under `exts/fdm`. The Python package `fdm` is the runtime.
- Key subsystems and their roles:
  - **model/** – learned forward-dynamics and multi-step predictors.
  - **planner/** – trajectory optimisation, cost functions and planner hooks.
  - **agents/** – high-level agent definitions (`plan`, `plan_reset`, `debug_viz`, etc.).
  - **data_buffers/** + top-level `data/` – datasets, replay buffers, transforms and downloaded terrain/robot data.
  - **runner/** – training/evaluation orchestration, environment stepping, data collection.
  - **env_cfg/** – datablock holding all environment and planner configuration.
  - **scripts/** (at workspace root) – entrypoints like `train.py`, `eval.py`, `plan_test.py`.
  - **ros/** – ROS bridge and planner node used for deploying on real hardware.
  - **nav-suite/** (sibling repo) contains unrelated navigation tasks; occasionally referenced for shared utilities.

The runtime is highly config-driven; `runner` starts a gym-like environment, calls an agent to query the `planner` which in turn uses the `model` to score candidate trajectories. Data flows from the simulator through `runner` into `data_buffers` and the model, while action proposals originate in `agents`.

## Developer workflows 🚀

1. **Setup**
   ```bash
   cd /path/to/repo
   pip install -e exts/fdm            # local editable install
   ```
   For simulation runs, source the IsaacSim environment and use its `python.sh` wrapper:
   ```bash
   source <ISAAC_SIM_PATH>/setup_python_env.sh
   <ISAAC_SIM_PATH>/python.sh scripts/train.py --cfg <your_cfg>
   ```
   A VSCode task `setup_python_env` in `.vscode/tasks.json` automates this.

2. **Running experiments**
   - `scripts/train.py` & `scripts/eval.py`: inspect top for `argparse` flags.
   - The training script no longer automatically evaluates a separate "baseline" model; add `--include_baseline` if you need to run that comparison at the end of a training job or during evaluation.
   - `scripts/plan_test.py` is used for planner diagnostics, metric runs, and figure generation.
   - Use `--headless` for CI/RTX-free training; `--num_envs` scales simulation parallelism.

3. **Debugging / plotting**
   - Planner scripts often modify `env_cfg` programmatically (see `plan_test.py` for examples of switching terrain generators, commands or planner cost terms).
   - `add_env_cameras` in `plan_test.py` adds omniverse cameras for high-resolution snapshots.

4. **ROS deployment**
   - `ros/fdm_navigation_ros/scripts/planner_node.py` shows how `env_cfg` is serialized into a ROS message and later re-used to rebuild the height-map sensor etc.

5. **Testing & static checks**
   - Pytest is used (see `nav-suite/exts/nav_tasks/test` for examples). Run `pytest` at repo root.
   - Code style: `pyproject.toml` defines `isort` groups (`ISAACLAB_THIRDPARTY`, `NAVSUITE_THIRDPARTY`) and flake8 configuration. Run `isort -rc .` / `flake8 exts/fdm`.

## Project conventions & patterns 📌

- Config classes end in `Cfg` and map one-to-one to implementation modules (e.g. `PaperFigureAgentCfg` → `paper_figure_agent.py`).
- `env_cfg` is the central configuration object; it is passed everywhere and mutated by helper utilities (`utils/args_cli_utils` functions such as `planner_cfg_init`, `cfg_modifier_pre_init`, `env_modifier_post_init`).
- Use `env_cfg` rather than hard-coding constants in code; behaviour is switched via configuration (terrain generators, sensors, commands, events, episode length, etc.).
- Data loaders and transforms centralised in `data_buffers/dataset/trajectory_dataset.py` (`TrajectoryDataset`).
- Agent composition example: `agents/mixed_agent.py`.
- Avoid top-level imports of IsaacSim native modules (`omni.*`, `pxr`, `warp`) in code paths executed by CI; wrap with try/except.

## Integration & dependencies 🔗

- **IsaacSim/IsaacLab native packages** are required at runtime but not available in CI. Guard imports and provide informative errors.
- **ROS**: planner node, launch files, and dynamic reconfigure exist under `ros/`. Expect serialized `env_cfg` values (sensor pattern, offset, max_distance, etc).
- **Data**: large dataset directories under `exts/fdm/data/`. Do not commit large binaries.
- **Nav-suite**: sibling workspace used for other navigation research; occasionally referenced but not required for FDM.
- **Unitree model**: separate folder with robot-specific parameters.

## Cross-component communication

- `env_cfg` is the glue – the same object is manipulated in the planner scripts, baked into checkpoints, and read by ROS nodes.
- `runner` serialises model checkpoints; `planner` and `agents` load them during evaluation.
- `scripts/*` use `utils/args_cli_utils` to keep flag logic consistent.

## Pitfalls to avoid ⚠️

- Changing import group names without editing `pyproject.toml` breaks `isort`.
- Individual files executed in CI must not import IsaacSim natives at top level.
- Avoid committing large terrain data or model checkpoints.

## Maintenance notes 🛠

- Add new top-level scripts to the "Common commands" section.
- Update this document when new runtime/native dependencies appear or when workspace structure changes.

---

If you'd like help with anything specific (quickstart, annotating files, static checks), just ask for a number (1-3) as before.
