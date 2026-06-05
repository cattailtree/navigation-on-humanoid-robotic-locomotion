from __future__ import annotations

import csv
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


@dataclass
class BoxObstacle:
    name: str
    x: float
    y: float
    length: float
    width: float
    height: float
    yaw: float = 0.0
    rgba: str = "0.75 0.22 0.16 1"
    kind: str = "box"

    @property
    def z(self) -> float:
        return self.height * 0.5

    @property
    def half_size(self) -> tuple[float, float, float]:
        return self.length * 0.5, self.width * 0.5, self.height * 0.5


def parse_obstacle_box(values: list[float], index: int) -> BoxObstacle:
    if len(values) not in (5, 6):
        raise ValueError("--obstacle-box expects X Y LENGTH WIDTH HEIGHT [YAW].")
    yaw = float(values[5]) if len(values) == 6 else 0.0
    return BoxObstacle(
        name=f"terrain_box_{index}",
        x=float(values[0]),
        y=float(values[1]),
        length=float(values[2]),
        width=float(values[3]),
        height=float(values[4]),
        yaw=yaw,
    )


def load_obstacle_csv(path: Path) -> list[BoxObstacle]:
    obstacles: list[BoxObstacle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"x", "y", "length", "width", "height"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Obstacle CSV needs columns: x,y,length,width,height. "
                "Optional columns: name,yaw,rgba."
            )
        for index, row in enumerate(reader):
            name = row.get("name") or f"terrain_box_{index}"
            obstacles.append(
                BoxObstacle(
                    name=name,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    length=float(row["length"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                    yaw=float(row.get("yaw") or 0.0),
                    rgba=row.get("rgba") or "0.75 0.22 0.16 1",
                )
            )
    return obstacles


def generate_scene_with_obstacles(base_xml: Path, obstacles: list[BoxObstacle], output_xml: Path) -> Path:
    base_xml = base_xml.resolve()
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    _expand_relative_includes(root, base_xml.parent)
    _relocate_relative_compiler_meshdir(root, base_xml.parent, output_xml.parent.resolve())
    _relocate_relative_includes(root, base_xml.parent, output_xml.parent.resolve())
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    for obstacle in obstacles:
        sx, sy, sz = obstacle.half_size
        if obstacle.kind == "cylinder":
            geom_type = "cylinder"
            radius = max(obstacle.length, obstacle.width) * 0.5
            size = f"{radius:.6g} {sz:.6g}"
        else:
            geom_type = "box"
            size = f"{sx:.6g} {sy:.6g} {sz:.6g}"
        attrib = {
            "name": obstacle.name,
            "type": geom_type,
            "pos": f"{obstacle.x:.6g} {obstacle.y:.6g} {obstacle.z:.6g}",
            "size": size,
            "rgba": obstacle.rgba,
            "friction": "1.0 0.005 0.0001",
        }
        if abs(obstacle.yaw) > 1e-8:
            attrib["euler"] = f"0 0 {obstacle.yaw:.6g}"
        ET.SubElement(worldbody, "geom", attrib)

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)
    return output_xml


def _relocate_relative_includes(root: ET.Element, base_dir: Path, output_dir: Path) -> None:
    for include in root.findall("include"):
        file_attr = include.attrib.get("file")
        if not file_attr:
            continue
        include_path = Path(file_attr)
        if include_path.is_absolute():
            continue
        resolved = (base_dir / include_path).resolve()
        try:
            include.attrib["file"] = Path(os.path.relpath(resolved, output_dir)).as_posix()
        except ValueError:
            include.attrib["file"] = resolved.as_posix()


def _expand_relative_includes(root: ET.Element, base_dir: Path) -> None:
    for include in list(root.findall("include")):
        file_attr = include.attrib.get("file")
        if not file_attr:
            continue
        include_path = Path(file_attr)
        if include_path.is_absolute():
            continue
        resolved_include = include_path if include_path.is_absolute() else (base_dir / include_path).resolve()
        if not resolved_include.exists():
            continue
        try:
            include_root = ET.parse(resolved_include).getroot()
        except ET.ParseError:
            continue
        _relocate_relative_compiler_meshdir(include_root, resolved_include.parent, base_dir)
        insert_at = list(root).index(include)
        root.remove(include)
        for child in list(include_root):
            root.insert(insert_at, copy.deepcopy(child))
            insert_at += 1


def _relocate_relative_compiler_meshdir(root: ET.Element, base_dir: Path, output_dir: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return
    meshdir = compiler.attrib.get("meshdir")
    if not meshdir:
        return
    compiler.attrib["meshdir"] = _relocated_path(meshdir, base_dir, output_dir)


def _relocated_path(path_text: str, base_dir: Path, output_dir: Path) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return path.as_posix()
    resolved = (base_dir / path).resolve()
    try:
        return Path(os.path.relpath(resolved, output_dir)).as_posix()
    except ValueError:
        return resolved.as_posix()


def generate_fdm_terrain_obstacles(preset: str = "planner_eval", seed: int = 0) -> list[BoxObstacle]:
    """Approximate the FDM terrain_cfg presets with MuJoCo primitives."""
    normalized = preset.lower()
    if normalized not in {
        "planner_eval",
        "planner_eval_2d",
        "planner_eval_calib",
        "planner_eval_humanoid",
        "paper_figure",
        "sparse_boxes",
        "humanoid_plan_test",
    }:
        raise ValueError(f"Unknown FDM terrain preset: {preset}")

    rng = np.random.default_rng(seed)
    if normalized == "humanoid_plan_test":
        return _generate_humanoid_plan_test()
    if normalized == "sparse_boxes":
        x_shift = float(rng.uniform(-0.08, 0.08))
        y_shift = float(rng.uniform(-0.08, 0.08))
        return [
            BoxObstacle(
                "fdm_sparse_box_left_near",
                1.9 + x_shift,
                -0.75 + y_shift,
                length=0.65,
                width=0.65,
                height=1.20,
            ),
            BoxObstacle(
                "fdm_sparse_box_right_mid",
                2.8 + x_shift,
                0.68 + y_shift,
                length=0.65,
                width=0.65,
                height=1.20,
            ),
            BoxObstacle(
                "fdm_sparse_box_center_gap",
                3.5 + x_shift,
                -0.28 + y_shift,
                length=0.55,
                width=0.55,
                height=1.20,
            ),
            BoxObstacle(
                "fdm_sparse_box_left_far",
                4.4 + x_shift,
                -0.92 + y_shift,
                length=0.65,
                width=0.65,
                height=1.20,
            ),
            BoxObstacle(
                "fdm_sparse_box_right_far",
                5.0 + x_shift,
                0.78 + y_shift,
                length=0.65,
                width=0.65,
                height=1.20,
            ),
        ]
    obstacles = _generate_planner_eval_tile(
        rng,
        humanoid_feasible=normalized in {"planner_eval_humanoid", "planner_eval_calib"},
        calibrated=normalized == "planner_eval_calib",
    )
    if normalized == "paper_figure":
        return _generate_planner_eval_outdoor(rng, difficulty=0.65)
    if normalized == "planner_eval_2d":
        for obstacle in obstacles:
            obstacle.height = max(obstacle.height, 1.0)
        return obstacles
    return obstacles


def _generate_planner_eval_tile(
    rng: np.random.Generator,
    humanoid_feasible: bool = False,
    calibrated: bool = False,
) -> list[BoxObstacle]:
    subterrain = _sample_planner_eval_subterrain(rng)
    difficulty = float(rng.random())
    if humanoid_feasible:
        difficulty = min(difficulty, 0.45)
    if subterrain == "outdoor":
        return _generate_planner_eval_outdoor(rng, difficulty, calibrated=calibrated)
    if subterrain == "single_box":
        dim_low, dim_high = (0.65, 1.05) if calibrated else (1.0, 1.8)
        dim = _lerp(dim_low, dim_high, difficulty)
        height = float(rng.uniform(0.8, 1.3) if calibrated else rng.uniform(0.5, 1.5))
        return [BoxObstacle("fdm_planner_eval_single_box", 2.5, 0.0, dim, dim, height)]
    if subterrain == "single_cylinder":
        radius_low, radius_high = (0.32, 0.50) if calibrated else (0.5, 0.9)
        radius = _lerp(radius_low, radius_high, difficulty)
        height = float(rng.uniform(0.8, 1.3) if calibrated else rng.uniform(0.5, 1.5))
        diameter = 2.0 * radius
        return [
            BoxObstacle(
                "fdm_planner_eval_single_cylinder",
                2.5,
                0.0,
                diameter,
                diameter,
                height,
                kind="cylinder",
                rgba="0.18 0.45 0.85 1",
            )
        ]
    if subterrain == "single_wall":
        wall_low, wall_high = (0.70, 1.10) if calibrated else (1.0, 1.6)
        wall_length = _lerp(wall_low, wall_high, difficulty)
        height = float(rng.uniform(0.8, 1.3) if calibrated else rng.uniform(0.5, 1.5))
        return [
            BoxObstacle(
                "fdm_planner_eval_single_wall",
                2.5,
                0.0,
                0.10,
                wall_length,
                height,
                rgba="0.55 0.55 0.58 1",
            )
        ]
    if subterrain == "box_cross_pattern":
        dim_low, dim_high = (0.40, 0.62) if calibrated else (0.5, 0.9)
        dim = _lerp(dim_low, dim_high, difficulty)
        height = float(rng.uniform(0.8, 1.3) if calibrated else rng.uniform(0.5, 1.5))
        if calibrated:
            centers = [(2.5, 0.0), (4.6, 1.8), (4.6, -1.8)]
        else:
            centers = [(2.5, 0.0), (4.5, 2.0), (0.5, 2.0), (4.5, -2.0), (0.5, -2.0)]
        return [
            BoxObstacle(f"fdm_planner_eval_cross_box_{index}", x, y, dim, dim, height)
            for index, (x, y) in enumerate(centers)
        ]
    return _generate_planner_eval_maze(rng)


def _sample_planner_eval_subterrain(rng: np.random.Generator) -> str:
    names = [
        "outdoor",
        "single_box",
        "single_cylinder",
        "single_wall",
        "box_cross_pattern",
        "maze",
    ]
    weights = np.asarray([0.45, 0.22, 0.13, 0.01, 0.07, 0.01], dtype=np.float64)
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def _generate_planner_eval_outdoor(
    rng: np.random.Generator,
    difficulty: float,
    calibrated: bool = False,
) -> list[BoxObstacle]:
    origin = np.asarray([5.0, 5.0], dtype=np.float64)
    platform_width = 1.0
    platform_clearance = 0.1
    platform_min = origin - 0.5 * platform_width
    platform_max = origin + 0.5 * platform_width
    platform_min *= 1.0 - platform_clearance
    platform_max *= 1.0 + platform_clearance

    obstacles: list[BoxObstacle] = []
    num_boxes = 1 if calibrated else int(rng.integers(2, 4))
    box_x = _lerp(0.35, 0.65 if calibrated else 0.9, difficulty)
    box_y = _lerp(0.2, 0.35 if calibrated else 0.45, difficulty)
    for index, center in enumerate(_sample_centers_outside_platform(rng, num_boxes, platform_min, platform_max)):
        yaw = float(rng.uniform(-np.deg2rad(10.0), np.deg2rad(10.0)))
        obstacles.append(
            BoxObstacle(
                f"fdm_planner_eval_outdoor_box_{index}",
                float(center[0] - origin[0]),
                float(center[1] - origin[1]),
                box_x,
                box_y,
                2.5,
                yaw=yaw,
            )
        )

    num_cylinders = 1 if calibrated else int(rng.integers(2, 4))
    radius = _lerp(0.22, 0.35 if calibrated else 0.45, difficulty)
    for index, center in enumerate(_sample_centers_outside_platform(rng, num_cylinders, platform_min, platform_max)):
        obstacles.append(
            BoxObstacle(
                f"fdm_planner_eval_outdoor_cylinder_{index}",
                float(center[0] - origin[0]),
                float(center[1] - origin[1]),
                2.0 * radius,
                2.0 * radius,
                2.5,
                kind="cylinder",
                rgba="0.18 0.45 0.85 1",
            )
        )
    return obstacles


def _sample_centers_outside_platform(
    rng: np.random.Generator,
    count: int,
    platform_min: np.ndarray,
    platform_max: np.ndarray,
) -> np.ndarray:
    centers: list[np.ndarray] = []
    while len(centers) < count:
        candidate = rng.uniform(0.0, 10.0, size=2)
        within = np.all(candidate >= platform_min) and np.all(candidate <= platform_max)
        if not within:
            centers.append(candidate)
    return np.asarray(centers, dtype=np.float64)


def _generate_planner_eval_maze(rng: np.random.Generator) -> list[BoxObstacle]:
    resolution = 2.0
    wall_width = 0.2
    wall_height = 3.0
    open_probability = 0.6
    grid_shape = (int(10.0 / resolution), int(10.0 / resolution))
    maze = rng.random(grid_shape) > open_probability
    maze[2, 2] = False
    obstacles: list[BoxObstacle] = []
    for ix, iy in np.argwhere(maze):
        x = (float(ix) + 0.5) * resolution - 5.0
        y = (float(iy) + 0.5) * resolution - 5.0
        obstacles.append(
            BoxObstacle(
                f"fdm_planner_eval_maze_wall_{ix}_{iy}",
                x,
                y,
                wall_width,
                wall_width,
                wall_height,
                rgba="0.55 0.55 0.58 1",
            )
        )
    return obstacles


def _generate_humanoid_plan_test() -> list[BoxObstacle]:
    """A deterministic, humanoid-friendly local planning test.

    The default goal is (5, 0, 0). A cylinder sits directly on that straight
    line, while both side routes keep enough clearance for the G1 footprint
    plus the 0.30 m soft obstacle margin used by the MPPI scan cost.
    """
    return [
        BoxObstacle(
            "fdm_humanoid_test_center_blocker",
            2.35,
            0.0,
            length=0.55,
            width=0.55,
            height=1.20,
            kind="cylinder",
            rgba="0.18 0.45 0.85 1",
        ),
        BoxObstacle(
            "fdm_humanoid_test_lower_context",
            3.55,
            -1.55,
            length=0.45,
            width=0.45,
            height=1.20,
        ),
        BoxObstacle(
            "fdm_humanoid_test_upper_context",
            3.85,
            1.55,
            length=0.45,
            width=0.45,
            height=1.20,
        ),
        BoxObstacle(
            "fdm_humanoid_test_far_context",
            4.85,
            -1.25,
            length=0.50,
            width=0.50,
            height=1.20,
            kind="cylinder",
            rgba="0.18 0.45 0.85 1",
        ),
    ]


def _lerp(low: float, high: float, difficulty: float) -> float:
    return float(low + difficulty * (high - low))
