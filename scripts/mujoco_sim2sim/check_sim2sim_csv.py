from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from config import Sim2SimConfig


NUMERIC_COLUMNS = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "vx_cmd",
    "vy_cmd",
    "wz_cmd",
    "ctrl_abs_max",
    "height_min",
    "height_max",
    "height_mean",
    "height_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MuJoCo sim2sim CSV logs for pose/control/height-scan health.")
    parser.add_argument("csv_path", nargs="?", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Sim2SimConfig.log_dir)
    return parser.parse_args()


def latest_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("sim2sim_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No sim2sim CSV files found in {log_dir}")
    return candidates[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return math.nan


def column_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [to_float(row, key) for row in rows]


def finite_stats(values: list[float]) -> tuple[float, float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan, math.nan
    return min(finite), max(finite), sum(finite) / len(finite)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path or latest_csv(args.log_dir)
    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    header = set(rows[0].keys())
    missing = [column for column in NUMERIC_COLUMNS if column not in header]
    print(f"[CSV] path={csv_path}")
    print(f"[CSV] rows={len(rows)}")
    if missing:
        print(f"[CSV] missing_columns={missing}")

    bad_columns: list[str] = []
    for column in NUMERIC_COLUMNS:
        if column not in header:
            continue
        values = column_values(rows, column)
        if any(not math.isfinite(value) for value in values):
            bad_columns.append(column)
    print(f"[CSV] nonfinite_columns={bad_columns}")

    last = rows[-1]
    print(
        "[CSV] final_pose="
        f"x={to_float(last, 'x'):.3f}, y={to_float(last, 'y'):.3f}, z={to_float(last, 'z'):.3f}, "
        f"roll={to_float(last, 'roll'):.3f}, pitch={to_float(last, 'pitch'):.3f}, yaw={to_float(last, 'yaw'):.3f}"
    )
    print(
        "[CSV] final_cmd="
        f"vx={to_float(last, 'vx_cmd'):.3f}, vy={to_float(last, 'vy_cmd'):.3f}, wz={to_float(last, 'wz_cmd'):.3f}"
    )

    for column in ("ctrl_abs_max", "height_min", "height_max", "height_mean", "height_std"):
        if column in header:
            min_v, max_v, mean_v = finite_stats(column_values(rows, column))
            print(f"[CSV] {column}=min {min_v:.3f}, max {max_v:.3f}, mean {mean_v:.3f}")

    if "height_std" in header:
        height_std = column_values(rows, "height_std")
        all_flat = all(math.isfinite(value) and abs(value) < 1e-6 for value in height_std)
        print(f"[CSV] height_scan_flat={all_flat}")

    if "fdm_best_risk_max" in header:
        risk_max = column_values(rows, "fdm_best_risk_max")
        min_v, max_v, mean_v = finite_stats(risk_max)
        print(f"[CSV] fdm_best_risk_max=min {min_v:.3f}, max {max_v:.3f}, mean {mean_v:.3f}")
    if "fdm_cost_collision" in header:
        min_v, max_v, mean_v = finite_stats(column_values(rows, "fdm_cost_collision"))
        print(f"[CSV] fdm_cost_collision=min {min_v:.3f}, max {max_v:.3f}, mean {mean_v:.3f}")
    if "fdm_progress_guard" in header:
        guard_count = sum(1 for value in column_values(rows, "fdm_progress_guard") if math.isfinite(value) and value > 0.5)
        print(f"[CSV] fdm_progress_guard_count={guard_count}")
    if "fdm_cost_goal_distance" in header:
        final_goal_dist = to_float(last, "fdm_cost_goal_distance")
        print(f"[CSV] final_goal_distance={final_goal_dist:.3f}")


if __name__ == "__main__":
    main()
