from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class TargetTrajectory:
    source: str
    target_id: int
    timestamps: np.ndarray
    positions: np.ndarray

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def interpolate(self, query_time: float) -> np.ndarray:
        if not len(self.timestamps):
            raise ValueError("空轨迹不能插值")
        value = float(query_time)
        if value < self.timestamps[0] or value > self.timestamps[-1]:
            raise ValueError("query_time 超出轨迹范围")
        return np.array(
            [
                np.interp(value, self.timestamps, self.positions[:, axis])
                for axis in range(3)
            ],
            dtype=np.float32,
        )


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} 不是有效 JSON"
                ) from exc
    return rows


def load_ros_target_trajectories(
    paths: Iterable[Path],
    maximum_target_age: float = 0.25,
    minimum_samples: int = 8,
) -> List[TargetTrajectory]:
    """从控制器 JSONL 中提取仍由新鲜雷达观测支持的目标轨迹。"""

    trajectories: List[TargetTrajectory] = []
    for path_value in paths:
        path = Path(path_value)
        rows = _read_jsonl(path)
        if not rows:
            continue
        count = len(rows[0].get("target_pos", []))
        samples = [[] for _ in range(count)]
        initial_time = float(rows[0]["wall_time"])
        for row in rows:
            positions = np.asarray(row.get("target_pos"), dtype=np.float64)
            active = np.asarray(
                row.get("target_active", np.ones(count)), dtype=bool
            )
            retired = np.asarray(
                row.get("target_retired", np.zeros(count)), dtype=bool
            )
            age = np.asarray(
                row.get("target_age", np.zeros(count)), dtype=np.float64
            )
            if positions.shape != (count, 3):
                continue
            timestamp = float(row["wall_time"]) - initial_time
            for target_id in range(count):
                fresh = (
                    active[target_id]
                    and not retired[target_id]
                    and age[target_id] <= maximum_target_age
                    and np.isfinite(positions[target_id]).all()
                )
                if fresh:
                    samples[target_id].append(
                        (timestamp, positions[target_id].copy())
                    )
        for target_id, target_samples in enumerate(samples):
            if len(target_samples) < minimum_samples:
                continue
            timestamps = np.asarray(
                [sample[0] for sample in target_samples], dtype=np.float64
            )
            positions = np.asarray(
                [sample[1] for sample in target_samples], dtype=np.float32
            )
            increasing = np.concatenate(
                ([True], np.diff(timestamps) > 1.0e-6)
            )
            timestamps = timestamps[increasing]
            positions = positions[increasing]
            if len(timestamps) >= minimum_samples:
                trajectories.append(
                    TargetTrajectory(
                        source=str(path),
                        target_id=target_id,
                        timestamps=timestamps,
                        positions=positions,
                    )
                )
    return trajectories


def instantaneous_speed(
    trajectory: TargetTrajectory, sample_index: int
) -> float:
    index = int(sample_index)
    if index <= 0 or index >= len(trajectory.timestamps):
        return 0.0
    dt = float(
        trajectory.timestamps[index] - trajectory.timestamps[index - 1]
    )
    if dt <= 1.0e-6:
        return 0.0
    return float(
        np.linalg.norm(
            trajectory.positions[index] - trajectory.positions[index - 1]
        )
        / dt
    )
