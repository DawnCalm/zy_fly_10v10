from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


HISTORY_LENGTH = 4
FEATURE_COUNT = 31


def _as_history(value: np.ndarray, name: str) -> np.ndarray:
    history = np.asarray(value, dtype=np.float64)
    if history.shape[-2:] != (HISTORY_LENGTH, 3):
        raise ValueError(f"{name} 末两维必须是 (4, 3)")
    if not np.isfinite(history).all():
        raise ValueError(f"{name} 必须全部有限")
    return history


def _horizontal_frame(velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    horizontal = np.asarray(velocity, dtype=np.float64)[..., :2]
    speed = np.linalg.norm(horizontal, axis=-1, keepdims=True)
    forward = np.divide(
        horizontal,
        np.maximum(speed, 1.0e-8),
        out=np.zeros_like(horizontal),
    )
    stopped = speed[..., 0] < 1.0e-6
    forward[stopped] = (1.0, 0.0)
    lateral = np.stack((-forward[..., 1], forward[..., 0]), axis=-1)
    return forward, lateral


def _to_local(
    vector: np.ndarray, forward: np.ndarray, lateral: np.ndarray
) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return np.stack(
        (
            np.sum(value[..., :2] * forward, axis=-1),
            np.sum(value[..., :2] * lateral, axis=-1),
            value[..., 2],
        ),
        axis=-1,
    )


def _to_world(
    vector: np.ndarray, forward: np.ndarray, lateral: np.ndarray
) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    horizontal = (
        value[..., 0, None] * forward
        + value[..., 1, None] * lateral
    )
    return np.concatenate((horizontal, value[..., 2, None]), axis=-1)


def flight_response_features(
    velocity_history: np.ndarray,
    command_history: np.ndarray,
    history_dt: np.ndarray,
    command_dt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造只依赖当前和历史观测的机体系飞控响应特征。

    history 均按“最新帧在前”排列。history_dt 的两个值分别对应
    v[0]-v[1] 和 v[1]-v[2] 的时间间隔；command_dt 是当前指令到
    下一观测的时间间隔。
    """

    velocities = _as_history(velocity_history, "velocity_history")
    commands = _as_history(command_history, "command_history")
    previous_dt = np.asarray(history_dt, dtype=np.float64)
    current_dt = np.asarray(command_dt, dtype=np.float64)
    batch_shape = velocities.shape[:-2]
    if previous_dt.shape != batch_shape + (2,):
        raise ValueError("history_dt 形状必须与 history batch 匹配并以 2 结尾")
    if current_dt.shape != batch_shape:
        raise ValueError("command_dt 形状必须与 history batch 匹配")
    if (
        not np.isfinite(previous_dt).all()
        or not np.isfinite(current_dt).all()
        or np.any(previous_dt <= 0.0)
        or np.any(current_dt <= 0.0)
    ):
        raise ValueError("所有时间间隔必须是有限正数")

    forward, lateral = _horizontal_frame(velocities[..., 0, :])
    local_velocity = [
        _to_local(velocities[..., index, :], forward, lateral)
        for index in range(HISTORY_LENGTH)
    ]
    local_command = [
        _to_local(commands[..., index, :], forward, lateral)
        for index in range(HISTORY_LENGTH)
    ]
    acceleration_1 = (
        local_velocity[0] - local_velocity[1]
    ) / previous_dt[..., 0, None]
    acceleration_2 = (
        local_velocity[1] - local_velocity[2]
    ) / previous_dt[..., 1, None]
    features = np.concatenate(
        (
            local_velocity[0],
            *local_command,
            *local_velocity[1:],
            acceleration_1,
            acceleration_2,
            current_dt[..., None],
        ),
        axis=-1,
    )
    if features.shape[-1] != FEATURE_COUNT:
        raise RuntimeError("飞控响应特征维数错误")
    return features, forward, lateral


@dataclass(frozen=True)
class RidgeFlightResponseModel:
    """标准化岭回归飞控模型；输出下一周期的实际加速度。"""

    input_mean: np.ndarray
    input_scale: np.ndarray
    coefficient: np.ndarray
    output_mean: np.ndarray
    acceleration_limit: float = 8.0

    def __post_init__(self) -> None:
        input_mean = np.asarray(self.input_mean, dtype=np.float64)
        input_scale = np.asarray(self.input_scale, dtype=np.float64)
        coefficient = np.asarray(self.coefficient, dtype=np.float64)
        output_mean = np.asarray(self.output_mean, dtype=np.float64)
        if input_mean.shape != (FEATURE_COUNT,):
            raise ValueError("input_mean 维数错误")
        if input_scale.shape != (FEATURE_COUNT,):
            raise ValueError("input_scale 维数错误")
        if coefficient.shape != (FEATURE_COUNT, 3):
            raise ValueError("coefficient 维数错误")
        if output_mean.shape != (3,):
            raise ValueError("output_mean 维数错误")
        if (
            not np.isfinite(input_mean).all()
            or not np.isfinite(input_scale).all()
            or not np.isfinite(coefficient).all()
            or not np.isfinite(output_mean).all()
            or np.any(input_scale <= 0.0)
            or not np.isfinite(self.acceleration_limit)
            or self.acceleration_limit <= 0.0
        ):
            raise ValueError("飞控响应模型参数必须有限且尺度为正")

    @classmethod
    def load(cls, path: Path) -> "RidgeFlightResponseModel":
        with np.load(Path(path), allow_pickle=False) as source:
            return cls(
                input_mean=source["input_mean"],
                input_scale=source["input_scale"],
                coefficient=source["coefficient"],
                output_mean=source["output_mean"],
                acceleration_limit=float(source["acceleration_limit"]),
            )

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        target_acceleration: np.ndarray,
        ridge: float = 10.0,
        acceleration_limit: float = 8.0,
    ) -> "RidgeFlightResponseModel":
        values = np.asarray(features, dtype=np.float64)
        targets = np.asarray(target_acceleration, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise ValueError("features 必须是 N x 31")
        if targets.shape != (len(values), 3) or len(values) == 0:
            raise ValueError("target_acceleration 必须是非空 N x 3")
        if (
            not np.isfinite(values).all()
            or not np.isfinite(targets).all()
            or not np.isfinite(ridge)
            or ridge < 0.0
        ):
            raise ValueError("训练数据和 ridge 必须有限，ridge 必须非负")
        input_mean = np.mean(values, axis=0)
        input_scale = np.std(values, axis=0)
        input_scale = np.maximum(input_scale, 1.0e-6)
        output_mean = np.mean(targets, axis=0)
        normalized = (values - input_mean) / input_scale
        system = normalized.T @ normalized
        system.flat[:: FEATURE_COUNT + 1] += float(ridge)
        coefficient = np.linalg.solve(
            system, normalized.T @ (targets - output_mean)
        )
        return cls(
            input_mean=input_mean,
            input_scale=input_scale,
            coefficient=coefficient,
            output_mean=output_mean,
            acceleration_limit=float(acceleration_limit),
        )

    def predict_local_acceleration(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != FEATURE_COUNT or not np.isfinite(values).all():
            raise ValueError("features 必须以 31 结尾且全部有限")
        acceleration = (
            (values - self.input_mean) / self.input_scale
        ) @ self.coefficient + self.output_mean
        norm = np.linalg.norm(acceleration, axis=-1, keepdims=True)
        scale = np.minimum(
            1.0, self.acceleration_limit / np.maximum(norm, 1.0e-8)
        )
        return acceleration * scale

    def predict_world_acceleration(
        self,
        velocity_history: np.ndarray,
        command_history: np.ndarray,
        history_dt: np.ndarray,
        command_dt: np.ndarray,
    ) -> np.ndarray:
        features, forward, lateral = flight_response_features(
            velocity_history,
            command_history,
            history_dt,
            command_dt,
        )
        local = self.predict_local_acceleration(features)
        return _to_world(local, forward, lateral).astype(np.float32)
