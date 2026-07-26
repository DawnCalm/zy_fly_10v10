from __future__ import annotations

from typing import Tuple

import numpy as np

from .assignment import intercept_time


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > max_norm > 0.0:
        value = value * (float(max_norm) / norm)
    return value


def apn_zem_velocity(
    interceptor_pos: np.ndarray,
    interceptor_velocity: np.ndarray,
    target_pos: np.ndarray,
    target_velocity: np.ndarray,
    target_acceleration: np.ndarray,
    classic_velocity: np.ndarray,
    interceptor_speed: float,
    navigation_constant: float = 3.0,
    maximum_acceleration: float = 5.0,
    maximum_time_to_go: float = 5.0,
    response_lead_seconds: float = 2.5,
    activation_distance: float = 300.0,
    full_distance: float = 80.0,
) -> Tuple[np.ndarray, float, float]:
    """由 3D APN/ZEM 加速度构造速度设定点。

    远距离保持经典提前量；进入 activation_distance 后逐步混合，
    full_distance 内完全使用 APN 候选。目标加速度只使用垂直 LOS 的
    分量，避免沿视线加速度引起无意义的横向指令。
    """

    interceptor = np.asarray(interceptor_pos, dtype=np.float64)
    interceptor_vel = np.asarray(interceptor_velocity, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    target_vel = np.asarray(target_velocity, dtype=np.float64)
    target_accel = np.asarray(target_acceleration, dtype=np.float64)
    classic = np.asarray(classic_velocity, dtype=np.float64)
    relative = target - interceptor
    distance = float(np.linalg.norm(relative))
    if distance < 1.0e-6:
        return np.zeros(3, dtype=np.float32), 0.0, 1.0
    line_of_sight = relative / distance
    relative_velocity = target_vel - interceptor_vel
    raw_time_to_go = intercept_time(
        interceptor, target, target_vel, interceptor_speed
    )
    time_to_go = float(
        np.clip(raw_time_to_go, 0.25, maximum_time_to_go)
    )

    zero_effort_miss = (
        relative
        + relative_velocity * time_to_go
        + 0.5 * target_accel * time_to_go * time_to_go
    )
    perpendicular_miss = zero_effort_miss - (
        np.dot(zero_effort_miss, line_of_sight) * line_of_sight
    )
    acceleration = (
        float(navigation_constant)
        * perpendicular_miss
        / max(time_to_go * time_to_go, 1.0e-6)
    )
    acceleration = _clip_norm(acceleration, maximum_acceleration)

    candidate = (
        interceptor_vel + acceleration * float(response_lead_seconds)
    )
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm < 1.0:
        candidate = classic.copy()
    else:
        candidate *= float(interceptor_speed) / candidate_norm

    if activation_distance <= full_distance:
        blend = float(distance <= full_distance)
    else:
        blend = float(
            np.clip(
                (activation_distance - distance)
                / (activation_distance - full_distance),
                0.0,
                1.0,
            )
        )
    command = (1.0 - blend) * classic + blend * candidate
    command = _clip_norm(command, interceptor_speed)
    return command.astype(np.float32), raw_time_to_go, blend
