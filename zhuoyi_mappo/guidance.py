from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .assignment import intercept_time


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > max_norm > 0.0:
        value *= float(max_norm) / norm
    return value


def _smoothstep(value: float) -> float:
    fraction = float(np.clip(value, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def line_of_sight_kinematics(
    interceptor_pos: np.ndarray,
    interceptor_velocity: np.ndarray,
    target_pos: np.ndarray,
    target_velocity: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """返回 LOS 角速度、闭合速度和距离。"""

    relative = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        interceptor_pos, dtype=np.float64
    )
    relative_velocity = np.asarray(
        target_velocity, dtype=np.float64
    ) - np.asarray(interceptor_velocity, dtype=np.float64)
    distance = float(np.linalg.norm(relative))
    if distance < 1.0e-6:
        return np.zeros(3, dtype=np.float32), 0.0, distance
    line_of_sight = relative / distance
    los_rate = np.cross(relative, relative_velocity) / (distance * distance)
    closing_speed = max(
        0.0, -float(np.dot(relative_velocity, line_of_sight))
    )
    return los_rate.astype(np.float32), closing_speed, distance


@dataclass(frozen=True)
class LOSRatePNResult:
    velocity: np.ndarray
    intercept_time: float
    blend: float
    los_rate: np.ndarray
    closing_speed: float
    acceleration: np.ndarray


def los_rate_pn_velocity(
    interceptor_pos: np.ndarray,
    interceptor_velocity: np.ndarray,
    target_pos: np.ndarray,
    target_velocity: np.ndarray,
    classic_velocity: np.ndarray,
    interceptor_speed: float,
    navigation_constant: float = 3.0,
    maximum_acceleration: float = 5.0,
    response_lead_seconds: float = 2.5,
    activation_time_to_go: float = 6.0,
    full_time_to_go: float = 4.0,
    minimum_closing_speed: float = 1.0,
    regularization_distance: float = 5.0,
    close_fade_distance: float = 5.0,
    close_cutoff_distance: float = 1.0,
    los_rate_override: Optional[np.ndarray] = None,
) -> LOSRatePNResult:
    """在 classic 闭合速度上叠加受限的 LOS-rate PN 横向响应。"""

    positive = (
        interceptor_speed,
        navigation_constant,
        maximum_acceleration,
        response_lead_seconds,
        activation_time_to_go,
        full_time_to_go,
        minimum_closing_speed,
        regularization_distance,
        close_fade_distance,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("LOS PN 参数必须是有限正数")
    if activation_time_to_go <= full_time_to_go:
        raise ValueError("LOS PN 开始介入时间必须大于完全介入时间")
    if (
        not np.isfinite(close_cutoff_distance)
        or close_cutoff_distance < 0.0
        or close_fade_distance <= close_cutoff_distance
    ):
        raise ValueError("LOS PN 近距退出参数无效")

    interceptor = np.asarray(interceptor_pos, dtype=np.float64)
    interceptor_vel = np.asarray(interceptor_velocity, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    target_vel = np.asarray(target_velocity, dtype=np.float64)
    classic = _clip_norm(classic_velocity, interceptor_speed)
    kinematic_los_rate, closing_speed, distance = line_of_sight_kinematics(
        interceptor, interceptor_vel, target, target_vel
    )
    if los_rate_override is None:
        los_rate = np.asarray(kinematic_los_rate, dtype=np.float64)
    else:
        los_rate = np.asarray(los_rate_override, dtype=np.float64)
        if los_rate.shape != (3,) or not np.isfinite(los_rate).all():
            raise ValueError("LOS-rate override 必须是有限三维向量")
    raw_intercept_time = intercept_time(
        interceptor, target, target_vel, interceptor_speed
    )
    if distance < 1.0e-6:
        return LOSRatePNResult(
            classic.astype(np.float32),
            raw_intercept_time,
            0.0,
            los_rate,
            closing_speed,
            np.zeros(3, dtype=np.float32),
        )

    relative = target - interceptor
    relative_velocity = target_vel - interceptor_vel
    line_of_sight = relative / distance
    closing_time = distance / max(closing_speed, minimum_closing_speed)
    time_blend = _smoothstep(
        (activation_time_to_go - closing_time)
        / (activation_time_to_go - full_time_to_go)
    )
    close_blend = _smoothstep(
        (distance - close_cutoff_distance)
        / (close_fade_distance - close_cutoff_distance)
    )
    closing_blend = _smoothstep(closing_speed / minimum_closing_speed)
    blend = time_blend * close_blend * closing_blend

    if los_rate_override is None:
        control_los_rate = np.cross(
            relative, relative_velocity
        ) / max(distance * distance, regularization_distance**2)
    else:
        control_los_rate = los_rate * (
            distance * distance
            / max(distance * distance, regularization_distance**2)
        )
    acceleration = (
        navigation_constant
        * closing_speed
        * np.cross(control_los_rate, line_of_sight)
    )
    acceleration = _clip_norm(acceleration, maximum_acceleration)

    # classic 保留 LOS 方向闭合；横向速度差按一阶响应转换为 PN 加速度。
    classic_delta = classic - interceptor_vel
    longitudinal_delta = (
        float(np.dot(classic_delta, line_of_sight)) * line_of_sight
    )
    candidate = (
        interceptor_vel
        + longitudinal_delta
        + acceleration * response_lead_seconds
    )
    candidate = _clip_norm(candidate, interceptor_speed)
    command = _clip_norm(
        (1.0 - blend) * classic + blend * candidate,
        interceptor_speed,
    )
    return LOSRatePNResult(
        command.astype(np.float32),
        raw_intercept_time,
        blend,
        los_rate,
        closing_speed,
        (blend * acceleration).astype(np.float32),
    )
