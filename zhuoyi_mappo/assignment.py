from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def intercept_time(
    interceptor_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    interceptor_speed: float,
) -> float:
    """求解恒速拦截时间 ||r + v*t|| = interceptor_speed*t。

    无正实根时返回一个有限的追赶时间近似，便于匈牙利算法继续分配。
    """

    relative = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        interceptor_pos, dtype=np.float64
    )
    velocity = np.asarray(target_vel, dtype=np.float64)
    speed = max(float(interceptor_speed), 1.0e-6)

    a = float(np.dot(velocity, velocity) - speed * speed)
    b = float(2.0 * np.dot(relative, velocity))
    c = float(np.dot(relative, relative))
    if c < 1.0e-12:
        return 0.0

    roots = []
    if abs(a) < 1.0e-9:
        if abs(b) > 1.0e-9:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_disc = float(np.sqrt(discriminant))
            roots.extend(((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)))

    positive = [root for root in roots if np.isfinite(root) and root >= 0.0]
    if positive:
        return float(min(positive))

    # 目标在当前恒速假设下不可严格截获时，仍给出连续且有限的代价。
    closing_speed = max(speed - float(np.linalg.norm(velocity)), speed * 0.1)
    return float(np.sqrt(c) / closing_speed)


def lead_velocity(
    interceptor_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    interceptor_speed: float,
    max_prediction_s: float = 20.0,
    max_vertical_prediction_s: Optional[float] = None,
    terminal_distance: float = 0.0,
    terminal_gain: float = 0.7,
    terminal_minimum_closing_speed: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """返回经典提前量制导速度和预计拦截时间。

    远距离用恒速提前点；进入 terminal_distance 后改为目标速度前馈加
    位置误差反馈。可选的最小闭合速度避免碰撞前与目标匹配速度。
    """

    minimum_closing_speed = float(terminal_minimum_closing_speed)
    if not np.isfinite(minimum_closing_speed) or minimum_closing_speed < 0.0:
        raise ValueError("末端最小闭合速度必须是有限非负数")

    tti = intercept_time(interceptor_pos, target_pos, target_vel, interceptor_speed)
    relative = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        interceptor_pos, dtype=np.float64
    )
    velocity = np.asarray(target_vel, dtype=np.float64)
    distance = float(np.linalg.norm(relative))
    if terminal_distance > 0.0 and distance <= float(terminal_distance):
        closing_command = float(terminal_gain) * distance
        if minimum_closing_speed > 0.0:
            closing_command = max(closing_command, minimum_closing_speed)
            if distance > 1.0e-8:
                closing_direction = relative / distance
            else:
                target_speed = float(np.linalg.norm(velocity))
                closing_direction = (
                    velocity / target_speed
                    if target_speed > 1.0e-8
                    else np.zeros(3, dtype=np.float64)
                )
            desired = velocity + closing_command * closing_direction
        else:
            # 保持已验证 benchmark 的原始计算路径完全不变。
            desired = velocity + float(terminal_gain) * relative
        desired_norm = float(np.linalg.norm(desired))
        if desired_norm > float(interceptor_speed):
            desired *= float(interceptor_speed) / desired_norm
        return desired.astype(np.float32), tti

    prediction_t = min(max(tti, 0.0), max_prediction_s)
    vertical_limit = (
        max_prediction_s
        if max_vertical_prediction_s is None
        else max_vertical_prediction_s
    )
    vertical_prediction_t = min(max(tti, 0.0), vertical_limit)
    aim_point = np.asarray(target_pos, dtype=np.float64).copy()
    aim_point[:2] += velocity[:2] * prediction_t
    aim_point[2] += velocity[2] * vertical_prediction_t
    direction = aim_point - np.asarray(interceptor_pos, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8:
        return np.zeros(3, dtype=np.float32), tti
    return (direction / norm * float(interceptor_speed)).astype(np.float32), tti


def assign_targets(
    interceptor_pos: np.ndarray,
    interceptor_alive: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_alive: np.ndarray,
    interceptor_speed: float,
    previous_assignment: Optional[np.ndarray] = None,
    switch_penalty_s: float = 0.0,
    target_deadline_s: Optional[np.ndarray] = None,
    infeasible_penalty: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """用预计拦截时间构造代价，进行一对一目标分配。

    返回：
      assignment: [N_agent]，未分配为 -1
      cost_matrix: 仅包含当前有效拦截机和目标的代价矩阵
    """

    agent_ids = np.flatnonzero(np.asarray(interceptor_alive, dtype=bool))
    target_ids = np.flatnonzero(np.asarray(target_alive, dtype=bool))
    assignment = np.full(len(interceptor_alive), -1, dtype=np.int64)
    if len(agent_ids) == 0 or len(target_ids) == 0:
        return assignment, np.empty((len(agent_ids), len(target_ids)), dtype=np.float64)

    cost = np.zeros((len(agent_ids), len(target_ids)), dtype=np.float64)
    for row, agent_id in enumerate(agent_ids):
        for col, target_id in enumerate(target_ids):
            tti = intercept_time(
                interceptor_pos[agent_id],
                target_pos[target_id],
                target_vel[target_id],
                interceptor_speed,
            )
            candidate_cost = tti
            if target_deadline_s is not None:
                missed_by = max(0.0, tti - float(target_deadline_s[target_id]))
                candidate_cost += infeasible_penalty * missed_by
            if (
                previous_assignment is not None
                and int(previous_assignment[agent_id]) >= 0
                and int(previous_assignment[agent_id]) != int(target_id)
            ):
                candidate_cost += float(switch_penalty_s)
            cost[row, col] = candidate_cost

    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        assignment[agent_ids[row]] = target_ids[col]
    return assignment, cost
