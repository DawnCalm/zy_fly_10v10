from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .assignment import assign_targets, lead_velocity
from .config import EnvConfig
from .env import DIFFICULTIES
from .residual import terminal_observation_features


@dataclass
class GuidanceResult:
    assignment: np.ndarray
    assignment_changed: np.ndarray
    guide_velocity: np.ndarray
    intercept_time: np.ndarray
    observation: np.ndarray
    global_state: np.ndarray


def _target_deadline(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    safe_center: np.ndarray,
    safety_radius: float,
) -> np.ndarray:
    horizontal_distance = np.maximum(
        0.0,
        np.linalg.norm(target_pos[:, :2] - safe_center[:2], axis=1)
        - float(safety_radius),
    )
    horizontal_speed = np.maximum(np.linalg.norm(target_vel[:, :2], axis=1), 1.0)
    return horizontal_distance / horizontal_speed


def _nearest_friend_relative(
    agent_id: int,
    agent_pos: np.ndarray,
    agent_active: np.ndarray,
) -> np.ndarray:
    if not agent_active[agent_id]:
        return np.zeros(3, dtype=np.float32)
    candidates = np.flatnonzero(agent_active)
    candidates = candidates[candidates != agent_id]
    if len(candidates) == 0:
        return np.zeros(3, dtype=np.float32)
    delta = agent_pos[candidates] - agent_pos[agent_id]
    return delta[int(np.argmin(np.linalg.norm(delta, axis=1)))].astype(np.float32)


def build_guidance_inputs(
    config: EnvConfig,
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    agent_active: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_active: np.ndarray,
    safe_center: np.ndarray,
    previous_assignment: Optional[np.ndarray],
    elapsed_s: float,
    difficulty: str,
) -> GuidanceResult:
    """由真实 ROS 状态构造与训练环境一致的分配、观测和全局状态。"""

    if difficulty not in DIFFICULTIES:
        raise ValueError(f"未知难度: {difficulty}")
    agent_pos = np.asarray(agent_pos, dtype=np.float32)
    agent_vel = np.asarray(agent_vel, dtype=np.float32)
    agent_active = np.asarray(agent_active, dtype=bool)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    target_vel = np.asarray(target_vel, dtype=np.float32)
    target_active = np.asarray(target_active, dtype=bool)
    safe_center = np.asarray(safe_center, dtype=np.float32)

    previous = (
        np.full(config.num_agents, -1, dtype=np.int64)
        if previous_assignment is None
        else np.asarray(previous_assignment, dtype=np.int64)
    )
    assignment, _ = assign_targets(
        agent_pos,
        agent_active,
        target_pos,
        target_vel,
        target_active,
        config.interceptor_max_speed,
        previous_assignment=previous,
        switch_penalty_s=config.assignment_switch_penalty_s,
        target_deadline_s=_target_deadline(
            target_pos, target_vel, safe_center, config.safety_radius
        ),
        infeasible_penalty=config.infeasible_intercept_penalty,
    )
    changed = ((previous >= 0) & (assignment != previous)).astype(np.float32)

    guide = np.zeros((config.num_agents, 3), dtype=np.float32)
    tti = np.full(config.num_agents, 60.0, dtype=np.float32)
    prediction_xy, prediction_z = config.lead_prediction_horizons(difficulty)
    terminal_distance, terminal_gain = config.terminal_guidance_params(
        difficulty
    )
    for agent_id, target_id in enumerate(assignment):
        if (
            agent_active[agent_id]
            and target_id >= 0
            and target_active[target_id]
        ):
            guide[agent_id], tti[agent_id] = lead_velocity(
                agent_pos[agent_id],
                target_pos[target_id],
                target_vel[target_id],
                config.interceptor_max_speed,
                max_prediction_s=prediction_xy,
                max_vertical_prediction_s=prediction_z,
                terminal_distance=terminal_distance,
                terminal_gain=terminal_gain,
            )

    observation = np.zeros(
        (config.num_agents, config.obs_dim), dtype=np.float32
    )
    remaining_fraction = float(target_active.mean())
    # 必须与训练环境使用同一时间归一化；正式控制超过训练时域后保持 0。
    trained_episode_s = max(config.max_steps * config.dt, config.dt)
    time_remaining = float(
        np.clip(1.0 - elapsed_s / trained_episode_s, 0.0, 1.0)
    )
    for agent_id, target_id in enumerate(assignment):
        if target_id >= 0 and target_active[target_id]:
            rel_pos = target_pos[target_id] - agent_pos[agent_id]
            rel_vel = target_vel[target_id] - agent_vel[agent_id]
            target_valid = 1.0
        else:
            rel_pos = np.zeros(3, dtype=np.float32)
            rel_vel = np.zeros(3, dtype=np.float32)
            target_valid = 0.0
        observation[agent_id, 0:3] = rel_pos / config.arena_half_size
        observation[agent_id, 3:6] = rel_vel / config.interceptor_max_speed
        observation[agent_id, 6:9] = (
            agent_vel[agent_id] / config.interceptor_max_speed
        )
        observation[agent_id, 9:12] = (
            guide[agent_id] / config.interceptor_max_speed
        )
        observation[agent_id, 12] = np.clip(tti[agent_id] / 60.0, 0.0, 2.0)
        observation[agent_id, 13] = target_valid
        observation[agent_id, 14] = remaining_fraction
        observation[agent_id, 15] = time_remaining
        observation[agent_id, 16:19] = (
            _nearest_friend_relative(agent_id, agent_pos, agent_active)
            / config.arena_half_size
        )
        observation[agent_id, 19] = changed[agent_id]
        observation[agent_id, 20] = float(agent_active[agent_id])
        observation[
            agent_id, 21 + DIFFICULTIES.index(difficulty)
        ] = 1.0
        activation_distance, full_distance = config.residual_gate_distances(
            difficulty
        )
        observation[agent_id, 24:32] = terminal_observation_features(
            rel_pos[None],
            rel_vel[None],
            guide[agent_id : agent_id + 1],
            config.interceptor_max_speed,
            activation_distance,
            full_distance,
        )[0]
        if not target_valid:
            observation[agent_id, 31] = 0.0

    global_state = np.zeros(config.global_state_dim, dtype=np.float32)
    cursor = 0
    for agent_id in range(config.num_agents):
        global_state[cursor : cursor + 3] = (
            agent_pos[agent_id] - safe_center
        ) / config.arena_half_size
        global_state[cursor + 3 : cursor + 6] = (
            agent_vel[agent_id] / config.interceptor_max_speed
        )
        global_state[cursor + 6] = float(agent_active[agent_id])
        cursor += 7
    target_speed_scale = max(
        config.target_speed_low,
        config.target_speed_mid,
        config.target_speed_high,
    )
    for target_id in range(config.num_targets):
        global_state[cursor : cursor + 3] = (
            target_pos[target_id] - safe_center
        ) / config.arena_half_size
        global_state[cursor + 3 : cursor + 6] = (
            target_vel[target_id] / target_speed_scale
        )
        global_state[cursor + 6] = float(target_active[target_id])
        cursor += 7
    global_state[cursor] = time_remaining
    global_state[cursor + 1 + DIFFICULTIES.index(difficulty)] = 1.0

    return GuidanceResult(
        assignment=assignment,
        assignment_changed=changed,
        guide_velocity=guide,
        intercept_time=tti,
        observation=observation,
        global_state=global_state,
    )
