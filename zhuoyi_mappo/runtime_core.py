from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .assignment import assign_targets, lead_velocity
from .config import ControllerConfig
from .guidance import (
    line_of_sight_kinematics,
    los_rate_pn_velocity,
)


DIFFICULTIES = ("low", "mid", "high")
GUIDANCE_MODES = ("classic", "los_pn", "los_pn_kf")


def default_guidance_mode(difficulty: str) -> str:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"未知难度: {difficulty}")
    return "los_pn_kf" if difficulty == "high" else "los_pn"


@dataclass
class GuidanceResult:
    assignment: np.ndarray
    assignment_changed: np.ndarray
    guide_velocity: np.ndarray
    intercept_time: np.ndarray
    guidance_blend: np.ndarray
    target_deadline: np.ndarray
    los_angular_rate: np.ndarray
    closing_speed: np.ndarray
    los_pn_acceleration: np.ndarray


def _target_deadline(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    safe_center: np.ndarray,
    safety_radius: float,
) -> np.ndarray:
    horizontal_distance = np.maximum(
        0.0,
        np.linalg.norm(target_pos[:, :2] - safe_center[:2], axis=1)
        - safety_radius,
    )
    horizontal_speed = np.maximum(
        np.linalg.norm(target_vel[:, :2], axis=1), 1.0
    )
    return horizontal_distance / horizontal_speed


def build_guidance_inputs(
    config: ControllerConfig,
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    agent_active: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_active: np.ndarray,
    safe_center: np.ndarray,
    previous_assignment: Optional[np.ndarray],
    difficulty: str,
    guidance_mode: str = "classic",
    use_target_deadline: bool = True,
    los_rate_override: Optional[np.ndarray] = None,
) -> GuidanceResult:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"未知难度: {difficulty}")
    if guidance_mode not in GUIDANCE_MODES:
        raise ValueError(f"未知制导模式: {guidance_mode}")
    if guidance_mode == "los_pn_kf" and difficulty != "high":
        raise ValueError("自适应 LOS KF 候选仅允许 High 难度")

    agent_pos = np.asarray(agent_pos, dtype=np.float32)
    agent_vel = np.asarray(agent_vel, dtype=np.float32)
    agent_active = np.asarray(agent_active, dtype=bool)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    target_vel = np.asarray(target_vel, dtype=np.float32)
    target_active = np.asarray(target_active, dtype=bool)
    safe_center = np.asarray(safe_center, dtype=np.float32)
    if los_rate_override is not None:
        los_rate_override = np.asarray(los_rate_override, dtype=np.float32)
        if (
            los_rate_override.shape != (config.num_agents, 3)
            or not np.isfinite(los_rate_override).all()
        ):
            raise ValueError(
                "los_rate_override 必须是有限的 (num_agents, 3) 数组"
            )
    previous = (
        np.full(config.num_agents, -1, dtype=np.int64)
        if previous_assignment is None
        else np.asarray(previous_assignment, dtype=np.int64)
    )

    if use_target_deadline:
        target_deadline = _target_deadline(
            target_pos,
            target_vel,
            safe_center,
            config.safety_radius,
        ).astype(np.float32)
        assignment_deadline = target_deadline
        infeasible_penalty = config.infeasible_intercept_penalty
    else:
        target_deadline = np.full(
            config.num_targets, np.inf, dtype=np.float32
        )
        assignment_deadline = None
        infeasible_penalty = 0.0

    assignment, _ = assign_targets(
        agent_pos,
        agent_active,
        target_pos,
        target_vel,
        target_active,
        config.interceptor_max_speed,
        previous_assignment=previous,
        switch_penalty_s=config.assignment_switch_penalty_s,
        target_deadline_s=assignment_deadline,
        infeasible_penalty=infeasible_penalty,
    )
    changed = ((previous >= 0) & (assignment != previous)).astype(np.float32)

    guide = np.zeros((config.num_agents, 3), dtype=np.float32)
    tti = np.full(config.num_agents, 60.0, dtype=np.float32)
    blend = np.zeros(config.num_agents, dtype=np.float32)
    los_rate = np.zeros(config.num_agents, dtype=np.float32)
    closing_speed = np.zeros(config.num_agents, dtype=np.float32)
    pn_acceleration = np.zeros((config.num_agents, 3), dtype=np.float32)
    prediction_xy, prediction_z = config.lead_prediction_horizons(difficulty)
    terminal_distance, terminal_gain = config.terminal_guidance_params(
        difficulty
    )

    for agent_id, target_id in enumerate(assignment):
        if (
            not agent_active[agent_id]
            or target_id < 0
            or not target_active[target_id]
        ):
            continue
        classic, intercept = lead_velocity(
            agent_pos[agent_id],
            target_pos[target_id],
            target_vel[target_id],
            config.interceptor_max_speed,
            max_prediction_s=prediction_xy,
            max_vertical_prediction_s=prediction_z,
            terminal_distance=terminal_distance,
            terminal_gain=terminal_gain,
        )
        omega, pair_closing_speed, _ = line_of_sight_kinematics(
            agent_pos[agent_id],
            agent_vel[agent_id],
            target_pos[target_id],
            target_vel[target_id],
        )
        los_rate[agent_id] = float(np.linalg.norm(omega))
        closing_speed[agent_id] = pair_closing_speed
        if guidance_mode in ("los_pn", "los_pn_kf"):
            result = los_rate_pn_velocity(
                agent_pos[agent_id],
                agent_vel[agent_id],
                target_pos[target_id],
                target_vel[target_id],
                classic,
                config.interceptor_max_speed,
                navigation_constant=config.los_pn_navigation_constant,
                maximum_acceleration=config.interceptor_max_acceleration,
                response_lead_seconds=(
                    config.los_pn_response_lead_seconds
                ),
                activation_time_to_go=(
                    config.los_pn_activation_time_to_go
                ),
                full_time_to_go=config.los_pn_full_time_to_go,
                minimum_closing_speed=(
                    config.los_pn_minimum_closing_speed
                ),
                regularization_distance=(
                    config.los_pn_regularization_distance
                ),
                close_fade_distance=config.los_pn_close_fade_distance,
                close_cutoff_distance=config.los_pn_close_cutoff_distance,
                los_rate_override=(
                    los_rate_override[agent_id]
                    if (
                        guidance_mode == "los_pn_kf"
                        and los_rate_override is not None
                    )
                    else None
                ),
            )
            guide[agent_id] = result.velocity
            tti[agent_id] = result.intercept_time
            blend[agent_id] = result.blend
            pn_acceleration[agent_id] = result.acceleration
            los_rate[agent_id] = float(np.linalg.norm(result.los_rate))
        else:
            guide[agent_id] = classic
            tti[agent_id] = intercept

    return GuidanceResult(
        assignment=assignment,
        assignment_changed=changed,
        guide_velocity=guide,
        intercept_time=tti,
        guidance_blend=blend,
        target_deadline=target_deadline,
        los_angular_rate=los_rate,
        closing_speed=closing_speed,
        los_pn_acceleration=pn_acceleration,
    )
