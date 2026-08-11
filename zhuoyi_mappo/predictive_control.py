from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flight_response import RidgeFlightResponseModel


def _clip_norm_rows(vectors: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float64).copy()
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    scale = np.minimum(
        1.0, float(maximum) / np.maximum(norms, 1.0e-8)
    )
    return value * scale


@dataclass(frozen=True)
class PredictiveCPAConfig:
    horizon_s: float = 1.5
    step_s: float = 0.1
    actions_mps: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    maximum_distance_m: float = 80.0
    minimum_pn_acceleration_mps2: float = 0.05
    minimum_improvement_m: float = 0.2

    def __post_init__(self) -> None:
        positive = (
            self.horizon_s,
            self.step_s,
            self.maximum_distance_m,
            self.minimum_pn_acceleration_mps2,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("预测 CPA 的时域、步长和门控必须是有限正数")
        if (
            not np.isfinite(self.minimum_improvement_m)
            or self.minimum_improvement_m < 0.0
        ):
            raise ValueError("预测 CPA 改善门槛必须有限非负")
        actions = np.asarray(self.actions_mps, dtype=np.float64)
        if (
            actions.ndim != 1
            or len(actions) == 0
            or not np.isfinite(actions).all()
            or not np.any(np.isclose(actions, 0.0, atol=1.0e-12))
        ):
            raise ValueError("预测 CPA 动作必须是一维有限数组并包含零")


@dataclass(frozen=True)
class PredictiveCPABatchResult:
    correction_velocity: np.ndarray
    selected_action_mps: np.ndarray
    baseline_cpa_m: np.ndarray
    selected_cpa_m: np.ndarray
    improvement_m: np.ndarray
    evaluated: np.ndarray


def _rollout_cpa_batch(
    model: RidgeFlightResponseModel,
    interceptor_position: np.ndarray,
    velocity_history: np.ndarray,
    previous_command_history: np.ndarray,
    history_dt: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    desired_velocity: np.ndarray,
    limiter_value: np.ndarray,
    first_dt: float,
    maximum_speed: float,
    maximum_acceleration: float,
    config: PredictiveCPAConfig,
) -> np.ndarray:
    velocities = np.asarray(velocity_history, dtype=np.float64).copy()
    previous_commands = np.asarray(
        previous_command_history, dtype=np.float64
    ).copy()
    previous_dt = np.asarray(history_dt, dtype=np.float64).copy()
    interceptor = np.asarray(interceptor_position, dtype=np.float64).copy()
    target = np.asarray(target_position, dtype=np.float64).copy()
    target_vel = np.asarray(target_velocity, dtype=np.float64)
    command_value = np.asarray(limiter_value, dtype=np.float64).copy()
    desired = np.asarray(desired_velocity, dtype=np.float64)
    minimum_distance = np.linalg.norm(target - interceptor, axis=1)
    elapsed = 0.0
    step_index = 0

    while elapsed < config.horizon_s - 1.0e-9:
        nominal_dt = float(first_dt) if step_index == 0 else config.step_s
        dt = min(
            max(nominal_dt, 1.0e-3),
            config.horizon_s - elapsed,
        )
        desired_limited = _clip_norm_rows(desired, maximum_speed)
        delta = _clip_norm_rows(
            desired_limited - command_value,
            maximum_acceleration * max(dt, 1.0e-3),
        )
        command_value = _clip_norm_rows(
            command_value + delta, maximum_speed
        )
        commands = np.concatenate(
            (command_value[:, None, :], previous_commands), axis=1
        )
        acceleration = model.predict_world_acceleration(
            velocities,
            commands,
            previous_dt,
            np.full(len(velocities), dt, dtype=np.float64),
        ).astype(np.float64)
        interceptor += velocities[:, 0] * dt + 0.5 * acceleration * dt * dt
        target += target_vel * dt
        next_velocity = velocities[:, 0] + acceleration * dt
        velocities[:, 1:] = velocities[:, :-1]
        velocities[:, 0] = next_velocity
        previous_commands[:, 1:] = previous_commands[:, :-1]
        previous_commands[:, 0] = command_value
        previous_dt[:, 1] = previous_dt[:, 0]
        previous_dt[:, 0] = dt
        elapsed += dt
        step_index += 1
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(target - interceptor, axis=1),
        )
    return minimum_distance


def select_predictive_cpa_corrections_batch(
    model: RidgeFlightResponseModel,
    interceptor_position: np.ndarray,
    velocity_history: np.ndarray,
    previous_command_history: np.ndarray,
    history_dt: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    base_desired_velocity: np.ndarray,
    limiter_value: np.ndarray,
    pn_acceleration: np.ndarray,
    first_dt: float,
    maximum_speed: float,
    maximum_acceleration: float,
    config: PredictiveCPAConfig = PredictiveCPAConfig(),
) -> PredictiveCPABatchResult:
    interceptor = np.asarray(interceptor_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    pn = np.asarray(pn_acceleration, dtype=np.float64)
    count = len(interceptor)
    correction = np.zeros((count, 3), dtype=np.float32)
    selected_action = np.zeros(count, dtype=np.float64)
    initial_distance = np.linalg.norm(target - interceptor, axis=1)
    baseline_cpa = initial_distance.copy()
    selected_cpa = initial_distance.copy()
    improvement = np.zeros(count, dtype=np.float64)
    horizontal_pn = pn.copy()
    horizontal_pn[:, 2] = 0.0
    pn_norm = np.linalg.norm(horizontal_pn, axis=1)
    evaluated = (
        (initial_distance <= config.maximum_distance_m)
        & (pn_norm >= config.minimum_pn_acceleration_mps2)
    )
    indices = np.flatnonzero(evaluated)
    if len(indices) == 0:
        return PredictiveCPABatchResult(
            correction,
            selected_action,
            baseline_cpa,
            selected_cpa,
            improvement,
            evaluated,
        )

    direction = horizontal_pn[indices] / pn_norm[indices, None]
    actions = np.unique(np.asarray(config.actions_mps, dtype=np.float64))
    action_count = len(actions)
    base_desired = np.asarray(base_desired_velocity, dtype=np.float64)[indices]
    candidate_desired = _clip_norm_rows(
        base_desired[:, None, :]
        + actions[None, :, None] * direction[:, None, :],
        maximum_speed,
    ).reshape(-1, 3)

    def repeat(value: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(value)[indices], action_count, axis=0)

    cpa = _rollout_cpa_batch(
        model=model,
        interceptor_position=repeat(interceptor),
        velocity_history=repeat(velocity_history),
        previous_command_history=repeat(previous_command_history),
        history_dt=repeat(history_dt),
        target_position=repeat(target),
        target_velocity=repeat(target_velocity),
        desired_velocity=candidate_desired,
        limiter_value=repeat(limiter_value),
        first_dt=first_dt,
        maximum_speed=maximum_speed,
        maximum_acceleration=maximum_acceleration,
        config=config,
    ).reshape(len(indices), action_count)
    zero_index = int(np.argmin(np.abs(actions)))
    local_baseline = cpa[:, zero_index]
    best = np.argmin(cpa, axis=1)
    local_action = actions[best]
    local_selected = cpa[np.arange(len(indices)), best]
    local_improvement = local_baseline - local_selected
    accepted = (
        (np.abs(local_action) >= 1.0e-12)
        & (local_improvement >= config.minimum_improvement_m)
    )
    local_action = np.where(accepted, local_action, 0.0)
    local_selected = np.where(accepted, local_selected, local_baseline)
    local_improvement = np.where(accepted, local_improvement, 0.0)
    baseline_cpa[indices] = local_baseline
    selected_cpa[indices] = local_selected
    selected_action[indices] = local_action
    improvement[indices] = local_improvement
    correction[indices] = (local_action[:, None] * direction).astype(np.float32)
    return PredictiveCPABatchResult(
        correction,
        selected_action,
        baseline_cpa,
        selected_cpa,
        improvement,
        evaluated,
    )


class CausalFlightHistory:
    def __init__(self, count: int):
        self.velocity = np.zeros((int(count), 3, 3), dtype=np.float64)
        self.command = np.zeros((int(count), 3, 3), dtype=np.float64)
        self.timestamp = np.full((int(count), 3), -np.inf, dtype=np.float64)
        self.samples = np.zeros(int(count), dtype=np.int64)

    def ready(self, index: int, timestamp: float) -> bool:
        agent = int(index)
        if self.samples[agent] < 3:
            return False
        times = self.timestamp[agent]
        gaps = np.array(
            (float(timestamp) - times[0], times[0] - times[1]),
            dtype=np.float64,
        )
        return bool(
            np.isfinite(gaps).all()
            and np.all((gaps > 0.04) & (gaps <= 0.35))
        )

    def inputs(
        self, index: int, velocity: np.ndarray, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        agent = int(index)
        if not self.ready(agent, timestamp):
            raise ValueError("飞控历史尚未就绪")
        velocities = np.concatenate(
            (
                np.asarray(velocity, dtype=np.float64)[None, :],
                self.velocity[agent],
            ),
            axis=0,
        )
        history_dt = np.array(
            (
                float(timestamp) - self.timestamp[agent, 0],
                self.timestamp[agent, 0] - self.timestamp[agent, 1],
            ),
            dtype=np.float64,
        )
        return velocities, self.command[agent].copy(), history_dt

    def update(
        self,
        velocity: np.ndarray,
        command: np.ndarray,
        timestamp: float,
        active: np.ndarray,
    ) -> None:
        velocities = np.asarray(velocity, dtype=np.float64)
        commands = np.asarray(command, dtype=np.float64)
        active_mask = np.asarray(active, dtype=bool)
        for agent in range(len(self.samples)):
            if not active_mask[agent]:
                self.velocity[agent] = 0.0
                self.command[agent] = 0.0
                self.timestamp[agent] = -np.inf
                self.samples[agent] = 0
                continue
            if float(timestamp) <= self.timestamp[agent, 0]:
                continue
            self.velocity[agent, 1:] = self.velocity[agent, :-1]
            self.command[agent, 1:] = self.command[agent, :-1]
            self.timestamp[agent, 1:] = self.timestamp[agent, :-1]
            self.velocity[agent, 0] = velocities[agent]
            self.command[agent, 0] = commands[agent]
            self.timestamp[agent, 0] = float(timestamp)
            self.samples[agent] = min(self.samples[agent] + 1, 3)
