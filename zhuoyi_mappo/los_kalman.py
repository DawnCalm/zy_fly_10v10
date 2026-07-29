from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .guidance import line_of_sight_kinematics


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1.0e-6:
        return None
    return value / norm


@dataclass
class _LOSFilterState:
    target_id: int
    timestamp: float
    direction: np.ndarray
    direction_rate: np.ndarray
    covariance: np.ndarray
    samples: int = 1


@dataclass(frozen=True)
class AdaptiveLOSKalmanResult:
    los_rate: np.ndarray
    used: np.ndarray
    direction_innovation: np.ndarray
    rate_correction: np.ndarray
    process_scale: np.ndarray


class AdaptiveLOSKalmanObserver:
    """用原始 LOS 方向和稳定运动学角速度估计及时、受限的 LOS-rate。

    每个拦截机维护 ``[LOS 单位向量, LOS 向量变化率]`` 常速度模型。
    原始雷达位置只作为 LOS 方向测量，Alpha-Beta 速度反推的 LOS-rate
    作为方向导数伪测量。创新量大时提高过程噪声以跟随机动，同时对方向
    创新和最终角速度修正做门控，避免退化为高噪声的直接方向差分。
    """

    def __init__(
        self,
        count: int,
        process_acceleration_std: float = 0.30,
        direction_measurement_std: float = 0.006,
        rate_measurement_std: float = 0.20,
        innovation_gate_sigma: float = 4.0,
        maximum_process_scale: float = 25.0,
        maximum_rate_correction: float = 0.04,
        maximum_angular_rate: float = 2.0,
        minimum_dt: float = 0.02,
        maximum_dt: float = 0.5,
        maximum_measurement_age: float = 0.25,
    ):
        positive = (
            process_acceleration_std,
            direction_measurement_std,
            rate_measurement_std,
            innovation_gate_sigma,
            maximum_process_scale,
            maximum_rate_correction,
            maximum_angular_rate,
            minimum_dt,
            maximum_dt,
            maximum_measurement_age,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("自适应 LOS KF 参数必须是有限正数")
        if maximum_dt <= minimum_dt:
            raise ValueError("自适应 LOS KF maximum_dt 必须大于 minimum_dt")

        self.count = int(count)
        if self.count <= 0:
            raise ValueError("自适应 LOS KF count 必须大于 0")
        self.process_acceleration_std = float(process_acceleration_std)
        self.direction_measurement_std = float(direction_measurement_std)
        self.rate_measurement_std = float(rate_measurement_std)
        self.innovation_gate_sigma = float(innovation_gate_sigma)
        self.maximum_process_scale = float(maximum_process_scale)
        self.maximum_rate_correction = float(maximum_rate_correction)
        self.maximum_angular_rate = float(maximum_angular_rate)
        self.minimum_dt = float(minimum_dt)
        self.maximum_dt = float(maximum_dt)
        self.maximum_measurement_age = float(maximum_measurement_age)
        self.states: list[Optional[_LOSFilterState]] = [None] * self.count

    def reset(self) -> None:
        self.states = [None] * self.count

    def _initialize(
        self,
        agent_id: int,
        target_id: int,
        timestamp: float,
        direction: np.ndarray,
        kinematic_rate: np.ndarray,
    ) -> None:
        direction_rate = np.cross(kinematic_rate, direction)
        covariance = np.diag(
            [
                self.direction_measurement_std**2,
                self.rate_measurement_std**2,
            ]
        )
        self.states[agent_id] = _LOSFilterState(
            target_id=target_id,
            timestamp=float(timestamp),
            direction=direction.copy(),
            direction_rate=direction_rate,
            covariance=covariance,
        )

    def _measurement_update(
        self,
        state: _LOSFilterState,
        timestamp: float,
        measured_direction: np.ndarray,
        kinematic_rate: np.ndarray,
    ) -> tuple[float, float]:
        raw_dt = float(timestamp) - state.timestamp
        dt = float(np.clip(raw_dt, self.minimum_dt, self.maximum_dt))
        transition = np.array(((1.0, dt), (0.0, 1.0)))
        predicted_direction = (
            state.direction + dt * state.direction_rate
        )
        normalized_prediction = _unit(predicted_direction)
        if normalized_prediction is None:
            normalized_prediction = state.direction
        innovation_norm = float(
            np.linalg.norm(measured_direction - normalized_prediction)
        )
        normalized_innovation = (
            innovation_norm / self.direction_measurement_std
        )
        process_scale = 1.0 + min(
            self.maximum_process_scale - 1.0,
            normalized_innovation**2 / 9.0,
        )
        noise_gain = np.array((0.5 * dt * dt, dt))
        covariance = (
            transition @ state.covariance @ transition.T
            + (
                self.process_acceleration_std**2
                * process_scale
                * np.outer(noise_gain, noise_gain)
            )
        )
        direction = predicted_direction
        direction_rate = state.direction_rate.copy()

        innovation = measured_direction - direction
        gate = (
            self.innovation_gate_sigma * self.direction_measurement_std
        )
        innovation_length = float(np.linalg.norm(innovation))
        if innovation_length > gate:
            innovation *= gate / innovation_length
        direction_variance = self.direction_measurement_std**2
        direction_gain = covariance[:, 0] / (
            covariance[0, 0] + direction_variance
        )
        direction += direction_gain[0] * innovation
        direction_rate += direction_gain[1] * innovation
        covariance = (
            np.eye(2) - np.outer(direction_gain, (1.0, 0.0))
        ) @ covariance

        normalized_direction = _unit(direction)
        if normalized_direction is None:
            normalized_direction = measured_direction
        measured_direction_rate = np.cross(
            kinematic_rate, normalized_direction
        )
        rate_innovation = measured_direction_rate - direction_rate
        rate_variance = self.rate_measurement_std**2
        rate_gain = covariance[:, 1] / (
            covariance[1, 1] + rate_variance
        )
        direction += rate_gain[0] * rate_innovation
        direction_rate += rate_gain[1] * rate_innovation
        covariance = (
            np.eye(2) - np.outer(rate_gain, (0.0, 1.0))
        ) @ covariance

        normalized_direction = _unit(direction)
        if normalized_direction is None:
            normalized_direction = measured_direction
        direction_rate -= normalized_direction * float(
            np.dot(normalized_direction, direction_rate)
        )
        state.timestamp = float(timestamp)
        state.direction = normalized_direction
        state.direction_rate = direction_rate
        state.covariance = 0.5 * (covariance + covariance.T)
        state.samples += 1
        return innovation_norm, process_scale

    def update(
        self,
        timestamp: float,
        assignment: np.ndarray,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        agent_active: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_active: np.ndarray,
        raw_target_pos: np.ndarray,
        raw_target_time: np.ndarray,
    ) -> AdaptiveLOSKalmanResult:
        now = float(timestamp)
        assignment_value = np.asarray(assignment, dtype=np.int64)
        agent_position = np.asarray(agent_pos, dtype=np.float64)
        agent_velocity = np.asarray(agent_vel, dtype=np.float64)
        agent_mask = np.asarray(agent_active, dtype=bool)
        target_position = np.asarray(target_pos, dtype=np.float64)
        target_velocity = np.asarray(target_vel, dtype=np.float64)
        target_mask = np.asarray(target_active, dtype=bool)
        raw_position = np.asarray(raw_target_pos, dtype=np.float64)
        raw_time = np.asarray(raw_target_time, dtype=np.float64)

        expected_agent_vector = (self.count,)
        expected_agent_matrix = (self.count, 3)
        target_count = len(target_position)
        if assignment_value.shape != expected_agent_vector:
            raise ValueError("assignment shape 与 LOS KF count 不一致")
        if (
            agent_position.shape != expected_agent_matrix
            or agent_velocity.shape != expected_agent_matrix
            or agent_mask.shape != expected_agent_vector
        ):
            raise ValueError("agent 状态 shape 与 LOS KF count 不一致")
        if (
            target_position.shape != (target_count, 3)
            or target_velocity.shape != (target_count, 3)
            or target_mask.shape != (target_count,)
            or raw_position.shape != (target_count, 3)
            or raw_time.shape != (target_count,)
        ):
            raise ValueError("target 状态 shape 不一致")

        output = np.zeros((self.count, 3), dtype=np.float32)
        used = np.zeros(self.count, dtype=bool)
        direction_innovation = np.zeros(self.count, dtype=np.float32)
        rate_correction = np.zeros(self.count, dtype=np.float32)
        process_scale = np.ones(self.count, dtype=np.float32)

        for agent_id, target_id_value in enumerate(assignment_value):
            target_id = int(target_id_value)
            if (
                not agent_mask[agent_id]
                or target_id < 0
                or target_id >= target_count
                or not target_mask[target_id]
            ):
                self.states[agent_id] = None
                continue

            kinematic_rate, _, _ = line_of_sight_kinematics(
                agent_position[agent_id],
                agent_velocity[agent_id],
                target_position[target_id],
                target_velocity[target_id],
            )
            kinematic_rate = np.asarray(kinematic_rate, dtype=np.float64)
            output[agent_id] = kinematic_rate

            measurement_time = float(raw_time[target_id])
            age = now - measurement_time
            if (
                not np.isfinite(measurement_time)
                or age < -1.0e-3
                or age > self.maximum_measurement_age
                or not np.isfinite(raw_position[target_id]).all()
            ):
                self.states[agent_id] = None
                continue

            interceptor_at_measurement = (
                agent_position[agent_id]
                - agent_velocity[agent_id] * max(0.0, age)
            )
            measured_relative = (
                raw_position[target_id] - interceptor_at_measurement
            )
            measured_direction = _unit(measured_relative)
            if measured_direction is None:
                self.states[agent_id] = None
                continue
            measurement_rate, _, _ = line_of_sight_kinematics(
                interceptor_at_measurement,
                agent_velocity[agent_id],
                raw_position[target_id],
                target_velocity[target_id],
            )
            measurement_rate = np.asarray(
                measurement_rate, dtype=np.float64
            )

            state = self.states[agent_id]
            if (
                state is None
                or state.target_id != target_id
                or measurement_time <= state.timestamp
                or measurement_time - state.timestamp > self.maximum_dt
            ):
                if (
                    state is None
                    or state.target_id != target_id
                    or measurement_time - state.timestamp > self.maximum_dt
                ):
                    self._initialize(
                        agent_id,
                        target_id,
                        measurement_time,
                        measured_direction,
                        measurement_rate,
                    )
                    continue
            else:
                innovation, adaptive_scale = self._measurement_update(
                    state,
                    measurement_time,
                    measured_direction,
                    measurement_rate,
                )
                direction_innovation[agent_id] = innovation
                process_scale[agent_id] = adaptive_scale

            state = self.states[agent_id]
            if state is None or state.samples < 2:
                continue
            prediction_age = float(
                np.clip(now - state.timestamp, 0.0, self.maximum_dt)
            )
            predicted_direction = _unit(
                state.direction + prediction_age * state.direction_rate
            )
            if predicted_direction is None:
                continue
            predicted_rate = np.cross(
                predicted_direction, state.direction_rate
            )
            current_direction = _unit(
                target_position[target_id] - agent_position[agent_id]
            )
            if current_direction is None:
                continue
            predicted_rate -= current_direction * float(
                np.dot(current_direction, predicted_rate)
            )
            correction = predicted_rate - kinematic_rate
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > self.maximum_rate_correction:
                correction *= (
                    self.maximum_rate_correction / correction_norm
                )
            fused_rate = kinematic_rate + correction
            fused_norm = float(np.linalg.norm(fused_rate))
            if fused_norm > self.maximum_angular_rate:
                fused_rate *= self.maximum_angular_rate / fused_norm
            if not np.isfinite(fused_rate).all():
                continue
            output[agent_id] = fused_rate
            used[agent_id] = True
            rate_correction[agent_id] = float(np.linalg.norm(correction))

        return AdaptiveLOSKalmanResult(
            los_rate=output,
            used=used,
            direction_innovation=direction_innovation,
            rate_correction=rate_correction,
            process_scale=process_scale,
        )
