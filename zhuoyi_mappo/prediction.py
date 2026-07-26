from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


MODE_NAMES = ("cv", "ca", "ct")


@dataclass(frozen=True)
class TargetEstimate:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    covariance: np.ndarray
    mode_probabilities: Dict[str, float]


@dataclass(frozen=True)
class TrajectoryPrediction:
    horizons: np.ndarray
    position: np.ndarray
    covariance: np.ndarray
    mode_positions: Dict[str, np.ndarray]
    mode_probabilities: Dict[str, float]


def constant_velocity_prediction(
    position: np.ndarray,
    velocity: np.ndarray,
    horizons: Iterable[float],
) -> np.ndarray:
    values = np.asarray(tuple(horizons), dtype=np.float64)
    return (
        np.asarray(position, dtype=np.float64)[None]
        + values[:, None] * np.asarray(velocity, dtype=np.float64)[None]
    ).astype(np.float32)


class IMMTargetPredictor:
    """CV/CA/协调转弯三模型交互式目标跟踪与轨迹预测。

    三个模型共享 [position(3), velocity(3), acceleration(3)] 状态，因此
    可以直接进行 IMM 状态混合。CT 模型从水平速度/加速度估计转弯率，
    并在预测时保持切向加速度和转弯率。
    """

    def __init__(
        self,
        measurement_std: float = 0.1,
        process_acceleration_std: tuple[float, float, float] = (0.1, 1.0, 1.0),
        max_acceleration: float = 15.0,
        max_turn_rate_deg_s: float = 20.0,
        min_dt: float = 0.02,
        max_dt: float = 0.6,
        transition_matrix: Optional[np.ndarray] = None,
    ):
        self.measurement_std = float(measurement_std)
        self.process_acceleration_std = np.asarray(
            process_acceleration_std, dtype=np.float64
        )
        if self.process_acceleration_std.shape != (3,):
            raise ValueError("process_acceleration_std 应包含 CV/CA/CT 三项")
        self.max_acceleration = float(max_acceleration)
        self.max_turn_rate = float(np.deg2rad(max_turn_rate_deg_s))
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        default_transition = np.array(
            [
                [0.94, 0.03, 0.03],
                [0.04, 0.92, 0.04],
                [0.04, 0.04, 0.92],
            ],
            dtype=np.float64,
        )
        self.transition_matrix = np.asarray(
            transition_matrix
            if transition_matrix is not None
            else default_transition,
            dtype=np.float64,
        )
        if self.transition_matrix.shape != (3, 3):
            raise ValueError("transition_matrix 必须为 3x3")
        row_sum = self.transition_matrix.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0.0) or np.any(self.transition_matrix < 0.0):
            raise ValueError("transition_matrix 必须非负且每行和大于 0")
        self.transition_matrix /= row_sum

        self.mode_states = np.zeros((3, 9), dtype=np.float64)
        self.mode_covariances = np.tile(np.eye(9), (3, 1, 1)).astype(
            np.float64
        )
        self.mode_probabilities = np.full(3, 1.0 / 3.0, dtype=np.float64)
        self.last_time: Optional[float] = None
        self.last_measurement: Optional[np.ndarray] = None
        self.samples = 0

    def reset(self) -> None:
        self.mode_states[:] = 0.0
        self.mode_covariances[:] = np.eye(9)
        self.mode_probabilities[:] = 1.0 / 3.0
        self.last_time = None
        self.last_measurement = None
        self.samples = 0

    def age(self, timestamp: float) -> float:
        if self.last_time is None:
            return float("inf")
        return max(0.0, float(timestamp) - self.last_time)

    @staticmethod
    def _clip_vector(value: np.ndarray, max_norm: float) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(result))
        if norm > max_norm > 0.0:
            result = result * (max_norm / norm)
        return result

    def _dynamics(self, mode_index: int, state: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64).copy()
        position = value[0:3]
        velocity = value[3:6]
        acceleration = self._clip_vector(
            value[6:9], self.max_acceleration
        )

        if mode_index == 0:
            position += velocity * dt
            acceleration[:] = 0.0
        elif mode_index == 1:
            position += velocity * dt + 0.5 * acceleration * dt * dt
            velocity += acceleration * dt
        else:
            horizontal_velocity = velocity[:2].copy()
            horizontal_acceleration = acceleration[:2].copy()
            speed = float(np.linalg.norm(horizontal_velocity))
            if speed > 1.0e-4:
                tangent = horizontal_velocity / speed
                tangential_acceleration = float(
                    np.dot(horizontal_acceleration, tangent)
                )
                cross_z = (
                    horizontal_velocity[0] * horizontal_acceleration[1]
                    - horizontal_velocity[1] * horizontal_acceleration[0]
                )
                turn_rate = float(
                    cross_z / max(speed * speed, 1.0e-8)
                )
                turn_rate = float(
                    np.clip(
                        turn_rate, -self.max_turn_rate, self.max_turn_rate
                    )
                )
                angle = turn_rate * dt
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.array(
                    ((cosine, -sine), (sine, cosine)), dtype=np.float64
                )
                new_direction = rotation @ tangent
                new_speed = max(0.0, speed + tangential_acceleration * dt)
                new_horizontal_velocity = new_direction * new_speed
                position[:2] += (
                    0.5
                    * (horizontal_velocity + new_horizontal_velocity)
                    * dt
                )
                velocity[:2] = new_horizontal_velocity
                normal = np.array(
                    (-new_direction[1], new_direction[0]), dtype=np.float64
                )
                acceleration[:2] = (
                    tangential_acceleration * new_direction
                    + turn_rate * new_speed * normal
                )
            else:
                position[:2] += horizontal_velocity * dt
            position[2] += velocity[2] * dt + 0.5 * acceleration[2] * dt * dt
            velocity[2] += acceleration[2] * dt

        value[0:3] = position
        value[3:6] = velocity
        value[6:9] = self._clip_vector(
            acceleration, self.max_acceleration
        )
        return value

    def _jacobian(
        self, mode_index: int, state: np.ndarray, dt: float
    ) -> np.ndarray:
        if mode_index in (0, 1, 2):
            # CT 状态传播保持非线性；协方差传播使用 CA 的一阶近似。
            # 雷达观测每 0.1 s 校正一次，该近似几乎不改变回放精度，
            # 但可去掉数值雅可比，使 10 目标在线更新满足实时预算。
            jacobian = np.zeros((9, 9), dtype=np.float64)
            jacobian[0:3, 0:3] = np.eye(3)
            jacobian[0:3, 3:6] = np.eye(3) * dt
            jacobian[3:6, 3:6] = np.eye(3)
            if mode_index in (1, 2):
                jacobian[0:3, 6:9] = np.eye(3) * (0.5 * dt * dt)
                jacobian[3:6, 6:9] = np.eye(3) * dt
                jacobian[6:9, 6:9] = np.eye(3)
            return jacobian

    def _process_noise(self, mode_index: int, dt: float) -> np.ndarray:
        sigma = float(self.process_acceleration_std[mode_index])
        variance = sigma * sigma
        diagonal = np.concatenate(
            (
                np.full(3, 0.25 * dt**4 * variance),
                np.full(3, dt**2 * variance),
                np.full(3, max(dt, 1.0e-3) * variance),
            )
        )
        return np.diag(np.maximum(diagonal, 1.0e-9))

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (covariance + covariance.T)
        symmetric.flat[:: symmetric.shape[0] + 1] += 1.0e-9
        return symmetric

    @staticmethod
    def _inverse_3x3(matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """避免小矩阵调用多线程 BLAS 带来的毫秒级启动开销。"""

        value = np.asarray(matrix, dtype=np.float64)
        a, b, c = value[0]
        d, e, f = value[1]
        g, h, i = value[2]
        cofactor_00 = e * i - f * h
        cofactor_01 = f * g - d * i
        cofactor_02 = d * h - e * g
        cofactor_10 = c * h - b * i
        cofactor_11 = a * i - c * g
        cofactor_12 = b * g - a * h
        cofactor_20 = b * f - c * e
        cofactor_21 = c * d - a * f
        cofactor_22 = a * e - b * d
        determinant = (
            a * cofactor_00 + b * cofactor_01 + c * cofactor_02
        )
        if not np.isfinite(determinant) or determinant <= 1.0e-18:
            regularized = value + np.eye(3) * 1.0e-6
            return (
                np.linalg.inv(regularized),
                max(float(np.linalg.det(regularized)), 1.0e-18),
            )
        inverse = np.array(
            [
                [cofactor_00, cofactor_10, cofactor_20],
                [cofactor_01, cofactor_11, cofactor_21],
                [cofactor_02, cofactor_12, cofactor_22],
            ],
            dtype=np.float64,
        )
        return inverse / determinant, float(determinant)

    @staticmethod
    def _gaussian_log_likelihood(
        innovation: np.ndarray,
        covariance_inverse: np.ndarray,
        covariance_determinant: float,
    ) -> float:
        if covariance_determinant <= 0.0:
            return -1.0e12
        mahalanobis = float(
            innovation.T @ covariance_inverse @ innovation
        )
        dimension = len(innovation)
        return -0.5 * (
            dimension * np.log(2.0 * np.pi)
            + np.log(covariance_determinant)
            + mahalanobis
        )

    def _initialize(
        self, measurement: np.ndarray, timestamp: float
    ) -> TargetEstimate:
        self.mode_states[:] = 0.0
        self.mode_states[:, 0:3] = measurement
        initial_diagonal = np.concatenate(
            (
                np.full(3, self.measurement_std**2),
                np.full(3, 25.0),
                np.full(3, 16.0),
            )
        )
        self.mode_covariances[:] = np.diag(initial_diagonal)
        self.mode_probabilities[:] = 1.0 / 3.0
        self.last_time = float(timestamp)
        self.last_measurement = measurement.copy()
        self.samples = 1
        return self.estimate()

    def update(
        self, measurement: np.ndarray, timestamp: float
    ) -> TargetEstimate:
        observed_position = np.asarray(measurement, dtype=np.float64)
        if observed_position.shape != (3,) or not np.isfinite(
            observed_position
        ).all():
            raise ValueError("measurement 必须是有限的三维坐标")
        now = float(timestamp)
        if self.last_time is None or now <= self.last_time:
            return self._initialize(observed_position, now)

        dt = float(np.clip(now - self.last_time, self.min_dt, self.max_dt))
        if self.samples == 1 and self.last_measurement is not None:
            measured_velocity = (
                observed_position - self.last_measurement
            ) / dt
            self.mode_states[:, 0:3] = observed_position
            self.mode_states[:, 3:6] = measured_velocity
            self.last_time = now
            self.last_measurement = observed_position.copy()
            self.samples = 2
            return self.estimate()

        predicted_mode_probability = (
            self.mode_probabilities @ self.transition_matrix
        )
        predicted_mode_probability = np.maximum(
            predicted_mode_probability, 1.0e-12
        )
        mixing = (
            self.mode_probabilities[:, None] * self.transition_matrix
        ) / predicted_mode_probability[None, :]

        mixed_states = np.zeros_like(self.mode_states)
        mixed_covariances = np.zeros_like(self.mode_covariances)
        for destination in range(3):
            weights = mixing[:, destination]
            mixed_state = np.sum(
                weights[:, None] * self.mode_states, axis=0
            )
            mixed_states[destination] = mixed_state
            covariance = np.zeros((9, 9), dtype=np.float64)
            for source in range(3):
                delta = self.mode_states[source] - mixed_state
                covariance += weights[source] * (
                    self.mode_covariances[source]
                    + np.outer(delta, delta)
                )
            mixed_covariances[destination] = (
                self._stabilize_covariance(covariance)
            )

        observation_matrix = np.zeros((3, 9), dtype=np.float64)
        observation_matrix[:, 0:3] = np.eye(3)
        measurement_covariance = (
            np.eye(3, dtype=np.float64) * self.measurement_std**2
        )
        identity = np.eye(9, dtype=np.float64)
        log_likelihood = np.zeros(3, dtype=np.float64)

        for mode_index in range(3):
            transition = self._jacobian(
                mode_index, mixed_states[mode_index], dt
            )
            predicted_state = self._dynamics(
                mode_index, mixed_states[mode_index], dt
            )
            predicted_covariance = (
                transition
                @ mixed_covariances[mode_index]
                @ transition.T
                + self._process_noise(mode_index, dt)
            )
            innovation = (
                observed_position
                - observation_matrix @ predicted_state
            )
            innovation_covariance = (
                observation_matrix
                @ predicted_covariance
                @ observation_matrix.T
                + measurement_covariance
            )
            covariance_observation = (
                predicted_covariance @ observation_matrix.T
            )
            innovation_inverse, innovation_determinant = (
                self._inverse_3x3(innovation_covariance)
            )
            kalman_gain = covariance_observation @ innovation_inverse
            updated_state = predicted_state + kalman_gain @ innovation
            residual_update = identity - kalman_gain @ observation_matrix
            updated_covariance = (
                residual_update
                @ predicted_covariance
                @ residual_update.T
                + kalman_gain
                @ measurement_covariance
                @ kalman_gain.T
            )
            self.mode_states[mode_index] = updated_state
            self.mode_covariances[mode_index] = (
                self._stabilize_covariance(updated_covariance)
            )
            log_likelihood[mode_index] = self._gaussian_log_likelihood(
                innovation,
                innovation_inverse,
                innovation_determinant,
            )

        log_weight = np.log(predicted_mode_probability) + log_likelihood
        log_weight -= float(log_weight.max())
        mode_probability = np.exp(log_weight)
        probability_sum = float(mode_probability.sum())
        if not np.isfinite(probability_sum) or probability_sum <= 0.0:
            mode_probability[:] = 1.0 / 3.0
        else:
            mode_probability /= probability_sum
        self.mode_probabilities[:] = np.maximum(mode_probability, 1.0e-8)
        self.mode_probabilities /= self.mode_probabilities.sum()
        self.last_time = now
        self.last_measurement = observed_position.copy()
        self.samples += 1
        return self.estimate()

    def estimate(self) -> TargetEstimate:
        state = np.sum(
            self.mode_probabilities[:, None] * self.mode_states, axis=0
        )
        covariance = np.zeros((9, 9), dtype=np.float64)
        for mode_index in range(3):
            delta = self.mode_states[mode_index] - state
            covariance += self.mode_probabilities[mode_index] * (
                self.mode_covariances[mode_index] + np.outer(delta, delta)
            )
        return TargetEstimate(
            position=state[0:3].astype(np.float32),
            velocity=state[3:6].astype(np.float32),
            acceleration=state[6:9].astype(np.float32),
            covariance=self._stabilize_covariance(covariance).astype(
                np.float32
            ),
            mode_probabilities={
                name: float(self.mode_probabilities[index])
                for index, name in enumerate(MODE_NAMES)
            },
        )

    def predict(self, horizons: Iterable[float]) -> TrajectoryPrediction:
        values = np.asarray(tuple(horizons), dtype=np.float64)
        if values.ndim != 1 or np.any(values < 0.0):
            raise ValueError("horizons 必须是一维非负序列")
        mode_positions = np.zeros((3, len(values), 3), dtype=np.float64)
        mode_covariances = np.zeros(
            (3, len(values), 3, 3), dtype=np.float64
        )
        for mode_index in range(3):
            for horizon_index, horizon in enumerate(values):
                predicted_state = self._dynamics(
                    mode_index,
                    self.mode_states[mode_index],
                    float(horizon),
                )
                transition = self._jacobian(
                    mode_index,
                    self.mode_states[mode_index],
                    float(horizon),
                )
                covariance = (
                    transition
                    @ self.mode_covariances[mode_index]
                    @ transition.T
                    + self._process_noise(mode_index, float(horizon))
                )
                mode_positions[
                    mode_index, horizon_index
                ] = predicted_state[0:3]
                mode_covariances[
                    mode_index, horizon_index
                ] = covariance[0:3, 0:3]

        fused_position = np.sum(
            self.mode_probabilities[:, None, None] * mode_positions,
            axis=0,
        )
        fused_covariance = np.zeros(
            (len(values), 3, 3), dtype=np.float64
        )
        for mode_index in range(3):
            delta = mode_positions[mode_index] - fused_position
            for horizon_index in range(len(values)):
                fused_covariance[horizon_index] += self.mode_probabilities[
                    mode_index
                ] * (
                    mode_covariances[mode_index, horizon_index]
                    + np.outer(
                        delta[horizon_index], delta[horizon_index]
                    )
                )
        return TrajectoryPrediction(
            horizons=values.astype(np.float32),
            position=fused_position.astype(np.float32),
            covariance=fused_covariance.astype(np.float32),
            mode_positions={
                name: mode_positions[index].astype(np.float32)
                for index, name in enumerate(MODE_NAMES)
            },
            mode_probabilities={
                name: float(self.mode_probabilities[index])
                for index, name in enumerate(MODE_NAMES)
            },
        )

    def predict_positions(self, horizons: Iterable[float]) -> np.ndarray:
        """只计算融合位置，供 10 Hz 在线制导和数据集构造使用。"""

        values = np.asarray(tuple(horizons), dtype=np.float64)
        if values.ndim != 1 or np.any(values < 0.0):
            raise ValueError("horizons 必须是一维非负序列")
        mode_position = np.zeros(
            (3, len(values), 3), dtype=np.float64
        )
        for mode_index in range(3):
            for horizon_index, horizon in enumerate(values):
                mode_position[mode_index, horizon_index] = self._dynamics(
                    mode_index,
                    self.mode_states[mode_index],
                    float(horizon),
                )[0:3]
        return np.sum(
            self.mode_probabilities[:, None, None] * mode_position,
            axis=0,
        ).astype(np.float32)
