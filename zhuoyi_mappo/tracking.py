from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class AlphaBetaTrack:
    """适合 10 Hz 雷达位置的轻量 α-β 跟踪器。"""

    alpha: float = 0.65
    beta: float = 0.12
    min_dt: float = 0.02
    max_dt: float = 0.5
    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    last_time: Optional[float] = None
    samples: int = 0

    def update(self, measurement: np.ndarray, timestamp: float) -> None:
        value = np.asarray(measurement, dtype=np.float64)
        now = float(timestamp)
        if self.last_time is None or now <= self.last_time:
            self.position[:] = value
            self.velocity[:] = 0.0
            self.last_time = now
            self.samples = 1
            return

        dt = float(np.clip(now - self.last_time, self.min_dt, self.max_dt))
        prediction = self.position + self.velocity * dt
        innovation = value - prediction
        self.position[:] = prediction + self.alpha * innovation
        self.velocity[:] = self.velocity + self.beta / dt * innovation
        self.last_time = now
        self.samples += 1

    def predict(self, timestamp: float, max_horizon: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        if self.last_time is None:
            return self.position.copy(), self.velocity.copy()
        horizon = float(np.clip(float(timestamp) - self.last_time, 0.0, max_horizon))
        return (
            (self.position + self.velocity * horizon).astype(np.float32),
            self.velocity.astype(np.float32),
        )

    def age(self, timestamp: float) -> float:
        if self.last_time is None:
            return float("inf")
        return max(0.0, float(timestamp) - self.last_time)


class TargetRetirementDetector:
    """从公开雷达水平运动状态识别已完成突防并停住的目标。

    击中目标会停止发布，由 target_active 超时处理；逃逸目标仍持续
    发布，但会从突防速度降为静止。只有“曾明显运动且持续静止”的
    目标才会退役；调用方应传入水平速度，避免把垂直爬升后的悬停
    误判为逃逸。
    """

    def __init__(
        self,
        count: int,
        moving_speed: float = 5.0,
        stationary_speed: float = 0.5,
        stationary_grace: float = 1.5,
    ):
        self.moving_speed = float(moving_speed)
        self.stationary_speed = float(stationary_speed)
        self.stationary_grace = float(stationary_grace)
        self.moved = np.zeros(count, dtype=bool)
        self.retired = np.zeros(count, dtype=bool)
        self.stationary_since = np.full(count, np.inf, dtype=np.float64)

    def update(
        self,
        speed: np.ndarray,
        active: np.ndarray,
        timestamp: float,
    ) -> np.ndarray:
        values = np.asarray(speed, dtype=np.float64)
        active_mask = np.asarray(active, dtype=bool)
        now = float(timestamp)
        self.moved |= active_mask & (values >= self.moving_speed)
        candidates = (
            active_mask
            & self.moved
            & ~self.retired
            & (values <= self.stationary_speed)
        )
        started = candidates & ~np.isfinite(self.stationary_since)
        self.stationary_since[started] = now
        self.stationary_since[~candidates & ~self.retired] = np.inf
        self.retired |= candidates & (
            (now - self.stationary_since) >= self.stationary_grace
        )
        return self.retired.copy()


class VelocityLimiter:
    """同时限制速度模长和每周期速度变化。"""

    def __init__(self, count: int, max_speed: float, max_acceleration: float):
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.value = np.zeros((count, 3), dtype=np.float32)
        self.initialized = np.zeros(count, dtype=bool)

    @staticmethod
    def _clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1.0e-8))
        return vectors * scale

    def reset(self) -> None:
        self.value[:] = 0.0
        self.initialized[:] = False

    def update(
        self, desired: np.ndarray, dt: float, active: Optional[np.ndarray] = None
    ) -> np.ndarray:
        desired_value = self._clip_norm(
            np.asarray(desired, dtype=np.float32), self.max_speed
        )
        if active is None:
            active_mask = np.ones(len(self.value), dtype=bool)
        else:
            active_mask = np.asarray(active, dtype=bool)

        first = active_mask & ~self.initialized
        # 首帧也从零开始做加速度限制，避免接管时速度指令跳变。
        self.value[first] = 0.0
        self.initialized[first] = True
        delta = desired_value - self.value
        delta = self._clip_norm(delta, self.max_acceleration * max(float(dt), 1.0e-3))
        self.value[active_mask] += delta[active_mask]
        self.value[~active_mask] = 0.0
        self.value = self._clip_norm(self.value, self.max_speed).astype(np.float32)
        return self.value.copy()
