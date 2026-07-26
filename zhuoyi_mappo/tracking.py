from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class PositionContinuityGuard:
    """拒绝不可能的里程计位置跳变，并将对应实体隔离到本局结束。

    ``update`` 返回当前测量是否可供控制器使用。乱序或时间戳重复的
    测量会被拒绝，但不会触发永久隔离；只有在时间单调前进时，同时
    超过最小跳变距离和物理可达距离的测量才会触发隔离。
    """

    max_speed: float = 60.0
    min_jump_distance: float = 50.0
    distance_margin: float = 5.0
    last_position: Optional[np.ndarray] = field(default=None, init=False)
    last_time: Optional[float] = field(default=None, init=False)
    quarantined: bool = field(default=False, init=False)
    quarantine_reason: Optional[str] = field(default=None, init=False)
    last_rejection_reason: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.max_speed = float(self.max_speed)
        self.min_jump_distance = float(self.min_jump_distance)
        self.distance_margin = float(self.distance_margin)
        if self.max_speed < 0.0:
            raise ValueError("max_speed must be non-negative")
        if self.min_jump_distance < 0.0:
            raise ValueError("min_jump_distance must be non-negative")
        if self.distance_margin < 0.0:
            raise ValueError("distance_margin must be non-negative")

    @property
    def initialized(self) -> bool:
        return self.last_position is not None

    def reset(self) -> None:
        """清除历史和隔离状态，供新一局开始时调用。"""

        self.last_position = None
        self.last_time = None
        self.quarantined = False
        self.quarantine_reason = None
        self.last_rejection_reason = None

    def quarantine(self, reason: str) -> None:
        """因同一状态源的其他无效字段而隔离实体。"""

        self._quarantine(str(reason))

    def update(self, position: np.ndarray, timestamp: float) -> bool:
        """记录一帧位置；返回 ``True`` 表示该帧可安全使用。"""

        if self.quarantined:
            self.last_rejection_reason = self.quarantine_reason
            return False

        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {value.shape}")

        now = float(timestamp)
        if not np.isfinite(now) or not np.isfinite(value).all():
            self._quarantine("non-finite position or timestamp")
            return False

        if self.last_position is None or self.last_time is None:
            self.last_position = value.copy()
            self.last_time = now
            self.last_rejection_reason = None
            return True

        if now <= self.last_time:
            self.last_rejection_reason = "non-monotonic timestamp"
            return False

        dt = now - self.last_time
        distance = float(np.linalg.norm(value - self.last_position))
        physical_limit = self.max_speed * dt + self.distance_margin
        if (
            distance >= self.min_jump_distance
            and distance > physical_limit
        ):
            self._quarantine(
                "impossible position jump: "
                f"{distance:.3f} m in {dt:.3f} s "
                f"(limit {physical_limit:.3f} m)"
            )
            return False

        self.last_position = value.copy()
        self.last_time = now
        self.last_rejection_reason = None
        return True

    def _quarantine(self, reason: str) -> None:
        self.quarantined = True
        self.quarantine_reason = reason
        self.last_rejection_reason = reason


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

        # 丢帧后必须传播完整间隔；截短 dt 却推进 last_time 会留下系统性
        # 位置滞后。min_dt 只防止极短到达间隔放大速度修正。
        dt = max(now - self.last_time, self.min_dt)
        prediction = self.position + self.velocity * dt # 预测位置
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


class TargetReacquisitionHold:
    """短暂屏蔽疑似与已重置拦截机同时消失的目标。

    若目标在 hold 之后又收到一帧雷达数据，立即解除屏蔽；否则保持到
    目标自身超时。这样既不会把已命中的幽灵目标重新分配，也不会长期
    丢弃仍存活的目标。
    """

    def __init__(self, count: int):
        self.held = np.zeros(int(count), dtype=bool)
        self.reference_time = np.full(
            int(count), -np.inf, dtype=np.float64
        )

    def hold(self, target_id: int, last_measurement_time: float) -> None:
        index = int(target_id)
        if index < 0 or index >= len(self.held):
            return
        self.held[index] = True
        self.reference_time[index] = float(last_measurement_time)

    def filter(
        self,
        active: np.ndarray,
        last_measurement_time: np.ndarray,
    ) -> np.ndarray:
        active_mask = np.asarray(active, dtype=bool)
        measurement_time = np.asarray(
            last_measurement_time, dtype=np.float64
        )
        if active_mask.shape != self.held.shape:
            raise ValueError("active shape does not match target count")
        if measurement_time.shape != self.held.shape:
            raise ValueError(
                "last_measurement_time shape does not match target count"
            )
        refreshed = self.held & (
            measurement_time > self.reference_time + 1.0e-9
        )
        self.held[refreshed | ~active_mask] = False
        return active_mask & ~self.held


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
