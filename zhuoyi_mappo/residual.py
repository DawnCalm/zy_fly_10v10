from __future__ import annotations

from typing import Tuple

import numpy as np


def guidance_frame_axes(
    guide_velocity: np.ndarray,
    relative_position: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """构造随经典制导方向旋转的水平前向/侧向单位向量。"""

    guide = np.asarray(guide_velocity, dtype=np.float32)
    rel_pos = np.asarray(relative_position, dtype=np.float32)
    heading = guide[..., :2].copy()
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    fallback = rel_pos[..., :2]
    use_fallback = heading_norm[..., 0] < 1.0e-6
    heading[use_fallback] = fallback[use_fallback]
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    use_default = heading_norm[..., 0] < 1.0e-6
    heading[use_default] = np.array([1.0, 0.0], dtype=np.float32)
    heading /= np.maximum(
        np.linalg.norm(heading, axis=-1, keepdims=True), 1.0e-6
    )
    lateral = np.stack((-heading[..., 1], heading[..., 0]), axis=-1)
    return heading.astype(np.float32), lateral.astype(np.float32)


def to_guidance_frame(
    vectors: np.ndarray,
    guide_velocity: np.ndarray,
    relative_position: np.ndarray,
) -> np.ndarray:
    """将世界坐标向量转换为前向、侧向、垂向分量。"""

    values = np.asarray(vectors, dtype=np.float32)
    forward, lateral = guidance_frame_axes(guide_velocity, relative_position)
    return np.stack(
        (
            np.sum(values[..., :2] * forward, axis=-1),
            np.sum(values[..., :2] * lateral, axis=-1),
            values[..., 2],
        ),
        axis=-1,
    ).astype(np.float32)


def residual_gate(
    relative_position: np.ndarray,
    activation_distance: float,
    full_distance: float,
) -> np.ndarray:
    """远距离为 0，进入 activation 后线性放开，到 full 时为 1。"""

    rel_pos = np.asarray(relative_position, dtype=np.float32)
    if activation_distance <= 0.0:
        return np.ones(rel_pos.shape[:-1], dtype=np.float32)
    distance = np.linalg.norm(rel_pos, axis=-1)
    if full_distance >= activation_distance:
        return (distance <= activation_distance).astype(np.float32)
    span = max(float(activation_distance - full_distance), 1.0e-6)
    return np.clip(
        (float(activation_distance) - distance) / span, 0.0, 1.0
    ).astype(np.float32)


def residual_action_to_world(
    action: np.ndarray,
    guide_velocity: np.ndarray,
    relative_position: np.ndarray,
    max_speed: float,
    residual_fraction: float,
    activation_distance: float,
    full_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """把策略的前/侧/垂残差转换为带距离门控的世界坐标速度。"""

    policy_action = np.asarray(action, dtype=np.float32)
    rel_pos = np.asarray(relative_position, dtype=np.float32)
    forward, lateral = guidance_frame_axes(guide_velocity, rel_pos)
    world = np.zeros_like(policy_action, dtype=np.float32)
    world[..., :2] = (
        policy_action[..., 0, None] * forward
        + policy_action[..., 1, None] * lateral
    )
    world[..., 2] = policy_action[..., 2]
    gate = residual_gate(rel_pos, activation_distance, full_distance)
    world *= (
        float(max_speed)
        * float(residual_fraction)
        * gate[..., None]
    )
    return world.astype(np.float32), gate


def terminal_observation_features(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    guide_velocity: np.ndarray,
    max_speed: float,
    activation_distance: float,
    full_distance: float,
) -> np.ndarray:
    """提高 250 m 内几何分辨率，避免 1500 m 归一化淹没末端误差。"""

    rel_pos = np.asarray(relative_position, dtype=np.float32)
    rel_vel = np.asarray(relative_velocity, dtype=np.float32)
    local_scale = max(float(activation_distance), 1.0)
    local_pos = to_guidance_frame(rel_pos, guide_velocity, rel_pos) / local_scale
    local_vel = (
        to_guidance_frame(rel_vel, guide_velocity, rel_pos)
        / max(float(max_speed), 1.0e-6)
    )
    distance = np.linalg.norm(rel_pos, axis=-1) / local_scale
    gate = residual_gate(rel_pos, activation_distance, full_distance)
    return np.concatenate(
        (
            np.clip(local_pos, -6.0, 6.0),
            np.clip(local_vel, -3.0, 3.0),
            np.clip(distance[..., None], 0.0, 6.0),
            gate[..., None],
        ),
        axis=-1,
    ).astype(np.float32)
