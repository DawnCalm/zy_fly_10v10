from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .assignment import assign_targets, lead_velocity
from .config import EnvConfig
from .residual import (
    residual_action_to_world,
    terminal_observation_features,
)


DIFFICULTIES = ("low", "mid", "high")


@dataclass
class EpisodeStats:
    hits: int = 0
    escapes: int = 0
    elapsed_steps: int = 0


def _clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1.0e-8))
    return vectors * scale


class Kinematic10v10Env:
    """用于从零训练的轻量 10v10 随机运动学环境。

    匈牙利算法提供一对一分配，经典提前量制导给出基础速度；MAPPO
    动作仅作为幅度受限的三维残差。环境模拟拦截阶段，不模拟 PX4
    解锁和起飞状态机。
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: int = 1):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self._seed = int(seed)

        n, m = self.cfg.num_agents, self.cfg.num_targets
        self.agent_pos = np.zeros((n, 3), dtype=np.float32)
        self.agent_origin_z = np.zeros(n, dtype=np.float32)
        self.agent_vel = np.zeros((n, 3), dtype=np.float32)
        self.agent_alive = np.ones(n, dtype=bool)
        self.target_pos = np.zeros((m, 3), dtype=np.float32)
        self.target_vel = np.zeros((m, 3), dtype=np.float32)
        self.target_alive = np.ones(m, dtype=bool)

        self.assignment = np.full(n, -1, dtype=np.int64)
        self.previous_assignment = np.full(n, -1, dtype=np.int64)
        self.assignment_changed = np.zeros(n, dtype=np.float32)
        self.target_phase = np.zeros(m, dtype=np.float32)
        self.target_base_alt = np.zeros(m, dtype=np.float32)
        self.target_speed_scale = np.ones(m, dtype=np.float32)
        self.target_breakout_delay = self.cfg.target_breakout_delay_min
        self.safe_center = np.zeros(3, dtype=np.float32)
        self.collision_radius = self.cfg.collision_radius_min
        self.interceptor_response_tau = self.cfg.interceptor_response_tau
        self.difficulty = "low"
        self.step_count = 0
        self.stats = EpisodeStats()
        self._previous_assigned_distance = 0.0

    def reset(
        self, seed: Optional[int] = None, difficulty: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        self.difficulty = difficulty or str(self.rng.choice(DIFFICULTIES))
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"未知难度: {self.difficulty}")

        n, m = self.cfg.num_agents, self.cfg.num_targets
        self.step_count = 0
        self.stats = EpisodeStats()
        self.agent_alive[:] = True
        self.target_alive[:] = True
        self.agent_vel[:] = 0.0

        # 每回合随机生成相对几何关系；不依赖比赛场地预设坐标。
        self.safe_center[:] = (
            self.rng.uniform(-80.0, 80.0),
            self.rng.uniform(-80.0, 80.0),
            0.0,
        )
        formation_angle = float(self.rng.uniform(-np.pi, np.pi))
        forward = np.array(
            [np.cos(formation_angle), np.sin(formation_angle)], dtype=np.float32
        )
        lateral = np.array([-forward[1], forward[0]], dtype=np.float32)

        agent_offsets_forward = self.rng.uniform(-20.0, 20.0, size=n)
        agent_offsets_lateral = self.rng.uniform(-45.0, 45.0, size=n)
        self.agent_pos[:, :2] = (
            self.safe_center[:2]
            + agent_offsets_forward[:, None] * forward
            + agent_offsets_lateral[:, None] * lateral
        )
        self.agent_pos[:, 2] = self.rng.uniform(38.0, 42.0, size=n)
        self.agent_origin_z[:] = self.agent_pos[:, 2]

        target_ranges = self.rng.uniform(
            self.cfg.target_range_min, self.cfg.target_range_max, size=m
        )
        target_lateral = self.rng.uniform(
            -self.cfg.target_lateral_spread,
            self.cfg.target_lateral_spread,
            size=m,
        )
        self.target_pos[:, :2] = (
            self.safe_center[:2]
            + target_ranges[:, None] * forward
            + target_lateral[:, None] * lateral
        )
        self.target_pos[:, 2] = self.rng.uniform(
            self.cfg.target_spawn_altitude_min,
            self.cfg.target_spawn_altitude_max,
            size=m,
        )
        self.target_base_alt[:] = self.rng.uniform(
            self.cfg.target_cruise_altitude_min,
            self.cfg.target_cruise_altitude_max,
            size=m,
        )
        self.target_breakout_delay = float(
            self.rng.uniform(
                self.cfg.target_breakout_delay_min,
                self.cfg.target_breakout_delay_max,
            )
        )
        self.target_phase[:] = self.rng.uniform(0.0, 2.0 * np.pi, size=m)
        self.target_speed_scale[:] = self.rng.uniform(0.9, 1.1, size=m)
        self.collision_radius = float(
            self.rng.uniform(
                self.cfg.collision_radius_min, self.cfg.collision_radius_max
            )
        )
        response_min = min(
            self.cfg.interceptor_response_tau_min,
            self.cfg.interceptor_response_tau_max,
        )
        response_max = max(
            self.cfg.interceptor_response_tau_min,
            self.cfg.interceptor_response_tau_max,
        )
        self.interceptor_response_tau = float(
            self.rng.uniform(response_min, response_max)
            if response_max > response_min
            else response_min
        )

        self._update_target_velocity(initial=True)
        self.assignment[:] = -1
        self.previous_assignment[:] = -1
        self._update_assignment()
        self._previous_assigned_distance = self._assigned_distance()
        return self._observation(), self._global_state()

    def _base_target_speed(self) -> float:
        return {
            "low": self.cfg.target_speed_low,
            "mid": self.cfg.target_speed_mid,
            "high": self.cfg.target_speed_high,
        }[self.difficulty]

    def _target_deadline(self) -> np.ndarray:
        delta_xy = self.target_pos[:, :2] - self.safe_center[:2]
        distance = np.maximum(
            0.0, np.linalg.norm(delta_xy, axis=1) - self.cfg.safety_radius
        )
        horizontal_speed = np.maximum(
            np.linalg.norm(self.target_vel[:, :2], axis=1), 1.0
        )
        return distance / horizontal_speed

    def _update_assignment(self) -> None:
        new_assignment, _ = assign_targets(
            self.agent_pos,
            self.agent_alive,
            self.target_pos,
            self.target_vel,
            self.target_alive,
            self.cfg.interceptor_max_speed,
            previous_assignment=self.assignment,
            switch_penalty_s=self.cfg.assignment_switch_penalty_s,
            target_deadline_s=self._target_deadline(),
            infeasible_penalty=self.cfg.infeasible_intercept_penalty,
        )
        self.previous_assignment[:] = self.assignment
        self.assignment[:] = new_assignment
        self.assignment_changed[:] = (
            (self.previous_assignment >= 0)
            & (self.assignment != self.previous_assignment)
        ).astype(np.float32)

    def _guide_velocity_and_tti(self) -> Tuple[np.ndarray, np.ndarray]:
        guide = np.zeros_like(self.agent_vel)
        tti = np.full(self.cfg.num_agents, 60.0, dtype=np.float32)
        prediction_xy, prediction_z = self.cfg.lead_prediction_horizons(
            self.difficulty
        )
        terminal_distance, terminal_gain = self.cfg.terminal_guidance_params(
            self.difficulty
        )
        for agent_id, target_id in enumerate(self.assignment):
            if (
                not self.agent_alive[agent_id]
                or target_id < 0
                or not self.target_alive[target_id]
            ):
                continue
            guide[agent_id], tti[agent_id] = lead_velocity(
                self.agent_pos[agent_id],
                self.target_pos[target_id],
                self.target_vel[target_id],
                self.cfg.interceptor_max_speed,
                max_prediction_s=prediction_xy,
                max_vertical_prediction_s=prediction_z,
                terminal_distance=terminal_distance,
                terminal_gain=terminal_gain,
            )
        return guide, tti

    def _update_target_velocity(self, initial: bool = False) -> None:
        elapsed = self.step_count * self.cfg.dt
        motion_elapsed = max(0.0, elapsed - self.target_breakout_delay)
        to_safe_xy = self.safe_center[:2] - self.target_pos[:, :2]
        xy_norm = np.linalg.norm(to_safe_xy, axis=1, keepdims=True)
        forward = to_safe_xy / np.maximum(xy_norm, 1.0e-6)
        lateral = np.stack((-forward[:, 1], forward[:, 0]), axis=1)
        base_speed = self._base_target_speed() * self.target_speed_scale

        if elapsed < self.target_breakout_delay:
            desired_xy = np.zeros((self.cfg.num_targets, 2), dtype=np.float32)
            desired_alt = self.target_base_alt
        elif self.difficulty == "low":
            desired_xy = forward * base_speed[:, None]
            desired_alt = self.target_base_alt
        elif self.difficulty == "mid":
            desired_xy = forward * base_speed[:, None]
            desired_alt = self.target_base_alt + 35.0 * np.sin(
                0.12 * motion_elapsed + self.target_phase
            )
        else:
            speed_wave = 1.0 + self.cfg.target_high_speed_wave_amplitude * np.sin(
                self.cfg.target_high_speed_wave_frequency
                * motion_elapsed
                + self.target_phase
            )
            weave = self.cfg.target_high_weave_amplitude * np.sin(
                self.cfg.target_high_weave_frequency
                * motion_elapsed
                + self.target_phase
            )
            direction = forward + weave[:, None] * lateral
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-6)
            desired_xy = direction * (base_speed * speed_wave)[:, None]
            desired_alt = (
                self.target_base_alt
                + self.cfg.target_high_altitude_amplitude
                * np.sin(
                    self.cfg.target_high_altitude_frequency
                    * motion_elapsed
                    + self.target_phase
                )
            )

        desired_vz = np.clip(
            (desired_alt - self.target_pos[:, 2]) * 0.35,
            -self.cfg.target_climb_speed,
            self.cfg.target_climb_speed,
        )
        desired = np.column_stack((desired_xy, desired_vz)).astype(np.float32)
        if initial:
            self.target_vel[:] = desired
        else:
            # 靶机速度也采用一阶响应，避免瞬时改变航向。
            blend = min(
                1.0, self.cfg.dt / self.cfg.target_velocity_response_tau
            )
            self.target_vel += blend * (desired - self.target_vel)
        self.target_vel[~self.target_alive] = 0.0

    def _assigned_distance(self) -> float:
        total = 0.0
        for agent_id, target_id in enumerate(self.assignment):
            if (
                self.agent_alive[agent_id]
                and target_id >= 0
                and self.target_alive[target_id]
            ):
                total += float(
                    np.linalg.norm(
                        self.target_pos[target_id] - self.agent_pos[agent_id]
                    )
                )
        return total

    def _resolve_hits(
        self,
        previous_agent_pos: np.ndarray,
        previous_target_pos: np.ndarray,
    ) -> int:
        agent_ids = np.flatnonzero(self.agent_alive)
        target_ids = np.flatnonzero(self.target_alive)
        if len(agent_ids) == 0 or len(target_ids) == 0:
            return 0
        relative_end = (
            self.agent_pos[agent_ids, None, :]
            - self.target_pos[None, target_ids, :]
        )
        relative_start = (
            previous_agent_pos[agent_ids, None, :]
            - previous_target_pos[None, target_ids, :]
        )
        relative_motion = relative_end - relative_start
        denominator = np.sum(np.square(relative_motion), axis=2)
        closest_fraction = np.divide(
            -np.sum(relative_start * relative_motion, axis=2),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1.0e-12,
        )
        closest_fraction = np.clip(closest_fraction, 0.0, 1.0)
        closest_relative = relative_start + (
            closest_fraction[..., None] * relative_motion
        )
        distances = np.linalg.norm(closest_relative, axis=2)
        candidates = np.argwhere(distances <= self.collision_radius)
        if len(candidates) == 0:
            return 0
        candidates = sorted(
            candidates, key=lambda pair: distances[pair[0], pair[1]]
        )
        used_agents, used_targets = set(), set()
        hits = 0
        for row, col in candidates:
            agent_id, target_id = int(agent_ids[row]), int(target_ids[col])
            if agent_id in used_agents or target_id in used_targets:
                continue
            used_agents.add(agent_id)
            used_targets.add(target_id)
            self.agent_alive[agent_id] = False
            self.target_alive[target_id] = False
            self.agent_vel[agent_id] = 0.0
            self.target_vel[target_id] = 0.0
            hits += 1
        return hits

    def _resolve_escapes(self) -> int:
        horizontal_distance = np.linalg.norm(
            self.target_pos[:, :2] - self.safe_center[:2], axis=1
        )
        escaped = self.target_alive & (horizontal_distance <= self.cfg.safety_radius)
        count = int(escaped.sum())
        self.target_alive[escaped] = False
        self.target_vel[escaped] = 0.0
        return count

    def _friend_penalty(self) -> float:
        alive_ids = np.flatnonzero(self.agent_alive)
        if len(alive_ids) < 2:
            return 0.0
        delta = (
            self.agent_pos[alive_ids, None, :]
            - self.agent_pos[None, alive_ids, :]
        )
        distance = np.linalg.norm(delta, axis=2)
        upper = distance[np.triu_indices(len(alive_ids), 1)]
        violations = np.maximum(0.0, self.cfg.friendly_separation - upper)
        return float(violations.sum() * self.cfg.close_friend_penalty)

    def step(
        self, residual_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict[str, float]]:
        action = np.asarray(residual_action, dtype=np.float32)
        expected_shape = (self.cfg.num_agents, 3)
        if action.shape != expected_shape:
            raise ValueError(f"动作形状应为 {expected_shape}，实际为 {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        guide, _ = self._guide_velocity_and_tti()
        assigned_target = self.assignment.copy()
        relative_position = np.zeros_like(self.agent_pos)
        distance_before = np.zeros(self.cfg.num_agents, dtype=np.float32)
        valid_pair = self.agent_alive & (assigned_target >= 0)
        for agent_id in np.flatnonzero(valid_pair):
            target_id = int(assigned_target[agent_id])
            if self.target_alive[target_id]:
                relative_position[agent_id] = (
                    self.target_pos[target_id] - self.agent_pos[agent_id]
                )
                distance_before[agent_id] = np.linalg.norm(
                    relative_position[agent_id]
                )
            else:
                valid_pair[agent_id] = False
        activation_distance, full_distance = self.cfg.residual_gate_distances(
            self.difficulty
        )
        residual, gate = residual_action_to_world(
            action,
            guide,
            relative_position,
            self.cfg.interceptor_max_speed,
            self.cfg.residual_fraction(self.difficulty),
            activation_distance,
            full_distance,
        )
        policy_active = valid_pair.copy()
        residual[~policy_active] = 0.0
        gate[~policy_active] = 0.0
        command = _clip_norm(guide + residual, self.cfg.interceptor_max_speed)
        command[~self.agent_alive] = 0.0

        # 与真实控制器一致：起飞时先爬升，局部高度达到 10 m 后才完全
        # 放开水平速度；达到 takeoff_altitude 后退出起飞保护。
        local_altitude = self.agent_pos[:, 2] - self.agent_origin_z
        taking_off = self.agent_alive & (
            local_altitude < self.cfg.agent_takeoff_altitude
        )
        horizontal_scale = np.clip(local_altitude / 10.0, 0.0, 1.0)
        command[taking_off, :2] *= horizontal_scale[taking_off, None]
        command[taking_off, 2] = np.maximum(
            command[taking_off, 2], self.cfg.agent_climb_speed
        )

        blend = min(1.0, self.cfg.dt / self.interceptor_response_tau)
        velocity_delta = blend * (command - self.agent_vel)
        velocity_delta = _clip_norm(
            velocity_delta,
            self.cfg.interceptor_max_acceleration * self.cfg.dt,
        )
        self.agent_vel += velocity_delta
        self.agent_vel = _clip_norm(
            self.agent_vel, self.cfg.interceptor_max_speed
        ).astype(np.float32)
        previous_agent_pos = self.agent_pos.copy()
        previous_target_pos = self.target_pos.copy()
        self.agent_pos += self.agent_vel * self.cfg.dt

        self._update_target_velocity()
        self.target_pos += self.target_vel * self.cfg.dt
        self.step_count += 1

        hits = self._resolve_hits(previous_agent_pos, previous_target_pos)
        escapes = self._resolve_escapes()
        self.stats.hits += hits
        self.stats.escapes += escapes
        self.stats.elapsed_steps = self.step_count

        individual_progress = np.zeros(self.cfg.num_agents, dtype=np.float32)
        for agent_id in np.flatnonzero(valid_pair):
            target_id = int(assigned_target[agent_id])
            distance_after = np.linalg.norm(
                self.target_pos[target_id] - self.agent_pos[agent_id]
            )
            individual_progress[agent_id] = np.clip(
                distance_before[agent_id] - distance_after,
                -2.0 * self.cfg.interceptor_max_speed * self.cfg.dt,
                2.0 * self.cfg.interceptor_max_speed * self.cfg.dt,
            )
        self._update_assignment()
        self._previous_assigned_distance = self._assigned_distance()

        team_event_reward = (
            hits * self.cfg.hit_reward
            - escapes * self.cfg.escape_penalty
            - self.cfg.time_penalty
            - self._friend_penalty()
        )
        rewards = (
            team_event_reward
            + individual_progress * self.cfg.progress_reward_scale
            - self.cfg.residual_penalty
            * np.mean(np.square(action), axis=1)
            * gate
        ).astype(np.float32)

        done = bool(
            not self.target_alive.any()
            or not self.agent_alive.any()
            or self.step_count >= self.cfg.max_steps
        )
        info: Dict[str, float] = {
            "hits": float(self.stats.hits),
            "escapes": float(self.stats.escapes),
            "episode_steps": float(self.step_count),
            "active_targets": float(self.target_alive.sum()),
            "active_agents": float(self.agent_alive.sum()),
            "difficulty_id": float(DIFFICULTIES.index(self.difficulty)),
        }
        return self._observation(), self._global_state(), rewards, done, info

    def _nearest_friend_relative(self, agent_id: int) -> np.ndarray:
        if not self.agent_alive[agent_id]:
            return np.zeros(3, dtype=np.float32)
        candidates = np.flatnonzero(self.agent_alive)
        candidates = candidates[candidates != agent_id]
        if len(candidates) == 0:
            return np.zeros(3, dtype=np.float32)
        delta = self.agent_pos[candidates] - self.agent_pos[agent_id]
        nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
        return delta[nearest]

    def _observation(self) -> np.ndarray:
        guide, tti = self._guide_velocity_and_tti()
        obs = np.zeros(
            (self.cfg.num_agents, self.cfg.obs_dim), dtype=np.float32
        )
        remaining_fraction = float(self.target_alive.mean())
        time_remaining = 1.0 - self.step_count / self.cfg.max_steps
        for agent_id, target_id in enumerate(self.assignment):
            cursor = 0
            if target_id >= 0 and self.target_alive[target_id]:
                rel_pos = self.target_pos[target_id] - self.agent_pos[agent_id]
                rel_vel = self.target_vel[target_id] - self.agent_vel[agent_id]
                target_valid = 1.0
            else:
                rel_pos = np.zeros(3, dtype=np.float32)
                rel_vel = np.zeros(3, dtype=np.float32)
                target_valid = 0.0

            features = (
                rel_pos / self.cfg.arena_half_size,
                rel_vel / self.cfg.interceptor_max_speed,
                self.agent_vel[agent_id] / self.cfg.interceptor_max_speed,
                guide[agent_id] / self.cfg.interceptor_max_speed,
            )
            for feature in features:
                obs[agent_id, cursor : cursor + 3] = feature
                cursor += 3
            obs[agent_id, cursor] = np.clip(tti[agent_id] / 60.0, 0.0, 2.0)
            obs[agent_id, cursor + 1] = target_valid
            obs[agent_id, cursor + 2] = remaining_fraction
            obs[agent_id, cursor + 3] = time_remaining
            obs[agent_id, cursor + 4 : cursor + 7] = (
                self._nearest_friend_relative(agent_id)
                / self.cfg.arena_half_size
            )
            obs[agent_id, cursor + 7] = self.assignment_changed[agent_id]
            obs[agent_id, cursor + 8] = float(self.agent_alive[agent_id])
            obs[
                agent_id,
                cursor + 9 + DIFFICULTIES.index(self.difficulty),
            ] = 1.0
            activation_distance, full_distance = (
                self.cfg.residual_gate_distances(self.difficulty)
            )
            obs[agent_id, cursor + 12 : cursor + 20] = (
                terminal_observation_features(
                    rel_pos[None],
                    rel_vel[None],
                    guide[agent_id : agent_id + 1],
                    self.cfg.interceptor_max_speed,
                    activation_distance,
                    full_distance,
                )[0]
            )
            if not target_valid:
                obs[agent_id, cursor + 19] = 0.0
        return obs

    def _global_state(self) -> np.ndarray:
        state = np.zeros(self.cfg.global_state_dim, dtype=np.float32)
        cursor = 0
        for agent_id in range(self.cfg.num_agents):
            state[cursor : cursor + 3] = (
                self.agent_pos[agent_id] - self.safe_center
            ) / self.cfg.arena_half_size
            state[cursor + 3 : cursor + 6] = (
                self.agent_vel[agent_id] / self.cfg.interceptor_max_speed
            )
            state[cursor + 6] = float(self.agent_alive[agent_id])
            cursor += 7
        speed_scale = max(
            self.cfg.target_speed_low,
            self.cfg.target_speed_mid,
            self.cfg.target_speed_high,
        )
        for target_id in range(self.cfg.num_targets):
            state[cursor : cursor + 3] = (
                self.target_pos[target_id] - self.safe_center
            ) / self.cfg.arena_half_size
            state[cursor + 3 : cursor + 6] = (
                self.target_vel[target_id] / speed_scale
            )
            state[cursor + 6] = float(self.target_alive[target_id])
            cursor += 7
        state[cursor] = 1.0 - self.step_count / self.cfg.max_steps
        state[cursor + 1 + DIFFICULTIES.index(self.difficulty)] = 1.0
        return state
