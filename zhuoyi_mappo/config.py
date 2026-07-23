from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class EnvConfig:
    """随机运动学环境参数。

    所有位置均为相对坐标，训练环境不包含比赛场地的预设位置点。
    真实平台接入前，需要用 ROS 实测数据标定速度、响应时间和碰撞半径。
    """

    num_agents: int = 10
    num_targets: int = 10
    dt: float = 0.2
    max_steps: int = 2500
    arena_half_size: float = 1500.0
    safety_radius: float = 30.0

    # 2026-07-23 low 实机基线以 20 m/s、5 m/s^2 安全完成 10/10。
    interceptor_max_speed: float = 20.0
    interceptor_max_acceleration: float = 5.0
    interceptor_response_tau: float = 0.8
    agent_takeoff_altitude: float = 30.0
    agent_climb_speed: float = 3.0

    target_speed_low: float = 12.0
    target_speed_mid: float = 16.0
    target_speed_high: float = 20.0
    target_range_min: float = 900.0
    target_range_max: float = 1300.0
    target_lateral_spread: float = 140.0
    target_spawn_altitude_min: float = 28.0
    target_spawn_altitude_max: float = 38.0
    target_cruise_altitude_min: float = 175.0
    target_cruise_altitude_max: float = 255.0
    target_breakout_delay_min: float = 25.0
    target_breakout_delay_max: float = 35.0
    target_climb_speed: float = 8.0

    # 裁判日志中的命中几何距离约 0.5--1.4 m。
    collision_radius_min: float = 0.5
    collision_radius_max: float = 1.6
    terminal_guidance_distance: float = 60.0
    terminal_guidance_gain: float = 0.7
    # high 实测中多个漏失目标在截止前已进入 2--4 m，但闭合速度不足。
    # 仅在更近的 15 m 内提高增益，不影响 low/mid 已验证的末制导。
    terminal_guidance_distance_high: float = 15.0
    terminal_guidance_gain_high: float = 2.0
    lead_prediction_xy_low: float = 20.0
    lead_prediction_xy_mid: float = 12.0
    lead_prediction_xy_high: float = 20.0
    lead_prediction_z_low: float = 20.0
    lead_prediction_z_mid: float = 2.0
    lead_prediction_z_high: float = 20.0
    assignment_switch_penalty_s: float = 1.0
    infeasible_intercept_penalty: float = 4.0

    # MAPPO 只修正经典制导速度，防止随机策略完全接管飞控。
    residual_speed_fraction: float = 0.15
    residual_speed_fraction_mid: float = 0.30
    residual_speed_fraction_high: float = 1.00

    # 团队奖励；比赛排名以拦截数优先，因此漏失惩罚高于时间惩罚。
    hit_reward: float = 20.0
    escape_penalty: float = 24.0
    progress_reward_scale: float = 0.015
    time_penalty: float = 0.004
    residual_penalty: float = 0.03
    close_friend_penalty: float = 0.02
    friendly_separation: float = 8.0

    @property
    def obs_dim(self) -> int:
        # rel_pos(3), rel_vel(3), self_vel(3), guide_vel(3), tti(1),
        # target_valid(1), remain(1), time(1), nearest_friend(3),
        # assignment_changed(1), self_alive(1), difficulty_one_hot(3)
        return 24

    def residual_fraction(self, difficulty: str) -> float:
        return {
            "low": self.residual_speed_fraction,
            "mid": self.residual_speed_fraction_mid,
            "high": self.residual_speed_fraction_high,
        }[difficulty]

    def lead_prediction_horizons(self, difficulty: str) -> tuple[float, float]:
        return {
            "low": (self.lead_prediction_xy_low, self.lead_prediction_z_low),
            "mid": (self.lead_prediction_xy_mid, self.lead_prediction_z_mid),
            "high": (self.lead_prediction_xy_high, self.lead_prediction_z_high),
        }[difficulty]

    def terminal_guidance_params(self, difficulty: str) -> tuple[float, float]:
        if difficulty == "high":
            return (
                self.terminal_guidance_distance_high,
                self.terminal_guidance_gain_high,
            )
        return self.terminal_guidance_distance, self.terminal_guidance_gain

    @property
    def global_state_dim(self) -> int:
        # 每架拦截机/靶机各 7 维（位置、速度、有效位），外加时间与难度。
        return self.num_agents * 7 + self.num_targets * 7 + 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    seed: int = 1
    total_updates: int = 300
    num_envs: int = 8
    rollout_steps: int = 256
    update_epochs: int = 5
    num_minibatches: int = 8

    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    actor_lr: float = 1.0e-4
    critic_lr: float = 5.0e-4
    entropy_coef: float = 1.0e-4
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    hidden_dim: int = 256
    device: str = "auto"
    log_interval: int = 1
    save_interval: int = 25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
