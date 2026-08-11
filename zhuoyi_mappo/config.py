from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """比赛控制器唯一配置源。"""

    num_agents: int = 10
    num_targets: int = 10
    safety_radius: float = 30.0

    interceptor_max_speed: float = 20.0
    interceptor_max_acceleration: float = 5.0

    terminal_guidance_distance: float = 60.0
    terminal_guidance_gain: float = 0.7
    terminal_guidance_distance_high: float = 15.0
    terminal_guidance_gain_high: float = 2.0
    # 实机总分有收益但存在单 seed -3 回归，默认保持 0 并仅显式启用。
    terminal_minimum_closing_speed_high: float = 0.0

    lead_prediction_xy_low: float = 20.0
    lead_prediction_xy_mid: float = 12.0
    lead_prediction_xy_high: float = 20.0
    lead_prediction_z_low: float = 20.0
    lead_prediction_z_mid: float = 2.0
    lead_prediction_z_high: float = 20.0

    assignment_switch_penalty_s: float = 1.0
    infeasible_intercept_penalty: float = 4.0

    los_pn_navigation_constant: float = 3.0
    los_pn_activation_time_to_go: float = 6.0
    los_pn_full_time_to_go: float = 4.0
    los_pn_response_lead_seconds: float = 2.5
    los_pn_minimum_closing_speed: float = 1.0
    los_pn_regularization_distance: float = 5.0
    los_pn_close_fade_distance: float = 5.0
    los_pn_close_cutoff_distance: float = 1.0

    def lead_prediction_horizons(
        self, difficulty: str
    ) -> tuple[float, float]:
        return {
            "low": (self.lead_prediction_xy_low, self.lead_prediction_z_low),
            "mid": (self.lead_prediction_xy_mid, self.lead_prediction_z_mid),
            "high": (
                self.lead_prediction_xy_high,
                self.lead_prediction_z_high,
            ),
        }[difficulty]

    def terminal_guidance_params(
        self, difficulty: str
    ) -> tuple[float, float, float]:
        if difficulty == "high":
            return (
                self.terminal_guidance_distance_high,
                self.terminal_guidance_gain_high,
                self.terminal_minimum_closing_speed_high,
            )
        return (
            self.terminal_guidance_distance,
            self.terminal_guidance_gain,
            0.0,
        )

# 兼容原有导入名，避免外部启动脚本失效。
EnvConfig = ControllerConfig
