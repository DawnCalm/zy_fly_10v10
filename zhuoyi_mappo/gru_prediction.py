from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


DEFAULT_HORIZONS = (0.5, 1.0, 2.0, 3.0)


def history_to_local_features(
    timestamps: np.ndarray,
    positions: np.ndarray,
    history_steps: int = 20,
    sample_dt: float = 0.2,
    position_scale: float = 100.0,
    velocity_scale: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """把世界坐标历史重采样到以当前航向为 x 轴的局部坐标系。"""

    times = np.asarray(timestamps, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1 or points.shape != (len(times), 3):
        raise ValueError("timestamps/positions 形状不匹配")
    if history_steps < 2 or sample_dt <= 0.0:
        raise ValueError("history_steps 至少为 2 且 sample_dt 必须为正")
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("轨迹时间必须严格递增")
    query = times[-1] - (
        np.arange(history_steps - 1, -1, -1, dtype=np.float64) * sample_dt
    )
    if query[0] < times[0] - 1.0e-6:
        raise ValueError("历史长度不足")
    sampled = np.stack(
        [
            np.interp(query, times, points[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    velocity = np.gradient(sampled, sample_dt, axis=0)
    recent_velocity = np.mean(velocity[-min(4, history_steps) :], axis=0)
    horizontal_speed = float(np.linalg.norm(recent_velocity[:2]))
    if horizontal_speed > 1.0e-4:
        forward = recent_velocity[:2] / horizontal_speed
    else:
        forward = np.array([1.0, 0.0], dtype=np.float64)
    left = np.array([-forward[1], forward[0]], dtype=np.float64)
    local_to_world = np.array(
        [
            [forward[0], left[0], 0.0],
            [forward[1], left[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local_position = (sampled - sampled[-1]) @ local_to_world
    local_velocity = velocity @ local_to_world
    features = np.concatenate(
        (
            local_position / float(position_scale),
            local_velocity / float(velocity_scale),
        ),
        axis=1,
    )
    return features.astype(np.float32), local_to_world.astype(np.float32)


class GRUResidualModel(nn.Module):
    """小型共享 GRU，只预测物理轨迹的局部坐标残差。"""

    def __init__(
        self,
        horizon_count: int = len(DEFAULT_HORIZONS),
        hidden_dim: int = 64,
        input_dim: int = 6,
    ):
        super().__init__()
        self.horizon_count = int(horizon_count)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon_count * 3),
            nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(history)
        return self.head(sequence[:, -1]).reshape(
            -1, self.horizon_count, 3
        )


@dataclass(frozen=True)
class HybridPrediction:
    position: np.ndarray
    residual: np.ndarray
    used_gru: bool


class GRUResidualPredictor:
    """在 IMM 等物理基线之上叠加受限 GRU 残差。"""

    def __init__(
        self,
        model: GRUResidualModel,
        horizons: Iterable[float] = DEFAULT_HORIZONS,
        history_steps: int = 20,
        sample_dt: float = 0.2,
        residual_scale: float = 20.0,
        residual_gain: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(torch.device(device)).eval()
        self.horizons = np.asarray(tuple(horizons), dtype=np.float32)
        self.history_steps = int(history_steps)
        self.sample_dt = float(sample_dt)
        self.residual_scale = float(residual_scale)
        self.residual_gain = float(residual_gain)
        self.device = torch.device(device)

    def predict(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        base_prediction: np.ndarray,
        horizons: Iterable[float],
    ) -> HybridPrediction:
        requested = np.asarray(tuple(horizons), dtype=np.float32)
        base = np.asarray(base_prediction, dtype=np.float32)
        if base.shape != (len(requested), 3):
            raise ValueError("base_prediction 形状应为 [horizon, 3]")
        if requested.shape != self.horizons.shape or not np.allclose(
            requested, self.horizons, atol=1.0e-5
        ):
            raise ValueError("请求的预测时域与 GRU 检查点不一致")
        try:
            features, local_to_world = history_to_local_features(
                timestamps,
                positions,
                history_steps=self.history_steps,
                sample_dt=self.sample_dt,
            )
            with torch.no_grad():
                normalized = self.model(
                    torch.as_tensor(
                        features[None],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )[0].cpu().numpy()
            local_residual = (
                normalized * self.residual_scale * self.residual_gain
            )
            world_residual = local_residual @ local_to_world.T
            if not np.isfinite(world_residual).all():
                raise ValueError("GRU 输出非有限数")
            return HybridPrediction(
                position=(base + world_residual).astype(np.float32),
                residual=world_residual.astype(np.float32),
                used_gru=True,
            )
        except (ValueError, RuntimeError):
            return HybridPrediction(
                position=base.copy(),
                residual=np.zeros_like(base),
                used_gru=False,
            )

    @classmethod
    def load(
        cls, checkpoint: Path, device: str = "cpu"
    ) -> "GRUResidualPredictor":
        payload = torch.load(
            Path(checkpoint), map_location=torch.device(device), weights_only=False
        )
        config = payload["model_config"]
        model = GRUResidualModel(
            horizon_count=len(config["horizons"]),
            hidden_dim=int(config["hidden_dim"]),
            input_dim=6,
        )
        model.load_state_dict(payload["model_state"])
        return cls(
            model=model,
            horizons=config["horizons"],
            history_steps=int(config["history_steps"]),
            sample_dt=float(config["sample_dt"]),
            residual_scale=float(config["residual_scale"]),
            residual_gain=float(config.get("residual_gain", 1.0)),
            device=device,
        )
