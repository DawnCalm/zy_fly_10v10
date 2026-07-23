from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    return module


class SharedActor(nn.Module):
    """所有拦截机共享的连续动作策略。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = _orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        # 初始残差接近零，降低随机策略破坏经典制导的程度。
        nn.init.constant_(self.mean_head.weight, 0.0)
        nn.init.constant_(self.mean_head.bias, 0.0)
        # 观测难度条件化后，low 的残差缩放会自动保护经典底座；
        # 较宽探索主要用于 high 中发现前置堵截等非局部追击动作。
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        latent = self.backbone(obs)
        mean = self.mean_head(latent)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp())

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        pre_tanh: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        base_log_prob = distribution.log_prob(pre_tanh)
        correction = torch.log(1.0 - action.square() + 1.0e-6)
        return (base_log_prob - correction).sum(dim=-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, action)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        safe_action = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        pre_tanh = torch.atanh(safe_action)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, safe_action)
        # 采用基础高斯熵作为稳定的探索强度代理。
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralCritic(nn.Module):
    """集中式 Critic：输入完整态势，为每架拦截机输出 V_i。"""

    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            _orthogonal_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, num_agents), gain=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MAPPOPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor = SharedActor(obs_dim, action_dim, hidden_dim)
        self.critic = CentralCritic(state_dim, num_agents, hidden_dim)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_probs = self.actor.act(obs, deterministic=deterministic)
        values = self.critic(state)
        return actions, log_probs, values
