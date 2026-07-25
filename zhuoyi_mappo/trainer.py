from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import EnvConfig, TrainConfig
from .env import DIFFICULTIES, Kinematic10v10Env
from .model import MAPPOPolicy


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 无法访问 GPU")
    return device


def set_global_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


class MAPPOTrainer:
    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        train_config: Optional[TrainConfig] = None,
    ):
        self.env_cfg = env_config or EnvConfig()
        self.train_cfg = train_config or TrainConfig()
        self.device = resolve_device(self.train_cfg.device)
        set_global_seed(
            self.train_cfg.seed, seed_cuda=self.device.type == "cuda"
        )

        self.policy = MAPPOPolicy(
            obs_dim=self.env_cfg.obs_dim,
            state_dim=self.env_cfg.global_state_dim,
            num_agents=self.env_cfg.num_agents,
            action_dim=3,
            hidden_dim=self.train_cfg.hidden_dim,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.train_cfg.actor_lr, eps=1.0e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.train_cfg.critic_lr, eps=1.0e-5
        )

        self.envs = [
            Kinematic10v10Env(
                self.env_cfg, seed=self.train_cfg.seed + 1009 * env_id
            )
            for env_id in range(self.train_cfg.num_envs)
        ]
        train_difficulties = self.train_cfg.difficulty_list()
        self.env_difficulties = [
            train_difficulties[env_id % len(train_difficulties)]
            for env_id in range(self.train_cfg.num_envs)
        ]
        reset_results = [
            env.reset(difficulty=difficulty)
            for env, difficulty in zip(self.envs, self.env_difficulties)
        ]
        self.obs = np.stack([item[0] for item in reset_results])
        self.state = np.stack([item[1] for item in reset_results])
        self.completed_episodes: List[Dict[str, float]] = []
        self.update_index = 0

    def _collect_rollout(self) -> Dict[str, np.ndarray]:
        t_steps = self.train_cfg.rollout_steps
        n_envs = self.train_cfg.num_envs
        n_agents = self.env_cfg.num_agents
        obs_dim = self.env_cfg.obs_dim
        state_dim = self.env_cfg.global_state_dim

        buffer = {
            "obs": np.zeros(
                (t_steps, n_envs, n_agents, obs_dim), dtype=np.float32
            ),
            "state": np.zeros(
                (t_steps, n_envs, state_dim), dtype=np.float32
            ),
            "actions": np.zeros(
                (t_steps, n_envs, n_agents, 3), dtype=np.float32
            ),
            "log_probs": np.zeros(
                (t_steps, n_envs, n_agents), dtype=np.float32
            ),
            "values": np.zeros(
                (t_steps, n_envs, n_agents), dtype=np.float32
            ),
            "rewards": np.zeros(
                (t_steps, n_envs, n_agents), dtype=np.float32
            ),
            "dones": np.zeros(
                (t_steps, n_envs, n_agents), dtype=np.float32
            ),
            "active_masks": np.zeros(
                (t_steps, n_envs, n_agents), dtype=np.float32
            ),
            "policy_masks": np.zeros(
                (t_steps, n_envs, n_agents), dtype=np.float32
            ),
        }

        for step in range(t_steps):
            buffer["obs"][step] = self.obs
            buffer["state"][step] = self.state
            buffer["active_masks"][step] = np.stack(
                [
                    (env.agent_alive & (env.assignment >= 0)).astype(np.float32)
                    for env in self.envs
                ]
            )
            buffer["policy_masks"][step] = np.stack(
                [
                    (
                        env.agent_alive
                        & (env.assignment >= 0)
                        & (self.obs[env_id, :, 31] > 1.0e-6)
                    ).astype(np.float32)
                    for env_id, env in enumerate(self.envs)
                ]
            )

            obs_tensor = torch.as_tensor(
                self.obs, dtype=torch.float32, device=self.device
            )
            state_tensor = torch.as_tensor(
                self.state, dtype=torch.float32, device=self.device
            )
            actions, log_probs, values = self.policy.act(
                obs_tensor, state_tensor
            )
            actions_np = actions.cpu().numpy()
            buffer["actions"][step] = actions_np
            buffer["log_probs"][step] = log_probs.cpu().numpy()
            buffer["values"][step] = values.cpu().numpy()

            next_obs, next_state = [], []
            for env_id, env in enumerate(self.envs):
                obs, state, rewards, done, info = env.step(actions_np[env_id])
                buffer["rewards"][step, env_id] = rewards
                if done:
                    buffer["dones"][step, env_id] = 1.0
                    self.completed_episodes.append(dict(info))
                    obs, state = env.reset(
                        difficulty=self.env_difficulties[env_id]
                    )
                next_obs.append(obs)
                next_state.append(state)
            self.obs = np.stack(next_obs)
            self.state = np.stack(next_state)

        with torch.no_grad():
            next_values = self.policy.critic(
                torch.as_tensor(
                    self.state, dtype=torch.float32, device=self.device
                )
            ).cpu().numpy()
        buffer["next_values"] = next_values.astype(np.float32)
        return buffer

    def _compute_gae(
        self, buffer: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = buffer["rewards"]
        values = buffer["values"]
        dones = buffer["dones"]
        advantages = np.zeros_like(rewards)
        last_advantage = np.zeros_like(buffer["next_values"])
        next_values = buffer["next_values"]

        for step in reversed(range(self.train_cfg.rollout_steps)):
            nonterminal = 1.0 - dones[step]
            following_values = (
                next_values
                if step == self.train_cfg.rollout_steps - 1
                else values[step + 1]
            )
            delta = (
                rewards[step]
                + self.train_cfg.gamma * following_values * nonterminal
                - values[step]
            )
            last_advantage = (
                delta
                + self.train_cfg.gamma
                * self.train_cfg.gae_lambda
                * nonterminal
                * last_advantage
            )
            advantages[step] = last_advantage
        returns = advantages + values
        return advantages, returns

    def _update_policy(
        self,
        buffer: Dict[str, np.ndarray],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        # minibatch 单位是“时间步×环境”，每个样本始终保留同一时刻的
        # 10 架智能体，避免旧代码打乱联合状态的问题。
        units = self.train_cfg.rollout_steps * self.train_cfg.num_envs
        n_agents = self.env_cfg.num_agents
        obs = torch.as_tensor(
            buffer["obs"].reshape(units, n_agents, self.env_cfg.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        state = torch.as_tensor(
            buffer["state"].reshape(units, self.env_cfg.global_state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            buffer["actions"].reshape(units, n_agents, 3),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            buffer["log_probs"].reshape(units, n_agents),
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(
            buffer["values"].reshape(units, n_agents),
            dtype=torch.float32,
            device=self.device,
        )
        returns_tensor = torch.as_tensor(
            returns.reshape(units, n_agents),
            dtype=torch.float32,
            device=self.device,
        )
        advantages_tensor = torch.as_tensor(
            advantages.reshape(units, n_agents),
            dtype=torch.float32,
            device=self.device,
        )
        active_masks = torch.as_tensor(
            buffer["active_masks"].reshape(units, n_agents),
            dtype=torch.float32,
            device=self.device,
        )
        policy_masks = torch.as_tensor(
            buffer["policy_masks"].reshape(units, n_agents),
            dtype=torch.float32,
            device=self.device,
        )

        active_advantages = advantages_tensor[policy_masks > 0.5]
        if active_advantages.numel() > 0:
            advantage_mean = active_advantages.mean()
            advantage_std = active_advantages.std(
                unbiased=False
            ).clamp_min(1.0e-6)
            advantages_tensor = (
                advantages_tensor - advantage_mean
            ) / advantage_std
        else:
            advantages_tensor = torch.zeros_like(advantages_tensor)

        minibatch_size = max(1, units // self.train_cfg.num_minibatches)
        metric_sums = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        metric_count = 0

        for _ in range(self.train_cfg.update_epochs):
            permutation = torch.randperm(units, device=self.device)
            for start in range(0, units, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                actor_mask = policy_masks[indices]
                actor_denominator = actor_mask.sum().clamp_min(1.0)

                new_log_probs, entropy = self.policy.actor.evaluate_actions(
                    obs[indices], actions[indices]
                )
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                surrogate_1 = ratio * advantages_tensor[indices]
                surrogate_2 = (
                    ratio.clamp(
                        1.0 - self.train_cfg.clip_ratio,
                        1.0 + self.train_cfg.clip_ratio,
                    )
                    * advantages_tensor[indices]
                )
                actor_loss = (
                    -torch.minimum(surrogate_1, surrogate_2) * actor_mask
                ).sum() / actor_denominator
                entropy_mean = (
                    entropy * actor_mask
                ).sum() / actor_denominator
                total_actor_loss = (
                    actor_loss - self.train_cfg.entropy_coef * entropy_mean
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(),
                    self.train_cfg.max_grad_norm,
                )
                self.actor_optimizer.step()

                values = self.policy.critic(state[indices])
                critic_mask = active_masks[indices]
                critic_denominator = critic_mask.sum().clamp_min(1.0)
                clipped_values = old_values[indices] + (
                    values - old_values[indices]
                ).clamp(
                    -self.train_cfg.value_clip, self.train_cfg.value_clip
                )
                value_loss_unclipped = F.mse_loss(
                    values, returns_tensor[indices], reduction="none"
                )
                value_loss_clipped = F.mse_loss(
                    clipped_values, returns_tensor[indices], reduction="none"
                )
                critic_loss = (
                    torch.maximum(value_loss_unclipped, value_loss_clipped)
                    * critic_mask
                ).sum() / critic_denominator

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.train_cfg.value_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(),
                    self.train_cfg.max_grad_norm,
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1.0) - log_ratio) * actor_mask
                    ).sum() / actor_denominator
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.train_cfg.clip_ratio).float()
                        * actor_mask
                    ).sum() / actor_denominator
                metric_sums["actor_loss"] += float(actor_loss.item())
                metric_sums["critic_loss"] += float(critic_loss.item())
                metric_sums["entropy"] += float(entropy_mean.item())
                metric_sums["approx_kl"] += float(approx_kl.item())
                metric_sums["clip_fraction"] += float(clip_fraction.item())
                metric_count += 1

        return {
            key: value / max(metric_count, 1)
            for key, value in metric_sums.items()
        }

    def train_update(self) -> Dict[str, float]:
        episode_start = len(self.completed_episodes)
        buffer = self._collect_rollout()
        advantages, returns = self._compute_gae(buffer)
        metrics = self._update_policy(buffer, advantages, returns)
        self.update_index += 1

        recent = self.completed_episodes[episode_start:]
        if recent:
            metrics.update(
                {
                    "episodes": float(len(recent)),
                    "mean_hits": float(np.mean([x["hits"] for x in recent])),
                    "mean_escapes": float(
                        np.mean([x["escapes"] for x in recent])
                    ),
                    "mean_steps": float(
                        np.mean([x["episode_steps"] for x in recent])
                    ),
                }
            )
            for difficulty_id, difficulty in enumerate(DIFFICULTIES):
                subset = [
                    row
                    for row in recent
                    if int(row["difficulty_id"]) == difficulty_id
                ]
                metrics[f"{difficulty}_hits"] = (
                    float(np.mean([row["hits"] for row in subset]))
                    if subset
                    else float("nan")
                )
                metrics[f"{difficulty}_escapes"] = (
                    float(np.mean([row["escapes"] for row in subset]))
                    if subset
                    else float("nan")
                )
                metrics[f"{difficulty}_steps"] = (
                    float(np.mean([row["episode_steps"] for row in subset]))
                    if subset
                    else float("nan")
                )
        else:
            metrics.update(
                {
                    "episodes": 0.0,
                    "mean_hits": float("nan"),
                    "mean_escapes": float("nan"),
                    "mean_steps": float("nan"),
                }
            )
            for difficulty in DIFFICULTIES:
                metrics[f"{difficulty}_hits"] = float("nan")
                metrics[f"{difficulty}_escapes"] = float("nan")
                metrics[f"{difficulty}_steps"] = float("nan")
        return metrics

    def checkpoint_dict(self) -> Dict[str, object]:
        return {
            "update": self.update_index,
            "env_config": asdict(self.env_cfg),
            "train_config": asdict(self.train_cfg),
            "policy": self.policy.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(), path)

    def load_training_state(self, path: Path) -> None:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        checkpoint_env = EnvConfig(**checkpoint["env_config"])
        if (
            checkpoint_env.num_agents != self.env_cfg.num_agents
            or checkpoint_env.num_targets != self.env_cfg.num_targets
            or checkpoint_env.obs_dim != self.env_cfg.obs_dim
            or checkpoint_env.global_state_dim != self.env_cfg.global_state_dim
        ):
            raise ValueError("检查点的环境/网络维度与当前训练配置不一致")
        self.policy.load_state_dict(checkpoint["policy"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_index = int(checkpoint.get("update", 0))

    def write_config(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "env_config": asdict(self.env_cfg),
            "train_config": asdict(self.train_cfg),
            "device": str(self.device),
            "current_update": self.update_index,
        }
        (output_dir / "config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_policy(
    checkpoint_path: Path, device_name: str = "auto"
) -> Tuple[MAPPOPolicy, EnvConfig, Dict[str, object], torch.device]:
    device = resolve_device(device_name)
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    env_cfg = EnvConfig(**checkpoint["env_config"])
    train_cfg = checkpoint.get("train_config", {})
    policy = MAPPOPolicy(
        obs_dim=env_cfg.obs_dim,
        state_dim=env_cfg.global_state_dim,
        num_agents=env_cfg.num_agents,
        action_dim=3,
        hidden_dim=int(train_cfg.get("hidden_dim", 256)),
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy, env_cfg, checkpoint, device
