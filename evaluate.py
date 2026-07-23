#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from zhuoyi_mappo.config import EnvConfig
from zhuoyi_mappo.env import DIFFICULTIES, Kinematic10v10Env
from zhuoyi_mappo.model import MAPPOPolicy
from zhuoyi_mappo.trainer import load_policy, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估经典零残差基线或训练后的 MAPPO 策略"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="每个难度局数")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=DIFFICULTIES,
        default=list(DIFFICULTIES),
        help="只评估指定难度，默认 low mid high",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=None
    )
    return parser.parse_args()


def run_episode(
    env: Kinematic10v10Env,
    difficulty: str,
    seed: int,
    policy: Optional[MAPPOPolicy],
    device: torch.device,
) -> Dict[str, float]:
    obs, state = env.reset(seed=seed, difficulty=difficulty)
    done = False
    info: Dict[str, float] = {}
    while not done:
        if policy is None:
            actions = np.zeros((env.cfg.num_agents, 3), dtype=np.float32)
        else:
            with torch.no_grad():
                action_tensor, _, _ = policy.act(
                    torch.as_tensor(
                        obs[None], dtype=torch.float32, device=device
                    ),
                    torch.as_tensor(
                        state[None], dtype=torch.float32, device=device
                    ),
                    deterministic=True,
                )
            actions = action_tensor[0].cpu().numpy()
        obs, state, _, done, info = env.step(actions)
    return info


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "hits": float(np.mean([row["hits"] for row in rows])),
        "escapes": float(np.mean([row["escapes"] for row in rows])),
        "steps": float(np.mean([row["episode_steps"] for row in rows])),
        "full_intercept_rate": float(
            np.mean([row["hits"] >= 10.0 for row in rows])
        ),
    }


def main() -> int:
    args = parse_args()
    if args.checkpoint:
        policy, env_cfg, checkpoint, device = load_policy(
            args.checkpoint, args.device
        )
        label = f"MAPPO update={checkpoint.get('update', '?')}"
    else:
        policy = None
        env_cfg = EnvConfig()
        device = resolve_device(args.device)
        label = "classic zero-residual baseline"
    if args.max_episode_steps is not None:
        env_cfg.max_steps = args.max_episode_steps

    print(label)
    print("difficulty  mean_hits  mean_escapes  mean_steps  full_rate")
    all_rows: List[Dict[str, float]] = []
    for difficulty in args.difficulties:
        difficulty_id = DIFFICULTIES.index(difficulty)
        rows = []
        for episode in range(args.episodes):
            seed = args.seed + difficulty_id * 1000 + episode
            env = Kinematic10v10Env(env_cfg, seed=seed)
            rows.append(
                run_episode(env, difficulty, seed, policy, device)
            )
        all_rows.extend(rows)
        stats = summarize(rows)
        print(
            f"{difficulty:10s}  {stats['hits']:9.2f}  "
            f"{stats['escapes']:12.2f}  {stats['steps']:10.1f}  "
            f"{stats['full_intercept_rate']:9.1%}"
        )
    total = summarize(all_rows)
    print(
        f"{'all':10s}  {total['hits']:9.2f}  "
        f"{total['escapes']:12.2f}  {total['steps']:10.1f}  "
        f"{total['full_intercept_rate']:9.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
