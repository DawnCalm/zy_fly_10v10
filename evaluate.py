#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        "--compare-classic",
        action="store_true",
        help="用完全相同的配置和种子同时评估零残差经典基线",
    )
    parser.add_argument(
        "--action-scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="确定性 MAPPO 动作缩放，可一次测试多档安全强度",
    )
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
    parser.add_argument(
        "--json-output",
        type=Path,
        help="保存逐种子结果、汇总和相对经典的配对统计",
    )
    return parser.parse_args()


def run_episode(
    env: Kinematic10v10Env,
    difficulty: str,
    seed: int,
    policy: Optional[MAPPOPolicy],
    device: torch.device,
    action_scale: float = 1.0,
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
            actions = action_tensor[0].cpu().numpy() * float(action_scale)
        obs, state, _, done, info = env.step(actions)
    return info


def run_episodes_batched(
    env_config: EnvConfig,
    difficulty: str,
    seeds: List[int],
    policy: Optional[MAPPOPolicy],
    device: torch.device,
    action_scale: float = 1.0,
) -> List[Dict[str, float]]:
    """批量执行独立环境，只合并 Actor 前向，不改变每局动力学。"""

    envs = [
        Kinematic10v10Env(env_config, seed=seed)
        for seed in seeds
    ]
    reset = [
        env.reset(seed=seed, difficulty=difficulty)
        for env, seed in zip(envs, seeds)
    ]
    observations = [item[0] for item in reset]
    states = [item[1] for item in reset]
    active = list(range(len(envs)))
    rows: List[Optional[Dict[str, float]]] = [None] * len(envs)
    while active:
        if policy is None:
            action_batch = np.zeros(
                (len(active), env_config.num_agents, 3), dtype=np.float32
            )
        else:
            with torch.no_grad():
                action_tensor, _, _ = policy.act(
                    torch.as_tensor(
                        np.stack([observations[index] for index in active]),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        np.stack([states[index] for index in active]),
                        dtype=torch.float32,
                        device=device,
                    ),
                    deterministic=True,
                )
            action_batch = (
                action_tensor.cpu().numpy() * float(action_scale)
            )
        next_active = []
        for batch_index, episode_index in enumerate(active):
            obs, state, _, done, info = envs[episode_index].step(
                action_batch[batch_index]
            )
            observations[episode_index] = obs
            states[episode_index] = state
            if done:
                rows[episode_index] = info
            else:
                next_active.append(episode_index)
        active = next_active
    return [row for row in rows if row is not None]


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
    else:
        policy = None
        env_cfg = EnvConfig()
        device = resolve_device(args.device)
    if args.max_episode_steps is not None:
        env_cfg.max_steps = args.max_episode_steps

    evaluations = []
    if policy is None or args.compare_classic:
        evaluations.append(("classic zero-residual baseline", None, 0.0))
    if policy is not None:
        for scale in args.action_scales:
            if scale < 0.0:
                raise ValueError("--action-scales 不能小于 0")
            evaluations.append(
                (
                    f"MAPPO update={checkpoint.get('update', '?')} "
                    f"action_scale={scale:g}",
                    policy,
                    scale,
                )
            )

    totals: Dict[str, Dict[str, float]] = {}
    detailed_results: Dict[
        str, Dict[str, List[Dict[str, float]]]
    ] = {}
    for label, evaluated_policy, action_scale in evaluations:
        print(label)
        print("difficulty  mean_hits  mean_escapes  mean_steps  full_rate")
        all_rows: List[Dict[str, float]] = []
        detailed_results[label] = {}
        for difficulty in args.difficulties:
            difficulty_id = DIFFICULTIES.index(difficulty)
            seeds = [
                args.seed + difficulty_id * 1000 + episode
                for episode in range(args.episodes)
            ]
            rows = run_episodes_batched(
                env_cfg,
                difficulty,
                seeds,
                evaluated_policy,
                device,
                action_scale,
            )
            for seed, row in zip(seeds, rows):
                row["seed"] = float(seed)
            detailed_results[label][difficulty] = rows
            all_rows.extend(rows)
            stats = summarize(rows)
            print(
                f"{difficulty:10s}  {stats['hits']:9.2f}  "
                f"{stats['escapes']:12.2f}  {stats['steps']:10.1f}  "
                f"{stats['full_intercept_rate']:9.1%}"
            )
        total = summarize(all_rows)
        totals[label] = total
        print(
            f"{'all':10s}  {total['hits']:9.2f}  "
            f"{total['escapes']:12.2f}  {total['steps']:10.1f}  "
            f"{total['full_intercept_rate']:9.1%}"
        )
        print()

    classic_label = "classic zero-residual baseline"
    if classic_label in totals and policy is not None:
        classic_hits = totals[classic_label]["hits"]
        print("paired mean-hit delta vs classic")
        for label, stats in totals.items():
            if label != classic_label:
                print(f"{label}: {stats['hits'] - classic_hits:+.3f}")
    if args.json_output:
        paired: Dict[str, Dict[str, Dict[str, float]]] = {}
        if classic_label in detailed_results:
            for label in detailed_results:
                if label == classic_label:
                    continue
                paired[label] = {}
                for difficulty in args.difficulties:
                    classic_rows = detailed_results[classic_label][difficulty]
                    policy_rows = detailed_results[label][difficulty]
                    deltas = np.asarray(
                        [
                            policy_row["hits"] - classic_row["hits"]
                            for classic_row, policy_row in zip(
                                classic_rows, policy_rows
                            )
                        ],
                        dtype=np.float32,
                    )
                    paired[label][difficulty] = {
                        "mean_hit_delta": float(deltas.mean()),
                        "standard_error": float(
                            deltas.std(ddof=1) / np.sqrt(len(deltas))
                            if len(deltas) > 1
                            else 0.0
                        ),
                        "improved_episodes": float((deltas > 0).sum()),
                        "equal_episodes": float((deltas == 0).sum()),
                        "worse_episodes": float((deltas < 0).sum()),
                    }
        payload = {
            "checkpoint": (
                str(args.checkpoint) if args.checkpoint else None
            ),
            "seed": args.seed,
            "episodes": args.episodes,
            "difficulties": args.difficulties,
            "environment": env_cfg.to_dict(),
            "summaries": totals,
            "paired": paired,
            "episodes_by_label": detailed_results,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
