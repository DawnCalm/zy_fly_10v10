#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from zhuoyi_mappo.config import EnvConfig, TrainConfig
from zhuoyi_mappo.trainer import MAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从零训练卓翼杯 10v10 匈牙利分配 + MAPPO 残差策略"
    )
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=EnvConfig().max_steps,
        help="运动学环境单回合最大步数",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "mappo_fresh",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="恢复模型与优化器；--updates 表示本次继续运行的更新次数",
    )
    parser.add_argument("--save-interval", type=int, default=25)
    return parser.parse_args()


def format_metric(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    env_cfg = EnvConfig(max_steps=args.max_episode_steps)
    train_cfg = TrainConfig(
        seed=args.seed,
        total_updates=args.updates,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        hidden_dim=args.hidden_dim,
        device=args.device,
        save_interval=args.save_interval,
    )
    trainer = MAPPOTrainer(env_cfg, train_cfg)
    if args.resume:
        trainer.load_training_state(args.resume)
    trainer.write_config(args.output)
    metrics_path = args.output / "metrics.jsonl"
    print(
        f"device={trainer.device} envs={train_cfg.num_envs} "
        f"rollout={train_cfg.rollout_steps} obs={env_cfg.obs_dim} "
        f"state={env_cfg.global_state_dim}"
    )

    for _ in range(train_cfg.total_updates):
        metrics = trainer.train_update()
        update = trainer.update_index
        record = {"update": update, **metrics}
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"update={update:04d} episodes={int(metrics['episodes']):3d} "
            f"hits={format_metric(metrics['mean_hits'], 2)} "
            f"escapes={format_metric(metrics['mean_escapes'], 2)} "
            f"steps={format_metric(metrics['mean_steps'], 1)} "
            f"L/M/H={format_metric(metrics['low_hits'], 1)}/"
            f"{format_metric(metrics['mid_hits'], 1)}/"
            f"{format_metric(metrics['high_hits'], 1)} "
            f"actor={metrics['actor_loss']:+.4f} "
            f"critic={metrics['critic_loss']:.4f} "
            f"entropy={metrics['entropy']:.3f} "
            f"kl={metrics['approx_kl']:.5f}"
        )
        if update % train_cfg.save_interval == 0:
            trainer.save(args.output / f"checkpoint_{update:05d}.pt")
        trainer.save(args.output / "latest.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
