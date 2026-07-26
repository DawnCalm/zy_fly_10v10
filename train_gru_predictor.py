#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from zhuoyi_mappo.config import EnvConfig
from zhuoyi_mappo.env import Kinematic10v10Env
from zhuoyi_mappo.gru_prediction import (
    DEFAULT_HORIZONS,
    GRUResidualModel,
    history_to_local_features,
)
from zhuoyi_mappo.prediction import IMMTargetPredictor
from zhuoyi_mappo.trajectory_data import TargetTrajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用随机 High 运动学轨迹训练 IMM 之上的小型 GRU 残差"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--validation-episodes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--history-steps", type=int, default=20)
    parser.add_argument("--sample-dt", type=float, default=0.2)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=list(DEFAULT_HORIZONS),
    )
    parser.add_argument("--residual-scale", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def generate_high_trajectories(
    episode_seed: int,
    sample_dt: float,
    duration: float,
) -> List[TargetTrajectory]:
    config = EnvConfig(
        dt=sample_dt,
        max_steps=int(math.ceil(duration / sample_dt)),
    )
    env = Kinematic10v10Env(config, seed=episode_seed)
    env.reset(seed=episode_seed, difficulty="high")
    times: List[float] = [0.0]
    history: List[np.ndarray] = [env.target_pos.copy()]
    active = np.ones(config.num_targets, dtype=bool)
    active_history: List[np.ndarray] = [active.copy()]
    for _ in range(config.max_steps):
        env._update_target_velocity()
        env.target_pos += env.target_vel * config.dt
        env.step_count += 1
        horizontal_distance = np.linalg.norm(
            env.target_pos[:, :2] - env.safe_center[:2], axis=1
        )
        active &= horizontal_distance > config.safety_radius
        times.append(env.step_count * config.dt)
        history.append(env.target_pos.copy())
        active_history.append(active.copy())
        if not active.any():
            break

    timestamps = np.asarray(times, dtype=np.float64)
    positions = np.asarray(history, dtype=np.float32)
    alive = np.asarray(active_history, dtype=bool)
    trajectories: List[TargetTrajectory] = []
    for target_id in range(config.num_targets):
        valid_count = int(np.count_nonzero(alive[:, target_id]))
        valid_count = max(valid_count, 2)
        trajectories.append(
            TargetTrajectory(
                source=f"synthetic-high-seed-{episode_seed}",
                target_id=target_id,
                timestamps=timestamps[:valid_count],
                positions=positions[:valid_count, target_id],
            )
        )
    return trajectories


def build_dataset(
    seeds: Iterable[int],
    horizons: np.ndarray,
    history_steps: int,
    sample_dt: float,
    sample_stride: int,
    residual_scale: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_features: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    history_duration = (history_steps - 1) * sample_dt
    maximum_horizon = float(np.max(horizons))
    for seed in seeds:
        for trajectory in generate_high_trajectories(
            int(seed), sample_dt, duration
        ):
            speed = np.linalg.norm(
                np.diff(trajectory.positions, axis=0), axis=1
            ) / np.diff(trajectory.timestamps)
            moving = np.flatnonzero(speed >= 2.0)
            if not len(moving):
                continue
            first_time = float(
                trajectory.timestamps[max(0, int(moving[0]) - history_steps)]
            )
            predictor = IMMTargetPredictor()
            for index, (timestamp, position) in enumerate(
                zip(trajectory.timestamps, trajectory.positions)
            ):
                if timestamp < first_time:
                    continue
                predictor.update(position, float(timestamp))
                if (
                    timestamp < first_time + history_duration
                    or index % max(1, sample_stride) != 0
                    or timestamp + maximum_horizon
                    > trajectory.timestamps[-1]
                ):
                    continue
                try:
                    features, local_to_world = history_to_local_features(
                        trajectory.timestamps[: index + 1],
                        trajectory.positions[: index + 1],
                        history_steps=history_steps,
                        sample_dt=sample_dt,
                    )
                except ValueError:
                    continue
                base = predictor.predict_positions(horizons)
                truth = np.stack(
                    [
                        trajectory.interpolate(float(timestamp + horizon))
                        for horizon in horizons
                    ],
                    axis=0,
                )
                local_residual = (truth - base) @ local_to_world
                normalized = np.clip(
                    local_residual / residual_scale, -1.0, 1.0
                )
                all_features.append(features)
                all_targets.append(normalized.astype(np.float32))
    if not all_features:
        raise RuntimeError("没有生成有效 GRU 训练样本")
    return np.stack(all_features), np.stack(all_targets)


def evaluate(
    model: GRUResidualModel,
    features: torch.Tensor,
    target: torch.Tensor,
    horizons: np.ndarray,
    residual_scale: float,
    residual_gain: float,
    device: torch.device,
) -> dict:
    model.eval()
    base_error = []
    hybrid_error = []
    with torch.no_grad():
        for start in range(0, len(features), 1024):
            batch_features = features[start : start + 1024].to(device)
            batch_target = target[start : start + 1024].to(device)
            prediction = model(batch_features) * residual_gain
            base_error.append(
                torch.linalg.vector_norm(
                    batch_target * residual_scale, dim=-1
                ).cpu()
            )
            hybrid_error.append(
                torch.linalg.vector_norm(
                    (batch_target - prediction) * residual_scale, dim=-1
                ).cpu()
            )
    base = torch.cat(base_error).numpy()
    hybrid = torch.cat(hybrid_error).numpy()
    return {
        str(float(horizon)): {
            "imm_mean_error": float(base[:, index].mean()),
            "imm_gru_mean_error": float(hybrid[:, index].mean()),
            "improvement": float(
                base[:, index].mean() - hybrid[:, index].mean()
            ),
        }
        for index, horizon in enumerate(horizons)
    }


def main() -> int:
    args = parse_args()
    if args.train_episodes <= 0 or args.validation_episodes <= 0:
        raise ValueError("训练和验证 episode 数必须为正")
    horizons = np.asarray(sorted(set(args.horizons)), dtype=np.float32)
    if np.any(horizons <= 0.0):
        raise ValueError("horizons 必须为正")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = torch.device(args.device)

    train_seeds = range(args.seed, args.seed + args.train_episodes)
    validation_seed_start = args.seed + 100000
    validation_seeds = range(
        validation_seed_start,
        validation_seed_start + args.validation_episodes,
    )
    started = time.perf_counter()
    train_x, train_y = build_dataset(
        train_seeds,
        horizons,
        args.history_steps,
        args.sample_dt,
        args.sample_stride,
        args.residual_scale,
        args.duration,
    )
    validation_x, validation_y = build_dataset(
        validation_seeds,
        horizons,
        args.history_steps,
        args.sample_dt,
        args.sample_stride,
        args.residual_scale,
        args.duration,
    )
    print(
        f"dataset train={len(train_x)} validation={len(validation_x)} "
        f"build_s={time.perf_counter() - started:.1f}"
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_features = torch.from_numpy(validation_x)
    validation_target = torch.from_numpy(validation_y)
    model = GRUResidualModel(
        horizon_count=len(horizons), hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
    )
    loss_function = nn.SmoothL1Loss(reduction="none")
    horizon_weight = torch.as_tensor(
        1.0 / np.sqrt(horizons),
        dtype=torch.float32,
        device=device,
    )
    horizon_weight /= horizon_weight.mean()
    best_validation = float("inf")
    best_gain = 0.0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    metrics = []
    baseline_validation = evaluate(
        model,
        validation_features,
        validation_target,
        horizons,
        args.residual_scale,
        residual_gain=0.0,
        device=device,
    )
    best_validation = float(
        np.mean(
            [
                value["imm_gru_mean_error"]
                for value in baseline_validation.values()
            ]
        )
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            prediction = model(features)
            element_loss = loss_function(prediction, target).mean(dim=-1)
            loss = (element_loss * horizon_weight[None]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        gain_results = {}
        validation_mean = float("inf")
        validation = {}
        epoch_gain = 0.0
        for candidate_gain in (0.0, 0.25, 0.5, 0.75, 1.0):
            candidate = evaluate(
                model,
                validation_features,
                validation_target,
                horizons,
                args.residual_scale,
                residual_gain=candidate_gain,
                device=device,
            )
            candidate_mean = float(
                np.mean(
                    [
                        value["imm_gru_mean_error"]
                        for value in candidate.values()
                    ]
                )
            )
            gain_results[str(candidate_gain)] = candidate_mean
            if candidate_mean < validation_mean:
                validation_mean = candidate_mean
                validation = candidate
                epoch_gain = candidate_gain
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mean_error": validation_mean,
                "residual_gain": epoch_gain,
                "gain_results": gain_results,
                "validation": validation,
            }
        )
        if validation_mean < best_validation - 1.0e-8:
            best_validation = validation_mean
            best_gain = epoch_gain
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.6f} "
            f"validation_error={validation_mean:.3f} gain={epoch_gain:.2f}"
        )

    model.load_state_dict(best_state)
    final_validation = evaluate(
        model,
        validation_features,
        validation_target,
        horizons,
        args.residual_scale,
        residual_gain=best_gain,
        device=device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model_config = {
        "horizons": horizons.tolist(),
        "hidden_dim": args.hidden_dim,
        "history_steps": args.history_steps,
        "sample_dt": args.sample_dt,
        "residual_scale": args.residual_scale,
        "residual_gain": best_gain,
    }
    torch.save(
        {
            "model_state": best_state,
            "model_config": model_config,
            "train_seeds": [args.seed, args.seed + args.train_episodes - 1],
            "validation_seeds": [
                validation_seed_start,
                validation_seed_start + args.validation_episodes - 1,
            ],
            "validation": final_validation,
        },
        args.output / "best.pt",
    )
    report = {
        "model_config": model_config,
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "train_seeds": [args.seed, args.seed + args.train_episodes - 1],
        "validation_seeds": [
            validation_seed_start,
            validation_seed_start + args.validation_episodes - 1,
        ],
        "best_validation_mean_error": best_validation,
        "best_residual_gain": best_gain,
        "validation": final_validation,
        "epochs": metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_validation, ensure_ascii=False, indent=2))
    print(f"saved {args.output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
