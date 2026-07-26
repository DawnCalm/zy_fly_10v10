#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

import numpy as np

from zhuoyi_mappo.gru_prediction import GRUResidualPredictor
from zhuoyi_mappo.prediction import IMMTargetPredictor
from zhuoyi_mappo.trajectory_data import (
    instantaneous_speed,
    load_ros_target_trajectories,
)


DEFAULT_LOG = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "ros"
    / "classic_high_speed30_terminal2_seed20260723.jsonl"
)


def portable_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parent
    try:
        return str(Path(path).resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "比较日志在线跟踪状态的恒速外推与离线 CV/CA/CT-IMM；"
            "旧日志 target_pos 已经过滤，不能当作原始雷达做无偏结论"
        )
    )
    parser.add_argument(
        "--logs", nargs="+", type=Path, default=[DEFAULT_LOG]
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0, 5.0],
    )
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-speed", type=float, default=2.0)
    parser.add_argument("--maximum-target-age", type=float, default=0.25)
    parser.add_argument("--measurement-std", type=float, default=0.1)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0.0,
            "mean": float("nan"),
            "rmse": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "count": float(len(array)),
        "mean": float(array.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def main() -> int:
    args = parse_args()
    horizons = np.asarray(sorted(set(args.horizons)), dtype=np.float64)
    if np.any(horizons <= 0.0):
        raise ValueError("预测时域必须大于 0")
    trajectories = load_ros_target_trajectories(
        args.logs, maximum_target_age=args.maximum_target_age
    )
    errors: DefaultDict[
        str, DefaultDict[float, List[float]]
    ] = defaultdict(lambda: defaultdict(list))
    mode_probability: DefaultDict[str, List[float]] = defaultdict(list)
    gru_predictor = (
        GRUResidualPredictor.load(args.gru_checkpoint, args.device)
        if args.gru_checkpoint is not None
        else None
    )
    if gru_predictor is not None and str(args.device).startswith("cpu"):
        import torch

        torch.set_num_threads(1)
    gru_used = 0
    gru_fallback = 0

    for trajectory in trajectories:
        imm = IMMTargetPredictor(
            measurement_std=args.measurement_std,
            process_acceleration_std=(0.1, 1.0, 1.0),
        )
        start_time = float(trajectory.timestamps[0])
        for index, (timestamp, position) in enumerate(
            zip(trajectory.timestamps, trajectory.positions)
        ):
            imm.update(position, float(timestamp))
            if (
                timestamp - start_time < args.warmup_seconds
                or instantaneous_speed(trajectory, index)
                < args.minimum_speed
            ):
                continue
            valid_horizons = [
                float(horizon)
                for horizon in horizons
                if timestamp + horizon <= trajectory.timestamps[-1]
            ]
            if not valid_horizons:
                continue
            prediction = imm.predict(valid_horizons)
            hybrid = None
            if (
                gru_predictor is not None
                and timestamp + float(gru_predictor.horizons.max())
                <= trajectory.timestamps[-1]
            ):
                base = imm.predict_positions(gru_predictor.horizons)
                hybrid = gru_predictor.predict(
                    trajectory.timestamps[: index + 1],
                    trajectory.positions[: index + 1],
                    base,
                    gru_predictor.horizons,
                )
                if hybrid.used_gru:
                    gru_used += 1
                else:
                    gru_fallback += 1
            logged_velocity = (
                np.asarray(
                    trajectory.velocities[index], dtype=np.float64
                )
                if trajectory.velocities is not None
                else np.zeros(3, dtype=np.float64)
            )
            logged_cv_prediction = (
                np.asarray(position, dtype=np.float64)[None]
                + np.asarray(valid_horizons)[:, None]
                * logged_velocity[None]
            )
            for horizon_index, horizon in enumerate(valid_horizons):
                truth = trajectory.interpolate(timestamp + horizon)
                errors["logged_tracker_cv"][horizon].append(
                    float(
                        np.linalg.norm(
                            logged_cv_prediction[horizon_index] - truth
                        )
                    )
                )
                errors["imm"][horizon].append(
                    float(
                        np.linalg.norm(
                            prediction.position[horizon_index] - truth
                        )
                    )
                )
                for mode_name, mode_position in (
                    prediction.mode_positions.items()
                ):
                    errors[f"imm_{mode_name}"][horizon].append(
                        float(
                            np.linalg.norm(
                                mode_position[horizon_index] - truth
                            )
                        )
                    )
                if hybrid is not None:
                    matching = np.flatnonzero(
                        np.isclose(
                            gru_predictor.horizons,
                            horizon,
                            atol=1.0e-5,
                        )
                    )
                    if len(matching):
                        model_horizon_index = int(matching[0])
                        errors["imm_gru_base"][horizon].append(
                            float(
                                np.linalg.norm(
                                    base[model_horizon_index] - truth
                                )
                            )
                        )
                        errors["imm_gru"][horizon].append(
                            float(
                                np.linalg.norm(
                                    hybrid.position[model_horizon_index] - truth
                                )
                            )
                        )
            for name, probability in (
                prediction.mode_probabilities.items()
            ):
                mode_probability[name].append(probability)

    methods = sorted(errors)
    payload = {
        "logs": [portable_path(path) for path in args.logs],
        "trajectory_count": len(trajectories),
        "horizons": horizons.tolist(),
        "minimum_speed": args.minimum_speed,
        "warmup_seconds": args.warmup_seconds,
        "caveat": (
            "target_pos/target_vel 是控制器在线跟踪状态，不是原始雷达；"
            "offline IMM 是对已过滤位置再次滤波，仅供诊断。"
        ),
        "methods": {
            method: {
                str(horizon): summarize(errors[method][float(horizon)])
                for horizon in horizons
            }
            for method in methods
        },
        "mean_mode_probability": {
            name: float(np.mean(values))
            for name, values in mode_probability.items()
        },
        "gru_checkpoint": (
            portable_path(args.gru_checkpoint)
            if args.gru_checkpoint is not None
            else None
        ),
        "gru_used": gru_used,
        "gru_fallback": gru_fallback,
    }

    print(
        "method           horizon   count   mean_error   rmse    p90"
    )
    for method in methods:
        for horizon in horizons:
            stats = payload["methods"][method][str(float(horizon))]
            print(
                f"{method:16s} {horizon:7.2f} "
                f"{int(stats['count']):7d} "
                f"{stats['mean']:11.3f} "
                f"{stats['rmse']:7.3f} "
                f"{stats['p90']:7.3f}"
            )
    print("mean mode probability:", payload["mean_mode_probability"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
