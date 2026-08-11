#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from zhuoyi_mappo.flight_response import (
    RidgeFlightResponseModel,
    flight_response_features,
)


@dataclass(frozen=True)
class TrajectoryData:
    run_id: np.ndarray
    seed: np.ndarray
    agent_id: np.ndarray
    time_s: np.ndarray
    dt_s: np.ndarray
    valid: np.ndarray
    assignment_valid: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    next_velocity: np.ndarray
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    guidance_blend: np.ndarray


@dataclass(frozen=True)
class SampleBatch:
    data: TrajectoryData
    current: np.ndarray
    history: np.ndarray
    features: np.ndarray
    target_local_acceleration: np.ndarray
    target_world_acceleration: np.ndarray
    distance: np.ndarray


def _load_npz(path: Path) -> TrajectoryData:
    with np.load(path, allow_pickle=False) as source:
        return TrajectoryData(
            run_id=source["run_id"].astype(np.int64),
            seed=source["seed"].astype(np.int64),
            agent_id=source["agent_id"].astype(np.int64),
            time_s=source["time_s"].astype(np.float64),
            dt_s=source["dt_s"].astype(np.float64),
            valid=np.ones(len(source["run_id"]), dtype=bool),
            assignment_valid=source["assignment_valid"].astype(bool),
            velocity=source["agent_velocity"].astype(np.float64),
            command=source["command_velocity"].astype(np.float64),
            next_velocity=source["next_agent_velocity"].astype(np.float64),
            relative_position=source["relative_position"].astype(np.float64),
            relative_velocity=source["relative_velocity"].astype(np.float64),
            guidance_blend=source["guidance_blend"].astype(np.float64),
        )


def _load_jsonl(path: Path, seed: int) -> TrajectoryData:
    frames = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(frames) < 2:
        raise ValueError(f"外部日志帧数不足: {path}")
    count = len(frames[0]["agent_vel"])
    base_time = float(frames[0]["wall_time"])
    fields: Dict[str, List[np.ndarray]] = {
        name: []
        for name in (
            "time",
            "dt",
            "valid",
            "assignment_valid",
            "velocity",
            "command",
            "next_velocity",
            "relative_position",
            "relative_velocity",
            "guidance_blend",
        )
    }
    for frame, next_frame in zip(frames[:-1], frames[1:]):
        now = float(frame["wall_time"])
        dt = float(next_frame["wall_time"]) - now
        velocity = np.asarray(frame["agent_vel"], dtype=np.float64)
        next_velocity = np.asarray(next_frame["agent_vel"], dtype=np.float64)
        command = np.asarray(frame["published_velocity"], dtype=np.float64)
        active = np.asarray(frame["agent_active"], dtype=bool)
        next_active = np.asarray(next_frame["agent_active"], dtype=bool)
        assignment = np.asarray(frame["assignment"], dtype=np.int64)
        target_active = np.asarray(frame["target_active"], dtype=bool)
        target_position = np.asarray(frame["target_pos"], dtype=np.float64)
        target_velocity = np.asarray(frame["target_vel"], dtype=np.float64)
        agent_position = np.asarray(frame["agent_pos"], dtype=np.float64)
        assigned = (
            active
            & (assignment >= 0)
            & (assignment < len(target_active))
        )
        assigned &= np.where(
            assignment >= 0,
            target_active[np.clip(assignment, 0, len(target_active) - 1)],
            False,
        )
        relative_position = np.zeros((count, 3), dtype=np.float64)
        relative_velocity = np.zeros((count, 3), dtype=np.float64)
        ids = np.flatnonzero(assigned)
        relative_position[ids] = (
            target_position[assignment[ids]] - agent_position[ids]
        )
        relative_velocity[ids] = (
            target_velocity[assignment[ids]] - velocity[ids]
        )
        fields["time"].append(np.full(count, now - base_time))
        fields["dt"].append(np.full(count, dt))
        fields["valid"].append(active & next_active)
        fields["assignment_valid"].append(assigned)
        fields["velocity"].append(velocity)
        fields["command"].append(command)
        fields["next_velocity"].append(next_velocity)
        fields["relative_position"].append(relative_position)
        fields["relative_velocity"].append(relative_velocity)
        fields["guidance_blend"].append(
            np.asarray(frame["guidance_blend"], dtype=np.float64)
        )

    frame_count = len(frames) - 1
    return TrajectoryData(
        run_id=np.zeros(frame_count * count, dtype=np.int64),
        seed=np.full(frame_count * count, int(seed), dtype=np.int64),
        agent_id=np.tile(np.arange(count, dtype=np.int64), frame_count),
        time_s=np.concatenate(fields["time"]),
        dt_s=np.concatenate(fields["dt"]),
        valid=np.concatenate(fields["valid"]),
        assignment_valid=np.concatenate(fields["assignment_valid"]),
        velocity=np.concatenate(fields["velocity"]),
        command=np.concatenate(fields["command"]),
        next_velocity=np.concatenate(fields["next_velocity"]),
        relative_position=np.concatenate(fields["relative_position"]),
        relative_velocity=np.concatenate(fields["relative_velocity"]),
        guidance_blend=np.concatenate(fields["guidance_blend"]),
    )


def _sequence_indices(data: TrajectoryData) -> Iterable[np.ndarray]:
    keys = np.stack((data.run_id, data.agent_id), axis=1)
    for key in np.unique(keys, axis=0):
        indices = np.flatnonzero(np.all(keys == key, axis=1))
        yield indices[np.argsort(data.time_s[indices])]


def _history_rows(data: TrajectoryData) -> np.ndarray:
    parts: List[np.ndarray] = []
    for indices in _sequence_indices(data):
        if len(indices) < 4:
            continue
        current = indices[3:]
        p1, p2, p3 = indices[2:-1], indices[1:-2], indices[:-3]
        gaps = np.stack(
            (
                data.time_s[current] - data.time_s[p1],
                data.time_s[p1] - data.time_s[p2],
                data.time_s[p2] - data.time_s[p3],
            ),
            axis=1,
        )
        valid = (
            np.all(data.valid[np.stack((current, p1, p2, p3), axis=1)], axis=1)
            & np.all((gaps > 0.04) & (gaps <= 0.35), axis=1)
            & (data.dt_s[current] > 0.04)
            & (data.dt_s[current] <= 0.35)
            & (data.time_s[current] > 10.0)
            & (np.linalg.norm(data.velocity[current, :2], axis=1) > 2.0)
        )
        parts.append(
            np.stack((current[valid], p1[valid], p2[valid], p3[valid]), axis=1)
        )
    if not parts:
        raise ValueError("没有满足连续性条件的飞控响应样本")
    return np.concatenate(parts)


def _build_samples(data: TrajectoryData) -> SampleBatch:
    history = _history_rows(data)
    current = history[:, 0]
    velocity_history = data.velocity[history]
    command_history = data.command[history]
    history_dt = np.stack(
        (
            data.time_s[history[:, 0]] - data.time_s[history[:, 1]],
            data.time_s[history[:, 1]] - data.time_s[history[:, 2]],
        ),
        axis=1,
    )
    features, forward, lateral = flight_response_features(
        velocity_history,
        command_history,
        history_dt,
        data.dt_s[current],
    )
    target_world = (
        data.next_velocity[current] - data.velocity[current]
    ) / data.dt_s[current, None]
    target_local = np.stack(
        (
            np.sum(target_world[:, :2] * forward, axis=1),
            np.sum(target_world[:, :2] * lateral, axis=1),
            target_world[:, 2],
        ),
        axis=1,
    )
    distance = np.where(
        data.assignment_valid[current],
        np.linalg.norm(data.relative_position[current], axis=1),
        np.inf,
    )
    return SampleBatch(
        data=data,
        current=current,
        history=history,
        features=features,
        target_local_acceleration=target_local,
        target_world_acceleration=target_world,
        distance=distance,
    )


def _metric(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _segments(batch: SampleBatch) -> Dict[str, np.ndarray]:
    current = batch.current
    return {
        "all": np.ones(len(current), dtype=bool),
        "pn": batch.data.guidance_blend[current] > 0.0,
        "near_80m": batch.distance < 80.0,
        "near_15m": batch.distance < 15.0,
    }


def _one_step_metrics(
    batch: SampleBatch, prediction: np.ndarray
) -> Dict[str, Dict[str, object]]:
    error = prediction - batch.target_local_acceleration
    vector_error = np.linalg.norm(error, axis=1)
    velocity_error = vector_error * batch.data.dt_s[batch.current]
    result: Dict[str, Dict[str, object]] = {}
    for name, mask in _segments(batch).items():
        if not np.any(mask):
            continue
        result[name] = {
            "samples": int(np.sum(mask)),
            "acceleration_vector_error_mps2": _metric(vector_error[mask]),
            "next_velocity_vector_error_mps": _metric(velocity_error[mask]),
            "lateral_acceleration_abs_error_mps2": _metric(
                np.abs(error[mask, 1])
            ),
        }
    return result


def _merge_predictions(
    batch: SampleBatch, ridge: float
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    groups = batch.data.seed[batch.current]
    response = np.zeros_like(batch.target_local_acceleration)
    fold_summary: List[Dict[str, object]] = []
    for seed in np.unique(groups):
        train = groups != seed
        test = groups == seed
        model = RidgeFlightResponseModel.fit(
            batch.features[train],
            batch.target_local_acceleration[train],
            ridge=ridge,
        )
        response[test] = model.predict_local_acceleration(batch.features[test])
        fold_summary.append(
            {
                "held_out_seed": int(seed),
                "train_samples": int(np.sum(train)),
                "test_samples": int(np.sum(test)),
            }
        )
    return {
        "persistence": np.zeros_like(batch.target_local_acceleration),
        "previous_acceleration": batch.features[:, 24:27],
        "response_ridge": response,
    }, fold_summary


def _angle_error_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(
        target, axis=1
    )
    cosine = np.divide(
        np.sum(prediction * target, axis=1),
        np.maximum(denominator, 1.0e-8),
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _rollout_starts(
    data: TrajectoryData, steps: int, stride: int = 3
) -> np.ndarray:
    starts: List[np.ndarray] = []
    for indices in _sequence_indices(data):
        if len(indices) < 4 + steps:
            continue
        positions = np.arange(3, len(indices) - steps, stride)
        rows = np.stack(
            [indices[positions + offset] for offset in range(-3, steps)],
            axis=1,
        )
        time_gap = np.diff(data.time_s[rows], axis=1)
        valid = (
            np.all(data.valid[rows], axis=1)
            & np.all((time_gap > 0.04) & (time_gap <= 0.35), axis=1)
            & np.all(
                (data.dt_s[rows[:, 3:]] > 0.04)
                & (data.dt_s[rows[:, 3:]] <= 0.35),
                axis=1,
            )
            & (data.time_s[rows[:, 3]] > 10.0)
            & (np.linalg.norm(data.velocity[rows[:, 3], :2], axis=1) > 2.0)
        )
        starts.append(rows[valid])
    if not starts:
        raise ValueError("没有满足连续性条件的 rollout 片段")
    return np.concatenate(starts)


def _rollout_metrics(
    data: TrajectoryData,
    model_for_seed: Dict[int, RidgeFlightResponseModel],
    steps: int = 12,
) -> Dict[str, Dict[str, object]]:
    rows = _rollout_starts(data, steps)
    current_rows = rows[:, 3:]
    velocity_history = data.velocity[rows[:, [3, 2, 1, 0]]].copy()
    response_velocity = velocity_history[:, 0].copy()
    persistence_velocity = response_velocity.copy()
    elapsed = np.zeros(len(rows), dtype=np.float64)
    checkpoints = (3, 6, 12)
    result: Dict[str, Dict[str, object]] = {}
    # Rollout rows and one-step rows use different strides, so calculate masks here.
    start_index = rows[:, 3]
    start_distance = np.where(
        data.assignment_valid[start_index],
        np.linalg.norm(data.relative_position[start_index], axis=1),
        np.inf,
    )
    rollout_segments = {
        "all": np.ones(len(rows), dtype=bool),
        "pn": data.guidance_blend[start_index] > 0.0,
        "near_80m": start_distance < 80.0,
        "near_15m": start_distance < 15.0,
    }
    for step in range(steps):
        current = current_rows[:, step]
        command_indices = rows[:, step : step + 4][:, ::-1]
        command_history = data.command[command_indices]
        history_dt = np.stack(
            (
                data.time_s[rows[:, step + 3]]
                - data.time_s[rows[:, step + 2]],
                data.time_s[rows[:, step + 2]]
                - data.time_s[rows[:, step + 1]],
            ),
            axis=1,
        )
        acceleration = np.zeros_like(response_velocity)
        seeds = data.seed[current]
        for seed in np.unique(seeds):
            mask = seeds == seed
            acceleration[mask] = model_for_seed[int(seed)].predict_world_acceleration(
                velocity_history[mask],
                command_history[mask],
                history_dt[mask],
                data.dt_s[current[mask]],
            )
        response_velocity += acceleration * data.dt_s[current, None]
        velocity_history[:, 1:] = velocity_history[:, :-1]
        velocity_history[:, 0] = response_velocity
        elapsed += data.dt_s[current]

        if step + 1 not in checkpoints:
            continue
        actual = data.next_velocity[current]
        target_velocity = data.relative_velocity[current] + data.velocity[current]
        relative_position = data.relative_position[current]
        distance = np.linalg.norm(relative_position, axis=1)
        los = np.divide(
            relative_position,
            np.maximum(distance[:, None], 1.0e-8),
        )
        actual_closing = -np.sum((target_velocity - actual) * los, axis=1)
        predictions = {
            "persistence": persistence_velocity,
            "response_ridge": response_velocity,
        }
        horizon: Dict[str, object] = {
            "elapsed_s": _metric(elapsed.copy()),
            "models": {},
        }
        for model_name, predicted in predictions.items():
            velocity_error = np.linalg.norm(predicted - actual, axis=1)
            direction_error = _angle_error_deg(predicted, actual)
            predicted_closing = -np.sum(
                (target_velocity - predicted) * los, axis=1
            )
            closing_error = np.abs(predicted_closing - actual_closing)
            model_metrics: Dict[str, object] = {}
            for segment_name, mask in rollout_segments.items():
                if not np.any(mask):
                    continue
                closing_mask = mask & data.assignment_valid[current]
                values: Dict[str, object] = {
                    "samples": int(np.sum(mask)),
                    "velocity_vector_error_mps": _metric(velocity_error[mask]),
                    "velocity_direction_error_deg": _metric(
                        direction_error[mask]
                    ),
                }
                if np.any(closing_mask):
                    values["closing_speed_abs_error_mps"] = _metric(
                        closing_error[closing_mask]
                    )
                model_metrics[segment_name] = values
            horizon["models"][model_name] = model_metrics
        result[f"step_{step + 1}"] = horizon
    return result


def _model_map_cross_validation(
    batch: SampleBatch, ridge: float
) -> Dict[int, RidgeFlightResponseModel]:
    groups = batch.data.seed[batch.current]
    return {
        int(seed): RidgeFlightResponseModel.fit(
            batch.features[groups != seed],
            batch.target_local_acceleration[groups != seed],
            ridge=ridge,
        )
        for seed in np.unique(groups)
    }


def _external_evaluation(
    train: SampleBatch,
    external: TrajectoryData,
    ridge: float,
) -> Dict[str, object]:
    external_batch = _build_samples(external)
    model = RidgeFlightResponseModel.fit(
        train.features, train.target_local_acceleration, ridge=ridge
    )
    predictions = {
        "persistence": np.zeros_like(external_batch.target_local_acceleration),
        "previous_acceleration": external_batch.features[:, 24:27],
        "response_ridge": model.predict_local_acceleration(
            external_batch.features
        ),
    }
    seed = int(np.unique(external.seed)[0])
    return {
        "samples": int(len(external_batch.features)),
        "one_step": {
            name: _one_step_metrics(external_batch, prediction)
            for name, prediction in predictions.items()
        },
        "rollout": _rollout_metrics(external, {seed: model}),
    }


def _save_model(path: Path, model: RidgeFlightResponseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_mean=model.input_mean.astype(np.float32),
        input_scale=model.input_scale.astype(np.float32),
        coefficient=model.coefficient.astype(np.float32),
        output_mean=model.output_mean.astype(np.float32),
        acceleration_limit=np.float32(model.acceleration_limit),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--external-jsonl", type=Path)
    parser.add_argument("--external-seed", type=int, default=20260723)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data = _load_npz(args.dataset)
    batch = _build_samples(data)
    predictions, folds = _merge_predictions(batch, args.ridge)
    result: Dict[str, object] = {
        "dataset": str(args.dataset),
        "causal_samples": int(len(batch.features)),
        "unique_seeds": [int(value) for value in np.unique(data.seed)],
        "split": "leave-one-seed-out; paired variants of a seed stay together",
        "features": (
            "current and previous 3 measured velocities, current and previous "
            "3 published commands, previous 2 measured accelerations, dt; "
            "horizontal velocity frame"
        ),
        "folds": folds,
        "cross_validation": {
            "one_step": {
                name: _one_step_metrics(batch, prediction)
                for name, prediction in predictions.items()
            },
            "rollout": _rollout_metrics(
                data, _model_map_cross_validation(batch, args.ridge)
            ),
        },
    }
    if args.external_jsonl is not None:
        external = _load_jsonl(args.external_jsonl, args.external_seed)
        result["external_validation"] = _external_evaluation(
            batch, external, args.ridge
        )
        result["external_validation"]["source"] = str(args.external_jsonl)
        result["external_validation"]["seed"] = int(args.external_seed)
    if args.model_output is not None:
        final_model = RidgeFlightResponseModel.fit(
            batch.features, batch.target_local_acceleration, ridge=args.ridge
        )
        _save_model(args.model_output, final_model)
        result["model_output"] = str(args.model_output)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()
