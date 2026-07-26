#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较两份真实 ROS 10v10 日志的命中、实时性和近失"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def distribution(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {}
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def infer_hit_targets(rows: List[dict]) -> List[int]:
    """命中时目标先停止，随后对应拦截机里程计消失。"""

    target_drop: List[tuple[float, int]] = []
    agent_drop: List[float] = []
    initial_time = float(rows[0]["wall_time"])
    previous_target = np.ones(10, dtype=bool)
    previous_agents = 10
    for row in rows:
        timestamp = float(row["wall_time"]) - initial_time
        target = np.asarray(
            row.get("target_active_control", row["target_active"]),
            dtype=bool,
        )
        for target_id in np.flatnonzero(previous_target & ~target):
            target_drop.append((timestamp, int(target_id)))
        agents = int(np.count_nonzero(row["agent_active"]))
        agent_drop.extend([timestamp] * max(0, previous_agents - agents))
        previous_target = target
        previous_agents = agents

    hits: List[int] = []
    unused = set(range(len(target_drop)))
    for agent_time in agent_drop:
        candidates = [
            index
            for index in unused
            if 0.0 <= agent_time - target_drop[index][0] <= 2.0
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda index: agent_time - target_drop[index][0],
            )
            hits.append(target_drop[selected][1])
            unused.remove(selected)
    return sorted(hits)


def online_cv_errors(
    rows: List[dict], horizons: Iterable[float]
) -> Dict[str, Dict[str, float]]:
    timestamps = np.asarray(
        [row["wall_time"] for row in rows], dtype=np.float64
    )
    timestamps -= timestamps[0]
    positions = np.asarray(
        [row["target_pos"] for row in rows], dtype=np.float64
    )
    velocities = np.asarray(
        [row["target_vel"] for row in rows], dtype=np.float64
    )
    active = np.asarray(
        [
            row.get("target_active_control", row["target_active"])
            for row in rows
        ],
        dtype=bool,
    )
    age = np.asarray(
        [row["target_age"] for row in rows], dtype=np.float64
    )
    errors = {float(horizon): [] for horizon in horizons}
    for index, timestamp in enumerate(timestamps):
        if timestamp < 4.0:
            continue
        for target_id in range(10):
            if not active[index, target_id] or age[index, target_id] > 0.25:
                continue
            for horizon in errors:
                query = timestamp + horizon
                future = int(np.searchsorted(timestamps, query))
                if future <= 0 or future >= len(timestamps):
                    continue
                if (
                    not active[future - 1, target_id]
                    or not active[future, target_id]
                ):
                    continue
                fraction = (
                    (query - timestamps[future - 1])
                    / (timestamps[future] - timestamps[future - 1])
                )
                truth = (
                    positions[future - 1, target_id] * (1.0 - fraction)
                    + positions[future, target_id] * fraction
                )
                predicted = (
                    positions[index, target_id]
                    + velocities[index, target_id] * horizon
                )
                errors[horizon].append(
                    float(np.linalg.norm(predicted - truth))
                )
    return {
        str(horizon): {
            "count": len(values),
            "mean": float(np.mean(values)),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "p90": float(np.percentile(values, 90)),
        }
        for horizon, values in errors.items()
    }


def analyze(path: Path) -> dict:
    rows = load_rows(path)
    timestamps = np.asarray(
        [row["wall_time"] for row in rows], dtype=np.float64
    )
    timestamps -= timestamps[0]
    actual_speed: List[float] = []
    desired_speed: List[float] = []
    published_speed: List[float] = []
    command_acceleration: List[float] = []
    command_turn_rate: List[float] = []
    target_acceleration: List[float] = []
    guidance_blend: List[float] = []
    minimum_distance = np.full(10, np.inf, dtype=np.float64)
    closing_at_minimum = np.full(10, np.nan, dtype=np.float64)
    time_at_minimum = np.full(10, np.nan, dtype=np.float64)
    assignment_switches = 0

    for index, row in enumerate(rows):
        agent_pos = np.asarray(row["agent_pos"], dtype=np.float64)
        agent_vel = np.asarray(row["agent_vel"], dtype=np.float64)
        agent_active = np.asarray(row["agent_active"], dtype=bool)
        target_pos = np.asarray(row["target_pos"], dtype=np.float64)
        target_vel = np.asarray(row["target_vel"], dtype=np.float64)
        target_active = np.asarray(
            row.get("target_active_control", row["target_active"]),
            dtype=bool,
        )
        published = np.asarray(
            row.get("published_velocity", np.zeros((10, 3))),
            dtype=np.float64,
        )
        desired = np.asarray(
            row.get("desired_velocity", np.zeros((10, 3))),
            dtype=np.float64,
        )
        assignment = np.asarray(
            row.get("assignment", np.full(10, -1)), dtype=np.int64
        )
        actual_speed.extend(np.linalg.norm(agent_vel[agent_active], axis=1))
        desired_speed.extend(np.linalg.norm(desired[agent_active], axis=1))
        published_speed.extend(
            np.linalg.norm(published[agent_active], axis=1)
        )
        acceleration = np.asarray(
            row.get("target_acceleration", np.zeros((10, 3))),
            dtype=np.float64,
        )
        target_acceleration.extend(
            np.linalg.norm(acceleration[target_active], axis=1)
        )
        if "guidance_blend" in row:
            blend = np.asarray(row["guidance_blend"], dtype=np.float64)
            valid = agent_active & (assignment >= 0)
            guidance_blend.extend(blend[valid])

        active_agents = np.flatnonzero(agent_active)
        for target_id in np.flatnonzero(target_active):
            if not len(active_agents):
                continue
            delta = agent_pos[active_agents] - target_pos[target_id]
            distance = np.linalg.norm(delta, axis=1)
            local_index = int(np.argmin(distance))
            value = float(distance[local_index])
            if value < minimum_distance[target_id]:
                agent_id = int(active_agents[local_index])
                relative = target_pos[target_id] - agent_pos[agent_id]
                relative_velocity = (
                    target_vel[target_id] - agent_vel[agent_id]
                )
                minimum_distance[target_id] = value
                closing_at_minimum[target_id] = -float(
                    np.dot(relative, relative_velocity)
                ) / max(value, 1.0e-6)
                time_at_minimum[target_id] = timestamps[index]

        if index == 0:
            continue
        previous = rows[index - 1]
        dt = float(timestamps[index] - timestamps[index - 1])
        previous_active = np.asarray(
            previous["agent_active"], dtype=bool
        )
        previous_published = np.asarray(
            previous.get("published_velocity", np.zeros((10, 3))),
            dtype=np.float64,
        )
        valid = agent_active & previous_active
        if dt > 0.0 and np.any(valid):
            command_acceleration.extend(
                np.linalg.norm(
                    published[valid] - previous_published[valid], axis=1
                )
                / dt
            )
            for before, after in zip(
                previous_published[valid], published[valid]
            ):
                before_norm = float(np.linalg.norm(before))
                after_norm = float(np.linalg.norm(after))
                if before_norm > 3.0 and after_norm > 3.0:
                    cosine = float(
                        np.clip(
                            np.dot(before, after)
                            / (before_norm * after_norm),
                            -1.0,
                            1.0,
                        )
                    )
                    command_turn_rate.append(
                        math.degrees(math.acos(cosine)) / dt
                    )
        previous_assignment = np.asarray(
            previous.get("assignment", np.full(10, -1)),
            dtype=np.int64,
        )
        valid_assignment = (
            valid & (assignment >= 0) & (previous_assignment >= 0)
        )
        assignment_switches += int(
            np.count_nonzero(
                valid_assignment & (assignment != previous_assignment)
            )
        )

    hit_targets = infer_hit_targets(rows)
    compute = [
        row.get("control_compute_ms", float("nan")) for row in rows
    ]
    return {
        "log": str(path),
        "rows": len(rows),
        "duration_s": float(timestamps[-1]),
        "mean_log_dt_s": float(np.diff(timestamps).mean()),
        "hits": len(hit_targets),
        "hit_targets": [target_id + 11 for target_id in hit_targets],
        "control_compute_ms": {
            **distribution(compute),
            "over_100_ms": int(np.count_nonzero(np.asarray(compute) > 100.0)),
        },
        "actual_speed": distribution(actual_speed),
        "desired_speed": distribution(desired_speed),
        "published_speed": distribution(published_speed),
        "command_acceleration": distribution(command_acceleration),
        "command_turn_rate_deg_s": distribution(command_turn_rate),
        "target_acceleration": distribution(target_acceleration),
        "guidance_blend": distribution(guidance_blend),
        "assignment_switches": assignment_switches,
        "online_tracker_cv_error": online_cv_errors(
            rows, (0.5, 1.0, 2.0, 3.0, 5.0)
        ),
        "targets": [
            {
                "target_id": target_id + 11,
                "hit": target_id in hit_targets,
                "sampled_minimum_distance_m": float(
                    minimum_distance[target_id]
                ),
                "closing_speed_at_minimum_m_s": float(
                    closing_at_minimum[target_id]
                ),
                "controller_time_at_minimum_s": float(
                    time_at_minimum[target_id]
                ),
            }
            for target_id in range(10)
        ],
        "caveat": (
            "距离使用控制器跟踪状态和约 0.16 s 日志采样，只是上界近似；"
            "命中以目标停止后 2 s 内拦截机消失推断。"
        ),
    }


def main() -> int:
    args = parse_args()
    baseline = analyze(args.baseline)
    candidate = analyze(args.candidate)
    payload = {
        args.baseline_label: baseline,
        args.candidate_label: candidate,
        "comparison": {
            "hit_delta": candidate["hits"] - baseline["hits"],
            "compute_p95_delta_ms": (
                candidate["control_compute_ms"]["p95"]
                - baseline["control_compute_ms"]["p95"]
            ),
            "assignment_switch_delta": (
                candidate["assignment_switches"]
                - baseline["assignment_switches"]
            ),
        },
    }
    print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2))
    for label, report in (
        (args.baseline_label, baseline),
        (args.candidate_label, candidate),
    ):
        print(
            f"{label}: hits={report['hits']} "
            f"targets={report['hit_targets']} "
            f"compute_p95={report['control_compute_ms']['p95']:.1f}ms "
            f"switches={report['assignment_switches']}"
        )
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
