#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from zhuoyi_mappo.config import high_real_env_config
from zhuoyi_mappo.env import Kinematic10v10Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配对比较 High classic 与 IMM + 3D APN/ZEM"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--navigation-constant", type=float, default=3.0)
    parser.add_argument("--maximum-time-to-go", type=float, default=5.0)
    parser.add_argument("--response-lead-seconds", type=float, default=2.5)
    parser.add_argument("--activation-distance", type=float, default=300.0)
    parser.add_argument("--full-distance", type=float, default=80.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("classic", "imm_lead", "apn", "apn_truth"),
        default=["classic", "imm_lead", "apn_truth", "apn"],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(seed: int, guidance_mode: str, config) -> Dict[str, float]:
    env = Kinematic10v10Env(
        config,
        seed=seed,
        guidance_mode=guidance_mode,
        simulate_command_limiter=True,
    )
    env.reset(seed=seed, difficulty="high")
    done = False
    info: Dict[str, float] = {}
    action = np.zeros((config.num_agents, 3), dtype=np.float32)
    while not done:
        _, _, _, done, info = env.step(action)
    return info


def main() -> int:
    args = parse_args()
    if "classic" not in args.modes:
        raise ValueError("--modes 必须包含 classic，才能进行配对比较")
    config = high_real_env_config(max_steps=args.max_steps)
    config.apn_navigation_constant = args.navigation_constant
    config.apn_maximum_time_to_go = args.maximum_time_to_go
    config.apn_response_lead_seconds = args.response_lead_seconds
    config.apn_activation_distance = args.activation_distance
    config.apn_full_distance = args.full_distance
    rows: List[dict] = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        results = {
            mode: run(seed, mode, config) for mode in args.modes
        }
        row = {"seed": seed}
        for mode, result in results.items():
            row[f"{mode}_hits"] = result["hits"]
            row[f"{mode}_steps"] = result["episode_steps"]
        if "classic" in results:
            for mode, result in results.items():
                if mode != "classic":
                    row[f"{mode}_delta"] = (
                        result["hits"] - results["classic"]["hits"]
                    )
        rows.append(row)
        print(
            f"seed={seed} "
            + " ".join(
                f"{mode}={results[mode]['hits']:.0f}"
                for mode in args.modes
            )
        )
    summaries = {}
    classic_hits = np.asarray(
        [row["classic_hits"] for row in rows], dtype=np.float64
    )
    for mode in args.modes:
        values = np.asarray(
            [row[f"{mode}_hits"] for row in rows], dtype=np.float64
        )
        summary = {"mean_hits": float(values.mean())}
        if mode != "classic":
            delta = values - classic_hits
            summary.update(
                {
                    "mean_delta": float(delta.mean()),
                    "standard_error": float(
                        delta.std(ddof=1) / np.sqrt(len(delta))
                        if len(delta) > 1
                        else 0.0
                    ),
                    "improved": int(np.count_nonzero(delta > 0)),
                    "equal": int(np.count_nonzero(delta == 0)),
                    "worse": int(np.count_nonzero(delta < 0)),
                }
            )
        summaries[mode] = summary
    payload = {
        "parameters": {
            "navigation_constant": args.navigation_constant,
            "maximum_time_to_go": args.maximum_time_to_go,
            "response_lead_seconds": args.response_lead_seconds,
            "activation_distance": args.activation_distance,
            "full_distance": args.full_distance,
        },
        "episodes": args.episodes,
        "seed": args.seed,
        "modes": args.modes,
        "summaries": summaries,
        "rows": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
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
