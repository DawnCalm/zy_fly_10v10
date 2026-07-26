import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zhuoyi_mappo.gru_prediction import (
    DEFAULT_HORIZONS,
    GRUResidualModel,
    GRUResidualPredictor,
    history_to_local_features,
)
from zhuoyi_mappo.guidance import apn_zem_velocity
from zhuoyi_mappo.prediction import (
    IMMTargetPredictor,
    constant_velocity_prediction,
)
from zhuoyi_mappo.trajectory_data import load_ros_target_trajectories


class PredictionTests(unittest.TestCase):
    def test_constant_velocity_helper(self):
        prediction = constant_velocity_prediction(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, -1.0, 0.5]),
            [0.0, 2.0],
        )
        np.testing.assert_allclose(
            prediction,
            [[1.0, 2.0, 3.0], [5.0, 0.0, 4.0]],
        )

    def test_imm_tracks_constant_velocity(self):
        predictor = IMMTargetPredictor(measurement_std=0.05)
        velocity = np.array([12.0, -3.0, 1.5])
        for step in range(101):
            timestamp = step * 0.1
            predictor.update(velocity * timestamp, timestamp)

        prediction = predictor.predict([0.5, 2.0])
        truth = np.stack(
            (velocity * 10.5, velocity * 12.0), axis=0
        )
        np.testing.assert_allclose(
            prediction.position, truth, atol=0.12
        )
        self.assertAlmostEqual(
            sum(prediction.mode_probabilities.values()), 1.0, places=6
        )
        self.assertTrue(np.isfinite(prediction.covariance).all())

    def test_imm_turn_model_improves_over_constant_velocity(self):
        radius = 80.0
        turn_rate = 0.12
        predictor = IMMTargetPredictor(measurement_std=0.05)
        last_position = None
        for step in range(201):
            timestamp = step * 0.1
            angle = turn_rate * timestamp
            last_position = np.array(
                [radius * np.cos(angle), radius * np.sin(angle), 50.0]
            )
            predictor.update(last_position, timestamp)

        horizon = 2.0
        future_angle = turn_rate * (20.0 + horizon)
        truth = np.array(
            [
                radius * np.cos(future_angle),
                radius * np.sin(future_angle),
                50.0,
            ]
        )
        estimate = predictor.estimate()
        cv = estimate.position + horizon * estimate.velocity
        imm = predictor.predict([horizon]).position[0]
        self.assertLess(
            float(np.linalg.norm(imm - truth)),
            float(np.linalg.norm(cv - truth)),
        )

    def test_gru_features_are_translation_and_rotation_local(self):
        timestamps = np.arange(21, dtype=np.float64) * 0.2
        positions = np.column_stack(
            (
                10.0 + timestamps * 5.0,
                np.full_like(timestamps, 20.0),
                30.0 + timestamps,
            )
        )
        features, basis = history_to_local_features(
            timestamps, positions, history_steps=20, sample_dt=0.2
        )
        self.assertEqual(features.shape, (20, 6))
        np.testing.assert_allclose(features[-1, :3], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            basis.T @ basis, np.eye(3), atol=1.0e-6
        )
        self.assertGreater(float(features[-1, 3]), 0.0)

    def test_gru_predictor_falls_back_when_history_is_short(self):
        model = GRUResidualModel(horizon_count=len(DEFAULT_HORIZONS))
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        predictor = GRUResidualPredictor(model)
        timestamps = np.arange(5, dtype=np.float64) * 0.2
        positions = np.zeros((5, 3), dtype=np.float32)
        base = np.ones((len(DEFAULT_HORIZONS), 3), dtype=np.float32)
        result = predictor.predict(
            timestamps, positions, base, DEFAULT_HORIZONS
        )
        self.assertFalse(result.used_gru)
        np.testing.assert_allclose(result.position, base)

    def test_apn_turns_toward_lateral_zero_effort_miss(self):
        command, time_to_go, blend = apn_zem_velocity(
            interceptor_pos=np.array([0.0, 0.0, 0.0]),
            interceptor_velocity=np.array([20.0, 0.0, 0.0]),
            target_pos=np.array([100.0, 0.0, 0.0]),
            target_velocity=np.array([0.0, 8.0, 0.0]),
            target_acceleration=np.zeros(3),
            classic_velocity=np.array([30.0, 0.0, 0.0]),
            interceptor_speed=30.0,
            activation_distance=200.0,
            full_distance=120.0,
        )
        self.assertGreater(time_to_go, 0.0)
        self.assertGreater(blend, 0.0)
        self.assertGreater(float(command[1]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(command)), 30.0001)


class TrajectoryDataTests(unittest.TestCase):
    def test_loader_keeps_only_fresh_active_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            rows = []
            for step in range(10):
                rows.append(
                    {
                        "wall_time": 100.0 + step * 0.1,
                        "target_pos": [
                            [float(step), 0.0, 10.0],
                            [0.0, float(step), 20.0],
                        ],
                        "target_active": [1, 1],
                        "target_retired": [0, 0],
                        "target_age": [
                            0.05,
                            0.05 if step < 4 else 0.5,
                        ],
                    }
                )
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            trajectories = load_ros_target_trajectories(
                [path], maximum_target_age=0.25, minimum_samples=5
            )
            self.assertEqual(len(trajectories), 1)
            self.assertEqual(trajectories[0].target_id, 0)
            self.assertEqual(len(trajectories[0].timestamps), 10)
            self.assertEqual(trajectories[0].velocities.shape, (10, 3))
            np.testing.assert_allclose(
                trajectories[0].interpolate(0.45),
                [4.5, 0.0, 10.0],
                atol=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
