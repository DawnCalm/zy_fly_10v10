import unittest

import numpy as np

from zhuoyi_mappo.flight_response import FEATURE_COUNT, RidgeFlightResponseModel
from zhuoyi_mappo.predictive_control import (
    CausalFlightHistory,
    PredictiveCPAConfig,
    select_predictive_cpa_corrections_batch,
)


def _command_response_model() -> RidgeFlightResponseModel:
    coefficient = np.zeros((FEATURE_COUNT, 3), dtype=np.float64)
    coefficient[3:6] = np.eye(3)
    return RidgeFlightResponseModel(
        input_mean=np.zeros(FEATURE_COUNT),
        input_scale=np.ones(FEATURE_COUNT),
        coefficient=coefficient,
        output_mean=np.zeros(3),
        acceleration_limit=20.0,
    )


class PredictiveControlTests(unittest.TestCase):
    def test_zero_action_is_exact_when_gate_is_closed(self):
        result = select_predictive_cpa_corrections_batch(
            model=_command_response_model(),
            interceptor_position=np.zeros((1, 3)),
            velocity_history=np.zeros((1, 4, 3)),
            previous_command_history=np.zeros((1, 3, 3)),
            history_dt=np.full((1, 2), 0.1),
            target_position=np.array([[100.0, 0.0, 0.0]]),
            target_velocity=np.zeros((1, 3)),
            base_desired_velocity=np.array([[10.0, 0.0, 0.0]]),
            limiter_value=np.zeros((1, 3)),
            pn_acceleration=np.array([[0.0, 1.0, 0.0]]),
            first_dt=0.1,
            maximum_speed=30.0,
            maximum_acceleration=5.0,
        )
        self.assertFalse(result.evaluated[0])
        np.testing.assert_array_equal(result.correction_velocity, 0.0)

    def test_selects_bounded_pn_direction_correction(self):
        config = PredictiveCPAConfig(
            horizon_s=1.0,
            actions_mps=(-1.0, 0.0, 1.0),
            minimum_improvement_m=0.0,
        )
        result = select_predictive_cpa_corrections_batch(
            model=_command_response_model(),
            interceptor_position=np.zeros((1, 3)),
            velocity_history=np.zeros((1, 4, 3)),
            previous_command_history=np.zeros((1, 3, 3)),
            history_dt=np.full((1, 2), 0.1),
            target_position=np.array([[10.0, 2.0, 0.0]]),
            target_velocity=np.zeros((1, 3)),
            base_desired_velocity=np.array([[10.0, 0.0, 0.0]]),
            limiter_value=np.zeros((1, 3)),
            pn_acceleration=np.array([[0.0, 1.0, 0.0]]),
            first_dt=0.1,
            maximum_speed=30.0,
            maximum_acceleration=20.0,
            config=config,
        )
        self.assertTrue(result.evaluated[0])
        self.assertEqual(result.selected_action_mps[0], 1.0)
        self.assertGreater(result.improvement_m[0], 0.0)
        np.testing.assert_allclose(result.correction_velocity[0], [0.0, 1.0, 0.0])

    def test_history_is_causal_and_newest_first(self):
        history = CausalFlightHistory(1)
        for index in range(3):
            history.update(
                np.array([[float(index), 0.0, 0.0]]),
                np.array([[0.0, float(index), 0.0]]),
                0.1 * index,
                np.ones(1, dtype=bool),
            )
        velocities, commands, dt = history.inputs(
            0, np.array([3.0, 0.0, 0.0]), 0.3
        )
        np.testing.assert_allclose(velocities[:, 0], [3.0, 2.0, 1.0, 0.0])
        np.testing.assert_allclose(commands[:, 1], [2.0, 1.0, 0.0])
        np.testing.assert_allclose(dt, [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
