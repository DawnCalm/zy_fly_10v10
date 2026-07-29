import unittest

import numpy as np

from zhuoyi_mappo.assignment import lead_velocity
from zhuoyi_mappo.config import EnvConfig
from zhuoyi_mappo.guidance import (
    line_of_sight_kinematics,
    los_rate_pn_velocity,
)
from zhuoyi_mappo.runtime_core import build_guidance_inputs


class LOSRatePNTests(unittest.TestCase):
    def test_los_rate_and_closing_speed_follow_relative_kinematics(self):
        los_rate, closing_speed, distance = line_of_sight_kinematics(
            interceptor_pos=np.zeros(3),
            interceptor_velocity=np.array([20.0, 0.0, 0.0]),
            target_pos=np.array([100.0, 0.0, 0.0]),
            target_velocity=np.array([0.0, 10.0, 0.0]),
        )

        np.testing.assert_allclose(los_rate, [0.0, 0.0, 0.1])
        self.assertAlmostEqual(closing_speed, 20.0)
        self.assertAlmostEqual(distance, 100.0)

    def test_pn_turns_in_correct_direction_and_limits_acceleration(self):
        result = los_rate_pn_velocity(
            interceptor_pos=np.zeros(3),
            interceptor_velocity=np.array([20.0, 0.0, 0.0]),
            target_pos=np.array([100.0, 0.0, 0.0]),
            target_velocity=np.array([0.0, 10.0, 0.0]),
            classic_velocity=np.array([30.0, 0.0, 0.0]),
            interceptor_speed=100.0,
            navigation_constant=3.0,
            maximum_acceleration=5.0,
            response_lead_seconds=2.0,
            activation_time_to_go=7.0,
            full_time_to_go=6.0,
        )

        self.assertAlmostEqual(result.blend, 1.0)
        np.testing.assert_allclose(
            result.acceleration, [0.0, 5.0, 0.0], atol=1.0e-6
        )
        np.testing.assert_allclose(
            result.velocity, [30.0, 10.0, 0.0], atol=1.0e-6
        )
        self.assertLessEqual(
            float(np.linalg.norm(result.acceleration)), 5.000001
        )
        self.assertAlmostEqual(float(result.acceleration[0]), 0.0)

    def test_pn_stays_off_before_activation_time(self):
        classic = np.array([29.0, 2.0, 0.0], dtype=np.float32)
        result = los_rate_pn_velocity(
            interceptor_pos=np.zeros(3),
            interceptor_velocity=np.array([10.0, 0.0, 0.0]),
            target_pos=np.array([200.0, 0.0, 0.0]),
            target_velocity=np.array([0.0, 10.0, 0.0]),
            classic_velocity=classic,
            interceptor_speed=30.0,
            activation_time_to_go=5.0,
            full_time_to_go=3.0,
        )

        self.assertEqual(result.blend, 0.0)
        np.testing.assert_allclose(result.velocity, classic)
        np.testing.assert_allclose(result.acceleration, 0.0)

    def test_pn_regularizes_and_returns_to_classic_at_capture_range(self):
        classic = np.array([5.0, 10.0, 0.0], dtype=np.float32)
        result = los_rate_pn_velocity(
            interceptor_pos=np.zeros(3),
            interceptor_velocity=np.array([20.0, 0.0, 0.0]),
            target_pos=np.array([1.0, 0.0, 0.0]),
            target_velocity=np.array([0.0, 10.0, 0.0]),
            classic_velocity=classic,
            interceptor_speed=30.0,
            activation_time_to_go=5.0,
            full_time_to_go=3.0,
            regularization_distance=5.0,
            close_fade_distance=5.0,
            close_cutoff_distance=1.0,
        )

        self.assertEqual(result.blend, 0.0)
        np.testing.assert_allclose(result.velocity, classic)
        self.assertTrue(np.isfinite(result.los_rate).all())
        np.testing.assert_allclose(result.acceleration, 0.0)

    def test_runtime_exposes_los_diagnostics_and_bounded_pn(self):
        config = EnvConfig(
            num_agents=1,
            num_targets=1,
            interceptor_max_speed=30.0,
            interceptor_max_acceleration=5.0,
        )
        agent_pos = np.zeros((1, 3), dtype=np.float32)
        agent_velocity = np.array([[20.0, 0.0, 0.0]], dtype=np.float32)
        target_pos = np.array([[80.0, 0.0, 0.0]], dtype=np.float32)
        target_velocity = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        classic, _ = lead_velocity(
            agent_pos[0],
            target_pos[0],
            target_velocity[0],
            interceptor_speed=30.0,
            terminal_distance=15.0,
            terminal_gain=2.0,
        )

        result = build_guidance_inputs(
            config,
            agent_pos,
            agent_velocity,
            np.ones(1, dtype=bool),
            target_pos,
            target_velocity,
            np.ones(1, dtype=bool),
            np.zeros(3, dtype=np.float32),
            None,
            difficulty="high",
            guidance_mode="los_pn",
            use_target_deadline=False,
        )

        self.assertGreater(float(result.guidance_blend[0]), 0.0)
        self.assertAlmostEqual(float(result.los_angular_rate[0]), 0.125)
        self.assertAlmostEqual(float(result.closing_speed[0]), 20.0)
        self.assertLessEqual(
            float(np.linalg.norm(result.los_pn_acceleration[0])),
            config.interceptor_max_acceleration + 1.0e-6,
        )
        self.assertFalse(
            np.allclose(result.guide_velocity[0], classic, atol=1.0e-6)
        )

    def test_runtime_allows_los_pn_in_low_and_mid(self):
        config = EnvConfig(
            num_agents=1,
            num_targets=1,
            interceptor_max_speed=30.0,
            interceptor_max_acceleration=5.0,
        )
        for difficulty in ("low", "mid"):
            with self.subTest(difficulty=difficulty):
                result = build_guidance_inputs(
                    config,
                    np.zeros((1, 3), dtype=np.float32),
                    np.array([[20.0, 0.0, 0.0]], dtype=np.float32),
                    np.ones(1, dtype=bool),
                    np.array([[80.0, 0.0, 0.0]], dtype=np.float32),
                    np.array([[0.0, 10.0, 0.0]], dtype=np.float32),
                    np.ones(1, dtype=bool),
                    np.zeros(3, dtype=np.float32),
                    None,
                    difficulty=difficulty,
                    guidance_mode="los_pn",
                    use_target_deadline=False,
                )

                self.assertGreater(float(result.guidance_blend[0]), 0.0)
                self.assertGreater(
                    float(np.linalg.norm(result.los_pn_acceleration[0])),
                    0.0,
                )

if __name__ == "__main__":
    unittest.main()
