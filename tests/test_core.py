import unittest

import numpy as np

from zhuoyi_mappo.assignment import (
    assign_targets,
    intercept_time,
    lead_velocity,
)
from zhuoyi_mappo.config import ControllerConfig
from zhuoyi_mappo.runtime_core import build_guidance_inputs
from zhuoyi_mappo.tracking import (
    AlphaBetaTrack,
    TargetRetirementDetector,
    VelocityLimiter,
)


class AssignmentTests(unittest.TestCase):
    def test_stationary_intercept_time(self):
        value = intercept_time(
            np.zeros(3),
            np.array([100.0, 0.0, 0.0]),
            np.zeros(3),
            10.0,
        )
        self.assertAlmostEqual(value, 10.0)

    def test_assignment_is_one_to_one(self):
        count = 10
        agents = np.column_stack(
            (np.zeros(count), np.arange(count) * 10.0, np.zeros(count))
        )
        targets = agents.copy()
        targets[:, 0] = 100.0
        assignment, _ = assign_targets(
            agents,
            np.ones(count, dtype=bool),
            targets,
            np.zeros((count, 3)),
            np.ones(count, dtype=bool),
            20.0,
        )
        self.assertEqual(len(set(assignment.tolist())), count)

    def test_terminal_guidance_uses_target_velocity_feedforward(self):
        velocity, _ = lead_velocity(
            np.zeros(3),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            interceptor_speed=20.0,
            terminal_distance=60.0,
            terminal_gain=0.5,
        )
        np.testing.assert_allclose(velocity, [1.0, 10.0, 0.0])

    def test_high_defaults_are_preserved(self):
        config = ControllerConfig()
        self.assertEqual(
            config.terminal_guidance_params("high"), (15.0, 2.0)
        )
        self.assertEqual(
            config.lead_prediction_horizons("high"), (20.0, 20.0)
        )


class TrackingTests(unittest.TestCase):
    def test_alpha_beta_estimates_constant_velocity(self):
        track = AlphaBetaTrack(alpha=0.8, beta=0.2)
        for step in range(20):
            timestamp = step * 0.1
            track.update(
                np.array([2.0 * timestamp, 0.0, 5.0]),
                timestamp,
            )
        position, velocity = track.predict(2.0)
        self.assertAlmostEqual(float(position[0]), 4.0, delta=0.15)
        self.assertAlmostEqual(float(velocity[0]), 2.0, delta=0.2)

    def test_velocity_limiter_limits_speed_and_acceleration(self):
        limiter = VelocityLimiter(1, max_speed=10.0, max_acceleration=2.0)
        command = limiter.update(np.array([[20.0, 0.0, 0.0]]), dt=0.5)
        self.assertLessEqual(float(np.linalg.norm(command[0])), 1.0001)
        for _ in range(20):
            command = limiter.update(
                np.array([[20.0, 0.0, 0.0]]), dt=0.5
            )
        self.assertLessEqual(float(np.linalg.norm(command[0])), 10.0001)

    def test_retirement_requires_motion_then_stationary(self):
        detector = TargetRetirementDetector(
            1,
            moving_speed=5.0,
            stationary_speed=0.5,
            stationary_grace=1.5,
        )
        self.assertFalse(detector.update([8.0], [True], 0.0)[0])
        self.assertFalse(detector.update([0.0], [True], 1.0)[0])
        self.assertTrue(detector.update([0.0], [True], 2.6)[0])


class RuntimeTests(unittest.TestCase):
    def test_classic_runtime_output_is_finite(self):
        config = ControllerConfig()
        count = config.num_agents
        agent_pos = np.column_stack(
            (np.zeros(count), np.arange(count) * 5.0, np.zeros(count))
        )
        target_pos = agent_pos.copy()
        target_pos[:, 0] = 500.0
        target_pos[:, 2] = 100.0
        result = build_guidance_inputs(
            config,
            agent_pos,
            np.zeros_like(agent_pos),
            np.ones(count, dtype=bool),
            target_pos,
            np.tile(np.array([-15.0, 0.0, 0.0]), (count, 1)),
            np.ones(count, dtype=bool),
            np.zeros(3),
            None,
            difficulty="high",
            guidance_mode="classic",
            use_target_deadline=False,
        )
        self.assertEqual(result.guide_velocity.shape, (count, 3))
        self.assertEqual(len(set(result.assignment.tolist())), count)
        self.assertTrue(np.isfinite(result.guide_velocity).all())
        np.testing.assert_allclose(result.guidance_blend, 0.0)


if __name__ == "__main__":
    unittest.main()
