import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zhuoyi_mappo.assignment import assign_targets, intercept_time, lead_velocity
from zhuoyi_mappo.config import EnvConfig, TrainConfig
from zhuoyi_mappo.env import Kinematic10v10Env
from zhuoyi_mappo.runtime_core import build_guidance_inputs
from zhuoyi_mappo.tracking import (
    AlphaBetaTrack,
    TargetRetirementDetector,
    VelocityLimiter,
)
from zhuoyi_mappo.trainer import MAPPOTrainer, load_policy


class AssignmentTests(unittest.TestCase):
    def test_intercept_time_for_stationary_target(self):
        value = intercept_time(
            np.zeros(3), np.array([100.0, 0.0, 0.0]), np.zeros(3), 10.0
        )
        self.assertAlmostEqual(value, 10.0, places=5)

    def test_assignment_is_one_to_one(self):
        count = 10
        agents = np.column_stack(
            (np.zeros(count), np.arange(count) * 10.0, np.zeros(count))
        )
        targets = np.column_stack(
            (np.ones(count) * 100.0, np.arange(count) * 10.0, np.zeros(count))
        )
        assignment, cost = assign_targets(
            agents,
            np.ones(count, dtype=bool),
            targets,
            np.zeros((count, 3)),
            np.ones(count, dtype=bool),
            20.0,
        )
        self.assertEqual(cost.shape, (count, count))
        self.assertEqual(len(set(assignment.tolist())), count)

    def test_terminal_guidance_matches_target_velocity_near_target(self):
        velocity, _ = lead_velocity(
            np.zeros(3),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            interceptor_speed=20.0,
            terminal_distance=60.0,
            terminal_gain=0.5,
        )
        np.testing.assert_allclose(velocity, [1.0, 10.0, 0.0], atol=1.0e-6)

    def test_vertical_prediction_horizon_is_independent(self):
        long_vertical, _ = lead_velocity(
            np.zeros(3),
            np.array([100.0, 0.0, 100.0]),
            np.array([0.0, 0.0, 8.0]),
            interceptor_speed=30.0,
            max_prediction_s=8.0,
            max_vertical_prediction_s=8.0,
        )
        short_vertical, _ = lead_velocity(
            np.zeros(3),
            np.array([100.0, 0.0, 100.0]),
            np.array([0.0, 0.0, 8.0]),
            interceptor_speed=30.0,
            max_prediction_s=8.0,
            max_vertical_prediction_s=1.0,
        )
        self.assertLess(float(short_vertical[2]), float(long_vertical[2]))

    def test_high_terminal_guidance_is_close_range_and_more_aggressive(self):
        config = EnvConfig()
        low_distance, low_gain = config.terminal_guidance_params("low")
        high_distance, high_gain = config.terminal_guidance_params("high")
        self.assertEqual(low_distance, 60.0)
        self.assertEqual(low_gain, 0.7)
        self.assertEqual(high_distance, 15.0)
        self.assertEqual(high_gain, 2.0)


class EnvironmentTests(unittest.TestCase):
    def test_shapes_and_finite_step(self):
        config = EnvConfig(max_steps=20)
        self.assertEqual(config.residual_fraction("low"), 0.15)
        self.assertEqual(config.residual_fraction("mid"), 0.30)
        self.assertEqual(config.residual_fraction("high"), 1.00)
        env = Kinematic10v10Env(config, seed=7)
        obs, state = env.reset(difficulty="high")
        self.assertEqual(obs.shape, (10, config.obs_dim))
        self.assertEqual(state.shape, (config.global_state_dim,))
        self.assertEqual(len(set(env.assignment.tolist())), 10)
        result = env.step(np.zeros((10, 3), dtype=np.float32))
        next_obs, next_state, rewards, done, info = result
        self.assertTrue(np.isfinite(next_obs).all())
        self.assertTrue(np.isfinite(next_state).all())
        self.assertTrue(np.isfinite(rewards).all())
        self.assertEqual(rewards.shape, (10,))
        self.assertIsInstance(done, bool)
        self.assertIn("hits", info)
        np.testing.assert_allclose(
            next_obs[:, 21:24], np.tile([0.0, 0.0, 1.0], (10, 1))
        )

    def test_swept_collision_does_not_tunnel(self):
        config = EnvConfig(collision_radius_min=0.5, collision_radius_max=0.5)
        env = Kinematic10v10Env(config, seed=9)
        env.reset(difficulty="low")
        env.agent_alive[1:] = False
        env.target_alive[1:] = False
        previous_agents = env.agent_pos.copy()
        previous_targets = env.target_pos.copy()
        previous_agents[0] = (-2.0, 0.0, 0.0)
        previous_targets[0] = (0.0, 0.0, 0.0)
        env.agent_pos[0] = (2.0, 0.0, 0.0)
        env.target_pos[0] = (0.0, 0.0, 0.0)
        self.assertEqual(env._resolve_hits(previous_agents, previous_targets), 1)

    def test_reset_uses_realistic_range_and_delayed_breakout(self):
        config = EnvConfig()
        env = Kinematic10v10Env(config, seed=13)
        env.reset(difficulty="low")
        ranges = np.linalg.norm(env.target_pos[:, :2] - env.safe_center[:2], axis=1)
        self.assertTrue(np.all(ranges >= config.target_range_min))
        self.assertTrue(np.all(ranges <= config.target_range_max + 20.0))
        np.testing.assert_allclose(env.target_vel[:, :2], 0.0, atol=1.0e-6)

    def test_short_mappo_update_and_checkpoint(self):
        env_config = EnvConfig(max_steps=25)
        train_config = TrainConfig(
            seed=3,
            total_updates=1,
            num_envs=2,
            rollout_steps=8,
            update_epochs=1,
            num_minibatches=2,
            hidden_dim=32,
            device="cpu",
        )
        trainer = MAPPOTrainer(env_config, train_config)
        metrics = trainer.train_update()
        self.assertTrue(np.isfinite(metrics["actor_loss"]))
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            trainer.save(checkpoint)
            policy, loaded_config, payload, device = load_policy(
                checkpoint, "cpu"
            )
            self.assertEqual(loaded_config.num_agents, 10)
            self.assertEqual(payload["update"], 1)
            self.assertEqual(device.type, "cpu")
            obs = torch.zeros((1, 10, loaded_config.obs_dim))
            state = torch.zeros((1, loaded_config.global_state_dim))
            actions, _, values = policy.act(obs, state, deterministic=True)
            self.assertEqual(tuple(actions.shape), (1, 10, 3))
            self.assertEqual(tuple(values.shape), (1, 10))
            resumed = MAPPOTrainer(env_config, train_config)
            resumed.load_training_state(checkpoint)
            self.assertEqual(resumed.update_index, 1)


class RuntimeCoreTests(unittest.TestCase):
    def test_tracker_estimates_constant_velocity(self):
        track = AlphaBetaTrack(alpha=0.8, beta=0.2)
        for step in range(20):
            timestamp = step * 0.1
            track.update(np.array([2.0 * timestamp, 0.0, 5.0]), timestamp)
        position, velocity = track.predict(2.0)
        self.assertAlmostEqual(float(position[0]), 4.0, delta=0.15)
        self.assertAlmostEqual(float(velocity[0]), 2.0, delta=0.2)

    def test_runtime_observation_matches_model_dimensions(self):
        config = EnvConfig()
        agent_pos = np.zeros((10, 3), dtype=np.float32)
        target_pos = np.zeros((10, 3), dtype=np.float32)
        agent_pos[:, 1] = np.arange(10) * 5.0
        target_pos[:, 0] = 500.0
        target_pos[:, 1] = np.arange(10) * 5.0
        target_pos[:, 2] = 100.0
        target_vel = np.tile(np.array([-15.0, 0.0, 0.0]), (10, 1))
        result = build_guidance_inputs(
            config,
            agent_pos,
            np.zeros((10, 3)),
            np.ones(10, dtype=bool),
            target_pos,
            target_vel,
            np.ones(10, dtype=bool),
            np.zeros(3),
            None,
            elapsed_s=0.0,
            difficulty="low",
        )
        self.assertEqual(result.observation.shape, (10, config.obs_dim))
        self.assertEqual(result.global_state.shape, (config.global_state_dim,))
        self.assertEqual(len(set(result.assignment.tolist())), 10)
        self.assertTrue(np.isfinite(result.guide_velocity).all())
        np.testing.assert_allclose(
            result.observation[:, 21:24],
            np.tile([1.0, 0.0, 0.0], (10, 1)),
        )

    def test_velocity_limiter_limits_acceleration(self):
        limiter = VelocityLimiter(2, max_speed=10.0, max_acceleration=2.0)
        command = limiter.update(
            np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]]),
            dt=0.5,
        )
        self.assertTrue(np.all(np.linalg.norm(command, axis=1) <= 1.0001))
        for _ in range(20):
            command = limiter.update(
                np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]]),
                dt=0.5,
            )
        self.assertTrue(np.all(np.linalg.norm(command, axis=1) <= 10.0001))

    def test_target_retirement_requires_motion_then_stationary_grace(self):
        detector = TargetRetirementDetector(
            2, moving_speed=5.0, stationary_speed=0.5, stationary_grace=1.5
        )
        active = np.ones(2, dtype=bool)
        self.assertFalse(detector.update([0.0, 0.0], active, 0.0).any())
        self.assertFalse(detector.update([8.0, 0.0], active, 1.0).any())
        self.assertFalse(detector.update([0.2, 0.0], active, 2.0).any())
        retired = detector.update([0.1, 0.0], active, 3.6)
        np.testing.assert_array_equal(retired, [True, False])


if __name__ == "__main__":
    unittest.main()
