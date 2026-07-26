import unittest

import numpy as np

from zhuoyi_mappo.config import EnvConfig
from zhuoyi_mappo.runtime_core import build_guidance_inputs
from zhuoyi_mappo.tracking import (
    AlphaBetaTrack,
    PositionContinuityGuard,
    TargetReacquisitionHold,
)


class PositionContinuityGuardTests(unittest.TestCase):
    def test_first_frame_is_accepted(self):
        guard = PositionContinuityGuard()

        self.assertTrue(guard.update(np.array([800.0, 20.0, 100.0]), 10.0))
        self.assertTrue(guard.initialized)
        self.assertFalse(guard.quarantined)
        np.testing.assert_allclose(guard.last_position, [800.0, 20.0, 100.0])

    def test_normal_thirty_meter_per_second_motion_is_accepted(self):
        guard = PositionContinuityGuard(
            max_speed=40.0,
            min_jump_distance=20.0,
            distance_margin=1.0,
        )
        self.assertTrue(guard.update(np.array([800.0, 0.0, 100.0]), 0.0))

        for step in range(1, 21):
            timestamp = step * 0.1
            position = np.array([800.0 - 30.0 * timestamp, 0.0, 100.0])
            self.assertTrue(guard.update(position, timestamp))

        self.assertFalse(guard.quarantined)

    def test_small_displacement_is_not_a_teleport_at_short_dt(self):
        guard = PositionContinuityGuard(
            max_speed=30.0,
            min_jump_distance=10.0,
            distance_margin=0.0,
        )
        self.assertTrue(guard.update(np.zeros(3), 1.0))
        self.assertTrue(guard.update(np.array([0.5, 0.0, 0.0]), 1.001))
        self.assertFalse(guard.quarantined)

    def test_eight_hundred_meter_reset_is_quarantined_for_the_round(self):
        guard = PositionContinuityGuard(
            max_speed=40.0,
            min_jump_distance=50.0,
            distance_margin=5.0,
        )
        self.assertTrue(guard.update(np.array([837.0, 0.0, 100.0]), 79.7))

        self.assertFalse(guard.update(np.zeros(3), 79.8))
        self.assertTrue(guard.quarantined)
        self.assertIn("impossible position jump", guard.quarantine_reason)

        # 后续看似正常的帧也不能让实体重新进入本局分配。
        self.assertFalse(guard.update(np.array([836.0, 0.0, 100.0]), 79.9))
        np.testing.assert_allclose(guard.last_position, [837.0, 0.0, 100.0])

    def test_non_monotonic_frame_is_rejected_without_permanent_quarantine(self):
        guard = PositionContinuityGuard()
        self.assertTrue(guard.update(np.array([100.0, 0.0, 0.0]), 5.0))

        self.assertFalse(guard.update(np.array([900.0, 0.0, 0.0]), 5.0))
        self.assertFalse(guard.quarantined)
        self.assertEqual(guard.last_rejection_reason, "non-monotonic timestamp")
        np.testing.assert_allclose(guard.last_position, [100.0, 0.0, 0.0])

        self.assertTrue(guard.update(np.array([103.0, 0.0, 0.0]), 5.1))

    def test_reset_starts_a_new_round(self):
        guard = PositionContinuityGuard()
        self.assertTrue(guard.update(np.array([900.0, 0.0, 0.0]), 0.0))
        self.assertFalse(guard.update(np.zeros(3), 0.1))
        self.assertTrue(guard.quarantined)

        guard.reset()

        self.assertFalse(guard.initialized)
        self.assertFalse(guard.quarantined)
        self.assertIsNone(guard.quarantine_reason)
        self.assertTrue(guard.update(np.zeros(3), 0.0))

    def test_quarantined_entity_is_excluded_from_runtime_assignment(self):
        config = EnvConfig()
        guards = [PositionContinuityGuard() for _ in range(config.num_agents)]
        for agent_id, guard in enumerate(guards):
            self.assertTrue(
                guard.update(
                    np.array([800.0, 10.0 * agent_id, 100.0]),
                    10.0,
                )
            )
        self.assertFalse(guards[0].update(np.zeros(3), 10.1))

        # 与 RosStateCache.snapshot 相同：quarantine 是 active mask 的硬条件。
        agent_active = ~np.asarray(
            [guard.quarantined for guard in guards], dtype=bool
        )
        agent_pos = np.column_stack(
            (
                np.zeros(config.num_agents),
                np.arange(config.num_agents) * 10.0,
                np.full(config.num_agents, 100.0),
            )
        )
        target_pos = agent_pos.copy()
        target_pos[:, 0] = 300.0
        previous_assignment = np.arange(config.num_agents, dtype=np.int64)

        guidance = build_guidance_inputs(
            config,
            agent_pos,
            np.zeros_like(agent_pos),
            agent_active,
            target_pos,
            np.zeros_like(target_pos),
            np.ones(config.num_targets, dtype=bool),
            np.zeros(3),
            previous_assignment,
            difficulty="high",
            use_target_deadline=False,
        )

        self.assertEqual(int(guidance.assignment[0]), -1)
        np.testing.assert_allclose(guidance.guide_velocity[0], 0.0)
        self.assertEqual(
            int(np.count_nonzero(guidance.assignment >= 0)),
            config.num_agents - 1,
        )


class AlphaBetaGapTests(unittest.TestCase):
    def test_update_propagates_the_full_gap(self):
        track = AlphaBetaTrack(alpha=0.0, beta=0.0, max_dt=0.5)
        track.position[:] = 0.0
        track.velocity[:] = (10.0, 0.0, 0.0)
        track.last_time = 0.0
        track.samples = 2

        track.update(np.array([10.0, 0.0, 0.0]), 1.0)

        np.testing.assert_allclose(track.position, [10.0, 0.0, 0.0])


class TargetReacquisitionHoldTests(unittest.TestCase):
    def test_stale_target_is_held_until_timeout(self):
        guard = TargetReacquisitionHold(3)
        guard.hold(1, 10.0)

        filtered = guard.filter(
            np.ones(3, dtype=bool),
            np.array([10.0, 10.0, 10.0]),
        )
        np.testing.assert_array_equal(filtered, [True, False, True])

        filtered = guard.filter(
            np.array([True, False, True]),
            np.array([10.1, 10.0, 10.1]),
        )
        np.testing.assert_array_equal(filtered, [True, False, True])
        self.assertFalse(guard.held[1])

    def test_new_radar_frame_immediately_releases_live_target(self):
        guard = TargetReacquisitionHold(2)
        guard.hold(0, 20.0)

        filtered = guard.filter(
            np.ones(2, dtype=bool),
            np.array([20.1, 20.0]),
        )

        np.testing.assert_array_equal(filtered, [True, True])
        self.assertFalse(guard.held[0])


if __name__ == "__main__":
    unittest.main()
