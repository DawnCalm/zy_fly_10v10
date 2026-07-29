import unittest

import numpy as np

from zhuoyi_mappo.config import EnvConfig
from zhuoyi_mappo.los_kalman import AdaptiveLOSKalmanObserver
from zhuoyi_mappo.runtime_core import (
    build_guidance_inputs,
    default_guidance_mode,
)


class AdaptiveLOSKalmanTests(unittest.TestCase):
    def test_default_guidance_uses_kf_only_for_high(self):
        self.assertEqual(default_guidance_mode("low"), "los_pn")
        self.assertEqual(default_guidance_mode("mid"), "los_pn")
        self.assertEqual(default_guidance_mode("high"), "los_pn_kf")

    @staticmethod
    def _update(
        observer: AdaptiveLOSKalmanObserver,
        timestamp: float,
        angle: float,
        kinematic_rate: float,
        target_id: int = 0,
    ):
        distance = 100.0
        direction = np.array(
            [np.cos(angle), np.sin(angle), 0.0], dtype=np.float64
        )
        target_position = distance * direction
        target_velocity = np.cross(
            np.array([0.0, 0.0, kinematic_rate]),
            target_position,
        )
        return observer.update(
            timestamp=timestamp,
            assignment=np.array([target_id]),
            agent_pos=np.zeros((1, 3)),
            agent_vel=np.zeros((1, 3)),
            agent_active=np.ones(1, dtype=bool),
            target_pos=target_position[None],
            target_vel=target_velocity[None],
            target_active=np.ones(1, dtype=bool),
            raw_target_pos=target_position[None],
            raw_target_time=np.array([timestamp]),
        )

    def test_direction_updates_reduce_a_lagged_kinematic_rate(self):
        observer = AdaptiveLOSKalmanObserver(count=1)
        result = None
        for step in range(30):
            timestamp = 0.1 * step
            result = self._update(
                observer,
                timestamp,
                angle=0.20 * timestamp,
                kinematic_rate=0.10,
            )

        self.assertIsNotNone(result)
        self.assertTrue(bool(result.used[0]))
        self.assertGreater(float(result.los_rate[0, 2]), 0.10)
        self.assertLessEqual(
            float(result.rate_correction[0]),
            observer.maximum_rate_correction + 1.0e-6,
        )

    def test_direction_outlier_is_gated_and_rate_correction_is_bounded(self):
        observer = AdaptiveLOSKalmanObserver(count=1)
        for step in range(10):
            result = self._update(
                observer,
                timestamp=0.1 * step,
                angle=0.02 * step,
                kinematic_rate=0.20,
            )
        result = self._update(
            observer,
            timestamp=1.0,
            angle=1.5,
            kinematic_rate=0.20,
        )

        self.assertTrue(np.isfinite(result.los_rate).all())
        self.assertLessEqual(
            float(result.rate_correction[0]),
            observer.maximum_rate_correction + 1.0e-6,
        )
        self.assertLessEqual(
            float(np.linalg.norm(result.los_rate[0])),
            observer.maximum_angular_rate + 1.0e-6,
        )
        self.assertGreater(float(result.process_scale[0]), 1.0)

    def test_stale_measurement_falls_back_to_kinematic_rate(self):
        observer = AdaptiveLOSKalmanObserver(count=1)
        result = observer.update(
            timestamp=1.0,
            assignment=np.array([0]),
            agent_pos=np.zeros((1, 3)),
            agent_vel=np.zeros((1, 3)),
            agent_active=np.ones(1, dtype=bool),
            target_pos=np.array([[100.0, 0.0, 0.0]]),
            target_vel=np.array([[0.0, 10.0, 0.0]]),
            target_active=np.ones(1, dtype=bool),
            raw_target_pos=np.array([[100.0, 0.0, 0.0]]),
            raw_target_time=np.array([0.0]),
        )

        self.assertFalse(bool(result.used[0]))
        np.testing.assert_allclose(result.los_rate[0], [0.0, 0.0, 0.1])

    def test_runtime_candidate_uses_override_and_is_high_only(self):
        config = EnvConfig(
            num_agents=1,
            num_targets=1,
            interceptor_max_speed=30.0,
            interceptor_max_acceleration=5.0,
        )
        common = (
            config,
            np.zeros((1, 3), dtype=np.float32),
            np.array([[20.0, 0.0, 0.0]], dtype=np.float32),
            np.ones(1, dtype=bool),
            np.array([[80.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 10.0, 0.0]], dtype=np.float32),
            np.ones(1, dtype=bool),
            np.zeros(3, dtype=np.float32),
            None,
        )
        result = build_guidance_inputs(
            *common,
            difficulty="high",
            guidance_mode="los_pn_kf",
            use_target_deadline=False,
            los_rate_override=np.array([[0.0, 0.0, 0.05]]),
        )

        self.assertAlmostEqual(float(result.los_angular_rate[0]), 0.05)
        with self.assertRaisesRegex(ValueError, "仅允许 High"):
            build_guidance_inputs(
                *common,
                difficulty="mid",
                guidance_mode="los_pn_kf",
                use_target_deadline=False,
            )


if __name__ == "__main__":
    unittest.main()
