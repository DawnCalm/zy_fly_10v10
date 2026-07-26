import unittest
from unittest.mock import patch

import numpy as np

from zhuoyi_mappo.config import ControllerConfig
from zhuoyi_mappo.runtime_core import build_guidance_inputs


class DeadlineControlTests(unittest.TestCase):
    def setUp(self):
        self.config = ControllerConfig(
            num_agents=2,
            num_targets=2,
            interceptor_max_speed=30.0,
            assignment_switch_penalty_s=0.0,
        )
        self.agent_pos = np.array(
            [[-200.0, -400.0, 0.0], [-100.0, -400.0, 0.0]]
        )
        self.target_pos = np.array(
            [[100.0, 0.0, 0.0], [400.0, 0.0, 0.0]]
        )
        self.target_vel = np.array(
            [[0.0, 20.0, 0.0], [0.0, 10.0, 0.0]]
        )

    def _build(self, enabled: bool):
        return build_guidance_inputs(
            self.config,
            self.agent_pos,
            np.zeros((2, 3)),
            np.ones(2, dtype=bool),
            self.target_pos,
            self.target_vel,
            np.ones(2, dtype=bool),
            np.zeros(3),
            None,
            difficulty="high",
            use_target_deadline=enabled,
        )

    def test_disabled_deadline_is_not_used_for_assignment(self):
        assignment = np.full(2, -1, dtype=np.int64)
        with patch(
            "zhuoyi_mappo.runtime_core.assign_targets",
            return_value=(assignment, np.zeros((2, 2))),
        ) as mocked:
            enabled = self._build(True)
            disabled = self._build(False)

        enabled_kwargs = mocked.call_args_list[0].kwargs
        disabled_kwargs = mocked.call_args_list[1].kwargs
        np.testing.assert_allclose(
            enabled_kwargs["target_deadline_s"], [3.5, 37.0]
        )
        self.assertIsNone(disabled_kwargs["target_deadline_s"])
        self.assertTrue(np.isfinite(enabled.target_deadline).all())
        self.assertTrue(np.isinf(disabled.target_deadline).all())

    def test_deadline_switch_changes_assignment(self):
        np.testing.assert_array_equal(self._build(True).assignment, [1, 0])
        np.testing.assert_array_equal(self._build(False).assignment, [0, 1])


if __name__ == "__main__":
    unittest.main()
