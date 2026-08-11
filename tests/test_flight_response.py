import unittest

import numpy as np

from zhuoyi_mappo.flight_response import (
    FEATURE_COUNT,
    RidgeFlightResponseModel,
    flight_response_features,
)


class FlightResponseTests(unittest.TestCase):
    def _history(self):
        velocity = np.array(
            [
                [10.0, 0.0, 0.0],
                [9.8, 0.0, 0.0],
                [9.6, 0.0, 0.0],
                [9.4, 0.0, 0.0],
            ]
        )
        command = np.array(
            [
                [12.0, 1.0, 0.0],
                [11.8, 0.8, 0.0],
                [11.6, 0.6, 0.0],
                [11.4, 0.4, 0.0],
            ]
        )
        return velocity, command

    def test_features_are_rotation_invariant(self):
        velocity, command = self._history()
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        original, _, _ = flight_response_features(
            velocity, command, np.array([0.1, 0.1]), np.array(0.1)
        )
        rotated, _, _ = flight_response_features(
            velocity @ rotation.T,
            command @ rotation.T,
            np.array([0.1, 0.1]),
            np.array(0.1),
        )
        np.testing.assert_allclose(rotated, original, atol=1.0e-8)
        self.assertEqual(original.shape, (FEATURE_COUNT,))

    def test_world_prediction_rotates_with_history(self):
        velocity, command = self._history()
        coefficient = np.zeros((FEATURE_COUNT, 3))
        model = RidgeFlightResponseModel(
            input_mean=np.zeros(FEATURE_COUNT),
            input_scale=np.ones(FEATURE_COUNT),
            coefficient=coefficient,
            output_mean=np.array([1.0, 2.0, 3.0]),
        )
        predicted = model.predict_world_acceleration(
            velocity, command, np.array([0.1, 0.1]), np.array(0.1)
        )
        np.testing.assert_allclose(predicted, [1.0, 2.0, 3.0])

        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rotated = model.predict_world_acceleration(
            velocity @ rotation.T,
            command @ rotation.T,
            np.array([0.1, 0.1]),
            np.array(0.1),
        )
        np.testing.assert_allclose(rotated, predicted @ rotation.T)

    def test_fit_learns_simple_causal_response(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(500, FEATURE_COUNT))
        coefficient = rng.normal(scale=0.05, size=(FEATURE_COUNT, 3))
        target = features @ coefficient + np.array([0.2, -0.1, 0.05])
        model = RidgeFlightResponseModel.fit(
            features, target, ridge=1.0e-6, acceleration_limit=100.0
        )
        prediction = model.predict_local_acceleration(features)
        self.assertLess(float(np.mean(np.abs(prediction - target))), 1.0e-5)

    def test_rejects_noncausal_history_shape(self):
        velocity, command = self._history()
        with self.assertRaisesRegex(ValueError, "\(4, 3\)"):
            flight_response_features(
                velocity[:3], command, np.array([0.1, 0.1]), np.array(0.1)
            )


if __name__ == "__main__":
    unittest.main()
