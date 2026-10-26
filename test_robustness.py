"""Check temporal/geographic isolation and weighted-target training contracts."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from robustness import temporal_split, geographic_split
from solution import fit, predict


class RobustnessTests(unittest.TestCase):
    def test_future_labels_cannot_enter_development_evaluation(self):
        data = pd.DataFrame({'date': ['2025-01-01', '2025-07-01', '2025-09-01']})
        training, evaluation = temporal_split(data, '2025-07-01', '2025-09-01')
        self.assertEqual(training.date.tolist(), ['2025-01-01'])
        self.assertEqual(evaluation.date.tolist(), ['2025-07-01'])
        with self.assertRaises(ValueError):
            temporal_split(data, '2025-07-01', '2025-10-01')

    def test_held_city_is_removed_from_both_route_endpoints(self):
        training = pd.DataFrame({'pickup': ['A', 'B', 'B'], 'delivery': ['B', 'A', 'C']})
        evaluation = pd.DataFrame({'pickup': ['A', 'C'], 'delivery': ['C', 'B']})
        fitting, unseen = geographic_split(training, evaluation, ['A'])
        self.assertEqual(fitting.index.tolist(), [2])
        self.assertEqual(unseen.index.tolist(), [0])

    def test_weighted_rpm_training_and_dollar_inference(self):
        data = pd.read_csv('train-test.csv', nrows=4)
        with patch('solution.CatBoostRegressor') as constructor:
            model = constructor.return_value
            fit(data, 'core', 'weighted_per_mile')
            args, kwargs = model.fit.call_args
            np.testing.assert_allclose(args[1], data.posted_rate / data.distance)
            np.testing.assert_allclose(kwargs['sample_weight'], data.distance)
            model.predict.return_value = np.full(len(data), 2.0)
            np.testing.assert_allclose(predict(model, data, 'core', 'weighted_per_mile'), 2 * data.distance)

    def test_saved_experiment_metrics_and_label_boundaries(self):
        source = pd.read_csv('train-test.csv').set_index('load_id')
        for prefix, partition in [('temporal', 'fold'), ('geographic', 'seed')]:
            predictions = pd.read_csv(f'output/robustness/{prefix}_predictions.csv')
            reported = pd.read_csv(f'output/robustness/{prefix}_metrics.csv').set_index([partition, 'candidate'])
            actual = source.loc[predictions.load_id]
            self.assertTrue((actual.date < '2025-09-01').all())
            np.testing.assert_allclose(predictions.actual, actual.posted_rate)
            self.assertFalse(predictions.duplicated([partition, 'candidate', 'load_id']).any())
            for key, group in predictions.groupby([partition, 'candidate']):
                error = group.prediction - group.actual
                row = reported.loc[key]
                self.assertEqual(len(group), row['n'])
                self.assertAlmostEqual(abs(error).mean(), row.MAE, places=8)
                self.assertAlmostEqual(np.sqrt((error ** 2).mean()), row.RMSE, places=8)
                if prefix == 'geographic':
                    cities = row.held_cities.split('|')
                    evaluation = source.loc[group.load_id]
                    self.assertTrue((evaluation.pickup.isin(cities) | evaluation.delivery.isin(cities)).all())


if __name__ == '__main__':
    unittest.main()
