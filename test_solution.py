"""Small regression checks for transformations and the submitted artifacts."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from solution import features, predict
from score import validate_december, validate_predictions


class SolutionTests(unittest.TestCase):
    def test_features_do_not_use_labels_or_ids(self):
        data = pd.read_csv('train-test.csv', nrows=8)
        original = features(data, 'full')
        data['posted_rate'] = -999
        data['load_id'] = 'changed'
        pd.testing.assert_frame_equal(original, features(data, 'full'))

    def test_core_requires_only_supplied_chart_inputs(self):
        data = pd.read_csv('december-chart-inputs.csv')
        x = features(data, 'core')
        self.assertEqual(len(x), 31)
        self.assertFalse({'quote_signal', 'market_index', 'posted_rate'} & set(x))
        self.assertGreater(x.day_index.nunique(), 1)

    def test_invalid_weight_becomes_missing(self):
        data = pd.read_csv('december-chart-inputs.csv').iloc[:3].copy()
        data['weight'] = [-5, 0, np.nan]
        x = features(data, 'core')
        self.assertTrue(x.weight.isna().all())
        self.assertEqual(x.weight_invalid.tolist(), [1, 1, 0])

    def test_no_quote_model_does_not_require_quote_column(self):
        data = pd.read_csv('train-test.csv', nrows=8)
        expected = features(data, 'no_quote')
        pd.testing.assert_frame_equal(expected, features(data.drop(columns='quote_signal'), 'no_quote'))

    def test_output_contract_and_template_order(self):
        final = pd.read_csv('output/validation_predictions.csv')
        validate_predictions(final)
        validate_december(pd.read_csv('output/december_predictions.csv'))
        self.assertEqual(final.load_id.tolist(), pd.read_csv('validation-predictions-template.csv').load_id.tolist())

    def test_saved_models_reproduce_outputs(self):
        specs = json.loads(Path('output/metrics.json').read_text())['model_specs']
        for role, input_path, output_path in [
            ('full', 'validation.csv', 'output/validation_predictions.csv'),
            ('core', 'december-chart-inputs.csv', 'output/december_predictions.csv')]:
            model = CatBoostRegressor()
            model.load_model(f'output/{role}_model.cbm')
            data = pd.read_csv(input_path)
            spec = specs[role]
            actual = predict(model, data, spec['mode'], spec['kind'])
            expected = pd.read_csv(output_path)
            if role == 'full':
                expected = expected.set_index('load_id').loc[data.load_id]
            np.testing.assert_allclose(actual, expected.predicted_rate, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
