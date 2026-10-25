"""Post-evaluation diagnostics from saved holdout predictions; never fit a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(frame):
    error = (frame.full_prediction - frame.posted_rate).to_numpy()
    absolute = np.abs(error)
    n_worst = int(np.ceil(len(error) * 0.01))
    squared = error ** 2
    return {
        'n': len(error), 'MAE': float(absolute.mean()),
        'median_absolute_error': float(np.median(absolute)),
        'p95_absolute_error': float(np.quantile(absolute, 0.95)),
        'mean_signed_error': float(error.mean()),
        'median_signed_error': float(np.median(error)),
        'underpredicted_pct': float(100 * (error < 0).mean()),
        'worst_1pct_rows': n_worst,
        'worst_1pct_squared_error_pct': float(100 * np.sort(squared)[-n_worst:].sum() / squared.sum()),
    }


def unseen_rows(reference, target):
    cities = set(reference.pickup) | set(reference.delivery)
    unseen = ~target.pickup.isin(cities) | ~target.delivery.isin(cities)
    return {'rows': len(target), 'unseen_city_rows': int(unseen.sum()),
            'unseen_city_pct': float(100 * unseen.mean()),
            'unseen_cities': sorted((set(target.pickup) | set(target.delivery)) - cities)}


def main():
    out = Path('output')
    train = pd.read_csv('train-test.csv')
    final = pd.read_csv('validation.csv')
    saved = pd.read_csv(out / 'holdout_predictions.csv')
    holdout = train[train.date >= '2025-09-01']
    assert saved.load_id.is_unique and holdout.load_id.is_unique
    assert set(saved.load_id) == set(holdout.load_id)
    joined = saved.merge(holdout[['load_id', 'distance', 'posted_rate', 'date']],
                         on='load_id', validate='one_to_one', suffixes=('', '_source'))
    np.testing.assert_allclose(joined.posted_rate, joined.posted_rate_source, rtol=0, atol=0)
    assert joined.date.equals(joined.date_source)
    joined['month'] = joined.date.str[:7]
    # Fixed descriptive bins, not boundaries optimized against residuals.
    joined['distance_band'] = pd.cut(joined.distance, [0, 500, 1000, 2000, np.inf],
                                     right=False,
                                     labels=['<500 mi', '500-999 mi', '1000-1999 mi', '>=2000 mi'])
    residuals = {'overall': summarize(joined)}
    rows = []
    for dimension in ['month', 'distance_band']:
        residuals[dimension] = {}
        for name, group in joined.groupby(dimension, observed=True, sort=True):
            stats = summarize(group)
            residuals[dimension][str(name)] = stats
            rows.append({'dimension': dimension, 'slice': str(name), **stats})
    metrics = json.loads((out / 'metrics.json').read_text())
    main_name, core_name = metrics['selected']['full'], metrics['selected']['core']
    main_mae = metrics['holdout'][main_name]['MAE']
    result = {
        'scope': 'Post-evaluation descriptive analysis; no model selection, calibration or prediction changes.',
        'residuals': residuals,
        'geography': {'holdout_vs_jan_aug': unseen_rows(train[train.date < '2025-09-01'], holdout),
                      'final_vs_jan_oct': unseen_rows(train, final)},
        'baseline_mae_reduction_pct': 100 * (1 - main_mae / metrics['holdout']['quote_baseline']['MAE']),
        'core_holdout_mae_reduction_pct': 100 * (1 - metrics['holdout'][core_name]['MAE'] / main_mae),
        'artifact_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                            for name in ['validation_predictions.csv', 'december_predictions.csv',
                                         'holdout_predictions.csv', 'full_model.cbm', 'core_model.cbm']},
    }
    (out / 'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out / 'residual_slices.csv', index=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
