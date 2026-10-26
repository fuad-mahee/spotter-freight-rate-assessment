"""Additional development-only experiments; never replace the submitted models.

Run after the original evaluation. These folds are retrospective development
evidence, not a new untouched test. All experiment labels precede September.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solution import fit, predict, metrics

FOLDS = [('Mar-Apr', '2025-03-01', '2025-05-01'),
         ('May-Jun', '2025-05-01', '2025-07-01'),
         ('Jul-Aug', '2025-07-01', '2025-09-01')]
CANDIDATES = {f'{mode}_{kind}': (mode, kind)
              for mode in ['no_quote', 'core']
              for kind in ['direct', 'per_mile', 'weighted_per_mile']}


def temporal_split(data, start, end):
    training = data[data.date < start].copy()
    evaluation = data[(data.date >= start) & (data.date < end)].copy()
    if training.empty or evaluation.empty or training.date.max() >= evaluation.date.min():
        raise ValueError('Expected nonempty strictly chronological windows')
    if evaluation.date.max() >= '2025-09-01':
        raise ValueError('Additional development experiments must end before September')
    return training, evaluation


def geographic_split(training, evaluation, cities):
    fitting = training[~training.pickup.isin(cities) & ~training.delivery.isin(cities)].copy()
    unseen = evaluation[evaluation.pickup.isin(cities) | evaluation.delivery.isin(cities)].copy()
    if fitting.empty or unseen.empty:
        raise ValueError('Empty geographic stress-test partition')
    if (set(fitting.pickup) | set(fitting.delivery)) & set(cities):
        raise ValueError('Held cities leaked into fitting')
    return fitting, unseen


def diagnostic(y, p):
    error = np.asarray(p) - np.asarray(y)
    return {**metrics(y, p), 'mean_signed_error': float(error.mean()),
            'median_absolute_error': float(np.median(abs(error))),
            'underpredicted_pct': float(100 * (error < 0).mean())}


def main():
    out = Path('output/robustness')
    out.mkdir(parents=True, exist_ok=True)
    source = Path('train-test.csv')
    data = pd.read_csv(source)
    data = data[data.date < '2025-09-01'].copy()
    protocol = {
        'status': 'Additional development analysis after the original holdout was inspected',
        'folds': FOLDS, 'candidates': CANDIDATES,
        'ranking': 'Equal-weight mean of three fold total-dollar MAEs; name breaks ties',
        'geography': 'Three fixed seeds (17, 29, 43), six cities each, chosen from Jan-Jun city names without targets; fit excludes all touching loads; evaluate Jul-Aug',
        'artifact_policy': 'Original selected models and submission predictions remain fixed; no holdout-based reselection or calibration',
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, predictions = [], []
    for fold, start, end in FOLDS:
        training, evaluation = temporal_split(data, start, end)
        for name, (mode, kind) in CANDIDATES.items():
            model = fit(training, mode, kind)
            p = predict(model, evaluation, mode, kind)
            row = {'fold': fold, 'candidate': name, 'fit_rows': len(training),
                   **diagnostic(evaluation.posted_rate, p)}
            rows.append(row)
            predictions.append(pd.DataFrame({'load_id': evaluation.load_id, 'fold': fold,
                'candidate': name, 'actual': evaluation.posted_rate, 'prediction': p}))
            print(f"{fold} {name}: MAE={row['MAE']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(out / 'temporal_metrics.csv', index=False)
    scores = pd.DataFrame(rows).groupby('candidate').agg(
        mean_MAE=('MAE', 'mean'), worst_MAE=('MAE', 'max'),
        mean_RMSE=('RMSE', 'mean'), mean_signed_error=('mean_signed_error', 'mean'))
    scores = scores.reset_index().sort_values(['mean_MAE', 'candidate'])
    scores.to_csv(out / 'ranking.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'temporal_predictions.csv', index=False)
    best = scores.iloc[0].candidate
    training, evaluation = temporal_split(data, '2025-07-01', '2025-09-01')
    pool = sorted(set(training.pickup) | set(training.delivery))
    geo_rows, geo_predictions = [], []
    for seed in [17, 29, 43]:
        cities = sorted(np.random.default_rng(seed).choice(pool, size=6, replace=False).tolist())
        fitting, unseen = geographic_split(training, evaluation, cities)
        for name in sorted({'no_quote_direct', best}):
            mode, kind = CANDIDATES[name]
            p = predict(fit(fitting, mode, kind), unseen, mode, kind)
            row = {'seed': seed, 'candidate': name, 'held_cities': '|'.join(cities),
                   'fit_rows': len(fitting), **diagnostic(unseen.posted_rate, p)}
            geo_rows.append(row)
            geo_predictions.append(pd.DataFrame({'load_id': unseen.load_id, 'seed': seed,
                'candidate': name, 'actual': unseen.posted_rate, 'prediction': p}))
            print(f"Cities seed={seed} {name}: MAE={row['MAE']:.2f}", flush=True)
    pd.DataFrame(geo_rows).to_csv(out / 'geographic_metrics.csv', index=False)
    pd.concat(geo_predictions, ignore_index=True).to_csv(out / 'geographic_predictions.csv', index=False)
    summary = {'development_leader': best, 'ranking': scores.to_dict('records'),
               'geographic_mean_MAE': pd.DataFrame(geo_rows).groupby('candidate').MAE.mean().to_dict(),
               'submission_changed': False,
               'limitations': 'Fold selection was designed after earlier results were known. City repetitions overlap and are not independent. This does not measure the actual unseen final cities or future December accuracy.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
