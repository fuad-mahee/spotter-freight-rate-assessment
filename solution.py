"""Reproducible freight-rate training, chronological evaluation and inference."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

SEED = 42
CAT = ['pickup', 'delivery', 'equipment', 'lane']
CORE = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date']
EXTRA = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'market_index', 'quote_signal']


def features(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Stateless transformations; CatBoost handles numerical missing values."""
    extras = [c for c in EXTRA if not (mode == 'no_quote' and c == 'quote_signal')]
    cols = CORE + ([] if mode == 'core' else extras)
    x = data[cols].copy()
    dates = pd.to_datetime(x.pop('date'), errors='raise')
    x['weight_invalid'] = (x.weight.notna() & (x.weight <= 0)).astype(int)
    x['weight'] = x.weight.where(x.weight > 0)
    x['weight_missing'] = x.weight.isna().astype(int)
    for c in ['pickup', 'delivery', 'equipment']:
        x[c] = x[c].fillna('UNKNOWN').astype(str)
    x['lane'] = x.pickup + ' -> ' + x.delivery
    x['day_of_week'] = dates.dt.dayofweek
    x['day_of_month'] = dates.dt.day
    x['month'] = dates.dt.month
    x['day_index'] = (dates - pd.Timestamp('2025-01-01')).dt.days
    x['weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    x['month_end'] = dates.dt.is_month_end.astype(int)
    x['annual_sin'] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    x['annual_cos'] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    x['log_distance'] = np.log1p(x.distance)
    if mode != 'core':
        x['market_missing'] = x.market_index.isna().astype(int)
        x['lat_delta'] = x.delivery_lat - x.pickup_lat
        x['lon_delta'] = x.delivery_lon - x.pickup_lon
        if mode != 'no_quote':
            x['quote_total'] = x.distance * x.quote_signal
    return x


def metrics(y, prediction) -> dict:
    y, prediction = np.asarray(y), np.asarray(prediction)
    error = prediction - y
    return {'n': len(y), 'MAE': float(np.mean(abs(error))),
            'RMSE': float(np.sqrt(np.mean(error ** 2))),
            'WAPE_pct': float(100 * abs(error).sum() / abs(y).sum()),
            'R2': float(1 - (error ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def offset(data, kind):
    return data.distance.to_numpy() * data.quote_signal.to_numpy() if kind == 'residual' else 0


def fit(data, mode, kind, loss='MAE'):
    model = CatBoostRegressor(iterations=650, depth=6, learning_rate=0.055,
                             loss_function=loss, random_seed=SEED, thread_count=4,
                             verbose=False, allow_writing_files=False)
    y = data.posted_rate.to_numpy() - offset(data, kind)
    if kind in {'per_mile', 'weighted_per_mile'}:
        y = y / data.distance.to_numpy()
    weights = data.distance.to_numpy() if kind == 'weighted_per_mile' else None
    model.fit(features(data, mode), y, cat_features=CAT, sample_weight=weights)
    return model


def predict(model, data, mode, kind):
    p = model.predict(features(data, mode))
    if kind in {'per_mile', 'weighted_per_mile'}:
        p *= data.distance.to_numpy()
    return np.maximum(p + offset(data, kind), 0.01)


def audit(data):
    return {'rows': len(data), 'columns': list(data),
            'date_min': data.date.min(), 'date_max': data.date.max(),
            'missing': {k: int(v) for k, v in data.isna().sum().items()},
            'invalid_weight': int((data.weight <= 0).sum()),
            'duplicate_ids': int(data.load_id.duplicated().sum()),
            'duplicate_features': int(data.drop(columns=['load_id', 'posted_rate'], errors='ignore').duplicated().sum())}


def check_inputs(train, final, template, december):
    for data in [train, final]:
        if data.load_id.isna().any() or data.load_id.duplicated().any():
            raise ValueError('Missing or duplicate load IDs')
        for c in ['distance', 'quote_signal']:
            if not np.isfinite(data[c]).all() or (data[c] <= 0).any():
                raise ValueError(f'Invalid {c}')
        pd.to_datetime(data.date, errors='raise')
    if not np.isfinite(train.posted_rate).all() or (train.posted_rate <= 0).any():
        raise ValueError('Invalid target')
    if set(train.load_id) & set(final.load_id):
        raise ValueError('Overlapping train/final IDs')
    if template.load_id.duplicated().any() or set(template.load_id) != set(final.load_id):
        raise ValueError('Template IDs do not match final data')
    if train.date.max() >= final.date.min():
        raise ValueError('Expected future-only final data')
    if list(december) != CORE + ['predicted_rate']:
        raise ValueError('Unexpected December schema')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    args = parser.parse_args()
    root, out = args.data_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / 'train-test.csv')
    final = pd.read_csv(root / 'validation.csv')
    template = pd.read_csv(root / 'validation-predictions-template.csv')
    december = pd.read_csv(root / 'december-chart-inputs.csv')
    check_inputs(train, final, template, december)
    development = train[train.date < '2025-07-01']
    selection = train[(train.date >= '2025-07-01') & (train.date < '2025-09-01')]
    pretest = train[train.date < '2025-09-01']
    test = train[train.date >= '2025-09-01']
    result = {'audit': {'train': audit(train), 'final': audit(final)},
              'split': {'train_jan_jun': len(development), 'selection_jul_aug': len(selection),
                        'refit_jan_aug': len(pretest), 'holdout_sep_oct': len(test)},
              'selection': {}, 'holdout': {}, 'seed': SEED,
              'environment': {'python': platform.python_version(), **{
                  name: importlib.metadata.version(name)
                  for name in ['numpy', 'pandas', 'catboost', 'scikit-learn', 'scipy']}}}
    cities = set(train.pickup) | set(train.delivery)
    result['audit']['new_final_cities'] = sorted((set(final.pickup) | set(final.delivery)) - cities)
    result['audit']['new_city_final_rows'] = int((~final.pickup.isin(cities) | ~final.delivery.isin(cities)).sum())
    candidates = {'full_direct_mae': ('full', 'direct', 'MAE'),
                  'full_residual_mae': ('full', 'residual', 'MAE'),
                  'full_direct_rmse': ('full', 'direct', 'RMSE'),
                  'no_quote_ablation': ('no_quote', 'direct', 'MAE'),
                  'core_direct': ('core', 'direct', 'MAE'),
                  'core_per_mile': ('core', 'per_mile', 'MAE')}
    result['selection']['quote_baseline'] = metrics(selection.posted_rate, selection.distance * selection.quote_signal)
    median = float(development.posted_rate.median())
    result['selection']['median_baseline'] = metrics(selection.posted_rate, np.full(len(selection), median))
    for name, (mode, kind, loss) in candidates.items():
        model = fit(development, mode, kind, loss)
        result['selection'][name] = metrics(selection.posted_rate, predict(model, selection, mode, kind))
        print(name, result['selection'][name], flush=True)
    # The supplied validator specifies no accuracy metric: predeclare MAE selection.
    full_name = min([n for n in candidates if n.startswith('full_') or n == 'no_quote_ablation'],
                    key=lambda n: result['selection'][n]['MAE'])
    core_name = min(['core_direct', 'core_per_mile'], key=lambda n: result['selection'][n]['MAE'])
    result['selected'] = {'full': full_name, 'core': core_name}
    result['holdout']['quote_baseline'] = metrics(test.posted_rate, test.distance * test.quote_signal)
    result['holdout']['median_baseline'] = metrics(test.posted_rate, np.full(len(test), pretest.posted_rate.median()))
    holdout = test[['load_id', 'date', 'equipment', 'posted_rate']].copy()
    for role, name in result['selected'].items():
        mode, kind, loss = candidates[name]
        model = fit(pretest, mode, kind, loss)
        p = predict(model, test, mode, kind)
        result['holdout'][name] = metrics(test.posted_rate, p)
        holdout[role + '_prediction'] = p
        result['holdout'][role + '_by_month'] = {
            month: metrics(test.loc[mask, 'posted_rate'], p[mask.to_numpy()])
            for month in sorted(test.date.str[:7].unique())
            for mask in [test.date.str[:7].eq(month)]}
        result['holdout'][role + '_by_equipment'] = {
            eq: metrics(test.loc[mask, 'posted_rate'], p[mask.to_numpy()])
            for eq in sorted(test.equipment.unique()) for mask in [test.equipment.eq(eq)]}
        print('HOLDOUT', name, result['holdout'][name], flush=True)
        # Refit only after the chronological holdout evaluation is recorded.
        model = fit(train, mode, kind, loss)
        model.save_model(str(out / f'{role}_model.cbm'))
        importance = pd.Series(model.feature_importances_, index=features(train, mode).columns)
        importance.sort_values(ascending=False).rename('importance').to_csv(out / f'{role}_feature_importance.csv')
        if role == 'full':
            predictions = pd.Series(predict(model, final, mode, kind), index=final.load_id)
            template['predicted_rate'] = template.load_id.map(predictions)
            template.to_csv(out / 'validation_predictions.csv', index=False)
        else:
            december['predicted_rate'] = predict(model, december, mode, kind)
            december.to_csv(out / 'december_predictions.csv', index=False)
    holdout.to_csv(out / 'holdout_predictions.csv', index=False)
    result['model_specs'] = {role: dict(zip(['mode', 'kind', 'loss'], candidates[name]))
                             for role, name in result['selected'].items()}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved predictions, models, diagnostics and metrics to', out, flush=True)


if __name__ == '__main__':
    main()
