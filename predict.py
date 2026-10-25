"""Predict using a saved full or six-input core model."""
import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from solution import predict

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('output'))
    p.add_argument('--role', choices=['full', 'core'], default='full')
    a = p.parse_args()
    spec = json.loads((a.model_dir / 'metrics.json').read_text())['model_specs'][a.role]
    model = CatBoostRegressor()
    model.load_model(str(a.model_dir / f'{a.role}_model.cbm'))
    data = pd.read_csv(a.input)
    rates = predict(model, data, spec['mode'], spec['kind'])
    result = data[['load_id']].copy() if a.role == 'full' else data.copy()
    result['predicted_rate'] = rates
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
