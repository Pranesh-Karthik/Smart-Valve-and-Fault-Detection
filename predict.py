"""Predict from a feature CSV using a locally trusted trained model."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import FEATURES

def predict(data, model='results/valve_model.joblib', out='results/predictions.csv'):
    bundle = joblib.load(model)  # Only load trusted local joblib files.
    df = pd.read_csv(data)
    x = df[FEATURES].astype(float)
    if not np.isfinite(x.to_numpy()).all() or (x < 0).any().any():
        raise ValueError('Features must be finite and nonnegative.')
    if not x.pressure.between(0, 10).all() or not x.dominant_freq.between(40, 300).all():
        raise ValueError('Pressure or frequency outside sensor schema.')
    pipe = bundle['pipeline']
    result = df.copy()
    result['prediction'] = pipe.predict(x)
    result['max_class_probability'] = pipe.predict_proba(x).max(axis=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result[['prediction', 'max_class_probability']].head(10).to_string(index=False))
    print(f'Saved {len(result)} predictions to {out}. Model source: {bundle["source"]}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='results/valve_model.joblib')
    p.add_argument('--out', default='results/predictions.csv')
    predict(**vars(p.parse_args()))
