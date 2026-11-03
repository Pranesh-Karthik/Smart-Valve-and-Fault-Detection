"""Generate explicitly synthetic one-second pump signals; no measured data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import LABELS, FEATURES, extract_features

def generate(out='data/synthetic_valve_dataset.csv', per_class=300, seed=42):
    if per_class < 10:
        raise ValueError('Use at least 10 windows per class.')
    rng = np.random.default_rng(seed)
    t = np.arange(1000) / 1000
    rows = []
    # Illustrative assumptions, not a calibrated hydraulic model.
    for label, amplitude, pressure, noise in zip(
            LABELS, [1.8, 2.5, 3.2, 3.9, 2.9], [5.5, 4.6, 3.8, 3.0, 1.7],
            [.07, .10, .12, .15, .55]):
        for _ in range(per_class):
            a = max(.1, rng.normal(amplitude, .22))
            f = rng.normal(50, .18)
            phase = rng.uniform(0, 2*np.pi)
            x = a*np.sin(2*np.pi*f*t + phase)
            x += a*rng.uniform(.02, .08)*np.sin(2*np.pi*150*t + phase)
            x += a*rng.uniform(.01, .05)*np.sin(2*np.pi*250*t)
            x += rng.normal(0, noise*rng.uniform(.8, 1.2), 1000)
            if label == 'leak_detected':
                x *= 1 + .15*np.sin(2*np.pi*7*t)
            row = extract_features(x, float(np.clip(rng.normal(pressure, .4), 0, 10)))
            rows.append({**row, 'label': label})
    frame = pd.DataFrame(rows, columns=FEATURES + ['label']).sample(frac=1, random_state=seed)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix('.metadata.json').write_text(json.dumps({
        'source': 'synthetic', 'seed': seed, 'windows_per_class': per_class,
        'sample_rate_hz': 1000, 'samples_per_window': 1000,
        'warning': 'Simulated signals; results do not establish real pump performance.'
    }, indent=2), encoding='utf-8')
    print(f'SYNTHETIC dataset: {len(frame)} rows -> {path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/synthetic_valve_dataset.csv')
    p.add_argument('--per-class', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    generate(**vars(p.parse_args()))
