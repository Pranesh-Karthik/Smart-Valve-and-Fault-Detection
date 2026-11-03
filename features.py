"""Shared feature definition, matching the report's firmware DFT convention."""
import numpy as np

FEATURES = ['rms_current', 'mean_abs_current', 'std_current', 'peak_current',
            'crest_factor', 'variance', 'dominant_freq', 'dominant_mag',
            'harmonic3_energy', 'harmonic5_energy', 'total_energy', 'pressure']
LABELS = ['all_closed', 'valve1_open', 'valve2_open', 'valve3_open', 'leak_detected']

def extract_features(samples, pressure, sample_rate=1000):
    x = np.asarray(samples, dtype=float)
    if x.shape != (1000,) or sample_rate != 1000 or not np.isfinite(x).all():
        raise ValueError('Expected 1000 finite samples at 1000 Hz.')
    if not np.isfinite(pressure) or not 0 <= pressure <= 10:
        raise ValueError('Pressure must be in [0, 10] bar.')
    spectrum = np.abs(np.fft.rfft(x)) / len(x)
    candidates = np.arange(40, 301, 5)
    dominant = candidates[np.argmax(spectrum[candidates])]
    rms = np.sqrt(np.mean(x*x))
    peak = np.max(np.abs(x))
    values = [rms, np.mean(np.abs(x)), np.std(x), peak, peak/rms if rms else 0,
              np.var(x), dominant, spectrum[dominant],
              np.sum(spectrum[145:156]**2), np.sum(spectrum[245:256]**2),
              np.sum(x*x), pressure]
    return dict(zip(FEATURES, map(float, values)))

def validate_frame(df):
    if set(df.columns) != set(FEATURES + ['label']):
        raise ValueError('CSV must contain exactly the 12 feature columns and label.')
    x = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Features must be finite and nonnegative.')
    if not df['pressure'].between(0, 10).all():
        raise ValueError('Pressure outside 0..10 bar.')
    if not df['dominant_freq'].between(40, 300).all():
        raise ValueError('Dominant frequency outside 40..300 Hz.')
    if not df['label'].isin(LABELS).all():
        raise ValueError('Unknown or missing class label.')
