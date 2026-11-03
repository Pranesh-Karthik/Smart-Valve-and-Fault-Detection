import unittest
import numpy as np
from features import extract_features, FEATURES
from receiver import parse_packet

class FeatureTests(unittest.TestCase):
    def test_sine_matches_analytical_values(self):
        x = 2*np.sin(2*np.pi*50*np.arange(1000)/1000)
        f = extract_features(x, 3)
        self.assertAlmostEqual(f['rms_current'], np.sqrt(2))
        self.assertEqual(f['dominant_freq'], 50)
        self.assertAlmostEqual(f['dominant_mag'], 1)
        self.assertAlmostEqual(f['total_energy'], 2000)
        self.assertAlmostEqual(f['harmonic3_energy'], 0)

    def test_harmonic_normalization_matches_firmware(self):
        t = np.arange(1000)/1000
        f = extract_features(np.sin(2*np.pi*50*t)+.4*np.sin(2*np.pi*150*t), 2)
        self.assertAlmostEqual(f['harmonic3_energy'], .04)

    def test_zero_signal_is_finite(self):
        self.assertTrue(all(np.isfinite(v) for v in extract_features(np.zeros(1000), 0).values()))

    def test_receiver_rejects_nonfinite_and_unknown_label(self):
        row = extract_features(np.sin(2*np.pi*50*np.arange(1000)/1000), 2)
        parts = [str(row[k]) for k in FEATURES] + ['all_closed']
        self.assertEqual(parse_packet(','.join(parts).encode())['label'], 'all_closed')
        for index, value in [(0, 'nan'), (11, '11'), (12, 'unlabeled')]:
            bad = parts.copy()
            bad[index] = value
            with self.assertRaises(ValueError):
                parse_packet(','.join(bad).encode())

if __name__ == '__main__':
    unittest.main()
