import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
from worker.sports_nova_v3.calibration import fit_component_calibration, apply_frozen_calibration

class Phase07CalibrationTests(unittest.TestCase):
    def test_cutoff_gates_future_rows(self):
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        times = [cutoff - timedelta(days=2)] * 450 + [cutoff + timedelta(days=2)] * 450
        p = [.2] * 450 + [.9] * 450
        y = [0, 1] * 225 + [1, 0] * 225
        artifact = fit_component_calibration(p, y, times, cutoff=cutoff, min_n=400)
        self.assertEqual(artifact.n_fit, 450)
        self.assertEqual(artifact.status, "FIT")

    def test_insufficient_support_is_explicit(self):
        now = datetime.now(timezone.utc)
        artifact = fit_component_calibration([.2], [1], [now], cutoff=now, min_n=400)
        self.assertEqual(artifact.status, "INSUFFICIENT_SUPPORT")
        np.testing.assert_allclose(apply_frozen_calibration(artifact, [.1, .5]), [.1, .5])

if __name__ == "__main__":
    unittest.main()
