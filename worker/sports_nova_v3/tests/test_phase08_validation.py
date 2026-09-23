import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
from worker.sports_nova_v3.validation import chronological_folds, metric_report, run_dev_ablations, validate_walk_forward

class Phase08ValidationTests(unittest.TestCase):
    def test_folds_are_strictly_chronological(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [{"game_id": f"G{i}", "kickoff": t + timedelta(days=i), "actual": i, "predicted": i, "probability": .5, "binary_outcome": i % 2} for i in range(4)]
        folds = chronological_folds(rows)
        self.assertTrue(folds)
        self.assertTrue(all(f.cutoff_ok for f in folds))
        result = validate_walk_forward(rows)
        self.assertEqual(result["temporal_firewall"], "PASS")

    def test_required_metrics_and_ablations_are_explicit(self):
        report = metric_report([0, 1], [0, 1], probabilities=[.1, .9], binary_outcomes=[0, 1], samples=np.array([[0, 1], [0, 2]]))
        for key in ("MAE", "RMSE", "Brier", "LogLoss", "ECE", "CRPS"):
            self.assertIn(key, report)
        ablations = run_dev_ablations()
        self.assertEqual(len(ablations), 7)
        self.assertTrue(all(v["status"] == "NOT_RUN" for v in ablations.values()))

if __name__ == "__main__":
    unittest.main()
