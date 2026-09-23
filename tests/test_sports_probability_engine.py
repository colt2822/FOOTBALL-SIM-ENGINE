import unittest
from datetime import datetime, timedelta, timezone

from worker.sports_probability_engine import (
    POSITION_TARGETS, apply_calibrator, assert_price_blind,
    build_joint_simulation_schema, build_player_rolling_features, build_team_rolling_features,
    distribution_from_samples, fit_calibrator, freeze_model_artifact,
    probability_metrics, validate_feature_row, verify_frozen_artifact,
    walk_forward_ridge,
)


UTC = timezone.utc


def _rows(n=52):
    start = datetime(2020, 9, 1, 18, tzinfo=UTC)
    rows = []
    for index in range(n):
        event = start + timedelta(days=7 * index)
        value = 40 + (index % 8) * 4 + index * 0.25
        rows.append({
            "player_id": "p1", "game_id": f"g{index}", "season": 2020 + index // 17,
            "position": "RB", "source": "fixture://causal-player-log",
            "event_time": event.isoformat(), "available_at": (event + timedelta(hours=4)).isoformat(),
            "as_of_time": (event - timedelta(hours=1)).isoformat(),
            "carries": 8 + index % 6, "rushing_yards": value,
            "targets": 2 + index % 4, "receptions": 1 + index % 3,
            "receiving_yards": 8 + index % 10, "rushing_tds": index % 2,
            "receiving_tds": 1 if index % 11 == 0 else 0,
        })
    return rows


class PriceBlindContractTests(unittest.TestCase):
    def test_market_derived_key_is_rejected_recursively(self):
        with self.assertRaisesRegex(ValueError, "PROHIBITED_MARKET_INPUT"):
            assert_price_blind({"features": {"sportsbook_odds": -110}})
        with self.assertRaisesRegex(ValueError, "PROHIBITED_MARKET_INPUT"):
            assert_price_blind({"features": {"closing_line_used": 45.5}})

    def test_football_features_are_allowed(self):
        assert_price_blind({"passing_epa": 0.2, "home": True, "rest_days": 7})


class TemporalFeatureTests(unittest.TestCase):
    def test_rolling_features_exclude_current_game_and_keep_provenance(self):
        rows = build_player_rolling_features(_rows(6), target="RUSH_YARDS")
        self.assertEqual(rows[0]["features"]["rush_yards_trailing_4"]["status"], "NO_PRIOR_DATA")
        last = rows[-1]
        cell = last["features"]["rush_yards_trailing_4"]
        self.assertEqual(cell["support_n"], 4)
        self.assertEqual(cell["value"], sum(r["rushing_yards"] for r in _rows(6)[1:5]) / 4)
        self.assertLess(cell["event_time"], last["event_time"])
        validate_feature_row(last)

    def test_late_available_prior_outcome_is_excluded(self):
        rows = _rows(3)
        event2 = datetime.fromisoformat(rows[2]["event_time"])
        rows[1]["available_at"] = (event2 + timedelta(hours=1)).isoformat()
        built = build_player_rolling_features(rows, target="RUSH_YARDS")
        self.assertEqual(built[2]["features"]["rush_yards_career_prior"]["support_n"], 1)

    def test_as_of_after_event_fails_closed(self):
        rows = _rows(1)
        rows[0]["as_of_time"] = (datetime.fromisoformat(rows[0]["event_time"]) + timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "AS_OF_AFTER_EVENT"):
            build_player_rolling_features(rows, target="RUSH_YARDS")

    def test_team_targets_use_same_temporal_contract(self):
        rows = []
        for index, base in enumerate(_rows(5)):
            rows.append({
                **{key: base[key] for key in ("game_id", "season", "source", "event_time", "available_at", "as_of_time")},
                "team_id": "NE", "offensive_plays": 60 + index,
                "pass_attempts": 30 + index, "rush_attempts": 25, "points": 20 + index,
            })
        built = build_team_rolling_features(rows, target="OFFENSIVE_PLAYS")
        self.assertEqual(built[-1]["features"]["offensive_plays_trailing_4"]["support_n"], 4)
        validate_feature_row(built[-1])


class DistributionAndCalibrationTests(unittest.TestCase):
    def test_distribution_contract_and_half_unit_curve(self):
        result = distribution_from_samples([0, 1, 2, 3])
        for key in ("EXPECTED_VALUE", "MEDIAN", "STD", "P10", "P25", "P50", "P75", "P90"):
            self.assertIn(key, result)
        self.assertEqual(result["DENSE_PROBABILITY_CURVE"][0]["threshold"], 0.5)
        self.assertAlmostEqual(result["DENSE_PROBABILITY_CURVE"][0]["P_OVER"], 0.75)

    def test_yardage_distribution_can_preserve_negative_outcomes(self):
        result = distribution_from_samples([-3, 0, 2])
        self.assertIn(-3.0, result["RAW_DISTRIBUTION_SAMPLES"])
        bounded = distribution_from_samples([-3, 0, 2], lower_bound=0)
        self.assertNotIn(-3.0, bounded["RAW_DISTRIBUTION_SAMPLES"])

    def test_probability_metrics_include_required_calibration_outputs(self):
        result = probability_metrics([0.1, 0.3, 0.7, 0.9], [0, 0, 1, 1], bins=2)
        for key in ("BRIER", "LOGLOSS", "ECE", "CALIBRATION_TABLE"):
            self.assertIn(key, result)

    def test_platt_and_isotonic_are_training_cutoff_gated(self):
        available = [datetime(2021, 1, day, tzinfo=UTC) for day in range(1, 7)]
        p, y = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], [0, 0, 1, 0, 1, 1]
        for method in ("PLATT", "ISOTONIC"):
            calibrator = fit_calibrator(method, p, y, available_at=available, training_cutoff=datetime(2021, 1, 7, tzinfo=UTC))
            calibrated = apply_calibrator(calibrator, [0.25, 0.75])
            self.assertTrue(all(0 <= value <= 1 for value in calibrated))
        with self.assertRaisesRegex(ValueError, "CALIBRATION_FUTURE_OUTCOME_LEAKAGE"):
            fit_calibrator("PLATT", p, y, available_at=available, training_cutoff=datetime(2021, 1, 5, tzinfo=UTC))


class WalkForwardTests(unittest.TestCase):
    def test_expanding_walk_forward_reports_point_probability_and_baselines(self):
        built = build_player_rolling_features(_rows(), target="RUSH_YARDS")
        features = ("rush_yards_trailing_4", "rush_yards_season_to_date", "rush_yards_career_prior")
        result = walk_forward_ridge(built, feature_names=features, thresholds=(49.5, 59.5), min_train_rows=8)
        self.assertEqual(result["STATUS"], "COMPLETE")
        self.assertGreater(result["METRICS"]["N"], 0)
        self.assertIn("MAE", result["METRICS"])
        self.assertIn("BRIER", result["PROBABILITY_METRICS_BY_THRESHOLD"]["49.5"])
        self.assertEqual(set(result["BASELINES"]), set(features))

    def test_position_target_matrix_contains_mission_targets(self):
        self.assertEqual(POSITION_TARGETS["QB"], ("PASS_ATTEMPTS", "COMPLETIONS", "PASS_YARDS", "PASS_TD", "RUSH_YARDS"))
        self.assertIn("TD", POSITION_TARGETS["TE"])


class JointAndArtifactTests(unittest.TestCase):
    def test_joint_schema_forbids_independence_and_has_10000_target(self):
        schema = build_joint_simulation_schema(game_id="g1", home_team_id="A", away_team_id="B")
        self.assertFalse(schema["INDEPENDENCE_ASSUMED"])
        self.assertEqual(schema["SIMULATION_COUNT_TARGET"], 10000)
        self.assertGreaterEqual(len(schema["DEPENDENCIES"]), 6)

    def test_frozen_artifact_has_required_fields_and_detects_tampering(self):
        artifact = freeze_model_artifact(
            model_id="m1", version="1.0.0", target="RUSH_YARDS",
            training_cutoff=datetime(2021, 1, 1, tzinfo=UTC),
            feature_schema={"version": "v1"}, data_provenance={"source": "fixture"},
            model_parameters={"alpha": 1.0}, validation_metrics={"N": 50},
            calibration_metrics={"ECE": 0.1}, created_at=datetime(2021, 1, 2, tzinfo=UTC),
        )
        self.assertEqual(verify_frozen_artifact(artifact)["STATUS"], "VALID")
        artifact["MODEL_PARAMETERS"] = {"alpha": 2.0}
        self.assertEqual(verify_frozen_artifact(artifact)["STATUS"], "CORRUPT")


if __name__ == "__main__":
    unittest.main()
