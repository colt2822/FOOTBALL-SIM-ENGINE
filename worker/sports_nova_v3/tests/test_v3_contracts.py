"""Contract fixtures only: no empirical evidence. Pending runtime gates are explicit."""
import unittest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError
from worker.sports_nova_v3.schemas import (
    Evidence, Feature, Uncertainty, OpportunityShares, TeamState, PlayerState,
    PregameState, PlayerOutcome, FrozenModelRef, ProbabilityResult)
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game
from worker.sports_nova_v3.joint_queries import probability_from_masks
from worker.sports_nova_v3.artifact_freeze import canonical_bytes, content_sha256, freeze_simulation

def fixture():
    t = datetime(2026, 9, 1, tzinfo=timezone.utc)
    evidence = Evidence(source_id="TEST_ONLY", raw_sha256="a" * 64,
        event_end=t, available_at=t, retrieved_at=t, availability_basis="LIVE_RECEIPT")
    uncertainty = Uncertainty(effective_sample_size=0., personnel_unknown=True)
    player = PlayerState(player_id="TEST_QB", team_id="TEST_HOME", position="QB",
        availability="UNKNOWN", identity_evidence=evidence, uncertainty=uncertainty)
    residual = OpportunityShares(player_ids=(), shares=(), residual_share=1., evidence=evidence)
    home = TeamState(team_id="TEST_HOME", identity_evidence=evidence, features=(),
        target_shares=residual, carry_shares=residual)
    away = TeamState(**{**home.model_dump(), "team_id": "TEST_AWAY"})
    return PregameState(game_id="TEST_GAME", kickoff=t + timedelta(days=1), as_of=t,
        ruleset_id="TEST_RULES", identity_registry_sha256="b" * 64, game_evidence=evidence,
        home=home, away=away, players=(player,))

def replace_validated(obj, **changes):
    return type(obj).model_validate({**obj.model_dump(), **changes})

class ContractTests(unittest.TestCase):
    def test_no_future_leakage(self):
        s = fixture()
        for kwargs in ({"as_of": s.kickoff}, {"as_of": s.as_of - timedelta(seconds=1)},
                       {"as_of": s.as_of.replace(tzinfo=None)}):
            with self.assertRaises(ValidationError):
                replace_validated(s, **kwargs)
        late = replace_validated(s.game_evidence, event_end=s.kickoff)
        with self.assertRaises(ValidationError):
            replace_validated(s, game_evidence=late)

    def test_no_sportsbook_fields_or_features(self):
        s = fixture()
        for key in ("sportsbook", "odds", "spread", "market_total"):
            with self.assertRaises(ValidationError):
                replace_validated(s, **{key: 1.0})
            with self.assertRaises(ValidationError):
                Feature(name=key, value=1., unit="TEST", evidence=s.game_evidence)

    def test_immutable_state(self):
        with self.assertRaises(ValidationError):
            fixture().home.team_id = "CHANGED"

    def test_stable_identity(self):
        s = fixture()
        self.assertEqual(content_sha256(s), content_sha256(fixture()))
        with self.assertRaises(ValidationError):
            replace_validated(s, players=s.players + s.players)
        with self.assertRaises(ValidationError):
            replace_validated(s, away=s.home)

    def test_valid_opportunity_shares(self):
        e = fixture().game_evidence
        OpportunityShares(player_ids=("P",), shares=(.6,), residual_share=.4, evidence=e)
        for shares in ((-.1,), (.8,), (float("nan"),)):
            with self.assertRaises(ValidationError):
                OpportunityShares(player_ids=("P",), shares=shares, residual_share=.4, evidence=e)

    def test_no_impossible_negative_counts(self):
        row = dict(player_id="P", team_id="T", pass_attempts=0, pass_yards=0,
            rush_attempts=1, rush_yards=-2, targets=0, receptions=0, receiving_yards=0,
            pass_tds=0, rush_tds=0, receiving_tds=0, other_tds=0)
        PlayerOutcome(**row)
        for change in ({"rush_attempts": -1}, {"receptions": 1}, {"pass_yards": 2}):
            with self.assertRaises(ValidationError):
                PlayerOutcome(**{**row, **change})

    def test_probability_bounds_and_joint_less_than_marginals(self):
        a, b = (True, True, False, False), (True, False, True, False)
        joint = probability_from_masks((a, b)).probability
        self.assertEqual(joint, .25)
        for event in (a, b):
            p = probability_from_masks((event,)).probability
            self.assertTrue(0 <= joint <= p <= 1)
        with self.assertRaises(ValidationError):
            ProbabilityResult(probability=1.1, numerator=1, denominator=1, status="OK")

    def test_conditional_and_empty_support(self):
        a, b = (True, False), (True, True)
        self.assertEqual(probability_from_masks((a,), (b,)).probability, .5)
        result = probability_from_masks((a,), ((False, False),))
        self.assertIsNone(result.probability)
        self.assertEqual(result.status, "NO_CONDITION_SUPPORT")

    def test_frozen_simulation_version(self):
        with self.assertRaises(ValueError):
            simulate_game(fixture(), 10_000, 1, "latest")
        with self.assertRaises(ValidationError):
            FrozenModelRef(version="latest", sha256="a" * 64,
                training_available_through=fixture().as_of,
                calibration_available_through=fixture().as_of)

    def test_reproducible_state_bytes(self):
        self.assertEqual(canonical_bytes(fixture()), canonical_bytes(fixture()))

    def test_fixed_seed_determinism_10k(self):
        a = simulate_game(fixture(), 10_000, 17, MODEL_VERSION)
        b = simulate_game(fixture(), 10_000, 17, MODEL_VERSION)
        self.assertEqual(a.n_sims, 10_000)
        self.assertEqual(canonical_bytes(a), canonical_bytes(b))

    def test_reproducible_simulation_artifact(self):
        # Luna supplies the typed fitted-model/runtime fixture, then verifies bytes,
        # on-disk readback, checksum rejection, and a second-process replay.
        a = simulate_game(fixture(), 10_000, 17, MODEL_VERSION)
        self.assertEqual(freeze_simulation(a, a.model, a.runtime),
                         freeze_simulation(a, a.model, a.runtime))

if __name__ == "__main__":
    import sys
    if "--require-runtime" in sys.argv:
        sys.argv.remove("--require-runtime")
    unittest.main()
