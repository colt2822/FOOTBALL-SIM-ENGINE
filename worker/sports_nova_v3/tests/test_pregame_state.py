import unittest
from datetime import datetime, timedelta, timezone
from worker.sports_nova_v3.pregame_state import PregameUnavailable, build_pregame_state
from worker.sports_nova_v3.tests.test_v3_contracts import fixture

class PregameStateTests(unittest.TestCase):
    def test_explicit_mapping_builds_immutable_state(self):
        state = fixture()
        payload = state.model_dump()
        built = build_pregame_state(game_id="TEST_GAME", as_of=state.as_of,
            feature_store={"game": payload}, identity_registry={"registry_sha256": "b" * 64})
        self.assertEqual(built.game_id, "TEST_GAME")
        with self.assertRaises(Exception):
            built.game_id = "other"

    def test_future_evidence_and_market_fields_fail_closed(self):
        state = fixture()
        payload = state.model_dump()
        payload["game_evidence"]["available_at"] = state.kickoff
        with self.assertRaises(PregameUnavailable):
            build_pregame_state(game_id="TEST_GAME", as_of=state.as_of,
                feature_store={"game": payload}, identity_registry={"registry_sha256": "b" * 64})
        payload = state.model_dump()
        payload["sportsbook_line"] = 1.0
        with self.assertRaises(PregameUnavailable):
            build_pregame_state(game_id="TEST_GAME", as_of=state.as_of,
                feature_store={"game": payload}, identity_registry={"registry_sha256": "b" * 64})

if __name__ == "__main__":
    unittest.main()
