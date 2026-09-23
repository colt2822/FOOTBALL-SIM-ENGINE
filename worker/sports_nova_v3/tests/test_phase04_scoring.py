import unittest
from worker.sports_nova_v3.game_state import GameState
from worker.sports_nova_v3.scoring import score_event, transition_block, winner, apply_points

class Phase04ScoringTests(unittest.TestCase):
    def test_score_and_clock_conservation(self):
        state = GameState(game_id="G", simulation_id=0, block_index=0, seconds_remaining=3600,
                          home_score=0, away_score=0, possession="H", field_position=25)
        nxt = transition_block(state, offense_team="H", home_team_id="H", away_team_id="A",
            seconds_elapsed=35, field_position=50, event=score_event("H", "TD"), next_possession="A").state
        self.assertEqual((nxt.home_score, nxt.away_score, nxt.seconds_remaining), (6, 0, 3565))
        self.assertEqual(winner(nxt.home_score, nxt.away_score), "HOME")

    def test_negative_points_rejected_and_ties_explicit(self):
        state = GameState(game_id="G", simulation_id=0, block_index=0, seconds_remaining=10,
                          home_score=3, away_score=3, possession="H", field_position=90)
        with self.assertRaises(ValueError):
            apply_points(state, home_points=-1)
        self.assertEqual(winner(3, 3), "TIE")

if __name__ == "__main__":
    unittest.main()
