"""V28 tests T28-T30: current-slate runner, NOVA-book freeze-before-market-join, operator CLI."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import scripts.sports_nova_m1_fg_causal_ablation_v1 as H
from worker.sports_nova_terminal import engine_v28 as eng
from worker.sports_nova_v28_game_state_scoring import config as cfg
from worker.sports_nova_v28_game_state_scoring import simulator as s28
from worker.sports_nova_v28_game_state_scoring.tests.test_v28 import PILOT, pilot_state

ROOT = Path(__file__).resolve().parents[3]


class Operator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = pilot_state()
        cls.batch = s28.simulate_scoring(cls.state, 60, H.game_seed(PILOT), cfg.VERSIONS[2])
        cls.home, cls.away = cls.batch.team_ids

    def test_t28_current_slate_runner(self):
        import scripts.sports_nova_m1_v28_slate_canary as C
        for name in ("cmd_sim", "cmd_analyze", "sim_worker", "out_dir"):
            self.assertTrue(callable(getattr(C, name)))
        src = (ROOT / "scripts/sports_nova_m1_v28_slate_canary.py").read_text(encoding="utf-8")
        self.assertIn("PANEL_SHA_EXPECTED", src)                       # inputs come only from the immutable panel V2
        self.assertIn("v28.simulate_game", src)                        # the FULL V26 -> V25 -> V28 pipeline, not the bare scoring boundary
        self.assertEqual(C.N_DEFAULT, 5000)

    def test_t29_nova_book_frozen_before_market_join(self):
        book = eng.build_nova_book(self.batch, self.home, self.away, cfg.VERSIONS[2])
        sealed = eng.seal(book)
        self.assertEqual(len(sealed.sha256), 64)
        q = [{"market": "WINNER_HOME", "price": 0.55}, {"market": "TOTAL_OVER", "line": 44.5, "price": 0.5, "basis": "REGULATION_ONLY"},
             {"market": "TEAM_TOTAL_OVER", "team": self.home, "line": 20.5, "price": 0.5}]
        out = eng.join_market(sealed, q)
        self.assertEqual({r["book_sha256"] for r in out["rows"]}, {sealed.sha256})
        self.assertFalse(out["LIVE_CAPITAL_AUTHORIZED"])
        with self.assertRaises(eng.EngineError):
            eng.join_market(book, q)                                    # an unsealed book cannot be joined
        tampered = eng.SealedBook(json.loads(json.dumps(book)), sealed.sha256)
        tampered.book["INCLUDING_OT"]["WINNER"]["HOME"] = 0.99          # any post-seal edit is refused
        with self.assertRaises(eng.EngineError):
            eng.join_market(tampered, q)
        with self.assertRaises(eng.EngineError):
            eng.join_market(sealed, [{"market": "WINNER_HOME", "price": 1.5}])
        # the book builder has no market parameter; the book carries no price-like key
        import inspect
        self.assertEqual(list(inspect.signature(eng.build_nova_book).parameters), ["batch", "home", "away", "version"])
        blob = json.dumps(book).lower()
        for w in ("moneyline", "sportsbook", "kalshi_yes", "bid", "ask"):
            self.assertNotIn(f'"{w}', blob)
        # both OT bases are exported and OT-including totals are >= regulation-only totals
        self.assertGreaterEqual(book["INCLUDING_OT"]["TOTAL_MEAN"], book["REGULATION_ONLY"]["TOTAL_MEAN"] - 1e-12)
        rows = eng.calibration_rows(book, {"home_score": 24, "away_score": 20, "reg_home": 24, "reg_away": 20})
        self.assertTrue(rows and all(set(("p_raw", "y")) <= set(r) for r in rows))
        self.assertIn("ISOTONIC", eng.CALIBRATORS["comparison_only"])

    def test_t30_operator_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = eng.main(["--engine", "v28", "--game", "2026_02_IND_KC", "--sims", "24", "--variant", "1", "--seed", "7"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        for k in ("engine_version", "engine_hash", "game", "sims", "NOVA_home_win", "NOVA_away_win", "NOVA_total_mean", "NOVA_team_totals", "calibration_status", "market_status"):
            self.assertIn(k, out)
        self.assertEqual(out["calibration_status"], "UNCALIBRATED_RESEARCH_ONLY")
        self.assertEqual(out["engine_version"], cfg.VERSIONS[1])
        self.assertEqual(out["engine_role"], "CANDIDATE_NOT_CHAMPION" if eng.champion_variant() != 1 else "GAME_SIM_CHAMPION")
        self.assertFalse(out["LIVE_CAPITAL_AUTHORIZED"])
        self.assertIn("NOT_JOINED", out["market_status"])
        # legacy launcher default untouched: without --engine the interactive terminal is used
        src = (ROOT / "sports_nova.py").read_text(encoding="utf-8")
        self.assertIn('"--engine" in sys.argv', src)
        self.assertIn("sys.exit(main())", src)


if __name__ == "__main__":
    unittest.main()
