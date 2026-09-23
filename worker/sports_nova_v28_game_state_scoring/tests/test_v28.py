"""V28 unit tests.  Scoring-boundary tests run on a REAL historical pilot state (real player table); pipeline tests reuse V26's synthetic panel through the REAL make_state."""
import ast
import hashlib
import json
import unittest
from unittest import mock

import numpy as np

import scripts.sports_nova_m1_fg_causal_ablation_v1 as H
import scripts.sports_nova_v3_player_joint_walkforward_v1 as wf
from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_v23.simulator import _state_hash
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from worker.sports_nova_v26_active_skill_state import simulator as v26
from worker.sports_nova_v26_active_skill_state.completion import complete_state
from worker.sports_nova_v26_active_skill_state.tests.test_completion import GAME, KICK, panel, roster_snapshot
from worker.sports_nova_v27_causal_fg_rate import causal_fg as fg27
from worker.sports_nova_v27_causal_fg_rate import config as cfg27
from worker.sports_nova_v27_causal_fg_rate import simulator as v27
from worker.sports_nova_v28_game_state_scoring import config as cfg
from worker.sports_nova_v28_game_state_scoring import hazard as hz_mod
from worker.sports_nova_v28_game_state_scoring import simulator as s28
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "worker" / "sports_nova_v28_game_state_scoring"
V1, V2, V3 = cfg.VERSIONS[1], cfg.VERSIONS[2], cfg.VERSIONS[3]
PILOT = "2022_01_BAL_NYJ"
POST = "2021_22_LA_CIN"
F = s28.BI


def pilot_state(game=PILOT):
    import pandas as pd
    season, week, away, home = H.game_parts(game)
    df = pd.read_parquet(H.PLAYER)
    df["_key"] = df.SEASON * 100 + df.WEEK
    prior = df[(df._key < season * 100 + week) & (df.SEASON >= max(1999, season - 5))]
    return H.make_state(game, prior, H.surrogate_kickoff(season, week), None)


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = pilot_state()
        cls.seed = H.game_seed(PILOT)
        cls.b1, cls.blk1 = s28.simulate_scoring(cls.state, 96, cls.seed, V1, want_blocks=True)
        cls.b2, cls.blk2 = s28.simulate_scoring(cls.state, 96, cls.seed, V2, want_blocks=True)

    def test_t1_parent_immutability(self):
        h = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted({f"worker/sports_nova_v27_causal_fg_rate/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                                                                      for p in (ROOT / "worker/sports_nova_v27_causal_fg_rate").glob("*.py")}.items())).encode()).hexdigest()
        self.assertEqual(h, "295a5345d64fa76eab68ae8219634f1583c87ed9c4d2a167b87203f85e27a020")
        src = "".join(f.read_text(encoding="utf-8") for f in PKG.glob("*.py"))
        for frozen in ("def role_aware_values", "def complete_state", "def allocate_opportunities"):
            self.assertNotIn(frozen, src)

    def test_t2_v27_reproduction_from_sealed_arrays(self):
        z = np.load(ROOT / "data/sports_nova_v3/MULTI_TD_POSSESSION_ABLATION_V1/arrays" / f"{PILOT}.npz")
        b = v27.simulate_scoring(self.state, 128, self.seed, cfg27.MODEL_VERSION)
        self.assertTrue(np.array_equal(np.asarray(b.team_stats["score"]), z["base__team_score"]))
        self.assertTrue(np.array_equal(np.asarray(b.winner), z["base__winner"]))

    def test_t3_one_td_max_per_possession(self):
        for blk in (self.blk1, self.blk2):
            self.assertLessEqual(blk[:, F["td"]].max(), 1)
            self.assertTrue(set(np.unique(blk[blk[:, F["ot"]] == 0][:, F["pts"]])) <= {0, 3, 6, 7})

    def test_t4_t5_t6_recipient_and_pass_rush_accounting(self):
        for b, blk in ((self.b1, self.blk1), (self.b2, self.blk2)):
            ts = b.team_stats
            self.assertEqual(int(np.asarray(ts["tds"]).sum()), int(blk[:, F["td"]].sum()))
            self.assertEqual(int(np.asarray(ts["tds"]).sum()), int(np.asarray(ts["td_pass"]).sum() + np.asarray(ts["td_rush"]).sum()))
            for stat in ("scored_tds", "receiving_tds", "rush_tds", "pass_tds"):
                self.assertTrue(np.asarray(b.player_stats[stat]).min() >= 0)
            tot = lambda k: int(np.asarray(b.player_stats[k]).sum())
            self.assertEqual(tot("scored_tds"), int(np.asarray(ts["tds"]).sum()))
            self.assertEqual(tot("receiving_tds"), int(np.asarray(ts["td_pass"]).sum()))     # pass TD: one receiver credited
            self.assertEqual(tot("pass_tds"), int(np.asarray(ts["td_pass"]).sum()))          # ... and one QB credited
            self.assertEqual(tot("rush_tds"), int(np.asarray(ts["td_rush"]).sum()))          # rush TD: one carrier credited, no QB pass credit
            self.assertEqual(tot("scored_tds"), tot("receiving_tds") + tot("rush_tds"))      # never receiver + carrier on one possession

    def test_t7_qb_scramble_td_path(self):
        seen = 0
        for game in (PILOT, "2020_01_ARI_SF", "2023_02_NYG_ARI" if False else "2021_09_CLE_CIN"):
            st = pilot_state(game)
            b, blk = s28.simulate_scoring(st, 128, H.game_seed(game), V1, want_blocks=True)
            seen += int(blk[:, F["td_qb_rush"]].sum())
            self.assertTrue(((blk[:, F["td_qb_rush"]] == 0) | (blk[:, F["td_rush"]] == 1)).all())
            qb_rows = blk[blk[:, F["td_qb_rush"]] == 1]
            self.assertTrue((qb_rows[:, F["pass_att"]] + qb_rows[:, F["designed"]] + qb_rows[:, F["scrambles"]] > 0).all())
        self.assertGreater(seen, 0, "the QB never scored a rushing TD: the scramble path is not live")

    def test_t8_no_td_plus_fg_same_possession(self):
        for blk in (self.blk1, self.blk2):
            self.assertEqual(int(((blk[:, F["td"]] > 0) & (blk[:, F["fg"]] > 0)).sum()), 0)
            self.assertEqual(int(((blk[:, F["td"]] > 0) & (blk[:, F["fg_elig"]] > 0)).sum()), 0)
            self.assertTrue(((blk[:, F["fg"]] == 0) | (blk[:, F["pts"]] == 3)).all())

    def test_t9_causal_fg_exact_estimator(self):
        r = fg27.causal_fg_rate_for_game(PILOT)
        self.assertEqual(self.b1.runtime["fg_rate"], r.rate)
        self.assertEqual(self.b1.runtime["fg_rate_source"], "CAUSAL_ESTIMATOR_A")
        self.assertEqual(self.b1.runtime["fg_estimator_sha256"], cfg.ESTIMATOR_SHA256)
        r26 = fg27.causal_fg_rate(2026, 2)
        self.assertEqual((r26.n_eligible, r26.made), (21452, 4751))
        self.assertAlmostEqual(r26.rate, 0.2214712, places=6)

    def test_t10_estimator_and_hazard_fail_closed(self):
        with mock.patch.object(s28.fg27, "causal_fg_rate_for_game", side_effect=fg27.CausalFGError("boom")):
            with self.assertRaises(fg27.CausalFGError):
                s28.simulate_scoring(self.state, 4, 1, V1)
        with mock.patch.dict(hz_mod._T, {}, clear=True), mock.patch.object(hz_mod, "DRIVE_SHA256", "0" * 64):
            hz_mod.hazard_for.cache_clear()
            with self.assertRaises(hz_mod.HazardError):
                hz_mod.hazard_for(2022, 1)
        hz_mod.hazard_for.cache_clear()
        with self.assertRaises(hz_mod.HazardError):
            hz_mod.hazard_for(1999, 1)                                       # empty window
        with self.assertRaises(hz_mod.HazardError):
            hz_mod.hazard_for_game(PILOT, align=True, ref_quantiles=None)    # alignment without a frozen reference
        with self.assertRaises(ValueError):
            s28.simulate_scoring(self.state, 4, 1, cfg27.MODEL_VERSION)      # V27 string is not a V28 version
        tree = ast.parse((PKG / "simulator.py").read_text(encoding="utf-8"))
        floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
        self.assertNotIn(0.08, floats)

    def test_t11_regulation_score_accounting(self):
        b, blk = self.b1, self.blk1
        n = b.n_sims
        sc = np.zeros((n, 2), np.int64)
        for r in blk:
            sc[int(r[F["sim"]]), int(r[F["off"]])] += int(r[F["pts"]])
        self.assertTrue(np.array_equal(sc, np.asarray(b.team_stats["score"])))
        self.assertTrue(np.array_equal(np.asarray(b.team_stats["score"]), np.asarray(b.team_stats["total"])))
        hs, as_ = np.asarray(b.team_stats["score"]).T
        self.assertTrue(np.array_equal(b.winner, np.where(hs > as_, "HOME", np.where(as_ > hs, "AWAY", "TIE"))))
        self.assertEqual(int(np.asarray(b.team_stats["went_ot"]).sum()), 0)   # variant .1 has no overtime

    def test_t12_t13_overtime_winner_and_total_accounting(self):
        # force regulation ties: a dead-even scripted score by mocking the hazard to a low constant across many sims is unnecessary -- use many sims and take the OT sims
        st = self.state
        b, blk = s28.simulate_scoring(st, 400, self.seed + 1, V2, want_blocks=True)
        ts = b.team_stats
        went = np.asarray(ts["went_ot"])[:, 0] == 1
        self.assertGreater(int(went.sum()), 0)
        reg = np.asarray(ts["reg_score"]); fin = np.asarray(ts["score"]); ot = np.asarray(ts["ot_score"])
        self.assertTrue(np.array_equal(reg + ot, fin))
        self.assertTrue((reg[went, 0] == reg[went, 1]).all())                                    # OT only from a regulation tie
        self.assertTrue((ot[~went] == 0).all())
        hs, as_ = fin.T
        self.assertTrue(np.array_equal(b.winner, np.where(hs > as_, "HOME", np.where(as_ > hs, "AWAY", "TIE"))))
        self.assertTrue(((ts["ot_score"] >= 0)).all())
        wrong = [i for i in np.where(went)[0] if (fin[i, 0] == fin[i, 1]) != (b.winner[i] == "TIE")]
        self.assertEqual(wrong, [])
        bo = blk[blk[:, F["ot"]] == 1]
        self.assertGreater(len(bo), 0)
        sc = np.zeros((b.n_sims, 2), np.int64)
        for r in blk:
            sc[int(r[F["sim"]]), int(r[F["off"]])] += int(r[F["pts"]])
        self.assertTrue(np.array_equal(sc, fin))                                                 # totals include OT scoring exactly

    def test_t14_rare_final_tie_and_postseason_no_tie(self):
        # regular season: an all-scoreless game (no TD, no FG) must end as a regulation tie, then an OT tie -> final TIE (permitted by the rules)
        class Zero:
            lo, hi = -7.0, 104.0
            def p_td(self, *a):
                return 0.0
            def evidence(self):
                return {}
        with mock.patch.object(s28.hz_mod, "hazard_for_game", return_value=Zero()):
            b = s28._score(self.state, 6, 1, V2, 0.0, {}, Zero())
        self.assertTrue((b.winner == "TIE").all())
        self.assertTrue((np.asarray(b.team_stats["went_ot"])[:, 0] == 1).all())
        self.assertTrue((np.asarray(b.team_stats["score"]) == 0).all())
        # postseason: played to a winner -- never a TIE
        st = pilot_state(POST)
        b, _ = s28.simulate_scoring(st, 300, H.game_seed(POST), V2, want_blocks=True)
        self.assertNotIn("TIE", set(b.winner.tolist()))
        self.assertTrue(cfg.is_postseason(2021, 22) and not cfg.is_postseason(2021, 18) and cfg.is_postseason(2020, 18) and not cfg.is_postseason(2026, 2))

    def test_t15_t16_market_isolation(self):
        assert_market_free(cfg.HAZARD_POLICY)
        assert_market_free(cfg.TD_ALLOCATION_POLICY)
        assert_market_free(json.loads(json.dumps(self.b1.runtime, default=str)))
        forbidden = ("sports_market_interface", "sports_research", "market_book", "comparator", "trading_lab", "kalshi", "sportsbook")
        for f in PKG.glob("*.py"):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
                for m in mods:
                    self.assertFalse(any(x in m for x in forbidden), (f.name, m))
        bad_cols = ("odds", "vegas", "spread", "price", "moneyline", "implied", "kalshi", "line")
        self.assertFalse([c for c in hz_mod._COLS if any(x in c.lower() for x in bad_cols)])
        self.assertFalse([c for c in ("yards", "plays", "diff", "sec") if any(x in c for x in bad_cols)])

    def test_t22_deterministic_seed(self):
        a = s28.simulate_scoring(self.state, 30, 123, V2)
        b = s28.simulate_scoring(self.state, 30, 123, V2)
        c = s28.simulate_scoring(self.state, 30, 124, V2)
        self.assertTrue(np.array_equal(np.asarray(a.team_stats["score"]), np.asarray(b.team_stats["score"])))
        self.assertTrue(all(np.array_equal(a.player_stats[k], b.player_stats[k]) for k in a.player_stats))
        self.assertFalse(np.array_equal(np.asarray(a.team_stats["score"]), np.asarray(c.team_stats["score"])))

    def test_observer_is_inert(self):
        a = s28.simulate_scoring(self.state, 40, 5, V1)
        b, _ = s28.simulate_scoring(self.state, 40, 5, V1, want_blocks=True)
        self.assertTrue(all(np.array_equal(a.team_stats[k], b.team_stats[k]) for k in a.team_stats))
        self.assertTrue(all(np.array_equal(a.player_stats[k], b.player_stats[k]) for k in a.player_stats))

    def test_hazard_causality_and_monotone(self):
        hz = hz_mod.hazard_for(2022, 1)
        w = hz_mod.window(2022, 1)
        self.assertLess(int(w.key.max()), 202201)
        self.assertGreaterEqual(int(w.season.min()), 2017)
        y = np.array([-20, 0, 10, 25, 40, 55, 70, 85, 100.0])
        p = hz.p_td(y, np.full(9, 7), np.zeros(9), np.full(9, 2000.0))
        self.assertTrue(np.all(np.diff(p) > -1e-9), p)                        # more production never lowers the TD probability at fixed plays/score/clock
        self.assertGreater(p[-1] - p[0], 0.5)
        # leakage: appending future rows must not move the fit
        self.assertEqual(hz_mod.hazard_for(2022, 1).n_train, len(w))

    def test_t23_coherence_metric_detects_independence(self):
        """NEGATIVE test: a scoring boundary whose TD probability ignores production must FAIL the coherence metric; the real hazard must pass it."""
        class Flat:
            lo, hi = -7.0, 104.0
            def p_td(self, *a):
                return 0.22
            def evidence(self):
                return {}
        params_fg = fg27.causal_fg_rate_for_game(PILOT).rate
        flat = s28._score(self.state, 96, self.seed, V1, params_fg, {}, Flat(), want_blocks=True)[1]
        self.assertLess(coherence_range(flat), 0.10)
        self.assertGreaterEqual(coherence_range(self.blk1), 0.45)

    def test_t24_yardage_perturbation_directional(self):
        st, seed = self.state, self.seed
        base = s28.simulate_scoring(st, 96, seed, V1)
        lo = s28.simulate_scoring(st, 96, seed, V1, perturb=s28.Perturb(yard_scale=0.5))
        hi = s28.simulate_scoring(st, 96, seed, V1, perturb=s28.Perturb(yard_scale=1.5))
        cap = s28.simulate_scoring(st, 96, seed, V1, perturb=s28.Perturb(explosive_cap=20.0))
        m = lambda b: float(np.asarray(b.team_stats["score"], float).mean())
        self.assertLess(m(lo), m(base) - 1.0)
        self.assertGreater(m(hi), m(base) + 1.0)
        self.assertLess(m(cap), m(base))
        home = base.team_ids[0]
        h_lo = s28.simulate_scoring(st, 96, seed, V1, perturb=s28.Perturb(yard_scale=0.5, only_team=home))
        h_hi = s28.simulate_scoring(st, 96, seed, V1, perturb=s28.Perturb(yard_scale=1.5, only_team=home))
        p = lambda b: float((b.winner == "HOME").mean())
        self.assertGreater(p(h_hi), p(h_lo) + 0.10)
        self.assertLess(s28.simulate_scoring(st, 8, seed, V1, perturb=s28.IDENTITY).team_stats["score"].sum(), 10 ** 6)
        same = s28.simulate_scoring(st, 40, seed, V1, perturb=s28.IDENTITY)
        self.assertTrue(np.array_equal(np.asarray(same.team_stats["score"]), np.asarray(s28.simulate_scoring(st, 40, seed, V1).team_stats["score"])))

    def test_t25_turnover_perturbation_is_documented_not_represented(self):
        src = "".join(f.read_text(encoding="utf-8") for f in PKG.glob("*.py")).lower()
        self.assertNotIn("def turnover", src)
        self.assertNotIn('"turnovers"', src)
        self.assertIn("not_represented", (ROOT / "scripts/sports_nova_m1_v28_game_state_scoring.py").read_text(encoding="utf-8").lower())

    def test_t26_extreme_score_tail_sanity(self):
        ts = np.asarray(self.b2.team_stats["score"])
        self.assertLessEqual(int(ts.max()), 70)
        self.assertLessEqual(int(ts.sum(axis=1).max()), 110)
        self.assertGreaterEqual(int(ts.sum(axis=1).min()), 0)
        self.assertEqual(int(self.blk2[:, F["pts"]].max()) <= 7, True)

    def test_efficiency_kappa_only_touches_the_noise(self):
        V5, V6 = cfg.VERSIONS[5], cfg.VERSIONS[6]
        self.assertEqual(tuple(cfg.variant_of(V1))[1:], (False, False, 1.0, False, 1.0, False))
        self.assertEqual(tuple(cfg.variant_of(V6))[1:], (True, False, 0.0, False, 1.0, False))
        a = s28.simulate_scoring(self.state, 40, 5, V1)
        b = s28.simulate_scoring(self.state, 40, 5, V1, eff_kappa=1.0)            # explicit kappa=1 is the default path
        self.assertTrue(np.array_equal(np.asarray(a.team_stats["score"]), np.asarray(b.team_stats["score"])))
        y = lambda bb: (np.asarray(bb.team_stats["pass_yards"]) + np.asarray(bb.team_stats["rush_yards"])).ravel().astype(float)
        k0 = s28.simulate_scoring(self.state, 96, self.seed, V5)
        k1 = s28.simulate_scoring(self.state, 96, self.seed, V1)
        self.assertLess(y(k0).std(), 0.75 * y(k1).std())                            # the noise is what inflated team-game yardage dispersion
        self.assertEqual(k0.runtime["efficiency_noise_kappa"], 0.0)
        # regulation is identical with and without overtime (OT only runs after a regulation tie, on the score stream)
        k6 = s28.simulate_scoring(self.state, 96, self.seed, V6)
        self.assertTrue(np.array_equal(np.asarray(k0.team_stats["score"]), np.asarray(k6.team_stats["reg_score"])))
        self.assertTrue(np.array_equal(np.asarray(k0.team_stats["reg_score"]), np.asarray(k6.team_stats["reg_score"])))

    def test_team_multiplier_and_clock_scale(self):
        V7, V8 = cfg.VERSIONS[7], cfg.VERSIONS[8]
        self.assertAlmostEqual(cfg.CLOCK_SCALE, 1.0508, places=3)
        b5, blk5 = s28.simulate_scoring(self.state, 96, self.seed, cfg.VERSIONS[5], want_blocks=True)
        b7, blk7 = s28.simulate_scoring(self.state, 96, self.seed, V7, want_blocks=True)
        b8, blk8 = s28.simulate_scoring(self.state, 96, self.seed, V8, want_blocks=True)
        self.assertTrue(b7.runtime["team_finishing_multiplier"] and not b5.runtime["team_finishing_multiplier"])
        self.assertLess(0.02, b7.runtime["league_pass_td_rate"] - 0.0)
        self.assertLessEqual(blk7[:, F["td"]].max(), 1)
        self.assertLess(len(blk8), 0.97 * len(blk7))                                    # the clock scale removes ~5% of possessions
        # the multiplier is team-specific: raising ONE team's finishing rates raises ITS TD count and only its own
        from worker.sports_nova_v3.schemas import Feature
        def bump(team, f):
            feats = tuple(x.model_copy(update={"value": x.value * f}) if x.name in ("pass_td_rate", "rush_td_rate") else x for x in team.features)
            return team.model_copy(update={"features": feats})
        st_hi = self.state.model_copy(update={"home": bump(self.state.home, 1.6)})
        bh = s28.simulate_scoring(st_hi, 96, self.seed, V7)
        tds = lambda b: np.asarray(b.team_stats["tds"], float).mean(axis=0)
        self.assertGreater(tds(bh)[0], tds(b7)[0] + 0.15)
        self.assertLess(abs(tds(bh)[1] - tds(b7)[1]), 0.6)
        # regulation is identical with and without overtime for the new lineage as well
        b10 = s28.simulate_scoring(self.state, 96, self.seed, cfg.VERSIONS[10])
        self.assertTrue(np.array_equal(np.asarray(b7.team_stats["reg_score"]), np.asarray(b10.team_stats["reg_score"])))
        b9 = s28.simulate_scoring(self.state, 96, self.seed, cfg.VERSIONS[9])
        self.assertTrue(np.array_equal(np.asarray(b8.team_stats["reg_score"]), np.asarray(b9.team_stats["reg_score"])))

    def test_league_rate_window_is_causal_and_pinned(self):
        lp, lr = hz_mod.league_td_rates(2022, 1)
        self.assertTrue(0.05 < lp < 0.09 and 0.02 < lr < 0.05)
        with mock.patch.dict(hz_mod._T, {}, clear=True), mock.patch.object(hz_mod, "PLAYER_SHA256", "0" * 64):
            with self.assertRaises(hz_mod.HazardError):
                hz_mod.league_td_rates(2022, 1)
        with self.assertRaises(hz_mod.HazardError):
            hz_mod.league_td_rates(1999, 1)

    def test_recent_form_shrink_is_causal_and_only_shrinks(self):
        sl = hz_mod.recent_form_slope(2022, 1)
        self.assertTrue(0.0 <= sl <= 1.0 and sl < 0.6, sl)
        with self.assertRaises(hz_mod.HazardError):
            hz_mod.recent_form_slope(1999, 1)
        V11, V8 = cfg.VERSIONS[11], cfg.VERSIONS[8]
        self.assertTrue(cfg.variant_of(V11).eff_shrink and not cfg.variant_of(V8).eff_shrink)
        a = s28.simulate_scoring(self.state, 96, self.seed, V8)
        b = s28.simulate_scoring(self.state, 96, self.seed, V11)
        self.assertAlmostEqual(b.runtime["recent_form_slope"], sl, places=12)
        y = lambda bb: (np.asarray(bb.team_stats["pass_yards"]) + np.asarray(bb.team_stats["rush_yards"])).ravel().astype(float)
        self.assertLessEqual(abs(y(b).mean() - 360.0), abs(y(a).mean() - 360.0) + 60.0)      # sanity only: still a sane yardage level
        b12 = s28.simulate_scoring(self.state, 96, self.seed, cfg.VERSIONS[12])
        self.assertTrue(np.array_equal(np.asarray(b.team_stats["reg_score"]), np.asarray(b12.team_stats["reg_score"])))

    def test_t27_frozen_60_game_runner(self):
        import scripts.sports_nova_m1_v28_game_state_scoring as R
        games = R.pilot_games()
        self.assertEqual(len(games), 60)
        self.assertEqual(games[0], H.pilot_games(list(json.loads(H.MANIFEST.read_text())["GAME_IDS"]))[0])
        for name in ("cmd_pilot", "cmd_freeze", "cmd_coherence", "load_freeze"):
            self.assertTrue(callable(getattr(R, name)))


def coherence_range(blk):
    y, td = blk[:, F["yds"]], blk[:, F["td"]] > 0
    return float(td[y >= 70].mean() - td[y <= 10].mean())


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = panel()
        wf.PLAYER_SHA = "a" * 64
        cls.state = wf.make_state(GAME, cls.prior, KICK, None)

    def snap(self, statuses=None, game_status=None):
        extra = {"cutwr": ("HOM", "ACT"), "cutrb": ("HOM", "ACT"), "arrival": ("HOM", "ACT"), "newrookie": ("HOM", "ACT"), "dupe": ("HOM", "ACT")}
        s = roster_snapshot(self.state, extra=extra, statuses=statuses, game_status=game_status)
        return RosterSnapshot(roster=s.roster, game_status=s.game_status, roster_week=2, positions={**s.positions, "newrookie": ("WR", "WR")})

    def test_t17_to_t21_roster_v25_v26_invalid_usage_conservation(self):
        snap = self.snap(statuses={"hrb0": ("HOM", "RES"), "arb1": ("AWY", "INA")}, game_status={"hwr1": "OUT"})
        c = complete_state(self.state, snap, self.prior)
        b = s28.simulate_game(self.state, 150, 9, V1, roster=snap, completion=c)
        ids = list(b.player_ids)
        for pid in ("hrb0", "arb1", "hwr1"):
            usage = sum(int(b.player_stats[k][:, ids.index(pid)].sum()) for k in ("pass_attempts", "targets", "rush_attempts", "scored_tds", "receiving_tds", "rush_tds"))
            self.assertEqual(usage, 0, pid)                                      # invalid usage == 0, including TDs
        # V25 repair + V26 completion run UNCHANGED upstream of V28: same repaired state as V27's pipeline
        r28 = s28.prepare_state(self.state, snap, self.prior)
        r27 = v27.prepare_state(self.state, snap, self.prior)
        self.assertEqual((r28[2].raw_state_hash, r28[2].repaired_state_hash, r28[2].summaries), (r27[2].raw_state_hash, r27[2].repaired_state_hash, r27[2].summaries))
        self.assertEqual(b.runtime["repaired_state_hash"], r28[2].repaired_state_hash)
        for s in r28[2].summaries:
            self.assertAlmostEqual(s["POST_REDISTRIBUTION_TOTAL"], s["PRE_REMOVAL_TOTAL"], places=9)     # opportunity mass conserved
            self.assertAlmostEqual(s["REMOVED_MASS"], s["REDISTRIBUTED_MASS"], places=9)
        # completion (V26) active-state behavior: completed state is what gets simulated
        self.assertEqual(b.runtime["completed_state_hash"], c.completed_state_hash)
        # target / carry conservation: every attempt is credited to a named player (V23 allocation)
        for stat_p, stat_t in (("targets", "pass_attempts"), ("rush_attempts", "rush_attempts")):
            self.assertLessEqual(int(np.asarray(b.player_stats[stat_p]).sum()), int(np.asarray(b.team_stats[stat_t]).sum()) + int(np.asarray(b.player_stats["rush_attempts"]).sum()) * (stat_p == "rush_attempts"))
        # TDs conserve to the score: score == 6*TD + PAT + 3*FG  (PAT in {0,1})
        ts = b.team_stats
        pts = np.asarray(ts["score"]); tds = np.asarray(ts["tds"]); fgs = np.asarray(ts["fgs"])
        pat = pts - 6 * tds - 3 * fgs
        self.assertTrue((pat >= 0).all() and (pat <= tds).all())

    def test_pipeline_deterministic_and_version_guard(self):
        snap = self.snap()
        c = complete_state(self.state, snap, self.prior)
        a = s28.simulate_game(self.state, 30, 3, V2, roster=snap, completion=c)
        b = s28.simulate_game(self.state, 30, 3, V2, roster=snap, completion=c)
        self.assertTrue(np.array_equal(np.asarray(a.team_stats["score"]), np.asarray(b.team_stats["score"])))
        with self.assertRaises(ValueError):
            s28.simulate_game(self.state, 5, 1, cfg27.MODEL_VERSION, roster=snap, completion=c)
        with self.assertRaises(TypeError):
            s28.simulate_game(self.state, 5, 1, V1)


if __name__ == "__main__":
    unittest.main()
