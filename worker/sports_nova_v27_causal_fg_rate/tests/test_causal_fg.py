"""V27 unit tests T1-T13.  The synthetic panel goes through the REAL make_state (V26's fixture); the estimator tests run on the REAL frozen drive table."""
import ast
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import scripts.sports_nova_v3_player_joint_walkforward_v1 as wf
from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_v23.simulator import _qb_shares, simulate_game as v23_simulate_game, MODEL_VERSION as V23_MV
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from worker.sports_nova_v25_role_aware.redistribution import assert_repair_invariants, repair_pregame_state
from worker.sports_nova_v26_active_skill_state import config as cfg26
from worker.sports_nova_v26_active_skill_state import simulator as v26
from worker.sports_nova_v26_active_skill_state.completion import complete_state
from worker.sports_nova_v26_active_skill_state.tests.test_completion import GAME, KICK, panel, roster_snapshot
from worker.sports_nova_v27_causal_fg_rate import causal_fg as fg
from worker.sports_nova_v27_causal_fg_rate import config as cfg
from worker.sports_nova_v27_causal_fg_rate import simulator as v27
from worker.sports_nova_v3.distributions import fit_distributions

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "worker" / "sports_nova_v27_causal_fg_rate"
FG_ABL = ROOT / "data" / "sports_nova_v3" / "FG_CAUSAL_ABLATION_V1"
STATS = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "scored_tds")


def same_arrays(a, b):
    return (a.player_ids == b.player_ids and a.team_ids == b.team_ids and a.player_team == b.player_team and np.array_equal(a.winner, b.winner)
            and all(np.array_equal(a.player_stats[k], b.player_stats[k]) for k in a.player_stats)
            and all(np.array_equal(a.team_stats[k], b.team_stats[k]) for k in a.team_stats))


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = fg._estimator()
        cls.d = fg._drives()

    def test_t2_estimator_reproduction_2026_w02(self):
        r = fg.causal_fg_rate(2026, 2)
        self.assertEqual((r.n_eligible, r.made), (21452, 4751))
        self.assertAlmostEqual(r.rate, 0.221471, places=6)
        self.assertEqual(r.estimator_sha256, "cc3a9f20a6761f6f18202d3c8a47af54f207834c0085d6936dbe7957041bd569")
        self.assertEqual(r.drive_sha256, "47b71f2b40866e0d1dcba2191ff0fd14a601777548b357076a1179e57017e50d")

    def test_t2_matches_all_60_preregistered_rates(self):
        pre = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
        for g in pre["PILOT"]["game_ids"]:
            s, w = fg.parse_game_key(g)
            self.assertAlmostEqual(fg.causal_fg_rate(s, w, g).rate, pre["PER_GAME_ESTIMATES"][g]["rate"], delta=1e-15)

    def test_t3_no_leakage(self):
        for s, w in [(2020, 1), (2022, 11), (2025, 22), (2026, 2)]:
            p = self.mod.prior_window(self.d, s, w)
            self.assertTrue((p["key"] < s * 100 + w).all())
            r = fg.causal_fg_rate(s, w)
            self.assertLess(r.max_key_used, s * 100 + w)
        # a same-week / future row injected into the table must not move the estimate
        extra = self.d[(self.d.key == 202211)].copy()
        extra["made_fg"] = 1
        extra["touchdowns"] = 0
        extra["plays"] = 9
        aug = pd.concat([self.d, extra, self.d[self.d.key > 202211].assign(made_fg=1, touchdowns=0, plays=9)], ignore_index=True)
        self.assertEqual(self.mod.estimate(aug, 2022, 11)["rate"], self.mod.estimate(self.d, 2022, 11)["rate"])

    def test_t4_window_semantics(self):
        p = self.mod.prior_window(self.d, 2026, 2)
        self.assertEqual((int(p.season.min()), int(p.season.max())), (2021, 2025))
        p = self.mod.prior_window(self.d, 2022, 11)                      # five prior season labels + current-season-to-date
        self.assertEqual(int(p.season.min()), 2017)
        self.assertEqual(int(p[p.season == 2022].week.max()), 10)
        self.assertEqual(sorted(p.season.unique().tolist()), [2017, 2018, 2019, 2020, 2021, 2022])
        self.assertEqual(fg.causal_fg_rate(2003, 4).window_min_season, 1999)   # max(1999, s-5)

    def test_t5_postseason_included(self):
        p = self.mod.prior_window(self.d, 2026, 2)
        el = p[(p.touchdowns == 0) & (p.plays >= 3)]
        self.assertGreater(int((el.game_type == "POST").sum()), 0)
        reg = el[el.game_type == "REG"]
        self.assertNotEqual(len(reg), len(el))
        self.assertEqual(len(el), fg.causal_fg_rate(2026, 2).n_eligible)

    def test_t6_numerator_is_the_field_goal_label(self):
        p = self.mod.prior_window(self.d, 2026, 2)
        el = p[(p.touchdowns == 0) & (p.plays >= 3)]
        label = el.possession_result.astype(str).str.upper().str.replace(" ", "_", regex=False) == "FIELD_GOAL"
        self.assertEqual(int(label.sum()), fg.causal_fg_rate(2026, 2).made)
        self.assertGreaterEqual(int(label.sum()), int((el.points_scored == 3).sum()))      # not defined through points_scored==3

    def test_t7_fail_closed(self):
        with self.assertRaises(fg.CausalFGError):
            fg.parse_game_key("not-a-game")
        with self.assertRaises(fg.CausalFGError):
            fg.causal_fg_rate(1999, 1)                                    # empty window: no prior blocks
        # estimator hash mismatch
        with mock.patch.dict(fg._STATE, {}, clear=True), mock.patch.object(fg, "ESTIMATOR_SHA256", "0" * 64):
            with self.assertRaises(fg.CausalFGError):
                fg.causal_fg_rate(2026, 2)
        # estimator file missing
        with mock.patch.dict(fg._STATE, {}, clear=True), mock.patch.object(fg, "ESTIMATOR_RELPATH", "scripts/does_not_exist.py"):
            with self.assertRaises(fg.CausalFGError):
                fg.causal_fg_rate(2026, 2)
        # drive table hash mismatch
        with mock.patch.dict(fg._STATE, {}, clear=True):
            mod = fg._estimator()
            with mock.patch.object(mod, "DRIVE_SHA", "0" * 64), self.assertRaises(fg.CausalFGError):
                fg._drives()
        fg._estimator(), fg._drives()                                      # state restored for the other tests

    def test_t7_no_fallback_to_the_default_in_production_code(self):
        for name in ("causal_fg.py", "simulator.py"):
            tree = ast.parse((PKG / name).read_text(encoding="utf-8"))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            self.assertNotIn("V23_DEFAULT_FG_RATE", names, name)
            consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
            self.assertNotIn(0.08, consts, name)
        self.assertEqual(fit_distributions([], KICK).fg_rate, cfg.V23_DEFAULT_FG_RATE)      # the value the harness forces is really the V23 default


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = panel()
        wf.PLAYER_SHA = "a" * 64
        cls.state = wf.make_state(GAME, cls.prior, KICK, None)

    def snap(self, statuses=None, game_status=None):
        extra = {"cutwr": ("HOM", "ACT"), "cutrb": ("HOM", "ACT"), "arrival": ("HOM", "ACT"), "newrookie": ("HOM", "ACT"), "dupe": ("HOM", "ACT")}
        s = roster_snapshot(self.state, extra=extra, statuses=statuses, game_status=game_status)
        return RosterSnapshot(roster=s.roster, game_status=s.game_status, roster_week=2, positions={**s.positions, "newrookie": ("WR", "WR")})

    def comp(self, snap):
        return complete_state(self.state, snap, self.prior)

    def test_t1_v26_pipeline_preserved_at_forced_default(self):
        snap = self.snap()
        c = self.comp(snap)
        a = v26.simulate_game(self.state, 60, 11, cfg26.MODEL_VERSION, roster=snap, completion=c)
        b = v27.simulate_game_forced_fg_rate(self.state, 60, 11, cfg.MODEL_VERSION, roster=snap, completion=c, fg_rate=cfg.V23_DEFAULT_FG_RATE)
        self.assertTrue(same_arrays(a, b))
        self.assertEqual((a.state_hash, a.status, a.game_id, a.seed, a.model), (b.state_hash, b.status, b.game_id, b.seed, b.model))
        self.assertEqual(b.model_version, cfg.MODEL_VERSION)
        for k in ("raw_state_hash", "repaired_state_hash", "completed_state_hash", "eligibility_policy_hash", "redistribution_policy_hash", "completion_policy_hash", "roster_week"):
            self.assertEqual(a.runtime[k], b.runtime[k], k)
        self.assertEqual(b.runtime["fg_rate_source"], "FORCED_REGRESSION_HARNESS")

    def test_t1_scoring_boundary_equals_plain_v23_at_forced_default(self):
        a = v23_simulate_game(self.state, 60, 11, V23_MV)
        b = v27.simulate_scoring_forced_fg_rate(self.state, 60, 11, cfg.MODEL_VERSION, fg_rate=cfg.V23_DEFAULT_FG_RATE)
        self.assertTrue(same_arrays(a, b))
        self.assertEqual((a.state_hash, a.status, a.model), (b.state_hash, b.status, b.model))

    def test_injection_is_the_only_difference(self):
        snap = self.snap()
        c = self.comp(snap)
        forced = v27.simulate_game_forced_fg_rate(self.state, 200, 5, cfg.MODEL_VERSION, roster=snap, completion=c, fg_rate=0.08)
        prod = v27.simulate_game(self.state, 200, 5, cfg.MODEL_VERSION, roster=snap, completion=c)
        r = fg.causal_fg_rate_for_game(GAME)
        self.assertEqual(prod.runtime["fg_rate"], r.rate)
        self.assertEqual(prod.runtime["fg_rate_source"], "CAUSAL_ESTIMATOR_A")
        self.assertEqual((prod.runtime["fg_n_eligible"], prod.runtime["fg_made"]), (21452, 4751))
        self.assertGreater(prod.team_stats["score"].mean(), forced.team_stats["score"].mean() + 2.0)      # the FG substitution has its (large) intended effect
        # T5-component isolation: the parameters handed to _run_one differ only in fg_rate
        p0 = fit_distributions([], self.state.as_of)
        import dataclasses
        p1 = dataclasses.replace(p0, fg_rate=r.rate)
        diff = [f.name for f in dataclasses.fields(p0) if getattr(p0, f.name) != getattr(p1, f.name)]
        self.assertEqual(diff, ["fg_rate"])

    def test_t7_estimator_failure_means_no_simulation(self):
        snap = self.snap()
        c = self.comp(snap)
        with mock.patch.object(v27, "causal_fg_rate_for_game", side_effect=fg.CausalFGError("boom")):
            with self.assertRaises(fg.CausalFGError):
                v27.simulate_game(self.state, 5, 1, cfg.MODEL_VERSION, roster=snap, completion=c)
            with self.assertRaises(fg.CausalFGError):
                v27.simulate_scoring(self.state, 5, 1, cfg.MODEL_VERSION)
        with self.assertRaises(ValueError):
            v27.simulate_game(self.state, 5, 1, cfg26.MODEL_VERSION, roster=snap, completion=c)         # wrong version string
        with self.assertRaises(TypeError):
            v27.simulate_game(self.state, 5, 1, cfg.MODEL_VERSION)                                        # roster/completion are required
        with self.assertRaises(CausalFGErrorAlias):
            v27._score(self.state, 5, 1, "0.08", {})                                                       # only an explicit float is accepted

    def test_t8_score_accounting(self):
        b = v27.simulate_scoring(self.state, 80, 3, cfg.MODEL_VERSION)
        hs, as_ = b.team_stats["score"][:, 0], b.team_stats["score"][:, 1]
        self.assertTrue(np.array_equal(b.team_stats["score"], b.team_stats["total"]))
        self.assertTrue(np.array_equal(b.winner, np.where(hs > as_, "HOME", np.where(as_ > hs, "AWAY", "TIE"))))

    def test_t9_mass_conservation_and_v26_audit_identity(self):
        snap = self.snap(statuses={"hrb0": ("HOM", "RES")})
        c = self.comp(snap)
        repaired, audit = v27.prepare_state(self.state, snap, self.prior)[1:]
        r26, a26 = v26.prepare_state(self.state, snap, self.prior)[1:]
        assert_repair_invariants(v26.prepare_state(self.state, snap, self.prior)[0].state, repaired, audit)
        self.assertEqual((audit.raw_state_hash, audit.repaired_state_hash, audit.summaries), (a26.raw_state_hash, a26.repaired_state_hash, a26.summaries))
        for s in audit.summaries:
            self.assertAlmostEqual(s["POST_REDISTRIBUTION_TOTAL"], s["PRE_REMOVAL_TOTAL"], places=9)
            self.assertAlmostEqual(s["REMOVED_MASS"], s["REDISTRIBUTED_MASS"], places=9)
            self.assertLess(s["MASS_TO_HELD_ROLES"], 1e-9)

    def test_t10_roster_integrity_and_t11_qb(self):
        snap = self.snap(statuses={"hrb0": ("HOM", "RES"), "arb1": ("AWY", "INA")}, game_status={"hwr1": "OUT"})
        c = self.comp(snap)
        b = v27.simulate_game(self.state, 150, 9, cfg.MODEL_VERSION, roster=snap, completion=c)
        ids = list(b.player_ids)
        for pid in ("hrb0", "arb1", "hwr1"):
            usage = sum(int(b.player_stats[k][:, ids.index(pid)].sum()) for k in ("pass_attempts", "targets", "rush_attempts"))
            self.assertEqual(usage, 0, pid)
        # QB selection inputs are those of V26 (same completed+repaired state)
        r27 = v27.prepare_state(self.state, snap, self.prior)[1]
        r26 = v26.prepare_state(self.state, snap, self.prior)[1]
        for t in ("HOM", "AWY"):
            i27, s27 = _qb_shares(r27, t)
            i26, s26 = _qb_shares(r26, t)
            self.assertEqual(list(i27), list(i26))
            self.assertTrue(np.array_equal(s27, s26))

    def test_t12_market_isolation(self):
        assert_market_free(cfg.CAUSAL_FG_POLICY)
        snap = self.snap()
        b = v27.simulate_game(self.state, 20, 2, cfg.MODEL_VERSION, roster=snap, completion=self.comp(snap))
        assert_market_free(json.loads(json.dumps(b.runtime, default=str)))
        forbidden = ("sports_market_interface", "sports_research", "market_book", "comparator", "trading_lab", "kalshi", "sportsbook")
        for f in PKG.glob("*.py"):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
                for m in mods:
                    self.assertFalse(any(x in m for x in forbidden), (f.name, m))

    def test_t13_frozen_versions_unchanged(self):
        import hashlib
        sha = lambda rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel, want in cfg26.V25_PACKAGE_FILE_SHA256.items():
            self.assertEqual(sha(rel), want, rel)
        meta = json.loads((FG_ABL / "run_meta.json").read_text())              # engine hashes recorded before any V27 code existed
        for rel, want in meta["engine_hashes_before"].items():
            self.assertEqual(sha(rel), want, rel)
        pre26 = json.loads((ROOT / "data/sports_nova_v3/sunday_2026_09_20/V26_PREREG/SPORTS_NOVA_V26_PREREG.json").read_text())
        for rel, want in pre26["PACKAGE_FILE_SHA256"].items():
            self.assertEqual(sha(rel), want, rel)
        # V27 imports frozen code; it does not contain any of it
        src = "".join(f.read_text(encoding="utf-8") for f in PKG.glob("*.py"))
        self.assertNotIn("def role_aware_values", src)
        self.assertNotIn("def complete_state", src)
        self.assertNotIn("def _run_one", src)


CausalFGErrorAlias = fg.CausalFGError

if __name__ == "__main__":
    unittest.main()
