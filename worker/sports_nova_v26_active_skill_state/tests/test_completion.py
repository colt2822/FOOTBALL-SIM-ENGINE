"""V26 tests T1-T10 (unit level).  The synthetic panel goes through the REAL make_state, so the machinery restatement is checked against the true code."""
import hashlib
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.sports_nova_v3_player_joint_walkforward_v1 as wf
from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_v23.simulator import _qb_shares, _state_hash
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from worker.sports_nova_v26_active_skill_state import config as cfg
from worker.sports_nova_v26_active_skill_state.completion import (
    MachineryMismatchError, complete_state, expected_active_skill, verify_machinery)
from worker.sports_nova_v26_active_skill_state.simulator import MODEL_VERSION, prepare_state, simulate_game

ROOT = Path(__file__).resolve().parents[3]
GAME = "2026_02_AWY_HOM"
KICK = datetime(2026, 9, 20, 17, tzinfo=timezone.utc)


def panel():
    rng = np.random.default_rng(7)
    rows = []

    def add(pid, team, pos, weeks, tgt=0, car=0, pa=0):
        for w in weeks:
            rows.append(dict(GAME_ID=f"2025_{w:02d}_{team}", CANONICAL_GAME_ID="x", SEASON=2025, WEEK=w, PLAYER_ID=pid, PLAYER_NAME=pid, POSITION=pos, TEAM=team,
                             PASS_ATTEMPTS=pa, COMPLETIONS=0, PASS_YARDS=pa * 6, PASS_TD=0, INTERCEPTIONS=0, RUSH_ATTEMPTS=car, RUSH_YARDS=car * 4, RUSH_TD=0,
                             TARGETS=tgt, RECEPTIONS=int(tgt * .6), RECEIVING_YARDS=int(tgt * 7), RECEIVING_TD=0))
    wk = range(1, 9)
    add("hqb", "HOM", "QB", wk, pa=33, car=3)
    add("aqb", "AWY", "QB", wk, pa=31, car=2)
    for i, (t, c) in enumerate([(2, 14), (1, 6), (0, 3)]):
        add(f"hrb{i}", "HOM", "RB", wk, tgt=t + 1, car=c)
        add(f"arb{i}", "AWY", "RB", wk, tgt=t + 1, car=c)
    for i in range(6):
        add(f"hwr{i}", "HOM", "WR", wk, tgt=8 - i)
        add(f"awr{i}", "AWY", "WR", wk, tgt=8 - i)
    add("hte0", "HOM", "TE", wk, tgt=4)
    add("ate0", "AWY", "TE", wk, tgt=4)
    for i in range(45):                                    # >40 positive-volume HOM skill rows -> `.head(40)` deterministically cuts the tail
        add(f"hdust{i:02d}", "HOM", "WR", [1], tgt=1)
    add("cutwr", "HOM", "WR", [8], tgt=0)                  # zero volume, latest team HOM -> provably truncated
    add("cutrb", "HOM", "RB", [8], tgt=0, car=1)          # tiny carry share, latest team HOM -> truncated
    add("arrival", "OTH", "WR", wk, tgt=7)                 # played only for OTH -> IDENTITY_JOIN when rostered at HOM
    add("dupe", "AWY", "TE", wk, tgt=3)                    # in AWY's state, but rostered at HOM
    df = pd.DataFrame(rows)
    df["_key"] = df.SEASON * 100 + df.WEEK
    return df


def roster_snapshot(state, extra=None, statuses=None, game_status=None):
    roster = {p.player_id: (p.team_id, "ACT") for p in state.players}
    for pid, (t, st) in (extra or {}).items():
        roster[pid] = (t, st)
    for pid, v in (statuses or {}).items():
        roster[pid] = v
    pos = {}
    for pid in roster:
        pos[pid] = ("QB" if pid.endswith("qb") else "RB" if "rb" in pid else "TE" if "te" in pid or pid == "dupe" else "WR", "None")
    return RosterSnapshot(roster=roster, game_status=game_status or {}, roster_week=2, positions=pos)


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = panel()
        wf.PLAYER_SHA = "a" * 64
        cls.state = wf.make_state(GAME, cls.prior, KICK, None)
        cls.ids = {p.player_id for p in cls.state.players}

    def snap(self, **kw):
        extra = {"cutwr": ("HOM", "ACT"), "cutrb": ("HOM", "ACT"), "arrival": ("HOM", "ACT"), "newrookie": ("HOM", "ACT"), "dupe": ("HOM", "ACT")}
        extra.update(kw.pop("extra", {}))
        s = roster_snapshot(self.state, extra=extra, **kw)
        return RosterSnapshot(roster=s.roster, game_status=s.game_status, roster_week=2, positions={**s.positions, "newrookie": ("WR", "WR")})

    # ---- premise: the fixture actually exercises the defect
    def test_fixture_reproduces_the_defect(self):
        self.assertNotIn("cutwr", self.ids)
        self.assertNotIn("arrival", self.ids)
        self.assertIn("dupe", self.ids)

    # ---- machinery equals the real make_state
    def test_machinery_reproduces_make_state(self):
        r = verify_machinery(self.state, self.prior)
        self.assertGreater(r["SHARES_CHECKED"], 50)
        self.assertLess(r["MAX_SHARE_ERR"], 1e-12)

    def test_machinery_mismatch_fails_closed(self):
        tampered = self.prior.copy()
        tampered.loc[tampered.PLAYER_ID == "hrb0", "RUSH_ATTEMPTS"] += 5
        with self.assertRaises(MachineryMismatchError):
            verify_machinery(self.state, tampered)

    # ---- T1 / T9 / T8
    def test_t1_active_history_players_are_added_with_machinery_share(self):
        c = complete_state(self.state, self.snap(), self.prior)
        ids = {p.player_id for p in c.state.players}
        self.assertIn("cutwr", ids)
        self.assertIn("cutrb", ids)
        self.assertIn("arrival", ids)
        rb = next(r for r in c.rows if r["PLAYER_ID"] == "cutrb")
        self.assertEqual((rb["REASON"], rb["ACTION"]), ("HISTORY_FILTER", cfg.ACTION_ADDED))
        car = dict(zip(c.state.home.carry_shares.player_ids, c.state.home.carry_shares.shares))
        total = sum(self.prior[(self.prior.TEAM == "HOM") & (self.prior.PLAYER_ID.isin(self.ids | {"cutrb"}))].RUSH_ATTEMPTS)
        self.assertGreater(car["cutrb"], 0)
        self.assertAlmostEqual(sum(c.state.home.carry_shares.shares) + c.state.home.carry_shares.residual_share, 1.0, places=9)
        old = dict(zip(self.state.home.carry_shares.player_ids, self.state.home.carry_shares.shares))
        for pid, s in old.items():
            self.assertAlmostEqual(car[pid], s, places=12)                     # existing shares untouched
        self.assertLess(c.state.home.carry_shares.residual_share, self.state.home.carry_shares.residual_share)   # the share came out of the residual bucket

    def test_t9_no_history_is_blocked_not_weighted(self):
        c = complete_state(self.state, self.snap(), self.prior)
        self.assertNotIn("newrookie", {p.player_id for p in c.state.players})
        r = next(r for r in c.rows if r["PLAYER_ID"] == "newrookie")
        self.assertEqual((r["REASON"], r["ACTION"]), ("NO_HISTORY", cfg.ACTION_BLOCKED_COLD_START))
        self.assertNotIn("newrookie", c.state.home.target_shares.player_ids)

    def test_identity_join_arrival_gets_the_machinery_share_zero(self):
        c = complete_state(self.state, self.snap(), self.prior)
        r = next(r for r in c.rows if r["PLAYER_ID"] == "arrival")
        self.assertEqual((r["REASON"], r["ACTION"]), ("IDENTITY_JOIN", cfg.ACTION_ADDED_ZERO_SHARE))
        tgt = dict(zip(c.state.home.target_shares.player_ids, c.state.home.target_shares.shares))
        self.assertEqual(tgt["arrival"], 0.0)
        self.assertGreater(r["TARGETS_ALL_TEAMS"], 0)                          # the at-stake volume is reported, not simulated

    def test_t8_no_cross_team_insertion(self):
        c = complete_state(self.state, self.snap(), self.prior)
        r = next(r for r in c.rows if r["PLAYER_ID"] == "dupe")
        self.assertEqual(r["ACTION"], cfg.ACTION_BLOCKED_DUPLICATE)
        self.assertEqual(sum(1 for p in c.state.players if p.player_id == "dupe"), 1)
        self.assertEqual(next(p for p in c.state.players if p.player_id == "dupe").team_id, "AWY")
        for p in c.state.players:
            if p.player_id in {"cutwr", "cutrb", "arrival"}:
                self.assertEqual(p.team_id, "HOM")

    # ---- T2
    def test_t2_out_and_ineligible_are_never_added(self):
        snap = self.snap(extra={"cutwr": ("HOM", "RES"), "cutrb": ("HOM", "CUT"), "arrival": ("AWY", "ACT")}, game_status={"newrookie": "OUT"})
        exp = expected_active_skill(snap, "HOM", frozenset())
        for pid in ("cutwr", "cutrb", "arrival", "newrookie"):
            self.assertNotIn(pid, exp)
        c = complete_state(self.state, snap, self.prior)
        self.assertFalse({"cutwr", "cutrb", "newrookie"} & {p.player_id for p in c.state.players})
        exp2 = expected_active_skill(self.snap(), "HOM", frozenset({"cutwr"}))
        self.assertNotIn("cutwr", exp2)                                        # panel M1_AVAILABILITY OUT counts as out too

    def test_nothing_to_add_returns_same_object(self):
        snap = roster_snapshot(self.state)
        c = complete_state(self.state, snap, self.prior)
        self.assertIs(c.state, self.state)
        self.assertEqual(c.raw_state_hash, c.completed_state_hash)
        self.assertFalse(c.changed)

    # ---- T3
    def test_t3_qb_inputs_identical(self):
        c = complete_state(self.state, self.snap(), self.prior)
        for t in ("HOM", "AWY"):
            a, b = _qb_shares(self.state, t), _qb_shares(c.state, t)
            self.assertEqual((tuple(map(str, a[0])), tuple(map(float, a[1]))), (tuple(map(str, b[0])), tuple(map(float, b[1]))))

    # ---- T4 / T10
    def test_t4_v25_package_is_byte_identical(self):
        for rel, want in cfg.V25_PACKAGE_FILE_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), want, rel)

    def test_v26_does_not_import_or_edit_v25_logic(self):
        src = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "worker/sports_nova_v26_active_skill_state").glob("*.py"))
        self.assertNotIn("redistribution import role_aware_values", src)
        self.assertNotIn("def role_aware_values", src)

    # ---- T5 / T6 / T7
    def test_t5_t6_t7_conservation_accounting_market(self):
        # one RB and one WR per team is RES so every carry/target pool has removed mass (V25 hands unchanged pools to V23 untouched, residual bucket included;
        # the real slate has removed mass in all 56 pools, which is where the accounting identity is defined)
        snap = self.snap(statuses={"hwr5": ("HOM", "RES"), "awr5": ("AWY", "RES"), "hrb2": ("HOM", "RES"), "arb2": ("AWY", "RES")})
        c, repaired, audit = prepare_state(self.state, snap, self.prior)             # V25 invariants (mass conserved, held roles unchanged) assert inside
        for s in audit.summaries:
            self.assertAlmostEqual(s["POST_REDISTRIBUTION_TOTAL"], s["PRE_REMOVAL_TOTAL"], places=9)
            self.assertAlmostEqual(s["REMOVED_MASS"], s["REDISTRIBUTED_MASS"], places=9)
        assert_market_free(c.state.model_dump(mode="json"))
        b = simulate_game(self.state, 60, 11, MODEL_VERSION, roster=snap, completion=c)
        self.assertEqual(b.state_hash, _state_hash(self.state))                       # the RAW input hash is what replay checks compare
        team_of = dict(b.player_team)
        for j, tid in enumerate(b.team_ids):
            idx = [i for i, p in enumerate(b.player_ids) if team_of[p] == tid]
            self.assertTrue((b.player_stats["rush_attempts"][:, idx].sum(axis=1) == b.team_stats["rush_attempts"][:, j]).all())
            self.assertTrue((b.player_stats["targets"][:, idx].sum(axis=1) == b.team_stats["pass_attempts"][:, j]).all())
        self.assertTrue({"cutwr", "cutrb", "arrival"} <= set(b.player_ids))

    def test_wrong_raw_state_rejected(self):
        c = complete_state(self.state, self.snap(), self.prior)
        with self.assertRaises(ValueError):
            simulate_game(c.state, 5, 1, MODEL_VERSION, roster=self.snap(), completion=c)

    # ---- diagnostic-only counterfactual stays valid and never leaks into the official basis
    def test_portable_counterfactual_is_valid_and_distinct(self):
        off = complete_state(self.state, self.snap(), self.prior)
        cf = complete_state(self.state, self.snap(), self.prior, share_basis=cfg.SHARE_BASIS_COUNTERFACTUAL)
        tgt = dict(zip(cf.state.home.target_shares.player_ids, cf.state.home.target_shares.shares))
        self.assertGreater(tgt["arrival"], 0.0)
        self.assertLessEqual(sum(cf.state.home.target_shares.shares) + cf.state.home.target_shares.residual_share, 1.0 + 1e-9)
        self.assertEqual(dict(zip(off.state.home.target_shares.player_ids, off.state.home.target_shares.shares))["arrival"], 0.0)


if __name__ == "__main__":
    unittest.main()
